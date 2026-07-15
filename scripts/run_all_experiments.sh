#!/bin/bash
# ============================================================
# MoE-Ascend-Opt 结题实验全部自动化脚本
# 运行方式: nohup bash scripts/run_all_experiments.sh > logs/experiments.log 2>&1 &
# ============================================================
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="/home/xwj/workspace/MoE-Ascend-Opt/logs/experiments_${TIMESTAMP}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "MoE-Ascend-Opt 结题实验开始: $(date)"
echo "日志文件: $LOGFILE"
echo "=========================================="

# ==================== 固定参数 ====================
MODEL_MINIMAX="/mnt/models/MiniMax-M2.5-AWQ-4bit"
MODEL_QWEN="/mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32"
DATASET="/home/xwj/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
BASE_RESULT="/home/xwj/workspace/MoE-Ascend-Opt/docs/bench_results/raw/minimax_m2_5_closure_seed1_tp1"

# S_slot = 7.59 MB (from Python calculation)
# Experts per layer = 256
# Small K=256(1层, 2.45GB), Medium K=1536(6层, 11.94GB), Large K=3072(12层, 23.33GB)

declare -A C2_K
C2_K[small]=256
C2_K[medium]=1536
C2_K[large]=3072

declare -A C1_L
C1_L[small]=1
C1_L[medium]=6
C1_L[large]=12

CONCURRENCIES=(1 4 8)
PORT_BASE=31000
export NANOVLLM_TP_SIZE=2
export NANOVLLM_TP_THREADS_PER_NODE=${NANOVLLM_TP_THREADS_PER_NODE:-20}

# ==================== 工具函数 ====================

kill_server() {
    local port=$1
    # Find and kill process on this port
    local pids=$(lsof -t -i:${port} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing processes on port ${port}: $pids"
        kill -9 $pids 2>/dev/null || true
    fi
    sleep 3
    # Also kill any remaining sglang processes
    ps aux | grep "sglang serve" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    local port=$1
    local timeout=${2:-600}
    echo "Waiting for server on port $port (timeout=${timeout}s)..."
    local start=$(date +%s)
    while true; do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            local elapsed=$(($(date +%s) - start))
            echo "Server ready on port $port after ${elapsed}s"
            return 0
        fi
        local elapsed=$(($(date +%s) - start))
        if [ $elapsed -ge $timeout ]; then
            echo "TIMEOUT: Server on port $port not ready after ${timeout}s"
            return 1
        fi
        sleep 5
    done
}

run_benchmark() {
    local result_dir=$1
    local port=$2
    local concurrency=$3
    local warmup=${4:-1}
    local extra_args=${5:-}

    echo "=== Benchmark: concurrency=$concurrency warmup=$warmup ==="
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${port}" \
        --dataset-name sharegpt --dataset-path "$DATASET" \
        --num-prompts 32 --sharegpt-output-len 320 \
        --max-concurrency "$concurrency" --warmup-requests "$warmup" --seed 1 \
        --output-file "${result_dir}/sharegpt_c${concurrency}.jsonl" \
        $extra_args \
        2>&1 | tee "${result_dir}/bench_c${concurrency}.log"
    echo "Benchmark concurrency=$concurrency done."
}

# ==================== 实验 A: 模型适配 ====================
run_experiment_a_minimax() {
    echo ""
    echo "==================== 实验A: MiniMax M2.5 Smoke Test ===================="
    local port=$((PORT_BASE + 0))

    sglang serve \
        --host 127.0.0.1 --port "$port" \
        --model-path "$MODEL_MINIMAX" --trust-remote-code \
        --tp-size 1 --attention-backend ascend \
        --cuda-graph-bs 1 2 4 8 \
        --enable-moe-offload --moe-offload-start-layer 0 \
        --moe-offload-quant-type q4_0 \
        >"${BASE_RESULT}/cpu_q4/server_smoke.log" 2>&1 &

    wait_for_server $port 600 || { echo "MiniMax smoke server failed"; return 1; }

    # Short request test
    local smoke_result="${BASE_RESULT}/cpu_q4/smoke_response.json"
    curl -fsS "http://127.0.0.1:${port}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_MINIMAX}\",\"prompt\":\"Reply with exactly: OK\",\"max_tokens\":4,\"temperature\":0}" \
        >"$smoke_result" 2>&1

    echo "MiniMax smoke response:"
    cat "$smoke_result"
    kill_server $port
    echo "实验A MiniMax done."
}

run_experiment_a_qwen() {
    echo ""
    echo "==================== 实验A: Qwen3-30B-A3B Smoke Test ===================="
    local port=$((PORT_BASE + 1))

    sglang serve \
        --host 127.0.0.1 --port "$port" \
        --model-path "$MODEL_QWEN" --trust-remote-code \
        --tp-size 1 --attention-backend ascend \
        --cuda-graph-bs 1 2 4 8 \
        >"${BASE_RESULT}/qwen3_smoke.log" 2>&1 &

    wait_for_server $port 300 || { echo "Qwen3 smoke server failed"; return 1; }

    curl -fsS "http://127.0.0.1:${port}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_QWEN}\",\"prompt\":\"Reply with exactly: OK\",\"max_tokens\":4,\"temperature\":0}" \
        >"${BASE_RESULT}/qwen3_smoke_response.json" 2>&1

    echo "Qwen3 smoke response:"
    cat "${BASE_RESULT}/qwen3_smoke_response.json"
    kill_server $port
    echo "实验A Qwen3 done."
}

# ==================== 实验 B: 显存充足基线 (Qwen3) ====================
run_experiment_b() {
    echo ""
    echo "==================== 实验B: Qwen3 显存充足基线对照 ===================="

    for mode in B0 B1; do
        local port=$((PORT_BASE + 2))
        local result_dir="${BASE_RESULT}/../qwen3_closure_seed1_tp1/${mode}"
        mkdir -p "$result_dir"

        if [ "$mode" == "B0" ]; then
            echo "--- B0: 关闭项目 W4A16 MoE (使用官方路径) ---"
            # B0: Disable project MoE optimizations
            export MOE_ASCEND_NPU_DISABLE=1
            sglang serve \
                --host 127.0.0.1 --port "$port" \
                --model-path "$MODEL_QWEN" --trust-remote-code \
                --tp-size 1 --attention-backend ascend \
                --cuda-graph-bs 1 2 4 8 \
                >"${result_dir}/server.log" 2>&1 &
        else
            echo "--- B1: 启用项目 W4A16 Fused MoE ---"
            unset MOE_ASCEND_NPU_DISABLE
            sglang serve \
                --host 127.0.0.1 --port "$port" \
                --model-path "$MODEL_QWEN" --trust-remote-code \
                --tp-size 1 --attention-backend ascend \
                --cuda-graph-bs 1 2 4 8 \
                >"${result_dir}/server.log" 2>&1 &
        fi

        wait_for_server $port 300 || { echo "${mode} server failed"; kill_server $port; continue; }

        for c in "${CONCURRENCIES[@]}"; do
            run_benchmark "$result_dir" "$port" "$c" 1
        done

        kill_server $port
        unset MOE_ASCEND_NPU_DISABLE
        echo "实验B ${mode} done."
    done
}

# ==================== 实验 C: 显存受限异构路径 ====================

# C0: Full CPU Q4
run_experiment_c0() {
    echo ""
    echo "==================== 实验C0: 全量 CPU Q4 卸载 ===================="
    local port=$((PORT_BASE + 0))
    local result_dir="${BASE_RESULT}/cpu_q4"
    mkdir -p "$result_dir"

    sglang serve \
        --host 127.0.0.1 --port "$port" \
        --model-path "$MODEL_MINIMAX" --trust-remote-code \
        --tp-size 1 --attention-backend ascend \
        --cuda-graph-bs 1 2 4 8 \
        --enable-moe-offload --moe-offload-start-layer 0 \
        --moe-offload-quant-type q4_0 \
        >"${result_dir}/server.log" 2>&1 &

    wait_for_server $port 600 || { echo "C0 server failed"; kill_server $port; return 1; }

    for c in "${CONCURRENCIES[@]}"; do
        run_benchmark "$result_dir" "$port" "$c" 32
    done

    kill_server $port
    echo "实验C0 done."
}

# C1: Fixed layer
run_experiment_c1() {
    local size=$1  # small/medium/large
    local L=${C1_L[$size]}

    echo ""
    echo "==================== 实验C1 (${size}): 固定层驻留 L=${L} ===================="
    local port=$((PORT_BASE + 10 + L))
    local result_dir="${BASE_RESULT}/layer_l${L}"
    mkdir -p "$result_dir"

    sglang serve \
        --host 127.0.0.1 --port "$port" \
        --model-path "$MODEL_MINIMAX" --trust-remote-code \
        --tp-size 1 --attention-backend ascend \
        --cuda-graph-bs 1 2 4 8 \
        --enable-moe-offload --moe-offload-start-layer "$L" \
        --moe-offload-quant-type q4_0 \
        >"${result_dir}/server.log" 2>&1 &

    wait_for_server $port 600 || { echo "C1 L=${L} server failed"; kill_server $port; return 1; }

    for c in "${CONCURRENCIES[@]}"; do
        run_benchmark "$result_dir" "$port" "$c" 1
    done

    kill_server $port
    echo "实验C1 L=${L} done."
}

# C2: Dynamic cache
run_experiment_c2() {
    local size=$1  # small/medium/large
    local K=${C2_K[$size]}
    local SWAP=64
    local BOOTSTRAP=64

    echo ""
    echo "==================== 实验C2 (${size}): 动态缓存 K=${K} ===================="
    local port=$((PORT_BASE + 30 + K / 256))
    local result_dir="${BASE_RESULT}/cache_k${K}"
    mkdir -p "$result_dir"

    sglang serve \
        --host 127.0.0.1 --port "$port" \
        --model-path "$MODEL_MINIMAX" --trust-remote-code \
        --tp-size 1 --attention-backend ascend \
        --cuda-graph-bs 1 2 4 8 \
        --enable-moe-expert-cache \
        --moe-expert-cache-size "$K" \
        --moe-expert-cache-swap-per-update "$SWAP" \
        --moe-expert-cache-bootstrap-fill "$BOOTSTRAP" \
        --moe-expert-cache-update-interval 16 \
        --moe-expert-cache-warmup-steps 16 \
        --moe-expert-cache-decay 0.95 \
        >"${result_dir}/server.log" 2>&1 &

    wait_for_server $port 600 || { echo "C2 K=${K} server failed"; kill_server $port; return 1; }

    for c in "${CONCURRENCIES[@]}"; do
        run_benchmark "$result_dir" "$port" "$c" 32
    done

    kill_server $port
    echo "实验C2 K=${K} done."
}

# ==================== 实验 D: 长上下文与稳定性 ====================
run_experiment_d_long_context() {
    echo ""
    echo "==================== 实验D: 长上下文测试 ===================="
    local K=1536  # Use medium cache
    local port=$((PORT_BASE + 50))
    local result_dir="${BASE_RESULT}/long_context"
    mkdir -p "$result_dir"

    sglang serve \
        --host 127.0.0.1 --port "$port" \
        --model-path "$MODEL_MINIMAX" --trust-remote-code \
        --tp-size 1 --attention-backend ascend \
        --cuda-graph-bs 1 2 4 8 \
        --enable-moe-expert-cache \
        --moe-expert-cache-size "$K" \
        --moe-expert-cache-swap-per-update 8 \
        --moe-expert-cache-update-interval 16 \
        --moe-expert-cache-warmup-steps 16 \
        --moe-expert-cache-decay 0.95 \
        >"${result_dir}/server.log" 2>&1 &

    wait_for_server $port 600 || { echo "Long context server failed"; kill_server $port; return 1; }

    # 8K context
    echo "--- 8K long context ---"
    curl -fsS "http://127.0.0.1:${port}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_MINIMAX}\",\"prompt\":\"$(python3 -c "print('Long context test. ' * 1500)")Reply with: OK\",\"max_tokens\":512,\"temperature\":0}" \
        >"${result_dir}/long_8k_response.json" 2>&1
    echo "8K done: $(cat ${result_dir}/long_8k_response.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("usage",{}))' 2>/dev/null)"

    # 32K context
    echo "--- 32K long context ---"
    curl -fsS "http://127.0.0.1:${port}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL_MINIMAX}\",\"prompt\":\"$(python3 -c "print('Long context test. ' * 6000)")Reply with: OK\",\"max_tokens\":512,\"temperature\":0}" \
        >"${result_dir}/long_32k_response.json" 2>&1
    echo "32K done: $(cat ${result_dir}/long_32k_response.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("usage",{}))' 2>/dev/null)"

    kill_server $port
    echo "实验D long context done."
}

run_experiment_d_stability() {
    echo ""
    echo "==================== 实验D: 稳定性压力测试 ===================="
    local K=1536
    local port=$((PORT_BASE + 51))
    local result_dir="${BASE_RESULT}/stability_run"
    mkdir -p "$result_dir"

    sglang serve \
        --host 127.0.0.1 --port "$port" \
        --model-path "$MODEL_MINIMAX" --trust-remote-code \
        --tp-size 1 --attention-backend ascend \
        --cuda-graph-bs 1 2 4 8 \
        --enable-moe-expert-cache \
        --moe-expert-cache-size "$K" \
        --moe-expert-cache-swap-per-update 8 \
        --moe-expert-cache-update-interval 16 \
        --moe-expert-cache-warmup-steps 16 \
        --moe-expert-cache-decay 0.95 \
        >"${result_dir}/server.log" 2>&1 &

    wait_for_server $port 600 || { echo "Stability server failed"; kill_server $port; return 1; }

    # 200 requests at concurrency 4
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${port}" \
        --dataset-name sharegpt --dataset-path "$DATASET" \
        --num-prompts 200 --sharegpt-output-len 320 \
        --max-concurrency 4 --warmup-requests 0 --seed 1 \
        --output-file "${result_dir}/stability_200.jsonl" \
        2>&1 | tee "${result_dir}/bench_stability.log"

    kill_server $port
    echo "实验D stability done."
}

# ==================== 实验 E: 正确性 ====================
run_experiment_e() {
    echo ""
    echo "==================== 实验E: 正确性验证 ===================="
    local result_dir="${BASE_RESULT}/correctness"
    mkdir -p "$result_dir"

    cd /home/xwj/workspace/MoE-Ascend-Opt/moe_ascend_npu

    echo "--- Running test_cache_policy.py ---"
    python tests/test_cache_policy.py 2>&1 | tee "${result_dir}/test_cache_policy.log"

    echo "--- Running run_all.sh ---"
    bash tests/run_all.sh 2>&1 | tee "${result_dir}/test_run_all.log"

    cd -
    echo "实验E done."
}

# ==================== 主流程 ====================
main() {
    echo ""
    echo "=========================================="
    echo "开始执行实验流水线"
    echo "=========================================="

    # 确保目录存在
    mkdir -p "${BASE_RESULT}"
    mkdir -p /home/xwj/workspace/MoE-Ascend-Opt/logs

    # 清理残留进程
    kill_server 0  # Kill all sglang
    sleep 3

    # ===== 实验 A =====
    run_experiment_a_minimax
    run_experiment_a_qwen

    # ===== 实验 C0 =====
    run_experiment_c0

    # ===== 实验 C1 (固定层) =====
    for size in small medium large; do
        run_experiment_c1 "$size"
    done

    # ===== 实验 C2 (动态缓存) =====
    for size in small medium large; do
        run_experiment_c2 "$size"
    done

    # ===== 实验 B (Qwen3 基线) =====
    run_experiment_b

    # ===== 实验 D =====
    run_experiment_d_long_context
    run_experiment_d_stability

    # ===== 实验 E =====
    run_experiment_e

    echo ""
    echo "=========================================="
    echo "全部实验完成: $(date)"
    echo "结果目录: ${BASE_RESULT}"
    echo "日志文件: $LOGFILE"
    echo "=========================================="
}

main "$@"
