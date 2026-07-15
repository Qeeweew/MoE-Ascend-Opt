#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/xwj/workspace/MoE-Ascend-Opt
MODEL=/mnt/models/MiniMax-M2.5-AWQ-4bit
DATASET=/home/xwj/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
CPU_BASELINE=${ROOT}/docs/bench_results/raw/minimax_m2_5_closure_seed1_tp1/cpu_q4
PORT=${PORT:-31042}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_DIR=${RESULT_DIR:-${ROOT}/logs/cache_k3072_policy_v2_${RUN_ID}}

mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${RESULT_DIR}/run.log") 2>&1

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export NANOVLLM_TP_SIZE=2
export NANOVLLM_TP_THREADS_PER_NODE=${NANOVLLM_TP_THREADS_PER_NODE:-20}
unset MOE_ASCEND_NPU_DISABLE || true

SERVER_PID=

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        sleep 5
        kill -9 "${SERVER_PID}" 2>/dev/null || true
    fi
    local port_pids
    port_pids=$(lsof -t -i:"${PORT}" 2>/dev/null || true)
    if [[ -n "${port_pids}" ]]; then
        kill -9 ${port_pids} 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local deadline=$((SECONDS + 1200))
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "server exited before becoming ready"
            tail -100 "${RESULT_DIR}/server.log" || true
            return 1
        fi
        sleep 5
    done
    echo "server readiness timeout"
    tail -100 "${RESULT_DIR}/server.log" || true
    return 1
}

run_benchmark() {
    local concurrency=$1
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${PORT}" \
        --dataset-name sharegpt \
        --dataset-path "${DATASET}" \
        --num-prompts 32 \
        --sharegpt-output-len 320 \
        --max-concurrency "${concurrency}" \
        --warmup-requests 32 \
        --seed 1 \
        --output-file "${RESULT_DIR}/sharegpt_c${concurrency}.jsonl" \
        2>&1 | tee "${RESULT_DIR}/bench_c${concurrency}.log"
}

prime_cache() {
    echo "=== Untimed decode cache priming: concurrency=1 prompts=20 ==="
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://127.0.0.1:${PORT}" \
        --dataset-name sharegpt \
        --dataset-path "${DATASET}" \
        --num-prompts 20 \
        --sharegpt-output-len 320 \
        --max-concurrency 1 \
        --warmup-requests 0 \
        --seed 1 \
        --output-file "${RESULT_DIR}/cache_prime_c1.jsonl" \
        2>&1 | tee "${RESULT_DIR}/cache_prime_c1.log"
}

echo "run_id=${RUN_ID}"
echo "result_dir=${RESULT_DIR}"
echo "model=${MODEL}"
echo "dataset=${DATASET}"
echo "device=${ASCEND_RT_VISIBLE_DEVICES} cpu_tp=${NANOVLLM_TP_SIZE} cache_k=3072"

sglang serve \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --model-path "${MODEL}" \
    --trust-remote-code \
    --tp-size 1 \
    --attention-backend ascend \
    --cuda-graph-bs 1 2 4 8 \
    --enable-moe-expert-cache \
    --moe-expert-cache-size 3072 \
    --moe-expert-cache-swap-per-update 64 \
    --moe-expert-cache-update-interval 16 \
    --moe-expert-cache-warmup-steps 16 \
    --moe-expert-cache-decay 0.95 \
    >"${RESULT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "server_pid=${SERVER_PID}"
wait_for_server

curl -fsS "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Reply with exactly: OK\",\"max_tokens\":4,\"temperature\":0}" \
    >"${RESULT_DIR}/smoke_response.json"

prime_cache
run_benchmark 4
run_benchmark 8

python - "${CPU_BASELINE}" "${RESULT_DIR}" <<'PY'
import json
import pathlib
import sys

baseline_dir = pathlib.Path(sys.argv[1])
result_dir = pathlib.Path(sys.argv[2])
summary = {}
for concurrency in (4, 8):
    baseline = json.loads((baseline_dir / f"sharegpt_c{concurrency}.jsonl").read_text().splitlines()[-1])
    result = json.loads((result_dir / f"sharegpt_c{concurrency}.jsonl").read_text().splitlines()[-1])
    cpu = float(baseline["output_throughput"])
    cache = float(result["output_throughput"])
    gain = cache / cpu - 1.0
    summary[str(concurrency)] = {
        "cpu_output_throughput": cpu,
        "cache_output_throughput": cache,
        "throughput_gain": gain,
        "target_met": gain > 0.50,
        "mean_tpot_ms": result.get("mean_tpot_ms"),
        "median_tpot_ms": result.get("median_tpot_ms"),
    }
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if not all(item["target_met"] for item in summary.values()):
    raise SystemExit(2)
PY

echo "K=3072 c4/c8 both exceeded the CPU baseline by 50%."
