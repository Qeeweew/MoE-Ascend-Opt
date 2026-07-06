#!/bin/bash
#
# MoE-Ascend-Opt 端到端 Decoding 性能测试
# 测试两个模型：
#   1) MiniMax M2.5 AWQ (显存充足，纯 NPU 推理，TP=4)
#   2) GLM-5.1 AWQ (显存不足，CPU 卸载，TP=1)
#
# Batch sizes: 1, 2, 4, 8
# Input len: 1024
# Output len: 256
#
# Usage: bash bench_e2e.sh [minimax|glm|all]
#   默认 all: 先跑 MiniMax，再跑 GLM
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}"
mkdir -p "$RESULT_DIR"

DATASET_PATH="/mnt/nvme0n1/xwj-data/dataset/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"

# ============ Common benchmark args ============
INPUT_LEN=1024
OUTPUT_LEN=256
BATCH_SIZES="1 2 4 8"
CUDA_GRAPH_BS="1 2 4 8"

# Default: run all
RUN_MINIMAX=true
RUN_GLM=true

case "${1:-all}" in
    minimax) RUN_GLM=false ;;
    glm)     RUN_MINIMAX=false ;;
    all)     ;;
    *)       echo "Usage: $0 [minimax|glm|all]"; exit 1 ;;
esac

echo "===================================================="
echo "  MoE-Ascend-Opt E2E Decoding Benchmark"
echo "  Model: MiniMax-M2.5 (Run=${RUN_MINIMAX})"
echo "  Model: GLM-5.1     (Run=${RUN_GLM})"
echo "  Batch sizes: ${BATCH_SIZES}"
echo "  Input len:   ${INPUT_LEN}"
echo "  Output len:  ${OUTPUT_LEN}"
echo "===================================================="
echo ""

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# ================================================================
# 1. MiniMax M2.5 AWQ: 纯 NPU 推理 (TP=4, cards 4,5,6,7)
# ================================================================
if $RUN_MINIMAX; then
    echo "[$(timestamp)] ========== MiniMax M2.5 AWQ Benchmark =========="
    MODEL_PATH="/mnt/nvme0n1/xwj-data/models/MiniMax-M2.5-AWQ-4bit"
    RESULT_FILE="${RESULT_DIR}/result_minimax_m2.5_awq.jsonl"

    export SGLANG_SET_CPU_AFFINITY=1
    export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

    echo "[$(timestamp)] Starting MiniMax M2.5 server + benchmark..."
    echo "  Model:  ${MODEL_PATH}"
    echo "  Cards:  4,5,6,7 (TP=4)"
    echo "  Result: ${RESULT_FILE}"
    echo ""

    python3 -m sglang.bench_one_batch_server \
        --port 30001 \
        --model-path "${MODEL_PATH}" \
        --trust-remote-code \
        --tp-size 4 \
        --attention-backend ascend \
        --sampling-backend ascend \
        --cuda-graph-bs ${CUDA_GRAPH_BS} \
        --batch-size ${BATCH_SIZES} \
        --input-len ${INPUT_LEN} \
        --output-len ${OUTPUT_LEN} \
        --dataset-path "${DATASET_PATH}" \
        --result-filename "${RESULT_FILE}"

    echo ""
    echo "[$(timestamp)] MiniMax M2.5 benchmark completed, waiting for NPU memory release..."
    sleep 10
    echo ""

    # Print summary
    if [ -f "${RESULT_FILE}" ]; then
        echo "---------- MiniMax M2.5 Results ----------"
        python3 -c "
import json
with open('${RESULT_FILE}') as f:
    lines = f.readlines()
print(f'Records: {len(lines)}')
for line in lines:
    d = json.loads(line)
    print(f\"  bs={d.get('batch_size','?')}, il={d.get('input_len','?')}, ol={d.get('output_len','?')}, output_throughput={d.get('output_throughput','?')}, latency={d.get('latency','?')}\")
"
        echo ""
    fi
fi

# ================================================================
# 2. GLM-5.1 AWQ: CPU 卸载推理 (TP=1, card 4 only)
# ================================================================
if $RUN_GLM; then
    echo "[$(timestamp)] ========== GLM-5.1 AWQ CPU Offload Benchmark =========="
    MODEL_PATH="/mnt/nvme0n1/xwj-data/models/GLM-5.1-AWQ-4bit"
    RESULT_FILE="${RESULT_DIR}/result_glm5.1_awq_offload.jsonl"

    export SGLANG_SET_CPU_AFFINITY=1
    export ASCEND_RT_VISIBLE_DEVICES=4
    export NANOVLLM_TP_THREADS_PER_NODE=20
    export NANOVLLM_TP_SIZE=8
    export OMP_NUM_THREADS=20

    echo "[$(timestamp)] Starting GLM-5.1 server + benchmark (CPU offload)..."
    echo "  Model:  ${MODEL_PATH}"
    echo "  Cards:  4 (TP=1 + CPU offload)"
    echo "  Result: ${RESULT_FILE}"
    echo ""

    python3 -m sglang.bench_one_batch_server \
        --port 30002 \
        --model-path "${MODEL_PATH}" \
        --trust-remote-code \
        --tp-size 1 \
        --attention-backend ascend \
        --sampling-backend ascend \
        --cuda-graph-bs ${CUDA_GRAPH_BS} \
        --enable-moe-offload \
        --moe-offload-quant-type q4_0 \
        --batch-size ${BATCH_SIZES} \
        --input-len ${INPUT_LEN} \
        --output-len ${OUTPUT_LEN} \
        --dataset-path "${DATASET_PATH}" \
        --result-filename "${RESULT_FILE}"

    echo ""
    echo "[$(timestamp)] GLM-5.1 benchmark completed."
    echo ""

    # Print summary
    if [ -f "${RESULT_FILE}" ]; then
        echo "---------- GLM-5.1 Results ----------"
        python3 -c "
import json
with open('${RESULT_FILE}') as f:
    lines = f.readlines()
print(f'Records: {len(lines)}')
for line in lines:
    d = json.loads(line)
    print(f\"  bs={d.get('batch_size','?')}, il={d.get('input_len','?')}, ol={d.get('output_len','?')}, output_throughput={d.get('output_throughput','?')}, latency={d.get('latency','?')}\")
"
        echo ""
    fi
fi

echo "[$(timestamp)] ===================================================="
echo "[$(timestamp)] All benchmarks completed!"
echo "[$(timestamp)] Results directory: ${RESULT_DIR}"
echo "[$(timestamp)] ===================================================="
