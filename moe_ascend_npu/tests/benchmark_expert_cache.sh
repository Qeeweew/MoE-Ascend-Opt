#!/bin/bash
# End-to-end cache-size sweep. NPU graph remains enabled intentionally.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32}"
RESULT_DIR="${RESULT_DIR:-./expert_cache_results}"
DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-0}"
CACHE_SIZES="${CACHE_SIZES:-64 128}"
SWAP_PER_UPDATE="${SWAP_PER_UPDATE:-8}"
UPDATE_INTERVAL="${UPDATE_INTERVAL:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-320}"
INPUT_LEN="${INPUT_LEN:-128}"
BATCH_SIZES="${BATCH_SIZES:-1}"
mkdir -p "$RESULT_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export NANOVLLM_TP_SIZE="${NANOVLLM_TP_SIZE:-2}"

python -m sglang.bench_one_batch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --tp-size 1 \
    --attention-backend ascend \
    --enable-moe-offload \
    --moe-offload-quant-type q4_0 \
    --batch-size $BATCH_SIZES \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --result-filename "$RESULT_DIR/cpu_q4_0.jsonl" \
    2>&1 | tee "$RESULT_DIR/cpu_q4_0.log"

for cache_size in $CACHE_SIZES; do
    python -m sglang.bench_one_batch_server \
        --model-path "$MODEL_PATH" \
        --trust-remote-code \
        --tp-size 1 \
        --attention-backend ascend \
        --enable-moe-expert-cache \
        --moe-expert-cache-size "$cache_size" \
        --moe-expert-cache-swap-per-update "$SWAP_PER_UPDATE" \
        --moe-expert-cache-update-interval "$UPDATE_INTERVAL" \
        --moe-expert-cache-warmup-steps "$WARMUP_STEPS" \
        --batch-size $BATCH_SIZES \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --result-filename "$RESULT_DIR/cache_k${cache_size}.jsonl" \
        2>&1 | tee "$RESULT_DIR/cache_k${cache_size}.log"
done
