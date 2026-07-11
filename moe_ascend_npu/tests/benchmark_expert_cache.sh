#!/bin/bash
# End-to-end cache-size sweep. NPU graph remains enabled intentionally.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32}"
RESULT_DIR="${RESULT_DIR:-./expert_cache_results}"
DEVICE="${ASCEND_RT_VISIBLE_DEVICES:-0}"
CACHE_SIZES="${CACHE_SIZES:-0 128 256 512 1024}"
PLACEMENT="${PLACEMENT:-lfu}"
mkdir -p "$RESULT_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export NANOVLLM_TP_SIZE="${NANOVLLM_TP_SIZE:-2}"

for cache_size in $CACHE_SIZES; do
    python -m sglang.bench_one_batch_server \
        --model-path "$MODEL_PATH" \
        --trust-remote-code \
        --tp-size 1 \
        --attention-backend ascend \
        --enable-moe-expert-cache \
        --moe-expert-cache-size "$cache_size" \
        --moe-expert-cache-swap-per-update 64 \
        --moe-expert-cache-update-interval 32 \
        --moe-expert-cache-warmup-steps 16 \
        --moe-expert-cache-placement "$PLACEMENT" \
        --batch-size 1 4 8 \
        --input-len 128 \
        --output-len 1024 \
        --result-filename "$RESULT_DIR/cache_k${cache_size}.jsonl" \
        2>&1 | tee "$RESULT_DIR/cache_k${cache_size}.log"
done
