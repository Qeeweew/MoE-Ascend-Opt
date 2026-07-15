#!/bin/bash
set -e

# ========== C0: Full CPU Q4 Offload ==========
MODEL_PATH="/mnt/models/MiniMax-M2.5-AWQ-4bit"
DATASET_PATH="/home/xwj/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
RESULT_DIR="/home/xwj/workspace/MoE-Ascend-Opt/docs/bench_results/raw/minimax_m2_5_closure_seed1_tp1/cpu_q4"
PORT=31000
export ASCEND_RT_VISIBLE_DEVICES=0

mkdir -p "$RESULT_DIR"

# Start server
sglang serve \
  --host 127.0.0.1 --port "$PORT" \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --tp-size 1 --attention-backend ascend \
  --cuda-graph-bs 1 2 4 8 \
  --enable-moe-offload --moe-offload-start-layer 0 \
  --moe-offload-quant-type q4_0 \
  >"$RESULT_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "C0 Server PID: $SERVER_PID"

# Wait for server ready
echo "Waiting for server..."
for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "C0 Server ready after $((i*5)) seconds"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server died! Check log"
    tail -50 "$RESULT_DIR/server.log"
    exit 1
  fi
  sleep 5
done

# Run benchmarks
for CONCURRENCY in 1 4 8; do
  echo "=== C0 concurrency=$CONCURRENCY ==="
  python -m sglang.bench_serving \
    --backend sglang \
    --base-url "http://127.0.0.1:${PORT}" \
    --dataset-name sharegpt --dataset-path "$DATASET_PATH" \
    --num-prompts 32 --sharegpt-output-len 320 \
    --max-concurrency "$CONCURRENCY" --warmup-requests 0 --seed 1 \
    --output-file "$RESULT_DIR/sharegpt_c${CONCURRENCY}.jsonl" \
    2>&1 | tee "$RESULT_DIR/bench_c${CONCURRENCY}.log"
done

# Cleanup
kill -INT "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null || true
echo "C0 done"
