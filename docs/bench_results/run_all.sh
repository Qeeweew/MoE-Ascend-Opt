#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_DIR="$SCRIPT_DIR"
BENCH_SCRIPT="$SCRIPT_DIR/bench.sh"
SERVER_PID=""

cleanup_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    pkill -f "sglang serve.*--port 30001" 2>/dev/null || true
}

wait_for_server() {
    echo "Waiting for server to be ready on http://localhost:30001..."
    for i in $(seq 1 600); do
        if curl -s http://localhost:30001/health > /dev/null 2>&1; then
            echo "Server is ready."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start within 20 minutes."
    return 1
}

run_benchmark() {
    local threshold="$1"
    local label="$2"
    local result_file="$RESULT_DIR/bench_result_${label}.json"

    cleanup_server

    echo "============================================"
    echo "Starting server with NPU_W4A16_SMALL_BS_THRESHOLD=$threshold"
    echo "============================================"

    NPU_W4A16_SMALL_BS_THRESHOLD="$threshold" \
    SGLANG_SET_CPU_AFFINITY=1 \
    ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
    sglang serve \
        --model-path /mnt/nvme0n1/xwj-data/models/MiniMax-M2.5-AWQ-4bit \
        --trust-remote-code \
        --tp-size 4 \
        --attention-backend ascend \
        --sampling-backend ascend \
        --tool-call-parser minimax-m2 \
        --reasoning-parser minimax-append-think \
        --cuda-graph-bs 1 2 3 4 5 6 7 8 \
        --served-model-name MiniMax-M2.5-AWQ \
        --port 30001 &
    SERVER_PID=$!

    wait_for_server || exit 1

    echo "Running benchmark..."
    bash "$BENCH_SCRIPT" "$result_file"

    echo "Benchmark with threshold=$threshold completed. Result saved to $result_file"

    cleanup_server
    sleep 5
}

trap cleanup_server EXIT

run_benchmark "-1" "w4a16_no_optimize"
run_benchmark "8" "w4a16_optimize_bs8"

echo "All benchmarks done."
