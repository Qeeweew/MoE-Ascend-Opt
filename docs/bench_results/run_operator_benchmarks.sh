#!/usr/bin/env bash
# Reproducible operator benchmark suite for the project report.
#
# Linear uses the existing generic 8192x8192 shape. Fused-MoE and CPU MoE use
# Qwen3-30B-A3B-Instruct-2507 dimensions: H=2048, I=768, E=128, TopK=8.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/docs/bench_results/raw/operators_qwen3}"
MODE="${1:-all}"

mkdir -p "${RESULT_DIR}"

run_cpu() {
    export NANOVLLM_TP_THREADS_PER_NODE="${NANOVLLM_TP_THREADS_PER_NODE:-20}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
    local tp_size
    for tp_size in ${CPU_TP_SIZES:-2 4 8}; do
        export NANOVLLM_TP_SIZE="${tp_size}"

        python "${ROOT_DIR}/Int8-gemm/test/benchmark_fused_moe_int4.py" \
            --json-out "${RESULT_DIR}/cpu_fused_moe_q4_0_tp${tp_size}.json" \
            2>&1 | tee "${RESULT_DIR}/cpu_fused_moe_q4_0_tp${tp_size}.log"

        python "${ROOT_DIR}/Int8-gemm/test/benchmark_fused_moe_cpu.py" \
            --json-out "${RESULT_DIR}/cpu_fused_moe_q8_0_tp${tp_size}.json" \
            2>&1 | tee "${RESULT_DIR}/cpu_fused_moe_q8_0_tp${tp_size}.log"

        python "${ROOT_DIR}/Int8-gemm/test/benchmark_partial_moe_decode.py" \
            --hidden 2048 --intermediate 768 --top-k 8 \
            --json-out "${RESULT_DIR}/cpu_partial_moe_decode_q4_0_tp${tp_size}.json" \
            2>&1 | tee "${RESULT_DIR}/cpu_partial_moe_decode_q4_0_tp${tp_size}.log"
    done

    export NANOVLLM_TP_SIZE=8
    python "${ROOT_DIR}/Int8-gemm/test/fused_moe_cpu.py" \
        2>&1 | tee "${RESULT_DIR}/cpu_q8_0_correctness.log"
    python "${ROOT_DIR}/Int8-gemm/test/fused_moe_cpu_int4.py" \
        2>&1 | tee "${RESULT_DIR}/cpu_q4_0_correctness.log"
}

run_npu() {
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

    (
        cd "${ROOT_DIR}/moe_ascend_npu/tests"
        python test_repack.py
        python test_gemv_w4a16.py
        python test_fused_moe.py
    ) 2>&1 | tee "${RESULT_DIR}/npu_correctness.log"

    python "${ROOT_DIR}/moe_ascend_npu/tests/benchmark_fused_moe.py" \
        --json-out "${RESULT_DIR}/npu_fused_moe_w4a16.json" \
        2>&1 | tee "${RESULT_DIR}/npu_fused_moe_w4a16.log"

    python "${ROOT_DIR}/moe_ascend_npu/tests/benchmark_w4a16_linear.py" \
        --json-out "${RESULT_DIR}/npu_w4a16_linear_8192.json" \
        2>&1 | tee "${RESULT_DIR}/npu_w4a16_linear_8192.log"
}

case "${MODE}" in
    cpu) run_cpu ;;
    npu) run_npu ;;
    all)
        run_cpu
        run_npu
        ;;
    *)
        echo "Usage: $0 [cpu|npu|all]" >&2
        exit 2
        ;;
esac

echo "Operator benchmark results: ${RESULT_DIR}"
