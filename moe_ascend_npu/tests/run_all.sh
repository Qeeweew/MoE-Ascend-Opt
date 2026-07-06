#!/bin/bash
# Run all moe_ascend_npu NPU correctness tests.
set -e
cd "$(dirname "$0")"
for t in test_repack.py test_gemv_w4a16.py test_fused_moe.py; do
    echo "===== $t ====="
    python "$t"
done
echo
echo "All moe_ascend_npu tests passed."
