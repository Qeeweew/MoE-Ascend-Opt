#!/bin/bash

# 定义需要遍历的 TP_SIZE 值
TP_SIZES=(8)

for tp_size in "${TP_SIZES[@]}"; do
    echo "========================================"
    echo "🚀 正在执行: NANOVLLM_TP_SIZE=${tp_size}"
    echo "========================================"

    # 执行基准测试（环境变量内联设置，仅对当前命令生效）
    NANOVLLM_TP_THREADS_PER_NODE=20 \
    OMP_NUM_THREADS=20 \
    NANOVLLM_TP_SIZE=${tp_size} \
    python3 benchmark_fused_moe_int4.py

    # 打印完成状态
    if [ $? -eq 0 ]; then
        echo "✅ NANOVLLM_TP_SIZE=${tp_size} 运行成功"
    else
        echo "❌ NANOVLLM_TP_SIZE=${tp_size} 运行失败 (退出码: $?)"
    fi
    echo ""
done
