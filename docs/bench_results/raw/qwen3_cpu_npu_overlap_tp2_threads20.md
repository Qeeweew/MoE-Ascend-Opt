# Qwen3 TP2 CPU/NPU Fused-MoE overlap

测试固定 `NANOVLLM_TP_SIZE=2`、`NANOVLLM_TP_THREADS_PER_NODE=20`，使用
Qwen3-30B-A3B 的 `H=2048 / I(TP2)=768 / E=128 / TopK=8`、BF16 decode shape。
CPU 分支使用持久 `MoEGraphContext`，执行顺序与 cache method 一致：main stream 记录输入
event，side stream 提交 CPU callback，main stream 同时启动 cached NPU fused-MoE，最后通过
event join 并相加。每个点交错采样 600 次、轮换 32 组 CPU expert ids。

| Cache hits | CPU misses | CPU path P50 | NPU path P50 | Overlap P50 | Overlap - CPU | NPU fully hidden |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 8 | 0.5283 ms | 0.1465 ms | 0.5267 ms | -0.0015 ms | 是 |
| 1 | 7 | 0.5107 ms | 0.1466 ms | 0.5091 ms | -0.0017 ms | 是 |
| 2 | 6 | 0.4918 ms | 0.1463 ms | 0.4916 ms | -0.0002 ms | 是 |
| 3 | 5 | 0.4744 ms | 0.1469 ms | 0.4783 ms | +0.0038 ms | 是，边界 |
| 4 | 4 | 0.4572 ms | 0.1459 ms | 0.4756 ms | +0.0183 ms | 否 |
| 5 | 3 | 0.4369 ms | 0.1461 ms | 0.4740 ms | +0.0371 ms | 否 |
| 6 | 2 | 0.4231 ms | 0.1462 ms | 0.4727 ms | +0.0496 ms | 否 |
| 7 | 1 | 0.4044 ms | 0.1459 ms | 0.4720 ms | +0.0676 ms | 否 |
| 8 | 0 | 0.3701 ms | 0.1462 ms | 0.4708 ms | +0.1007 ms | 否 |

以 5 us 作为单层计时噪声阈值，0--3 hit 时 NPU 分支可以完全隐藏；4 hit 起无法完全
隐藏。Overlap 从 3 hit 后基本停在约 0.47--0.48 ms，说明当前性能下界不是 CPU 算术量，
而是 CPU callback/D2H-H2D、NPU kernel 和 event join 的组合关键路径。

本次 K=512 ShareGPT 填满后平均命中率为 22.62%，即每层 TopK=8 平均约 1.81 hit，平均点
位于 NPU 可完全隐藏区间。但命中分布并不均匀，出现 4 个及以上 hit 的层/step 时仍会暴露
约 0.018--0.101 ms 的单层残余延迟。

原始 JSON：`qwen3_cpu_npu_overlap_tp2_threads20.json`。

注意：尝试在独立微脚本中单独 capture callback graph 时，CPU callback 没有被可靠回放，
因此没有采用该口径。表中结果使用 graph-safe 持久 context，但逐次直接提交真实 stream
callback 和 event join；它验证的是 CPU/NPU 硬件并行关系，不宣称复现 SGLang 整图 replay
的固定启动开销。
