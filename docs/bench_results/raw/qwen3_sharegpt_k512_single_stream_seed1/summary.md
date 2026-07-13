# Qwen3-30B K=512：单 stream 双 callback A/B

测试日期：2026-07-13。

## 口径

- 模型：`Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32`
- 设备：1 张 Ascend 910B3，SGLang TP=1
- CPU MoE：`NANOVLLM_TP_SIZE=2`，每节点固定 20 线程
- NPU Graph：BS 1/2/4/8，保持开启
- 数据：同 seed=1 的 32 条 ShareGPT 请求，并发 1
- 总输入：10092 tokens；总输出：10240 tokens，每请求 320 tokens
- 缓存：全局 LFU，K=512，`swap_per_update=8`，`update_interval=16`，`warmup_steps=16`
- 基线：原双 stream 路径；实验：主 stream 上 `D2H -> callback_start -> NPU MoE -> callback_join -> H2D -> Add`

## 端到端结果

| 指标 | 双 stream 基线 | 单 stream 双 callback | 相对变化 |
|---|---:|---:|---:|
| Output throughput | 41.605 tok/s | 42.161 tok/s | +1.34% |
| Benchmark duration | 246.127 s | 242.881 s | -1.32% |
| Median E2E | 7369.55 ms | 7277.82 ms | -1.24% |
| P99 E2E | 11903.48 ms | 11726.10 ms | -1.49% |
| Median TTFT | 472.45 ms | 471.92 ms | -0.11% |
| P99 TTFT | 4674.92 ms | 4632.11 ms | -0.92% |
| Median TPOT | 21.610 ms | 21.432 ms | -0.82% |
| P99 TPOT | 23.501 ms | 22.550 ms | -4.05% |
| Median ITL | 21.572 ms | 21.337 ms | -1.09% |
| P99 ITL | 29.539 ms | 28.730 ms | -2.74% |
| Local decode tail-16 median | 46.65 tok/s | 46.79 tok/s | +0.29% |

相对同 workload 的纯 CPU Q4 结果 40.087 tok/s，单 stream 方案总吞吐提升 5.17%。

两次运行的缓存轨迹并非完全一致：双 stream 填满后窗口平均命中率为 22.62%，单 stream
为 21.53%。因此 1.34% 只能视为一次 matched A/B 的正向结果，不能当作稳定置信区间；但
实验组在命中率更低的情况下仍更快，没有通过更高缓存命中率占便宜。

## msprof 单次执行

profile 采用 warmup 1 次后只执行 1 次 2-hit/6-miss 单层调用。新路径的目标任务全部位于
物理 stream 46：

- 4 个 D2H：1.84--2.40 us/个
- start callback：HOSTFUNC 10.900 us，随后 EVENT_WAIT 25.120 us
- NPU cached fused-MoE：41.582 us
- join callback：HOSTFUNC 11.182 us，随后 EVENT_WAIT 29.760 us
- H2D：2.740 us
- Add：2.120 us

原双 stream profile 中，CPU callback 窗口为 269.924 us，NPU fused-MoE 为 41.002 us；
NPU 完全落在 CPU 窗口内，但主 stream 在 NPU 结束后仍等待 side stream 98.700 us。新路径
删除了 side stream/event join，profile 中只剩同一 stream 上的两个 callback。msprof 会显著
放大 host/runtime 间隙，所以绝对墙钟时间以无 profile 的端到端与微基准为准。

## 结论

本次实现值得暂时保留：它保持 Graph、CPU TP2/20 线程和权重容量不变，输出等价性及 1000
次 replay 烟测均已通过，并在完整 ShareGPT workload 上取得小幅正收益。当前收益只有 1.34%，
后续若要作为最终性能结论，需要补 A/B/A 重复测试并报告波动范围。

原始文件：`cache_k512.jsonl`、`server.log`；profile 位于
`../msprof_cpu_npu_overlap_h2_single_stream/`，基线位于
`../qwen3_sharegpt_k512_matched_seed1_current/` 和 `../msprof_cpu_npu_overlap_h2_current/`。
