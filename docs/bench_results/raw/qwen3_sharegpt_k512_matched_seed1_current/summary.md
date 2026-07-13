# Qwen3-30B ShareGPT：K=512 专家缓存对比纯 CPU

测试日期：2026-07-13。

## 口径

- 模型：`Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32`
- 设备：1 张 Ascend 910B3，SGLang TP=1
- CPU MoE：`NANOVLLM_TP_SIZE=2`，`NANOVLLM_TP_THREADS_PER_NODE=20`
- NPU Graph：BS 1/2/4/8，保持开启
- 数据：同 seed=1 的 32 条 ShareGPT 请求，并发 1
- 总输入：10092 tokens；总输出：10240 tokens，每请求强制 320 tokens
- 缓存：全局 LFU，K=512，`swap_per_update=8`，`update_interval=16`，`warmup_steps=16`

## 结果

| 指标 | 纯 CPU Q4 | K=512 | K=512 相对变化 |
|---|---:|---:|---:|
| Output throughput | 40.087 tok/s | 41.605 tok/s | +3.78% |
| Benchmark duration | 255.441 s | 246.127 s | -3.65% |
| Median E2E | 7560.08 ms | 7369.55 ms | -2.52% |
| P99 E2E | 12616.97 ms | 11903.48 ms | -5.65% |
| Median TTFT | 538.55 ms | 472.45 ms | -12.27% |
| P99 TTFT | 5170.50 ms | 4674.92 ms | -9.58% |
| Median TPOT | 22.147 ms | 21.610 ms | -2.42% |
| P99 TPOT | 23.378 ms | 23.501 ms | +0.52% |
| Median ITL | 22.095 ms | 21.572 ms | -2.37% |
| P99 ITL | 24.675 ms | 29.539 ms | +19.71% |
| Local decode tail-16 median | 45.20 tok/s | 46.65 tok/s | +3.21% |

## 缓存行为

- 512 个 active slots 加 8 个 spare slots 实际占用约 1.29 GiB，覆盖 6144 个 expert instances 的 8.33%。
- 正式测试从空缓存开始，约 30 秒后填满 512 个 active slots。
- 含冷启动的 136 个窗口平均命中率为 18.05%。
- 缓存填满后的 72 个窗口平均命中率为 22.62%；最后 32 个窗口平均为 25.37%。
- 总 swap 数为 1080，其中前 512 次用于填充，其余约 568 次为替换。
- 离线路由分析对同类 ShareGPT trace、K=512、decay=0.95、每窗口最多 8 次 swap 的预测命中率为 20.31%，与本次实测量级一致。

## 结论

K=512 已经在 matched ShareGPT 上超过纯 CPU：整体吞吐提升 3.78%，中位 TPOT 下降 2.42%，说明缓存命中带来的 CPU remainder 减少开始覆盖 NPU 双路执行和控制面开销。

但当前不能称为全面胜出：P99 TPOT 基本持平，P99 ITL 恶化 19.71%。日志显示缓存填满后仍持续发生替换，尾延迟问题仍指向替换窗口和双路同步。后续优化重点应是降低稳态替换频率或进一步隐藏替换，而不是继续扩大 CPU 线程数。

TTFT 的下降只记录为观测值，不能直接当成 decode 缓存收益。当前 cache method 在 eager prefill
同样读取 slot table，并让命中专家走 NPU、miss experts 走 CPU，所以连续请求在缓存预热后理论上
可以缩短 prefill；但本次只有两次独立服务启动、每种配置一次 32 请求，没有 A/B/A 重复和逐请求
明细，无法把 12.27% 与运行时抖动分离。缓存收益的主要可信证据仍是 TPOT、local decode 和总吞吐。

原始文件：`cpu_q4.jsonl`、`cache_k512.jsonl`、`cpu_q4.log`、`cache_k512.log`。
