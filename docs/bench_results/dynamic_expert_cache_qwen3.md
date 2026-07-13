# Qwen3-30B-A3B 小缓存专家分流实验记录

> **结论修订（2026-07-13）**：本文早期 40+ tok/s 数据来自重复固定 prompt 的稳态实验，
> 不代表 ShareGPT 泛化性能。同 seed、同请求的 matched ShareGPT A/B 显示：CPU Q4 为
> 37.79 tok/s，K=256 为 37.05 tok/s（-1.96%）。在当前 CPU decode GEMV 引擎上重新测试，
> CPU Q4 为 40.09 tok/s，K=512 为 41.60 tok/s（+3.78%）。因此 K=256 仍不足以覆盖缓存开销，
> 但 K=512 已在 matched ShareGPT、并发 1 下取得小幅净加速；P99 ITL 仍有回退。

## 实验环境

- 日期：2026-07-11
- 设备：1x Ascend 910B3 64 GB，鲲鹏 920
- 模型：`Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32`
- 框架：SGLang，TP=1，NPU Graph BS=1/2/4/8
- 请求：同一 prompt，`temperature=0`，强制生成 320 tokens
- 缓存策略：全局 LFU expert pool，所有 MoE 层共享同一个 slot 池；不做整层缓存、不做 layer-id 过滤

## 当前实现状态

- `fused_moe_w4a16_cached`：命中专家走 NPU W4A16 cache slot，`slot_id < 0` 跳过。
- CPU partial：命中专家在 `topk_ids` 中改为 `-1`，CPU 只计算 miss experts。
- 双路执行：CPU partial 在共享 NPU side stream 中提交，主 stream 同时执行 NPU cached kernel，最后通过 event join 后相加。
- CPU decode 小 batch 路径：全 miss 保持 `2 * TopK` 线程预算；存在 cache hit 时，把 NUMA 节点线程预算重分配给剩余 CPU miss experts。
- cache 默认值：`K=256`，`swap_per_update=8`。K=64/128/512 作为显存-命中率-吞吐曲线消融；K=1024 以上只保留为历史上界实验，不作为当前方案目标。

## 显存与命中率

每个 Qwen3 expert cache slot 约 2.53 MiB。Qwen3 共有 48 层 MoE、每层 128 个专家，共 6144 个 expert instances。

| 活动槽 K | 活动缓存 | 占全部专家实例 | 观测命中率 |
|---:|---:|---:|---:|
| 64 | 0.16 GiB | 1.04% | 约 13.2% |
| 128 | 0.32 GiB | 2.08% | 约 23.3% |
| 256 | 0.63 GiB | 4.17% | 约 38.3% |
| 512 | 1.27 GiB | 8.33% | 约 57.6% |

K=128 实际分配 `active=128, spare=8`，总 cache buffer 约 0.34 GiB。K=64 实际分配 `active=64, spare=8`，总 cache buffer 约 0.18 GiB，但命中率不足以覆盖双路调度与控制面开销，因此不作为默认。K=256 总 buffer 约 0.65 GiB，K=512 总 buffer 约 1.29 GiB。所有配置 NPU Graph capture 正常完成，服务日志持续显示 decode `npu graph: True`。

## CPU remainder 精确微基准

固定 `NANOVLLM_TP_SIZE=2`、`NANOVLLM_TP_THREADS_PER_NODE=20`，使用 Qwen3 的真实
`128 experts / TopK=8 / H=2048 / Ish=768`。测试轮换 32 组 expert ids，乱序采样 600 次，
使用 C++ 内部计算计时并给出 bootstrap 置信区间，避免 Python 调用和八个常驻热专家造成偏差。

| CPU routes | C++ compute P50 | P95 |
|---:|---:|---:|
| 0 | 0.0040 ms | 0.0053 ms |
| 1 | 0.1069 ms | 0.1262 ms |
| 4 | 0.1693 ms | 0.2255 ms |
| 7 | 0.2513 ms | 0.3559 ms |
| 8 | 0.2661 ms | 0.3877 ms |

8 -> 7 routes 只下降 5.56%，并不等于理论计算量的 12.5%；8 -> 4 routes 下降 36.4%。
原因是多个专家原本就在固定 20 线程池上并行执行，少一个专家主要减少占用线程，而不会按路由数
线性缩短关键路径；只有 miss 数下降到能明显增加每个剩余专家的线程份额时，wall latency 才显著下降。
因此缓存收益必须用完整 route-count 曲线和实际命中分布估计，不能再使用“命中率即延迟降幅”的线性假设。

复现命令：

```bash
python Int8-gemm/test/benchmark_partial_moe_decode.py --runs 120 --warmup 20
```

## 固定 prompt 端到端结果（工作负载特例）

同一 prompt、8 次连续 320-token 请求：

| 配置 | K | hit rate | E2E median | E2E tok/s | local decode median |
|---|---:|---:|---:|---:|---:|
| CPU Q4_0 offload | 0 | 0% | 8.297 s | 38.57 tok/s | 40.45 tok/s |
| 全局 LFU expert cache，自适应更新退避 | 64 | 约 13.2% | 8.564 s | 37.37 tok/s | 38.86 tok/s |
| 全局 LFU expert cache，固定 8x 稳态更新 | 128 | 约 23.3% | 8.255 s | 38.76 tok/s | 41.29 tok/s |
| 全局 LFU expert cache，自适应更新退避 | 128 | 约 23.2% | 8.123 s | 39.39 tok/s | 40.77 tok/s |
| 全局 LFU expert cache，自适应更新退避 | 256 | 约 38.3% | 7.750 s | 41.29 tok/s | 43.46 tok/s |
| 全局 LFU expert cache，自适应更新退避 | 512 | 约 57.6% | 7.211 s | 44.37 tok/s | 48.83 tok/s |

K=64 自适应退避时，日志显示更新周期同样退到 256/512 step，但稳态 hit rate 只有约 13.2%，E2E 低于 CPU Q4_0 baseline（speedup 0.969）。这说明默认 cache 不能只看显存占比，还必须达到足够 routing hit rate。

K=128 固定 8x 稳态更新时，local decode 相对 CPU 约 +2.1%，E2E median 约 +0.5%。加入无替换窗口的自适应退避后，日志显示更新周期从 128 step 退到 256 step，再退到 512 step；E2E median 提升到 39.39 tok/s，相对 CPU 约 +2.1%。K=128 是最小正收益点，但只覆盖约 23% expert routes，因此不作为默认。

K=256 在重复固定 prompt 下 hit rate 约 38.3%，E2E median 相对 CPU 提升约 7.1%；K=512 约提升 15.0%。这些数字只能作为高路由复用场景的上界实验，不能作为 ShareGPT 或一般服务负载的结论。

这不是整层缓存失败，而是小缓存热点分流的真实边界：当 K 只有 2.08% expert instances 时，收益首先应体现在 CPU remainder 和局部 decode；端到端稳定提升还需要更高的小 K 命中率或更低控制面开销。

## 默认路径验证

只传 `--enable-moe-expert-cache`、不显式传 `--moe-expert-cache-size` 和 `--moe-expert-cache-update-interval` 的启动路径已经单独验证。日志显示默认分配 `active=256, spare=8`，总 cache buffer 约 0.65 GiB；短请求期间 `ExpertCache` 更新 step 为 `32, 48, 64, 80, 96`，相邻差值均为 16，说明默认 `update_interval=16` 已生效。Graph capture 正常完成。

## 可用于论文的结论

1. 小缓存全局 expert pool 可以在只缓存 2.08% expert instances 时获得约 23.3% routing hit rate，说明 Qwen3 decode routing 存在可利用热点；1.04% 的 K=64 命中率只有约 13.2%，不足以端到端超过 CPU baseline。
2. CPU remainder 会随 miss route 数下降，但不是线性关系：固定 TP2/20 线程下 8→7 下降 5.56%，8→4 下降 36.4%。
3. 固定 prompt 下 K=128/K=256/K=512 曾观察到约 +2.1%/+7.1%/+15.0%；matched ShareGPT 中 K=256 为 -1.96%，当前 K=512 为 +3.78%，说明需要约 20% 以上的实际命中率才能覆盖双路执行和控制面开销。
4. 当前方案坚持全局 LFU expert pool，不采用整层缓存或 layer 过滤；大 K/整层实验仅作为历史上界，不作为论文第三部分的主方案。

## 复现

完整固定 prompt sweep（CPU baseline + K=64/128/256/512）：

```bash
python moe_ascend_npu/tests/benchmark_expert_cache_prompt.py \
  --result-dir docs/bench_results/raw/qwen3_fixed_prompt_sweep
```

该脚本会逐个启动服务，保持 NPU Graph BS=1/2/4/8，发 8 次固定 prompt 请求，并从 server log 解析 `window_hit`、退避步长、local decode throughput。每个配置输出独立 JSON，汇总结果写入 `summary.json`，其中直接包含相对 CPU speedup、expert instance fraction、physical cache GiB 和跨配置输出 hash 一致性。

脚本自身已做最小端到端烟测：

```bash
python moe_ascend_npu/tests/benchmark_expert_cache_prompt.py \
  --skip-cpu --cache-sizes 256 --requests 2 --output-tokens 32 \
  --result-dir /tmp/moe_cache_prompt_smoke
```

该烟测确认脚本可以自动启动 K=256 cache 服务、完成请求、解析 allocation / hit window / local decode throughput、验证单配置输出 hash 稳定、生成 `cache_k256.json` 与 `summary.json`，并正常停止服务。跨配置 hash 一致性只在实际 sweep 至少包含两个配置时给出。

## 2026-07-13 替换控制面优化

固定 TP2、每 NUMA 节点 20 线程的分项 profiling 显示，旧实现每轮 8 专家替换的 145 ms
控制暂停中，纯 repack 约 5 ms、H2D submit 约 1.8 ms、staging event 等待约 0.06 ms；
真正的主要开销是 LFU 对 6144 个分数逐个进行 PyTorch 标量索引和转换，单独约 123 ms。

优化包括：

- int4 8×8 tile 重排由“NEON lane spill 到栈后标量拼接”改为寄存器内 TBL gather + nibble pack；
- LFU 分数通过 NumPy 零拷贝 view 批量 mask/sort，Python 只处理最终最多 8 个候选。

结果：repack 随机冷专家由约 0.683 ms 降至 0.502 ms；控制面 P50 由 145.02 ms 降至
14.16 ms；同一 192-token 请求由 6.39 s 降至 5.09 s。完整原始数据位于
`docs/bench_results/raw/repack_qwen3_tp2_threads20/`、
`docs/bench_results/raw/qwen3_cache_replacement_profile/` 和
`docs/bench_results/raw/qwen3_cache_replacement_profile_optimized/`。

相同 32 条 ShareGPT 请求的完整 A/B 中，output throughput 从 34.59 提升到 37.50 tok/s
（+8.43%），P99 ITL 从 139.99 ms 降到 29.50 ms（-78.93%），median E2E 从
8882.85 ms 降到 8265.99 ms（-6.94%）。命中率仅由 13.60% 变为 13.93%，说明改进来自
控制面开销消除。优化后原始数据位于
`docs/bench_results/raw/qwen3_sharegpt_routing_optimized/`。

K=256 默认小缓存：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export NANOVLLM_TP_SIZE=2
export GLOO_SOCKET_IFNAME=lo

python -m sglang.launch_server \
  --host 127.0.0.1 --port 31000 \
  --model-path /mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32 \
  --trust-remote-code \
  --tp-size 1 \
  --attention-backend ascend \
  --enable-moe-expert-cache \
  --moe-expert-cache-size 256 \
  --moe-expert-cache-swap-per-update 8 \
  --moe-expert-cache-update-interval 16 \
  --moe-expert-cache-warmup-steps 16 \
  --cuda-graph-bs 1 2 4 8
```

CPU Q4_0 baseline：

```bash
python -m sglang.launch_server \
  --host 127.0.0.1 --port 31000 \
  --model-path /mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32 \
  --trust-remote-code \
  --tp-size 1 \
  --attention-backend ascend \
  --enable-moe-offload \
  --moe-offload-quant-type q4_0 \
  --cuda-graph-bs 1 2 4 8
```

## 2026-07-13 K=512 matched ShareGPT

使用当前 CPU decode GEMV 引擎，固定 `NANOVLLM_TP_SIZE=2`、
`NANOVLLM_TP_THREADS_PER_NODE=20`，服务与数据抽样 seed 均为 1。32 条请求共
10092 输入 tokens / 10240 输出 tokens，并发 1，NPU Graph BS=1/2/4/8 保持开启。

| 指标 | CPU Q4 | K=512 | K=512 相对变化 |
|---|---:|---:|---:|
| Output throughput | 40.09 tok/s | 41.60 tok/s | +3.78% |
| Median E2E | 7560.08 ms | 7369.55 ms | -2.52% |
| P99 E2E | 12616.97 ms | 11903.48 ms | -5.65% |
| Median TPOT | 22.15 ms | 21.61 ms | -2.42% |
| P99 TPOT | 23.38 ms | 23.50 ms | +0.52% |
| P99 ITL | 24.68 ms | 29.54 ms | +19.71% |
| Local decode tail-16 median | 45.20 tok/s | 46.65 tok/s | +3.21% |

K=512 实际分配 `active=512, spare=8`，约 1.29 GiB，只覆盖全部 6144 个 expert
instances 的 8.33%。测试从空缓存开始，约 30 秒填满；含冷启动的窗口平均命中率为
18.05%，填满后平均为 22.62%，最后 32 个窗口平均为 25.37%。这与离线 bounded LFU
预测的约 20.31% 命中率量级一致。

该结果证明 K=512 已跨过当前并发 1 ShareGPT 工作负载的净收益点，但仍持续发生替换：
总计 1080 次 swap，其中约 568 次发生在填满之后。吞吐和中位延迟改善的同时 P99 ITL
恶化，下一步应减少稳态替换或隐藏替换窗口，而不是把 +3.78% 外推到所有并发和请求分布。

原始数据位于 `docs/bench_results/raw/qwen3_sharegpt_k512_matched_seed1_current/`。
