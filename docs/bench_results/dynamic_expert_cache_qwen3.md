# Qwen3-30B-A3B 小缓存专家分流实验记录

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

## CPU remainder 微基准

单 token decode、TopK=8、H=2048、Ish=768 的 CPU partial benchmark 用于验证一个核心问题：少算一个 CPU expert 是否接近少 1/8 latency。

多轮乱序 median：

| CPU routes | 延迟 |
|---:|---:|
| 8 | 0.290 ms |
| 7 | 0.253 ms |

8 -> 7 routes 下降 12.54%，接近理论 1/8 = 12.5%。这说明 CPU remainder 本身已经能把 expert cache hit 转化成 latency 下降。

复现命令：

```bash
python Int8-gemm/test/benchmark_partial_moe_decode.py --runs 120 --warmup 20
```

## 端到端结果

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

K=256 时 hit rate 提升到约 38.3%，更新周期同样退到 256/512 step，E2E median 相对 CPU 提升约 7.1%，local decode 提升约 7.4%。它只缓存 4.17% expert instances，且已经进入稳定退避，因此作为当前默认小缓存配置。K=512 时 hit rate 提升到约 57.6%，E2E median 相对 CPU 提升约 15.0%，cache 填满后的后 5 次请求 median 为 7.092 s / 45.12 tok/s；但 8 次请求窗口内仍有 replacement，尚未触发稳态退避，因此应作为更大 cache 的收益上界点，而不是当前默认点。

这不是整层缓存失败，而是小缓存热点分流的真实边界：当 K 只有 2.08% expert instances 时，收益首先应体现在 CPU remainder 和局部 decode；端到端稳定提升还需要更高的小 K 命中率或更低控制面开销。

## 默认路径验证

只传 `--enable-moe-expert-cache`、不显式传 `--moe-expert-cache-size` 和 `--moe-expert-cache-update-interval` 的启动路径已经单独验证。日志显示默认分配 `active=256, spare=8`，总 cache buffer 约 0.65 GiB；短请求期间 `ExpertCache` 更新 step 为 `32, 48, 64, 80, 96`，相邻差值均为 16，说明默认 `update_interval=16` 已生效。Graph capture 正常完成。

## 可用于论文的结论

1. 小缓存全局 expert pool 可以在只缓存 2.08% expert instances 时获得约 23.3% routing hit rate，说明 Qwen3 decode routing 存在可利用热点；1.04% 的 K=64 命中率只有约 13.2%，不足以端到端超过 CPU baseline。
2. CPU remainder 已验证接近按 active expert 数线性下降：TopK=8 时少 1 个 CPU route，延迟下降 12.54%。
3. K=128 加入稳态更新退避后，E2E 约 +2.1%；K=256 在只缓存 4.17% expert instances 时达到约 +7.1%；K=512 在 8.33% expert instances 时达到约 +15.0%，但仍处在替换收敛期。
4. 当前方案坚持全局 LFU expert pool，不采用整层缓存或 layer 过滤；大 K/整层实验仅作为历史上界，不作为论文第三部分的主方案。

## 复现

完整固定 prompt sweep（CPU baseline + K=64/128/256/512）：

```bash
python moe_ascend_npu/tests/benchmark_expert_cache_prompt.py \
  --result-dir docs/bench_results/raw/qwen3_fixed_prompt_sweep
```

该脚本会逐个启动服务，保持 NPU Graph BS=1/2/4/8，发 8 次固定 prompt 请求，并从 server log 解析 `window_hit`、退避步长、local decode throughput。每个配置输出独立 JSON，汇总结果写入 `summary.json`，其中直接包含相对 CPU speedup、expert instance fraction、physical cache GiB 和跨配置输出 hash 一致性。

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
