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
- cache 默认值：`K=64`，`swap_per_update=8`。K=64/128 是目标小缓存区间；K=1024 以上只保留为历史上界实验，不作为当前方案目标。

## 显存与命中率

每个 Qwen3 expert cache slot 约 2.53 MiB。Qwen3 共有 48 层 MoE、每层 128 个专家，共 6144 个 expert instances。

| 活动槽 K | 活动缓存 | 占全部专家实例 | 观测命中率 |
|---:|---:|---:|---:|
| 64 | 0.16 GiB | 1.04% | 约 10%-12% |
| 128 | 0.32 GiB | 2.08% | 约 23.3% |

K=128 实际分配 `active=128, spare=8`，总 cache buffer 约 0.34 GiB。NPU Graph capture 正常完成，服务日志持续显示 decode `npu graph: True`。

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
| 全局 LFU expert cache | 128 | 约 23.3% | 8.255 s | 38.76 tok/s | 41.29 tok/s |

K=128 的 local decode 相对 CPU 约 +2.1%，E2E median 约 +0.5%，基本打平。原因是小缓存只覆盖约 23% expert routes，NPU cached kernel、event join、prefill/HTTP 开销和低频 cache update 波动会抵消局部 decode 收益。该结果是固定 8x 稳态更新周期下的基线；后续实现已加入无替换窗口的自适应退避，以降低稳定热点场景下的控制面开销。

这不是整层缓存失败，而是小缓存热点分流的真实边界：当 K 只有 2.08% expert instances 时，收益首先应体现在 CPU remainder 和局部 decode；端到端稳定提升还需要更高的小 K 命中率或更低控制面开销。

## 可用于论文的结论

1. 小缓存全局 expert pool 可以在只缓存 2.08% expert instances 时获得约 23.3% routing hit rate，说明 Qwen3 decode routing 存在可利用热点。
2. CPU remainder 已验证接近按 active expert 数线性下降：TopK=8 时少 1 个 CPU route，延迟下降 12.54%。
3. K=128 的端到端收益仍不足，local decode 约 +2.1%，E2E 约 +0.5%。当前瓶颈不再是“CPU 少算一个 expert 不变快”，而是小 K hit rate 和 cache 控制面开销；稳态无替换时的更新退避是下一步直接优化点。
4. 当前方案坚持全局 LFU expert pool，不采用整层缓存或 layer 过滤；大 K/整层实验仅作为历史上界，不作为论文第三部分的主方案。

## 复现

K=128 小缓存：

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
  --moe-expert-cache-size 128 \
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
