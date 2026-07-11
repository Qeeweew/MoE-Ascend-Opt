# Qwen3-30B-A3B 动态专家缓存实验记录

## 实验环境

- 日期：2026-07-10
- 设备：1× Ascend 910B3 64 GB，鲲鹏 920，CPU NUMA 并行度 2
- 模型：`Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32`
- 框架：SGLang，TP=1，NPU Graph BS=1/2/4/8
- 请求：固定 seed=1234、temperature=0、输入约 128 tokens、强制生成 320 tokens
- 缓存统计：warmup=16，填充周期=32 steps；满载后周期自动变为 256 steps，每轮稳态最多替换 8 个专家

## 正确性与静态图

- `fused_moe_w4a16_cached` 的全命中、混合命中和全 miss 测试全部通过；BS=1/2/4 相对误差约 0.13%–0.15%。
- Qwen3 服务完成 NPU Graph capture，动态替换期间日志始终显示 `npu graph: True`，没有 recapture、非法 slot 或半写权重。
- 相同 seed、prompt 和 320-token 生成长度下，K=1024 动态缓存与全 CPU Q4_0 的输出文件均为 1024 bytes，逐字节完全一致。

## 显存与命中率

每个 Qwen3 专家 slot 为 2.53 MiB。全量 MoE 权重下沉 CPU 后，NPU 模型权重占用由约 17.4 GB 降为 1.91 GB；缓存按 K 线性增加固定 HBM：

| 活动槽 K | 活动缓存 | 占 6144 个专家实例 | 观测窗口命中率 |
|---:|---:|---:|---:|
| 64 | 0.16 GiB | 1.04% | 10%–16% |
| 512 | 1.27 GiB | 8.33% | 45%–50% |
| 1024 | 2.53 GiB | 16.67% | 65%–72% |

K=1024 实验使用 128 个启动备用槽，物理分配约 2.85 GiB；默认 M=64 时物理分配约 2.69 GiB。缓存地址和 slot table 地址在整个服务期保持固定。

## 吞吐结果

| 配置 | 稳态命中率 | 320-token E2E | 输出吞吐 | 相对全 CPU |
|---|---:|---:|---:|---:|
| 全 CPU Q4_0 | 0% | 8.003 s | 39.98 tok/s | 1.000× |
| K=512，稳态每 32 step 更新 | 约 45%–50% | 9.687 s | 33.03 tok/s | 0.826× |
| K=1024，稳态每 32 step 更新 | 约 60%–68% | 11.153 s | 28.69 tok/s | 0.718× |
| K=1024，稳态每 256 step 更新 | 约 68.5% | 7.709 s | 41.51 tok/s | **1.038×** |
| K=2048，16 个完整层驻留 | 33.3% 层完全命中 | 7.486–7.542 s | 42.43–42.75 tok/s | 1.061–1.069× |
| K=4096，32 个完整层驻留 | 66.7% 层完全命中 | 6.105–6.392 s | 50.06–52.41 tok/s | **1.252–1.311×** |

K=1024 低频稳态窗口内，SGLang 日志观测到的局部 decode 吞吐为 42–44 tok/s。结果说明缓存收益同时取决于命中率和控制面更新频率：高命中率不足以抵消频繁同步/repack，满载后必须降低替换频率。

### Batch=1 层完整性优化（2026-07-11）

进一步审计发现，全局 LFU 会把槽位分散到几乎所有 MoE 层。即使单个专家的命中率很高，只要某层仍有一个 miss，该层的 D2H、CPU callback、线程池调度、归约和 H2D 固定链仍会执行。新增 `--moe-expert-cache-placement layer`，按观测流量选择层，并在进入下一层前缓存该层全部 128 个专家。这样完整命中的层会触发 CPU 引擎已有的 `total_expert_tokens == 0` 快路径，直接输出零增量，由 NPU 缓存结果承担该层 MoE 输出。

K=4096 占 6144 个专家实例的三分之二，活动权重约 10.1 GiB；实验用 512 个备用槽，物理缓存约 11.4 GiB。为避免 batch=1 场景仍为 50 万 token KV 池保留约 46 GiB，将 `--mem-fraction-static` 调为 0.60；NPU Graph BS=1/2/4/8 仍正常 capture 和 replay。固定 320-token 请求的三次稳态结果为 6.105 s、6.283 s、6.239 s，即 52.41、50.93、51.29 tok/s；另一组带 `top_k=1` 的三次结果为 51.85、52.29、50.06 tok/s。相对全 CPU 39.98 tok/s，稳态范围提升 **25.2%–31.1%**，超过 batch=1 提升 20% 的目标。

服务级多次生成在当前 Ascend/SGLang 环境下并非逐字节确定，因此这里不把跨请求文本相等作为新策略的精度证明。数值正确性仍由 `fused_moe_w4a16_cached` 的全命中、混合命中和全 miss 对照测试覆盖；此前相同 seed/prompt 的单次缓存与全 CPU 输出逐字节一致。层完整策略只改变 slot 所有权集合，不改变 kernel 数学路径。

## 可用于论文的结论

1. 在仅缓存 16.67% 专家实例时，路由命中率达到约 68.5%，证明 Qwen3 decode routing 存在明显时间局部性。
2. 固定 2.53 GiB 活动缓存使 MoE 权重主体留在 CPU，同时保持 NPU Graph 和精度无损双路径。
3. 动态缓存存在容量/控制开销拐点：K≤512 时收益不足；K=1024 且稳态低频更新时，相对全 CPU 获得约 3.8% 端到端提升。
4. 启动填充与稳态替换必须采用不同速率。实现据此使用快速批量填充、满载后 8× 周期和每轮最多 8 个替换。
5. 当前结论适用于 TP=1、单请求 decode。多并发、TP>1 和更长工作负载应作为后续实验，不外推未经验证的加速比。

## 复现

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export NANOVLLM_TP_SIZE=2
export GLOO_SOCKET_IFNAME=lo

sglang serve \
  --model-path /mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32 \
  --tp-size 1 --attention-backend ascend \
  --enable-moe-expert-cache \
  --moe-expert-cache-size 1024 \
  --moe-expert-cache-swap-per-update 128 \
  --moe-expert-cache-update-interval 32 \
  --moe-expert-cache-warmup-steps 16 \
  --moe-expert-cache-placement layer \
  --cuda-graph-bs 1 2 4 8 --random-seed 1234
```

达到 batch=1 目标的配置使用 `--moe-expert-cache-size 4096`、`--moe-expert-cache-swap-per-update 512`、`--moe-expert-cache-update-interval 16` 和 `--mem-fraction-static 0.60`。
