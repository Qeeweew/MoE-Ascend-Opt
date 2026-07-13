# MoE 专家粒度动态缓存调度设计方案

> 对应选题报告 §2.3「兼容静态图执行的动态专家管理与协同调度」
> 本文结合 sglang 框架代码与现有 `moe_ascend_npu` / `nanovllm_ext` 实现，给出工程可落地的设计。

---

## 0. 一句话定位

在 NPU 上维护一块**固定大小**的专家权重缓存（K 个 slot，跨所有 MoE 层共享），按 **routing 激活频率**动态决定哪些专家实例 `(layer, expert)` 驻留 NPU：命中专家走 NPU W4A16 kernel，未命中专家走 CPU MoEInfer，两者各算各的加权贡献再求和。**精度无损**（所有专家都计算），**不碰 KV Cache**（NPU 显存预算固定，不随上下文长度变化），**不破坏静态图**（缓存替换在图外 step 边界完成，图内只有固定 op）。

### 0.1 当前实现状态

工程实现已落在独立包和扩展中，不修改 SGLang / sgl-kernel-npu：

- `fused_moe_w4a16_cached` 使用 slot id 索引固定 cache，`-1` 在 kernel 内安全跳过。
- CPU partial 路径把 NPU hit 对应的 expert id 改成 `-1`，复用 MoEInfer 已有的无效路由过滤，不重复计算命中专家。
- graph callback 同时接收原始 routing ids，按真实 token 数排除 graph padding 后累计频率。
- `ExpertCacheManager` 跨层维护固定 cache、slot table、滑动窗口 LFU、10% 替换滞回和备用 slot。
- SGLang `CudaGraphRunner.replay` 通过 monkey patch 提供图外 step 边界；每次发布均排在下一次 replay 之前，不 recapture graph。
- Ascend 实际使用覆盖通用实现的 `NPUGraphRunner.replay`，实现同时 patch 两个 runner，避免 NPU 路径绕过控制器。
- decode 小 batch 的 CPU remainder 已专门优化：全 miss 保持原 2×TopK 线程预算；只要有 NPU hit，CPU 将 NUMA 节点线程预算重分配给剩余 miss experts，使“少一个 CPU expert”能转化成 wall latency 下降。

首版交付边界是 Qwen3 compressed-tensors 对称 Int4、TP=1。多 rank 的统一决策与 delta broadcast 留作后续扩展。

固定 prompt 实验曾观察到 K=256 为 41.29 tok/s、相对 CPU Q4 的 38.57 tok/s 提升约 7.1%，K=512 为 44.37 tok/s。该结果来自重复相同 prompt 的稳态特例，K=256 命中率约 38.3%，不能泛化到真实多请求分布。2026-07-13 的同 seed、同 32 条 ShareGPT matched A/B 中，CPU Q4 为 37.79 tok/s，K=256 为 37.05 tok/s（-1.96%），平均窗口命中率仅约 13.1%。因此 K=256 当前只是实验默认点，尚不能称为通用正收益配置。完整记录见 `docs/bench_results/dynamic_expert_cache_qwen3.md`。

固定 prompt 复现实验由 `moe_ascend_npu/tests/benchmark_expert_cache_prompt.py` 自动完成，输出每个 cache size 的请求延迟、输出 hash、cache hit window、更新退避和 local decode throughput。

---

## 1. 为什么是专家粒度

### 1.1 层粒度方案为何被放弃

前期设计曾考虑「KV Cache 水位驱动的层粒度弹性卸载」——运行时根据 KV 占用动态增减 NPU 驻留的 MoE 层数。该方案在 sglang 架构下碰到了无法绕开的死结：

- **KV 池不可运行时扩容**：sglang 的 `MHATokenToKVPool` 在启动时一次性 `torch.empty` 分配固定大小（`memory_pool.py:1010`），运行时腾出的显存进不了 KV 池。
- 试图用「统一显存池」（KV 与 MoE 权重共享一块 buffer、动态分界）来解，在 NPU 上验证发现 `set_(storage, offset, ...)` 的 offset 语义有坑（按元素而非字节算），别名陷阱遍布，且要拦截 sglang 整个启动流程，工程量与风险远超收益。
- 层粒度的「切档」还需 recapture 静态图，引入秒级阻塞。

### 1.2 专家粒度如何回避死结

专家粒度方案把"动态"从**显存预算变化**转移到**缓存内容变化**：

- NPU 专家缓存 buffer 大小**启动时固定**（K 个 slot），永不 realloc。腾不出显存给 KV，也**不需要**——本方案不追求"长上下文时多卸载"，而是追求"有限显存下把最热的专家留在 NPU 加速"。
- 缓存替换是**原地写入**固定 slot（换权重内容，不换地址），graph 捕获的 data_ptr 稳定，**无需 recapture**。
- 精度无损：所有专家都被计算，只是按所在位置（NPU cache / CPU store）拆成两路加权求和。

### 1.3 前提：routing 局部性

方案成立的前提是 MoE routing 有足够局部性——少数专家实例被高频激活，缓存它们能覆盖大部分计算。这一前提在部分模型的部分层上已确认成立。若某层 routing 均匀分布，该层缓存收益低（但不会出错，只是退化为大部分走 CPU）。

---

## 2. 核心数据结构

### 2.1 NPU 专家缓存（固定 buffer）

```
w13_cache: [K+M, hidden, 2*inter_dim/8]  int32  # NPU kernel layout
w13_scale: [K+M, hidden/GS, 2*inter_dim] bf16
w2_cache:  [K+M, inter_dim, hidden/8]    int32
w2_scale:  [K+M, inter_dim/GS, hidden]   bf16
```

K 由启动参数决定，M 是每轮最大换入数并兼作备用 slot 数。每个 slot 存一个完整专家实例，格式与 `repack_int4_npu` 输出一致，直接被 `fused_moe_w4a16_cached` 消费。

### 2.2 slot table（图内读取，图外更新）

```
slot_table: [num_layers, num_experts] int32   # NPU tensor，原地更新
            值 = slot_id ∈ [0, K) 表示该 (layer, expert) 驻留在该 slot
            值 = -1 表示未缓存（miss，走 CPU）
```

反向索引 `slot_owner: [K] -> (layer, expert)` 仅在 CPU 侧 Python 维护，用于替换决策，不进图。

### 2.3 频率统计（滑动窗口 LFU）

```
freq: [num_layers, num_experts] float32  # CPU tensor
      每 step 累加该层各 expert 的激活次数（按 topk_ids 统计）
      滑动窗口衰减（如 freq = 0.95*freq + new_count）
```

用于替换决策：找出高频未缓存（换入候选）与低频已缓存（换出候选）。

### 2.4 CPU 全量后备

现有 `MoEInfer`（per-layer handle）持**全部**专家权重（q4_0 或 q8_0 格式，CPU 内存）。新方案下 CPU 仍是全量后备——任何 miss 专家的权重都在 CPU，可直接计算。CPU store 与 NPU cache 格式独立（CPU 用自己的 q4_0 布局，NPU 用 W4A16 布局）。

---

## 3. forward 流程（图内，精度无损）

以单层 MoE 为例（`apply` 内）：

```python
x          = dispatch_output.hidden_states     # [BS, hidden]
topk_ids   = dispatch_output.topk_output.topk_ids      # [BS, TopK] int32
topk_w     = dispatch_output.topk_output.topk_weights  # [BS, TopK] float32

# (1) gather slot_ids：expert_id -> slot_id（-1 表 miss），图内固定 op
slot_ids = slot_table[layer_idx].gather(0, topk_ids.flatten()).view_as(topk_ids)
hit_mask = (slot_ids >= 0)                     # [BS, TopK] bool

# (2) NPU 路径：只算命中专家，miss 贡献 0
npu_out = fused_moe_w4a16_cached(
    x, w13_cache, w13_scale, w2_cache, w2_scale,
    slot_ids, topk_w,                          # kernel 内对 slot_id<0 跳过
)                                              # [BS, hidden]

# (3) CPU 路径：只算未命中专家，hit 贡献 0
cpu_out = moe_infer.compute_partial(
    x, topk_ids, topk_w, ~hit_mask,            # execute_fn 内对 hit 跳过
)                                              # [BS, hidden]

# (4) 合并：加权求和可分解
out = npu_out + cpu_out
return StandardCombineInput(hidden_states=out)
```

### 3.1 正确性

MoE 输出是加权求和：

```
out[i] = Σ_{k=0}^{TopK-1} topk_w[i,k] · Expert_{topk_ids[i,k]}(x[i])
```

按专家所在位置分解：

```
out[i] = Σ_{k, hit}  w[i,k]·Expert(x[i])    # NPU 算
       + Σ_{k, miss} w[i,k]·Expert(x[i])    # CPU 算
       = npu_out[i] + cpu_out[i]
```

所有专家都被计算，无近似、无丢弃，**精度无损**。

### 3.2 graph 友好性

- `slot_table` 是固定地址 NPU tensor，graph 捕获其 data_ptr；replay 时读到的是**当前内容**（图外 step 边界更新后的）。
- `w13_cache` / `w2_cache` 同理，固定地址，原地换内容。
- `gather` / `>=` / kernel / add 都是固定 op，图内序列不变。
- **无需 recapture**——这是相比层粒度方案最大的工程优势。

---

## 4. 缓存替换（图外，step 边界）

替换由 `CudaGraphRunner.replay` 的外部 hook 在低频更新周期执行，不改变当前捕获图。CPU 路由计数通过内部 mutex 获取一致快照（尚未完成的 callback 自然计入下一窗口），不做全设备同步；repack、copy 和 table 更新顺序提交到主 NPU stream：

```
每个更新周期、下一次 replay 前（图外）：
  (1) 统计：freq[layer] += topk_ids 的直方图（衰减后）
  (2) 决策：找出换入候选（高频未缓存）与换出候选（低频已缓存）
  (3) 执行（NPU stream，使用备用 slot）：
      for (layer, expert) in 换入:
          slot = 选一个换出 slot（释放其 (layer,expert)）
          w = cpu_store[layer].get_expert(expert)        # CPU 内存
          w_npu = repack_int4_npu(w)                      # 转成 NPU W4A16 格式
          cache[slot].copy_(w_npu, non_blocking=True)     # H2D，原地写入固定 slot
          slot_table[layer, expert] = slot                # 原地更新
          slot_table[old_layer, old_expert] = -1          # 原地更新
  (4) slot table 更新排在下一次 graph replay 之前
```

### 4.1 一致性

替换期间（异步 H2D 进行中），slot_table 尚未更新（先搬数据，搬完才改 table）。因此当前 step 的图 replay 读到的 slot_table 仍是旧值，指向旧权重——**旧权重在 slot 里直到新权重搬完才被覆盖**。只要保证"先搬完数据再改 table"，就不会读到半搬状态。

实现保留 M 个备用 slot：新权重始终写入未发布 slot，写入提交后才切换 table；旧 slot 解除映射后成为下一轮备用 slot，避免覆盖仍可能被读取的活动权重。

### 4.2 替换粒度与频率

- 每次 step 最多换 M 个专家（M 小，如 4-8），控制 H2D 带宽开销。
- 启动填充按配置周期执行；缓存满载后更新周期先放大 8 倍。若连续两个稳态窗口没有替换，周期继续按 2 倍退避，最高 32 倍；一旦发生替换或窗口命中率明显下滑，恢复到 8 倍。这样保留动态自适应能力，同时避免无效统计成为 decode 固定开销。
- 每个 expert 权重 ≈ 3MB（Qwen3-30B TP2 下），H2D 3MB ≈ 0.3ms（PCIe ~10GB/s），M=8 ≈ 2.4ms，可在 step 间隙吸收。
- 替换是**分钟级低频**事件的细粒度版——稳态下命中率稳定后替换率趋近 0。

---

## 5. 改造点（对应具体文件）

| # | 改造点 | 文件 | 说明 |
|---|--------|------|------|
| 1 | NPU kernel 支持 slot 索引 + miss 跳过 | `csrc/op_host/grouped_gemv_w4a16_moe.cpp` + kernel | 新增 `fused_moe_w4a16_cached`：入参 `slot_ids`（-1 表 miss）替代 `expert_ids`，权重张量改为 cache buffer `[K,...]`，kernel 内 `if (slot<0) continue` |
| 2 | CPU partial + routing telemetry | `Int8-gemm/` | 图内把 hit 路由改为 `-1`，复用现有无效 ID 过滤；graph callback 额外统计原始 routing ids |
| 3 | 新建 `ExpertCache` | `moe_ascend_npu/cache.py`（新） | 持 cache buffer + slot_table + 频率统计；提供 `get_slot_table_layer(l)` / `swap_in(layer, expert)` / `record_routing(layer, topk_ids)` |
| 4 | 新建 `CacheController` | `moe_ascend_npu/cache.py`（新） | 单例，按更新窗口做 LFU 决策，在 graph replay 边界发布替换 |
| 5 | 双路径 `apply` | `moe_ascend_npu/patches/expert_cache_method.py`（新） | 替换现有 `fused_moe_method` 的 apply：gather slot_ids → CPU side stream 提交 partial → NPU cached kernel → join/add |
| 6 | 接入 sglang MoE 层 | `patches/moe_layer.py` | cache 打开时所有 MoE 层装 `ExpertCacheFusedMoEMethod`，共同竞争同一个全局 expert pool |
| 7 | CLI 参数 | `patches/server_args.py` | `--moe-expert-cache-size K`、`--moe-expert-cache-swap-per-step M`、`--moe-expert-cache-decay 0.95` |

### 5.1 kernel 改动细节（最小）

当前 `fused_moe_w4a16_small_bs` 的 kernel 循环：

```cpp
// 现状
int expert = expert_ids[t];
auto* w = weight[expert];        // 直接索引
// 计算 + 累加
```

改为：

```cpp
// 新增 cached 版本
int slot = slot_ids[t];          // -1 表 miss
if (slot < 0) continue;          // 跳过，贡献 0
auto* w = cache[slot];           // 索引 cache buffer
// 计算 + 累加（逻辑不变）
```

权重张量从 `[num_experts, ...]` 变为 `[K, ...]`，索引语义由外部 gather 完成。**核心计算逻辑零改动**。

### 5.2 CPU MoEInfer 改动细节

CPU routing 预处理本来就忽略负 expert id，因此无需修改 GEMM 主循环：

```cpp
// 现状：对每个 (token, k) 都算
for (t ...) {
    int expert = topk_ids[t];
    // 算 + 累加到 y_out
}

cpu_ids = where(hit_mask, -1, topk_ids)
// preprocess_moe_routing 已忽略 expert_id < 0，对应 hit 不生成 CPU task
```

---

## 6. 与现有两块工作的关系

| 模块 | 角色 | 新方案下的定位 |
|------|------|---------------|
| NPU W4A16 kernel（§2.1） | 快路径 | **命中专家**走此路径，加 cached 变体 |
| CPU offload MoEInfer（§2.2） | 省显存路径 | **未命中专家**走此路径，加 partial 变体；仍是全量后备 |
| 专家缓存调度（§2.3，本设计） | 协同 | 决定哪些专家走 NPU、哪些走 CPU，动态调整 |

三块形成完整叙事：NPU 算子提供单专家高速计算，CPU 算子提供全量后备，缓存调度决定分工——**在固定 NPU 显存预算下，让最热的专家留在 NPU，其余下沉 CPU，两者协同**。

---

## 7. 实施路线图

| 阶段 | 内容 | 交付物 | 验证 |
|------|------|--------|------|
| **P0** | routing 统计工具 | 脚本收集 topk_ids，输出各层频率分布 + 不同 K 的命中率曲线 | 确认局部性、选定 K 值（非关卡，用户已确认局部性成立） |
| **P1** | NPU cached kernel | `fused_moe_w4a16_cached`（slot 索引 + miss 跳过） | 单层 correctness：构造全 hit / 全 miss / 混合，输出与原 kernel 一致 |
| **P2** | CPU partial MoEInfer | `compute_partial`（hit_mask 跳过） | 单层 correctness：partial + cached_kernel = 原 full MoE |
| **P3** | ExpertCache + 同步替换 | `cache.py`（buffer + slot_table + 频率 + swap_in） | 手动 swap 后 correctness；命中率随 swap 提升 |
| **P4** | 异步替换 + Controller | `maybe_swap` 异步 + 下 step 同步 | 压测下命中率稳定，替换不阻塞 decode |
| **P5** | 集成 sglang + CLI | patch moe_layer / server_args | 端到端正确性（Qwen3-30B），对比全 NPU / 全 CPU offload / 缓存 |
| **P6** | TP 同步 + 调优 | controller TP 一致性 | TP=2/4 切档一致；K/M/decay 参数扫描 |

---

## 8. 风险与开放问题

1. **swap 期间的一致性**：实现用 M 个备用 slot，并在 graph replay 外按 stream 顺序执行“写备用 slot → 更新 table → replay”；活动 slot 不被原地覆盖。

2. **CPU partial 的延迟**：未命中专家走 CPU，decode 小 batch 时 CPU 计算仍是瓶颈（现有全卸载 ~37 tok/s）。固定 TP2、每 NUMA 节点 20 线程的精确测试中，8 routes 为 0.2661 ms、7 routes 为 0.2513 ms，下降 5.56%；8 routes 到 4 routes 下降 36.4%。多个专家在固定线程池中并行执行，因此 wall latency 不会按 route 数线性下降。缓存收益必须结合完整 route-count 曲线和实际命中分布估计，不能直接把命中率当作延迟降幅。

3. **TP 一致性**：多卡下各 rank 的缓存内容必须一致（同一 (layer,expert) 要么全卡命中要么全卡 miss），否则 TP 通信错乱。Controller 决策在 rank-0，broadcast slot_table delta 给所有 rank。

4. **prefill vs decode 路由差异**：prefill 大 bs 与 decode 小 bs 的 routing 分布可能不同。缓存若按 decode 统计填充，prefill 时命中率可能下降。缓解：prefill 强制全走 CPU（反正 prefill 大 bs CPU 也扛得住），或独立统计。

5. **swap-in 的 repack 开销**：CPU store 是 q4_0 格式，NPU cache 是 W4A16 格式，swap-in 需 repack。repack 在 CPU 端做（`repack_int4_npu`）再 H2D，还是 H2D 原始格式再 NPU 上 repack，P3 实测择优。

---

## 9. 与选题报告的对应关系

| 选题报告 §2.3 原文 | 本方案对应 |
|-------------------|-----------|
| 「基于激活频率的动态缓存策略：高频专家常驻 NPU，长尾卸载 CPU」 | §2-4 专家实例粒度 LFU 缓存（工程主线） |
| 「KV Cache 驱动的弹性卸载」 | **放弃**（KV 池不可扩容，§1.1）；改为"固定显存预算下的专家缓存"，动态性体现在缓存内容而非显存预算 |
| 「静态图中的动态路由：虚拟专家节点」 | §3 图内 gather/mask/kernel 实现"运行时按 slot_table 分流"，对上层透明；无需 recapture（§3.2） |
| 「对上层框架透明的异构协同推理」 | NPU cached + CPU partial 双路 add，apply 接口不变，对 sglang 透明 |

**论文叙述**：在「静态图 + 精度无损」双约束下，token 级动态路由不可行（图内不能 if），层粒度弹性受限于 KV 池不可扩容；专家粒度缓存是约束下的可落地选择——"动态"体现为缓存内容的频率驱动替换，"协同"体现为 NPU/CPU 双路加权求和。与相关工作的差异只依据已经阅读全文后的约束和实验进行陈述，不作未经验证的“首次”断言。

---

## 10. 下一步

1. **P0 routing 统计**（半天）：写脚本跑一次推理收集 topk_ids，输出频率分布与命中率曲线，选定 K。
2. **P1 cached kernel**（1-2 天）：最小改动，单层 correctness。
3. **P2 CPU partial**（1 天）：correctness = cached + partial == full。
4. **P3-P5** 依次推进，每阶段单测 + 端到端。
5. 基准：对比「全 NPU（显存够时）」「全 CPU offload」「专家缓存（固定显存预算）」三档的吞吐与显存，量化缓存命中率 → 加速比的对应关系。
