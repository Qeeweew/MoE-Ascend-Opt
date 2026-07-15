# MoE-Ascend-Opt

面向 Ascend NPU + Kunpeng CPU 的 MoE 推理优化工程。核心交付是独立包
`moe_ascend_npu/`：它以 monkey patch 和 `.pth` 自动注入的方式扩展官方
SGLang，而不维护 SGLang 或 `sgl-kernel-npu` 的源码分叉。

## 架构边界

- **不得修改** `sglang` 或 `sgl-kernel-npu` 源码。集成逻辑只能放在
  `moe_ascend_npu/moe_ascend_npu/patches/`；通过 `.pth` 的延迟 bootstrap
  在 NPU 进程导入 SGLang 后生效。
- `moe_ascend_npu/` 提供 Ascend C W4A16 MoE kernel、Triton Int4 repack 和
  SGLang 补丁；`Int8-gemm/` 独立构建为 `nanovllm_ext`，提供 ARM NEON / NUMA
  感知的 CPU MoE 后备。
- 动态专家缓存由 `cache.py`、`expert_cache_method.py` 和 `cache_graph.py`
  实现。它是**固定显存预算的跨层 expert-instance 缓存**，不是 KV Cache
  动态扩容，也不是按层整体换入换出。

## 动态专家缓存规则

- NPU cache、备用 slot 和 `slot_table` 在启动时一次性分配；之后仅原地更新
  内容与表项，不能因缓存替换改变 tensor 地址或触发 graph recapture。
- 活动 slot 在 KV Cache 规划前按层均匀填满；Prefill 可以读取现有缓存，命中走
  NPU、未命中走 CPU，但不参与路由热度统计或替换，动态更新仅由 decode 驱动。
- 命中路由走 `fused_moe_w4a16_cached`；未命中路由走 CPU Q4_0 partial MoE；
  两路加权结果相加。所有被路由专家必须计算，不能以命中率为由丢弃贡献。
- 路由热度统计须排除 NPU Graph padding。替换在 graph replay 之前的图外边界
  发布，遵循“写备用 slot → 发布 slot table → 下一次 replay”的顺序。
- 缓存评分使用 decode route 频率；不能用 gate score 代替计算成本，所有被路由
  专家仍必须完整计算。缓存满载后每个更新窗口最多替换 8 个，避免大批量抖动。
- 保持 CUDA/NPU Graph 开启；**不要随意传 `--disable-cuda-graph`**。动态缓存
  与 `--enable-moe-offload`（全层 CPU 卸载）互斥。
- 当前已验证边界：Qwen3 compressed-tensors 对称 Int4、TP=1。TP>1 需要
  rank-0 决策与 slot delta 广播后才可宣称支持。
- kernel 不得硬编码 batch-size 上限。“小 batch”描述的是私有化 decode 场景，
  不是接口限制。

## 常用命令

```bash
# Python 补丁包与 CPU 后备（后者仅全层卸载/动态缓存需要）
pip install -e moe_ascend_npu/
pip install -e Int8-gemm/ --no-build-isolation

# 构建 Ascend C kernel；构建脚本会安装 .pth 自动注入钩子
cd moe_ascend_npu && bash build_kernels.sh

# 若需要手动重装/移除自动注入钩子
python -m moe_ascend_npu._install_pth install
python -m moe_ascend_npu._install_pth remove

# NPU 正确性与缓存策略测试
cd moe_ascend_npu && bash tests/run_all.sh
```

## 验证与文档

- 修改 cache policy、slot 发布或双路径计算后，至少运行
  `moe_ascend_npu/tests/test_cache_policy.py`；涉及 kernel / repack 时运行
  `bash moe_ascend_npu/tests/run_all.sh`。
- 端到端缓存实验使用
  `moe_ascend_npu/tests/benchmark_expert_cache_prompt.py`；报告性能时区分固定
  prompt 稳态结果与 matched ShareGPT A/B，不能将前者外推为通用收益。
- 包架构和完整参数：`moe_ascend_npu/docs/architecture.md`
- 专家缓存设计：`docs/design/dynamic_offload.md`
- 不要对未阅读全文的论文作实现、对比或“首次”断言。
