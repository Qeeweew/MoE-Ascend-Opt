# MoE-Ascend-Opt

面向昇腾 NPU 与鲲鹏 CPU 的 MoE 推理优化工程。项目在不修改上游 SGLang 和
`sgl-kernel-npu` 源码的前提下，通过 `moe_ascend_npu` 独立包、运行时 monkey
patch 与 `.pth` 自动注入接入优化能力。

当前重点是**专家粒度动态卸载**：在固定 NPU 显存预算内缓存高频
`(layer, expert)` 实例，命中时使用 NPU W4A16 kernel，未命中时由 CPU Q4_0
后备计算。它不是 KV Cache 卸载，缓存容量在启动时固定，运行中只替换 slot 内容。

## 能力概览

- NPU：Ascend C W4A16 fused MoE kernel，以及 Int4 权重 repack。
- CPU：鲲鹏 ARM NEON / NUMA 感知的 Int8、Int4 MoE 后备引擎（`Int8-gemm/`）。
- 动态专家缓存：跨所有 MoE 层共享固定 slot 池，按 decode route 频率的 LFU
  策略管理热点专家。
- 静态图兼容：缓存 buffer 和 `slot_table` 地址固定；替换在 graph replay 前的
  图外边界发布，不需要 recapture graph。
- 精度无损分流：NPU 只计算 cache hit，CPU 只计算 miss，二者的加权输出相加；
  不丢弃任何被路由的专家贡献。

```text
topk routing
    │
    ├─ slot_table 命中 ──> NPU cached W4A16 MoE ─┐
    └─ slot_table 未命中 -> CPU Q4_0 partial MoE ─┼─> add -> MoE 输出
                                                   │
图外 replay hook: 统计 decode route -> LFU 决策 -> 写备用 slot -> 发布 slot_table
```

## 支持范围

动态专家缓存当前已在 Qwen3 compressed-tensors 对称 Int4、TP=1 路径验证。
它与全层 CPU 卸载 `--enable-moe-offload` 互斥，因为缓存模式本身已包含 CPU
partial 后备。TP>1 的统一决策和 slot delta 广播尚未完成。

请保持 CUDA/NPU Graph 开启；不要为缓存模式传 `--disable-cuda-graph`。缓存替换
依赖固定地址 tensor 和 graph 外 replay hook，关闭 graph 会偏离该路径的设计目标。

## 安装与构建

运行环境需要 Ascend NPU、CANN 工具链和 NPU 版 SGLang。CPU 后备依赖 ARMv8.2+
dotprod 环境；若只使用 NPU kernel，`Int8-gemm` 可以不安装。

```bash
# 安装 Python 包
pip install -e moe_ascend_npu/

# 构建 Ascend C kernel；脚本会安装 .pth 自动注入钩子
cd moe_ascend_npu
bash build_kernels.sh Ascend910_9382

# 动态缓存或全层 CPU 卸载所需的 CPU 引擎
cd ..
pip install -e Int8-gemm/ --no-build-isolation
```

`.pth` 会在解释器启动后等待 SGLang 导入完成，并仅在 NPU 进程安装补丁。需要
手动重装或删除该钩子时：

```bash
python -m moe_ascend_npu._install_pth install
python -m moe_ascend_npu._install_pth remove
```

## 启动动态专家缓存

下面是 Qwen3 TP=1 的示例。默认值为 `K=256`、每轮最多替换 8 个专家、基础更新间隔
16 个 decode replay、warmup 16 step、route 频率衰减 0.95。服务启动时会在 KV Cache
规划前将 K 个 slot 按层均匀填满；实际部署应按显存预算和目标工作负载测量 K，
而不是把默认值视为最佳值。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export NANOVLLM_TP_SIZE=2

sglang serve \
  --model-path /path/to/Qwen3-MoE-compressed-tensors-int4 \
  --trust-remote-code \
  --tp-size 1 \
  --attention-backend ascend \
  --enable-moe-expert-cache \
  --moe-expert-cache-size 256 \
  --moe-expert-cache-swap-per-update 8 \
  --moe-expert-cache-update-interval 16 \
  --moe-expert-cache-warmup-steps 16 \
  --moe-expert-cache-decay 0.95 \
  --cuda-graph-bs 1 2 4 8
```

关键参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--enable-moe-expert-cache` | 关闭 | 启用固定预算的动态专家缓存 |
| `--moe-expert-cache-size` | 256 | 全局活动 expert-instance slot 数 |
| `--moe-expert-cache-swap-per-update` | 8 | 备用 slot 数和最大替换批量；缓存满载后每轮最多替换 8 个 |
| `--moe-expert-cache-update-interval` | 16 | decode replay 的基础控制周期；满载后自动退避 |
| `--moe-expert-cache-warmup-steps` | 16 | 开始动态替换前的有效 decode step 数 |
| `--moe-expert-cache-decay` | 0.95 | decode route 频率 EMA 衰减系数 |

Prefill 可以使用已有专家缓存，命中走 NPU、未命中走 CPU，但不进入热度统计，也不会触发替换；只有
decode 驱动控制器。均匀 seed 后只在低频 decode 窗口做最多 8 个替换。替换按
“写备用 slot → 发布 `slot_table` → 下一次 replay”执行，避免读到未完成搬运的权重。

全层 CPU 卸载是独立模式：

```bash
sglang launch-server \
  --model-path /path/to/model \
  --attention-backend ascend \
  --enable-moe-offload \
  --moe-offload-quant-type q4_0
```

`--enable-moe-expert-cache` 和 `--enable-moe-offload` 不能同时使用。

## 性能验证

缓存收益取决于实际路由局部性、cache size、请求分布与控制面开销，不能由命中率
线性推导。请用仓库中的 benchmark 脚本在目标模型与工作负载上复测；固定 prompt
稳态结果不能外推为通用服务性能。

## 测试

```bash
# NPU kernel、repack、缓存策略与 benchmark 脚本测试
cd moe_ascend_npu
bash tests/run_all.sh

# 动态缓存端到端 smoke test
python tests/benchmark_expert_cache_prompt.py \
  --skip-cpu --cache-sizes 256 --requests 2 --output-tokens 32 \
  --result-dir /tmp/moe_cache_prompt_smoke
```

## 项目结构

```text
MoE-Ascend-Opt/
├── moe_ascend_npu/
│   ├── csrc/                         # Ascend C W4A16 kernel
│   ├── moe_ascend_npu/
│   │   ├── cache.py                  # 固定 slot cache、LFU controller
│   │   ├── kernels/repack.py         # Triton Int4 repack
│   │   └── patches/                  # SGLang monkey patch 与 graph/cache 接入
│   ├── tests/                        # 正确性、策略与缓存 benchmark
│   └── docs/architecture.md          # 包架构与完整参数
├── Int8-gemm/                        # nanovllm_ext：ARM NEON / NUMA CPU 引擎
├── docs/design/dynamic_offload.md    # 动态专家缓存设计
└── docs/bench_results/               # 可复现实验记录
```

## 文档

- [包架构与参数](moe_ascend_npu/docs/architecture.md)
- [专家粒度动态缓存设计](docs/design/dynamic_offload.md)
