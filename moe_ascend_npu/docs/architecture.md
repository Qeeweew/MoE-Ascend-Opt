# moe_ascend_npu 架构与参数说明

本文档描述 `moe_ascend_npu` 包的整体架构、模块职责、数据流、以及所有可配置参数（构建参数、CLI 参数、环境变量、kernel 张量参数、布局约定）。

`moe_ascend_npu` 是一个**独立的 Python 包**，把两个 MoE 优化功能通过 **monkey patching** 在运行时注入**官方未修改的** SGLang，无需改动 `sglang` / `sgl-kernel-npu` 源码：

1. **W4A16 融合 MoE NPU kernel**（解码 / 小 batch 场景，加速 Ascend 910B3）
2. **W4A16 MoE CPU 计算卸载**（Int8 / Int4，利用 ARM NEON 多核 CPU 分担 NPU 压力）

---

## 1. 顶层架构

```
                       interpreter start-up
                                │
                ┌───────────────▼────────────────┐
                │  site-packages/moe_ascend_npu.pth │   (.pth 自动 import)
                │  -> import moe_ascend_npu._bootstrap │
                └───────────────┬────────────────┘
                                │ 安装 builtins.__import__ wrapper
                                │ (每次 import 计数, depth 回到 0 且
                                │  sglang.srt 已在 sys.modules 时触发)
                                ▼
                ┌────────────────────────────────┐
                │  patches.apply_patches()       │
                │  - is_npu() 守卫, 非 NPU 跳过    │
                │  - 7 个 setattr 形式的 patch    │
                └───────────────┬────────────────┘
                                │ 永久替换官方 SGLang 的方法/类
                                ▼
        ┌───────────────────────┴────────────────────────┐
        ▼                                                ▼
  功能 A: W4A16 NPU kernel                      功能 B: MoE CPU 卸载
  (Ascend C + Triton, torch.ops.moe_ascend_npu.*)  (nanovllm_ext, torch.ops.nanovllm.*)
```

**设计要点**

- **零侵入**：`sglang`（`/sgl-workspace/sglang`，0.5.13.post1）与 `sgl_kernel_npu`（2026.6.1）均为官方版本，不做任何修改。所有改动通过 `setattr` 在进程内替换官方类的方法。
- **延迟触发**：bootstrap 不在 import 时立刻打补丁，而是等 SGLang 整棵 import 树结束（depth 回到 0）再打，避免 `sglang.srt.__init__` 半初始化时的循环 import。
- **NPU 守卫**：`apply_patches()` 内调用 `sglang.srt.utils.is_npu()`，非 NPU 进程直接返回，不打任何补丁。
- **非 SGLang 进程零开销**：bootstrap 的 import wrapper 对普通 import 只多做一次 `depth==0` 的字典查找 `sglang.srt in sys.modules`；SGLang 永不 import 的进程（如 `pip`、测试脚本）完全无感知。触发后 wrapper 立即解包恢复原 `__import__`。
- **op 命名空间隔离**：NPU kernel 注册到 `torch.ops.moe_ascend_npu.*`，与官方 `sgl_kernel_npu` 的 `npu` 命名空间完全分离，不冲突、可共存。

---

## 2. 目录结构

```
moe_ascend_npu/
├── build_kernels.sh              # CMake 构建 Ascend C kernel + 安装 .pth
├── CMakeLists.txt / cmake/       # ascendc 构建配置
├── csrc/                         # Ascend C host + kernel 源码 (namespace moe_ascend_npu)
│   ├── pytorch_extensions.cpp    # 注册 torch.ops.moe_ascend_npu.*
│   ├── op_host/grouped_gemv_w4a16_moe.cpp   # 三个 op 的 host 端实现 (tiling/launch)
│   ├── op_kernel/grouped_gemv_w4a16_moe.cpp # device kernel
│   └── utils/                    # torch/ge helper, tiling, version 等
├── include/moe_ascend_npu_ops.h  # host API 声明
├── moe_ascend_npu.pth            # 内容: import moe_ascend_npu._bootstrap
├── moe_ascend_npu/               # Python 包
│   ├── __init__.py
│   ├── _bootstrap.py             # 延迟 __import__ hook -> apply_patches()
│   ├── _install_pth.py           # python -m moe_ascend_npu._install_pth [remove]
│   ├── kernels/
│   │   ├── __init__.py           # 加载 .so, 暴露 ensure_kernels_loaded()
│   │   └── repack.py             # Triton repack_int4_npu kernel
│   ├── layers/
│   │   ├── linear_method.py      # NPUW4A16LinearMethod (Linear 用)
│   │   └── wna16.py              # NPUCompressedTensorsW4A16 (Linear scheme)
│   └── patches/                  # 7 个 SGLang monkey patch + offload 实现
│       ├── __init__.py           # apply_patches() 编排
│       ├── server_args.py        # +CLI 参数
│       ├── moe_layer.py          # FusedMoE.__init__ 拦截卸载
│       ├── offload.py            # CPU 卸载两个 Method (依赖 nanovllm_ext)
│       ├── compressed_tensors.py # NPU W4A16 Linear scheme 路由
│       ├── wna16_moe.py          # scale/offset dtype 修正
│       ├── fused_moe_method.py   # repack + 小 batch kernel 路径
│       └── minimax_m2.py         # 去 is_cuda 断言
├── tests/                        # NPU 正确性测试
└── docs/                         # 本文档
```

**未改动的相关仓库**（保留为子模块 / 参考）：
- `sglang/`（`/sgl-workspace/sglang`）— 官方 SGLang，被 patch 的目标
- `sgl-kernel-npu/` — 官方 sgl_kernel_npu 源码参考
- `Int8-gemm/` — 独立构建为 `nanovllm_ext`（CPU offload 的运行时依赖）

---

## 3. 功能 A：W4A16 融合 MoE NPU Kernel

### 3.1 提供的 op

构建产物 `moe_ascend_npu/lib/libmoe_ascend_npu_kernels.so` 在 `torch.ops.moe_ascend_npu` 命名空间下注册 3 个 op（见 `csrc/pytorch_extensions.cpp`）：

#### op 1: `grouped_gemv_w4a16_moe`

单 token-per-expert 的 GEMV，供解码路径或 grouped 推理使用。

```
y = torch.ops.moe_ascend_npu.grouped_gemv_w4a16_moe(x, weight, scales, expert_ids)
```

| 参数 | 形状 | dtype | 说明 |
| :-- | :-- | :-- | :-- |
| `x` | `[BS, K]` 或 `[TotalTokens, K]` | fp16 / bf16 | 激活。若为 `[BS,K]` 会自动 expand 成 `[BS*TopK, K]` |
| `weight` | `[E, K, N//8]` | int32 | 每个 expert 的 W4A16 权重，8 个 int4 打包进 1 个 int32 |
| `scales` | `[E, K//G, N]` | 同 x | 每组 per-channel scale（G=GROUP_SIZE=32） |
| `expert_ids` | `[BS, TopK]` 或 `[TopK]` | int32 | 每个 token 路由到的 expert 编号 |
| 返回 `y` | `[TotalTokens, N]` | 同 x | GEMV 输出 |

约束：`TotalTokens = BS * TopK`；对称量化（无 offset 参数）。

#### op 2: `fused_moe_w4a16_small_bs`

完整两段式 MoE（W1·Gated + W3·Up → SwiGLU → W2·Down）+ topk 加权，单 kernel。小 batch（解码）专用。

```
y = torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs(
        x, w13_weight, w13_scales, w2_weight, w2_scales, expert_ids, topk_weights)
```

| 参数 | 形状 | dtype | 说明 |
| :-- | :-- | :-- | :-- |
| `x` | `[BS, InDim]` | fp16 / bf16 | 输入隐状态 |
| `w13_weight` | `[E, InDim, 2*InterDim//8]` | int32 | W1+W3 合并权重 |
| `w13_scales` | `[E, InDim//G, 2*InterDim]` | 同 x | W1/W3 scale |
| `w2_weight` | `[E, InterDim, InDim//8]` | int32 | W2 权重 |
| `w2_scales` | `[E, InterDim//G, InDim]` | 同 x | W2 scale |
| `expert_ids` | `[BS, TopK]` | int32 | 路由结果 |
| `topk_weights` | `[BS, TopK]` | fp32 | topk 路由权重（用于最终加权） |
| 返回 `y` | `[BS, InDim]` | 同 x | 加权融合后的输出 |

约束：`InDim == OutDim`（MoE 输出回到 hidden size）；kernel 内部分配 FP32 累加 workspace（W13 输出 + SwiGLU + W2 输出）。

#### op 3: `batch_gemm_w4a16_small_bs`

小 batch 的 W4A16 batched GEMM，作为 `npu_weight_quant_batchmatmul` 的替换（当前未接入 Linear 路径，仅测试验证）。

```
y = torch.ops.moe_ascend_npu.batch_gemm_w4a16_small_bs(x, weight, scales)
```

| 参数 | 形状 | dtype | 说明 |
| :-- | :-- | :-- | :-- |
| `x` | `[BS, K]` | fp16 / bf16 | **BS ≤ 4** |
| `weight` | `[K, N//8]` | int32 | 单权重（无 expert 维） |
| `scales` | `[K//G, N]` | 同 x | per-group per-channel scale |
| 返回 `y` | `[BS, N]` | 同 x | GEMM 输出 |

> **注意**：该 kernel 采用 split-K + atomic-add，因此**逐元素 max_diff 跨运行非确定性**（少数接近零的输出因灾难性抵消波动较大），但 `mean_diff` 稳定。测试用 `mean_diff` 断言。

### 3.2 Triton `repack_int4_npu`

`moe_ascend_npu/kernels/repack.py` 提供的权重重排 kernel，在 NPU Vector Core 上**一步完成** unpack + transpose + repack + uint4b8→int4 二进制补码转换。

```
repacked = repack_int4_npu(weight_packed_t)   # [K//8, N] -> [K, N//8]
```

- 输入 `[K//8, N]` int32（已转置，K 维是打包维，每个 int32 装 8 个 uint4b8 nibble）
- 输出 `[K, N//8]` int32（kernel 期望布局，每个 int32 装 8 个 int4 二进制补码 nibble）
- Persistent kernel：grid 锁定到 NPU Vector Core 数（默认 32，通过 `sgl_kernel_npu.utils.triton_utils.get_device_properties` 获取），`BLOCK_N8=256`

### 3.3 关键常量与布局约定

| 常量 | 值 | 出处 |
| :-- | :-- | :-- |
| `GROUP_SIZE` | **32** | kernel 硬编码常量（`csrc/op_host/grouped_gemv_w4a16_moe.cpp`），所有 scale 维度按此切组 |
| int4 打包 | 8 个 nibble / int32，低 nibble 在前 | packed = `nibble_i << (4*i)` |
| uint4b8 偏置 | 存储 `value + 8`（-8..7 → 0..15） | 对应 compressed-tensors `uint4b8` scalar type |
| repack 转换 | repack 过程中 `(nibble - 8) & 0xF` 把 uint4b8 转成 int4 二进制补码 | `repack.py` kernel 内 |

### 3.4 NPU W4A16 Linear scheme（`layers/`）

供 patch 路由使用，不直接由用户调用：

- **`NPUCompressedTensorsW4A16`**（`layers/wna16.py`）：`CompressedTensorsLinearScheme` 子类。构造时校验 `num_bits==4` 且 `symmetric==True`，内部持有一个 `NPUW4A16LinearMethod`。
  - `create_weights`：建 `weight_packed`（int32 `[N, K//8]`）、`weight_scale`、`weight_shape`，按 strategy（channel/group）选参数类型。
  - `process_weights_after_loading`：转交给 `NPUW4A16LinearMethod`。
  - `apply_weights`：调用 `torch.ops.npu.npu_weight_quant_batchmatmul`。
- **`NPUW4A16LinearMethod`**（`layers/linear_method.py`）：
  - `process_weights_after_loading`：`weight_packed` 转置后经 `repack_int4_npu` 重排成 `[K, N//8]`；`weight_scale` 转置成 `[K//G, N]`；补一个全零 `weight_offset`（对称量化下 op 仍要求非 None）。
  - `apply`：`torch.ops.npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, antiquant_group_size, bias)`。

---

## 4. 功能 B：MoE CPU 计算卸载

### 4.1 依赖与触发

- **硬依赖 `nanovllm_ext`**：由 `Int8-gemm/` 独立构建（`pip install -e Int8-gemm`），提供 `torch.ops.nanovllm.*` 与 `torch.classes.nanovllm.*`。
- **延迟 import**：`patches/offload.py` 在模块顶部 `import nanovllm_ext`（硬失败），但该模块**仅在启用 offload 的层构造时**才被 import（见 `patches/moe_layer.py`）。因此：
  - **不开 offload 的正常 NPU 服务**：不 import `offload.py`，无需 `nanovllm_ext`，7 个 patch 正常生效。
  - **开 offload 但没装 `nanovllm_ext`**：构造第一个命中层时直接 `ImportError: No module named 'nanovllm_ext'`（在模型加载阶段抛出，错误清晰）。

### 4.2 卸载机制（`patches/moe_layer.py`）

包装官方 `FusedMoE.__init__`。当 `--enable-moe-offload` 且 `layer_id >= moe_offload_start_layer` 时，在官方 `__init__` 调用 `quant_config.get_quant_method(self, prefix)` 之前，把该方法**临时替换**成返回我们的 offload method，使 offload method 接管 `create_weights`（权重直接量化存入 CPU 引擎，而非 NPU）：

- `quant_config is not None`：monkey-patch `quant_config.get_quant_method`，`try/finally` 还原。
- `quant_config is None`（非量化模型）：注入一个最小 shim `_OffloadQuantShim`（其 `get_quant_method` 返回 offload method），`__init__` 后把 `self.quant_config` 复原为 None。

根据 `quant_type` 选 method：`q4_0` → `MoEOffloadInt4FusedMoEMethod`；`q8_0` → `MoEOffloadFusedMoEMethod`。

### 4.3 两个 offload Method（`patches/offload.py`）

均继承官方 `FusedMoEMethodBase`，通过 `torch.classes.nanovllm.MoEInfer` 在 CPU 侧持有量化权重并执行 MoE。

#### `MoEOffloadFusedMoEMethod`（Q8_0，在线 Int8 量化）

- `create_weights`：建轻量 dummy 参数 `w13_weight`/`w2_weight`（CPU），设置 `weight_loader` 回调；`rank 0` 创建 `MoEInfer(E, H, I, quant_type=0)`。
- `_stream_quant_weight`（weight_loader 回调）：加载时按 shard（w1/w3/w2）调用 `MoEInfer.quantize_and_store_expert(expert_id, proj_name, w)`，在线量化成 Int8 存 CPU。
- `apply`：eager 模式 `torch.ops.nanovllm.moe_forward_npu_stream(x, topk_ids, topk_weights, handle)`；graph 模式用预分配 `MoEGraphContext` + `moe_forward_npu_graph_out`。

#### `MoEOffloadInt4FusedMoEMethod`（Q4_0，预量化 Int4）

- `group_size=32`（固定，与 kernel 一致），`packed_factor=8`。
- `create_weights`：建 compressed-tensors 格式参数 `w13_weight_packed`/`w2_weight_packed`（int32 `[E, out, in//8]`，CPU）、`w13_weight_scale`/`w2_weight_scale`（`[E, out, num_groups]`）；`MoEInfer(E, H, I, quant_type=1)`。
- `process_weights_after_loading`：调用 `MoEInfer.store_quantized_repack(w13_packed, w13_scale, w2_packed, w2_scale)` 把预量化权重转存进 CPU 引擎，随后删除 layer 上的临时参数释放显存。
- `apply`：与 Q8_0 相同的 stream/graph 双路径。

### 4.4 异步传输与 graph 兼容

- `_get_or_create_global_callback_manager(stream_ptr)`：按 `(device_id, stream_ptr)` 缓存 `torch.classes.nanovllm.NpuCallbackManager`，处理 NPU↔CPU 异步回调。
- graph 模式下用 `(num_tokens, top_k, dtype_int)` 作 key 缓存 `MoEGraphContext`，避免重复构造。
- `dtype_int`：`0=fp16`，`1=bf16`。

---

## 5. Monkey Patch 清单（`patches/`）

`apply_patches()`（`patches/__init__.py`）按顺序应用，幂等，NPU 守卫。TopK **不打补丁**（官方 `fused_topk_npu` 已正确）。

| # | patch 文件 | 官方目标 | 替换/新增 | 作用 |
| :-: | :-- | :-- | :-- | :-- |
| 1 | `server_args.py` | `ServerArgs.add_cli_args` / `from_cli_args` | AROUND | 注入 `--enable-moe-offload` / `--moe-offload-start-layer` / `--moe-offload-quant-type`；`from_cli_args` 把它们挂为实例属性 |
| 2 | `moe_layer.py` | `FusedMoE.__init__` | AROUND | 命中层在 `create_weights` 前把 `quant_method` 换成 offload method（见 §4.2） |
| 3 | `compressed_tensors.py` | `CompressedTensorsConfig._get_scheme_from_parts` | AROUND | NPU 上 W4A16 **Linear** 返回 `NPUCompressedTensorsW4A16`（官方默认返回 `CompressedTensorsWNA16`） |
| 4 | `wna16_moe.py` | `NPUCompressedTensorsW4A16Int4DynamicMoE.create_weights` | AROUND | 官方硬编码 scale/offset 为 bf16；本 patch 把 `w13/w2_weight_scale`、`w13/w2_weight_offset` cast 成 `params_dtype`（fp16 模型必需，否则 kernel dtype 不匹配） |
| 5 | `fused_moe_method.py` | `NPUW4A16Int4DynamicMoEMethod.{process_weights_after_loading, apply}` | 替换 | process：用 `repack_int4_npu` 一步重排（替代官方 unpack→transpose→`npu_convert_weight_to_int4pack`）；apply：`BS ≤ 阈值` 走 `fused_moe_w4a16_small_bs`，否则走官方 `npu_fused_experts` |
| 6 | `minimax_m2.py` | `models/minimax_m2.py` 的 `rms_sumsq_serial` / `rms_apply_serial` | 替换 | 去掉 `x.is_cuda` 断言（NPU 上为 False），其余逻辑调用同一组 triton kernel |
| 7 | `offload.py` | （新模块） | 新增 | `MoEOffloadFusedMoEMethod` / `MoEOffloadInt4FusedMoEMethod` / `create_moe_offload_config`（被 #2 调用） |

---

## 6. 参数总览

### 6.1 构建参数（`build_kernels.sh`）

```bash
bash build_kernels.sh [SOC_VERSION]   # 默认 Ascend910_9382
```

| 参数 | 默认值 | 说明 |
| :-- | :-- | :-- |
| `SOC_VERSION` | `Ascend910_9382` | CANN ascendc 目标 SoC。910B3 对应 `Ascend910_9382` |

脚本内部：
- `source /usr/local/Ascend/cann-9.0.0/set_env.sh`（找不到则读 `/etc/Ascend/ascend_cann_install.info`）
- 定位 `ASCConfig.cmake`，设置 `CMAKE_PREFIX_PATH` / `ASC_DIR`
- `cmake -DSOC_VERSION=... -B build -S .` && `cmake --build build -j 16`
- 产物输出到 `moe_ascend_npu/lib/libmoe_ascend_npu_kernels.so`
- 构建末尾自动 `python -m moe_ascend_npu._install_pth` 安装自动触发钩子

构建参数透传给 CMake 的还有 `ASCEND_HOME_PATH`、`ASCEND_INCLUDE_DIR`（`${ASCEND_TOOLKIT_HOME}/$(arch)-linux/include`）。

### 6.2 CLI 参数（由 `server_args.py` 注入）

仅当通过 `sglang launch-server` 等 CLI 入口走 `ServerArgs.add_cli_args` / `from_cli_args` 时出现。以编程方式构造 `ServerArgs` 需手动设置对应属性。

| CLI 参数 | 类型 | 默认 | 说明 |
| :-- | :-- | :-- | :-- |
| `--enable-moe-offload` | flag (store_true) | `False` | 开启 MoE CPU 卸载。需安装 `nanovllm_ext`，否则构造命中层时 `ImportError` |
| `--moe-offload-start-layer` | int | `0` | 从第几层 MoE 开始卸载（`layer_id >= 此值` 的层卸载，前面的仍在 NPU） |
| `--moe-offload-quant-type` | choice | `q8_0` | `q8_0`=在线 Int8 量化；`q4_0`=使用预量化 Int4（compressed-tensors 格式）。`q4_0` 若与 awq/gptq 等格式不兼容会自动回退 `q8_0` 并告警 |

**`create_moe_offload_config` 回退规则**（`offload.py`）：
- `quant_type=q4_0` 但模型量化格式名含 `awq`/`gptq`/`gptq_marlin`/`awq_marlin` → 回退 `q8_0` + warning
- `quant_type=q4_0` 但格式名非 `compressed_tensors`/`compressed-tensors` → 回退 `q8_0` + warning

### 6.3 环境变量

#### moe_ascend_npu 自身

| 变量 | 默认 | 说明 |
| :-- | :-- | :-- |
| `NPU_W4A16_SMALL_BS_THRESHOLD` | `8` | `fused_moe_method.apply` 中走小 batch kernel 的 batch 上限。`BS ≤ 阈值` 用 `fused_moe_w4a16_small_bs`；否则用官方 `npu_fused_experts`。设为 `0` 关闭小 batch 路径 |

#### nanovllm_ext（CPU offload，来自 `Int8-gemm/`）

| 变量 | 默认 | 说明 |
| :-- | :-- | :-- |
| `NANOVLLM_TP_SIZE` | `2` | CPU 侧张量并行度，用于 NUMA 线程池划分 |
| `NANOVLLM_TP_THREADS_PER_NODE` | （自动） | 每 NUMA 节点线程数 |

### 6.4 Kernel 张量参数与约束（速查）

见 §3.1 各 op 表格。汇总硬约束：

- `GROUP_SIZE = 32`（所有 scale 的组大小）
- int4 打包：8 nibble/int32，低 nibble 在前
- `weight` 必须 int32；`expert_ids`/`topk_ids` 必须 int32；`topk_weights` 在小 batch kernel 路径强制 fp32
- `batch_gemm_w4a16_small_bs`：`BS ≤ 4`
- `fused_moe_w4a16_small_bs`：`InDim == OutDim`
- 对称量化（grouped_gemv / fused_moe_small_bs 无 offset 参数；Linear 路径补零 offset）

---

## 7. 端到端数据流

### 7.1 W4A16 NPU 小 batch MoE（功能 A，默认）

```
模型加载:
  compressed_tensors 选 scheme -> 官方 NPUCompressedTensorsW4A16Int4DynamicMoE (MoE)
                            └─ wna16_moe patch: create_weights 时 scale/offset cast 到 params_dtype
  权重加载后:
    fused_moe_method patch: process_weights_after_loading
      w13/w2_weight [E,N,K//8] --transpose--> [E,K//8,N] --repack_int4_npu--> [E,K,N//8]
      w13/w2_weight_scale [E,N,num_groups] --transpose--> [E,num_groups,N]
      w13/w2_weight_offset 同上

推理 (BS <= 阈值):
  fused_moe_method patch: apply
    topk (官方 fused_topk_npu, 未 patch)
    -> fused_moe_w4a16_small_bs(x, w13, s13, w2, s2, topk_ids, topk_w)  [NPU kernel]
    -> StandardCombineInput
推理 (BS > 阈值):
  -> npu_fused_experts(...)  [官方路径]
```

### 7.2 MoE CPU 卸载（功能 B，需 `--enable-moe-offload`）

```
模型加载 (layer_id >= start_layer):
  moe_layer patch: FusedMoE.__init__ 拦截
    -> create_moe_offload_config(layer_id, server_args, quant_config)
    -> 选 MoEOffloadFusedMoEMethod (q8_0) 或 MoEOffloadInt4FusedMoEMethod (q4_0)
    -> 临时替换 quant_config.get_quant_method 返回 offload method
    -> 官方 __init__ 用 offload method.create_weights
         q8_0: dummy param + weight_loader 在线量化存入 MoEInfer(quant_type=0)
         q4_0: compressed-tensors 参数 + process_weights 后 store_quantized_repack 到 MoEInfer(quant_type=1)

推理:
  offload method.apply
    -> NpuCallbackManager (按 stream 缓存)
    -> eager: moe_forward_npu_stream(x, topk_ids, topk_w, handle)  [CPU 计算, NPU 流同步]
    -> graph: MoEGraphContext + moe_forward_npu_graph_out(..., out)
```

---

## 8. 测试

`tests/run_all.sh` 依次跑三个 NPU 正确性测试，均对比纯 PyTorch W4A16 参考（权重走同一 `repack_int4_npu` 流程）。

| 测试 | 验证 | 典型精度 |
| :-- | :-- | :-- |
| `test_repack.py` | `repack_int4_npu` vs PyTorch ref | bit-exact (`max_diff=0`) |
| `test_gemv_w4a16.py` | `grouped_gemv_w4a16_moe` + `batch_gemm_w4a16_small_bs` | grouped_gemv rel ≈ 0.1%；batch_gemm 用 `mean_diff<0.05` 断言（split-K 非确定性） |
| `test_fused_moe.py` | `fused_moe_w4a16_small_bs` BS=1/2/4 | rel ≈ 0.15% |

测试参数（`tests/_helpers.py` + 各 test）：
- `GROUP_SIZE=32`（与 kernel 一致）
- `pack_uint4b8`：`(val+8) & 0xF`，低 nibble 在前
- `test_gemv_w4a16.py`：`NUM_EXPERTS=8, IN_DIM=2048, OUT_DIM=1536, BATCH_SIZE=2, TOP_K=8`
- `test_fused_moe.py`：`NUM_EXPERTS=16, HIDDEN_SIZE=2048, INTER_SIZE=768, TOP_K=8`，遍历 `BS ∈ {1,2,4}`（W13 输出维为 `2*INTER_SIZE`，因 gate+up 合并）

---

## 9. 安装与使用

```bash
cd moe_ascend_npu
bash build_kernels.sh Ascend910_9382   # 构建 .so + 安装 .pth
pip install -e .                        # 安装 Python 包（含 .so）
pip install -e ../Int8-gemm             # 可选：启用 MoE CPU 卸载 (nanovllm_ext)
```

之后任何 SGLang 启动在 NPU 上自动生效，无需改代码：

```bash
# 默认（W4A16 NPU 小 batch kernel 自动启用）
sglang launch-server --model-path /path/to/Qwen3-MoE-AWQ --attention-backend ascend ...

# 启用 MoE CPU 卸载
sglang launch-server --model-path ... \
    --enable-moe-offload \
    --moe-offload-start-layer 0 \
    --moe-offload-quant-type q4_0 ...
```

卸载自动触发钩子：`python -m moe_ascend_npu._install_pth remove`。

---

## 10. 已知限制与注意事项

1. **`batch_gemm_w4a16_small_bs` 非确定性**：split-K + atomic-add 导致逐元素 `max_diff` 跨运行波动；该 op 当前未接入 Linear 路径，仅作预留/测试。`fused_moe_w4a16_small_bs` 与 `grouped_gemv_w4a16_moe` 无此问题。
2. **`GROUP_SIZE=32` 硬编码**：权重/scale 必须按 32 分组，不支持其它 group size。
3. **对称量化**：`NPUCompressedTensorsW4A16` 与三个 NPU op 均仅支持对称量化（Linear 路径补零 offset 满足 op 签名）。
4. **`nanovllm_ext` 是 offload 的硬依赖**：不开 offload 无需安装；开了未装则在模型加载阶段 `ImportError`（不会静默降级）。
5. **SOC 版本**：仅验证 `Ascend910_9382`（910B3）；其它 SoC 需调整 `build_kernels.sh` 参数并可能需修改 kernel tiling。
6. **patch 依赖 SGLang 内部 API**：基于 sglang 0.5.13.post1 的具体符号（类名/方法名/参数顺序）。SGLang 升级后若这些符号变动，patch 可能失效，需重新对齐。
7. **rank 0 假设**：offload 的 `create_weights` 当前仅 rank 0 处理权重存储，多卡场景需注意（`tp_rank != 0` 直接 return）。
