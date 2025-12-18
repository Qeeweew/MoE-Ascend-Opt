# MoE-Ascend-Opt: 华为 NPU 上的高性能 AWQ MoE 推理优化

![Platform](https://img.shields.io/badge/Platform-Huawei%20Ascend%20910B-red)
![Model](https://img.shields.io/badge/Model-Qwen3%20MoE-blue)
![Quantization](https://img.shields.io/badge/Quantization-AWQ%20Int4-green)

**MoE-Ascend-Opt** 是一个致力于在华为 Ascend 910B NPU 上加速混合专家模型（MoE），特别是 Qwen2.5/3-MoE 系列 AWQ (W4A16) 量化模型的开源优化项目。

本项目基于 [SGLang](https://github.com/sgl-project/sglang) 框架，通过引入自定义的 Ascend C 算子和 Triton kernel，解决了原生实现中小 Batch Size 下的显存带宽瓶颈问题。在 Decoding 阶段，端到端吞吐量实现了 **~2倍** 的提升。

---

## 🚀 性能表现 (Performance)

**测试模型**: `tclf90/Qwen3-30B-A3B-Thinking-2507-AWQ`
**硬件环境**: Huawei Ascend 910B

### 1. Kernel 微基准测试 (Fused MoE Layer)

对比原生 PyTorch/Ascend 实现 (Ref) 与本项目优化后的 Kernel (Custom)。

| Batch Size | Ref Latency (us) | Custom Latency (us) | Ref Bandwidth (GB/s) | Custom Bandwidth (GB/s) | **加速比 (Speedup)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 207.61 | **52.99** | 96.59 | 378.45 | **3.92x** |
| **2** | 279.09 | **72.59** | 143.71 | 552.53 | **3.84x** |
| **3** | 408.59 | **93.67** | 147.24 | 642.30 | **4.36x** |
| **4** | 519.75 | **117.09** | 154.34 | 685.05 | **4.44x** |

### 2. SGLang 端到端 Decoding 吞吐量
设置：Input = 1 token, Output = 1024 tokens。

| Batch Size | 原生吞吐量 (tokens/s) | 优化后吞吐量 (tokens/s) | **提升倍数 (Speedup)** |
| :---: | :---: | :---: | :---: |
| 1 | 52.99 | 91.92 | **1.73x** |
| 2 | 83.86 | 163.43 | **1.95x** |
| 4 | 127.58 | 272.99 | **2.14x** |
| 6 | 155.48 | 340.32 | **2.19x** |
| 8 | 181.69 | 403.48 | **2.22x** |

---

## 🧠 核心优化原理 (Optimization Principles)

### 1. 痛点：显存带宽瓶颈
在 Ascend NPU 的原生 AWQ 实现中，计算流程通常是：
1.  **反量化**：将 Int4 权重加载并转换为 FP16，结果**写回**全局内存 (Global Memory, GM)。
2.  **矩阵乘**：Cube Core 从 GM 读取 FP16 权重进行矩阵乘法。

在小 Batch Size（Decoding 阶段）下，这种 Read-Write-Read 的模式极大地浪费了 NPU 的显存带宽，导致计算受限于带宽而非算力。

### 2. 解决方案：基于 Vector Core 的 W4A16 GEMV
我们使用了 Ascend C 编写了自定义算子，利用 **Vector Core (AI Vector)** 替代 Cube Core 处理小 Batch 场景：
*   **寄存器级反量化**：权重以 Int4 形式从 GM 加载到片上内存 (UB)，直接在 Vector 单元寄存器中完成反量化。
*   **消除中间读写**：反量化后的 FP16 数据直接参与点积计算，无需写回 GM。
*   **结果**：大幅减少了对 GM 的访问次数，显著提升了带宽利用率。

### 3. 垂直算子融合 (Vertical Fusion)
为了进一步减少 Kernel Launch 开销和数据搬运，我们对 MoE 的 MLP 块进行了简易的垂直融合：
*   **原生流程**：`GEMM(Gate)` -> GM -> `GEMM(Up)` -> GM -> `SwiGLU` -> GM -> `GEMM(Down)`。
*   **融合流程**：一个 Kernel 完成所有操作。
    *   加载输入 X。
    *   计算 Gate 和 Up 投影。
    *   在片上快速进行 SwiGLU 激活。
    *   计算 Down 投影并累加结果。

### 4. Triton 在 Ascend 上的尝试
本项目还探索了 OpenAI Triton 在 NPU 上的应用，用于处理非计算密集型但逻辑复杂的操作，提高开发效率：
*   **MoE Gating (TopK Softmax)**: 使用 Triton 实现了 Router Logits 的 Softmax 和 TopK 选择，避免了手写复杂的 C++ Tiling 逻辑。
*   **Weight Repacking**: 使用 Triton 实现了权重的 Layout 转换 (`sgl_kernel_npu/repack_int4.py`)，将通用 AWQ 权重重排为 NPU Vector 指令所需的格式。

---

## 📂 项目结构

```text
MoE-Ascend-Opt
├── sglang/                 # 修改后的 SGLang 框架源码
│   ├── python/sglang/srt/layers/moe/topk.py      # Triton 实现的 Gating 逻辑
│   ├── python/sglang/srt/layers/quantization/awq.py # 适配 NPU 优化的 AWQ 逻辑
│   └── ...
├── sgl-kernel-npu/         # 自定义 NPU 算子库
│   ├── csrc/grouped_gemv/  # 核心 Ascend C 代码 (W4A16 GEMV & Fused MoE)
│   ├── python/             # Python 绑定与 Triton Kernel
│   ├── build.sh            # 编译脚本
│   └── ...
└── README.md
```

---

## 🛠️ 安装与构建 (Installation)

### 环境要求
*   **Hardware**: Huawei Ascend 910B
*   **Software**: CANN Toolkit 8.0+, PyTorch (Ascend version)

### 步骤 1: 编译自定义算子库
编译包含 Ascend C kernel 的 `sgl-kernel-npu` 扩展。

```bash
cd sgl-kernel-npu
# 可选：清理旧的构建
./build.sh -c 

# 编译并安装 whl 包
./build.sh
pip install dist/sgl_kernel_npu-*.whl
```

### 步骤 2: 安装修改版 SGLang
安装集成了上述算子调用的 SGLang 框架。

```bash
cd ../sglang/python
pip install -e .
```

---

## 🚀 启动服务 (Usage)

使用以下命令启动 SGLang Server。

**关键参数说明**:
*   `--sampling-backend ascend`: **必须开启**。启用针对 Ascend NPU 优化的采样后端。
*   `--dtype float16`: **必须设置**。Ascend NPU 上 float16 性能通常优于 bfloat16，且适配当前算子实现。
*   `--cuda-graph-bs`: 设置捕获 CUDA Graph 的 Batch Size 列表，建议覆盖常用的 Decoding Batch 大小。

**启动命令示例**:

```bash
python3 -m sglang.launch_server \
  --model ~/data/models/Qwen3-30B-A3B-Thinking-2507-AWQ \
  --attention-backend ascend \
  --mem-fraction-static 0.9 \
  --reasoning-parser qwen3-thinking \
  --tp-size 1 \
  --sampling-backend ascend \
  --cuda-graph-bs 1 2 3 4 5 6 7 8 \
  --dtype float16 \
  --chunked-prefill-size 4096
```

---

## ⚠️ 限制说明 (Limitations)

1.  **Batch Size**: 自定义的 Vector-based GEMV 主要针对 **BS <= 8** 的 Decoding 阶段优化。对于大 Batch (Prefill 阶段)，框架会自动回退到原生的 Cube-based 矩阵乘法以保证吞吐量。
2.  **模型支持**: 目前主要在 **Qwen3-MoE** 架构 + **AWQ** 量化下进行了验证。
3.  **硬件依赖**: 代码针对 Ascend 910B (AIV 架构) 进行了指令级优化，无法直接在旧版 NPU 或 GPU 上运行。

---
