***

# MoE-Ascend-Opt: 面向国产异构平台的混合专家模型推理优化

![Platform](https://img.shields.io/badge/Platform-Huawei%20Ascend%20910B%20%2B%20Kunpeng%20920-red)
![Model](https://img.shields.io/badge/Model-Qwen3%20MoE-blue)
![Quantization](https://img.shields.io/badge/Quantization-AWQ%20W4A16%20%2F%20Int8-green)

**MoE-Ascend-Opt** 是一个针对国产异构计算平台（华为昇腾 910B NPU + 鲲鹏 920 CPU）的混合专家模型（MoE）推理优化项目。

本项目旨在解决 MoE 模型在边缘计算或单卡部署场景下的**显存带宽瓶颈**与**容量限制**问题。通过在 NPU 端实现基于 Vector Core 的极致 W4A16 算子，以及在 CPU 端实现基于 ARM NEON 的高性能 Int8 算子，构建了一套“计算向数据靠拢”的动态异构推理系统。

> **背景**: 本项目隶属于课题《面向国产异构平台的混合专家模型推理优化与动态卸载机制研究》。

---

## 🔬 核心技术原理 (Core Principles)

### 1. NPU 端：基于 Vector Core 的 W4A16 融合算子
**痛点**：MoE 推理的 Decoding 阶段（Batch Size 通常为 1~8）具有极高的稀疏性。昇腾 NPU 的 **Cube Core**（矩阵乘法单元）专为大规模稠密矩阵设计，在处理此类“碎片化、小形状”的 GEMV（矩阵-向量乘）任务时，填充（Padding）开销巨大，且无法充分利用显存带宽。

**优化原理**：
*   **Vector Core (AIV) 替代 Cube Core**：
    我们放弃通用的 `matmul` 接口，使用 **Ascend C** 编写自定义算子，利用 **Vector Core** 直接处理 GEMV 任务。对于 Batch=1 的场景，Vector 单元的流水线效率远高于 Cube 单元的调度开销。
*   **寄存器级反量化 (Register-Level Dequantization)**：
    *   *传统流程*：`Load Int4 (GM)` -> `Convert to FP16` -> `Store FP16 (GM)` -> `Load FP16 (GM)` -> `Compute`。这导致了严重的显存读写浪费。
    *   *本项目流程*：`Load Int4 (GM)` -> `Deqant to FP16 (Vector Reg)` -> `FMA (Vector Reg)`。
    *   权重数据从全局内存（Global Memory）加载后，直接在**寄存器**中完成 Int4 到 FP16 的转换并立即参与计算，消除了所有中间数据的显存读写，将带宽利用率（MBU）提升至硬件极限。
*   **垂直算子融合 (Vertical Fusion)**：
    将 MoE MLP 的全流程 `Gate_Proj` + `Up_Proj` -> `SiLU` -> `Quant(Optional)` -> `Down_Proj` 融合为一个 Kernel。利用 NPU 的片上统一缓冲（Unified Buffer, UB）传递中间激活值，彻底消除了 Kernel Launch 开销。

### 2. CPU 端：基于 ARM NEON 的高性能计算卸载
**痛点**：当显存不足时，传统方案是将权重通过 PCIe 搬运至 NPU 计算。然而，PCIe 4.0 的实测带宽（~32GB/s）远低于鲲鹏 920 服务器的内存带宽（>200GB/s）。**“搬运数据”是最大的瓶颈。**

**优化原理**：**计算向数据靠拢 (Compute Offloading)**
*   **ARM NEON 指令集优化**：
    针对鲲鹏 CPU（ARMv8.2-A），手写了 `gemm_q8_0` 微内核。利用 `SDOT` (Signed Dot Product) 指令实现 4 路并行的 Int8 点积运算，并通过指令重排（Instruction Interleaving）掩盖内存加载延迟（Latency Hiding）。
*   **NUMA 感知 (NUMA-Awareness)**：
    鲲鹏 920 是多 NUMA 节点架构（多 Socket/Die）。跨 Socket 访问内存会导致带宽下降 40% 以上。
    *   我们实现了一个**NUMA 绑核线程池 (`numa_threadpool.h`)**。
    *   在推理前，将不同专家的权重物理地分配在不同的 NUMA 节点内存上。
    *   计算时，严格绑定线程到对应的 NUMA 节点核心上执行，确保 **100% 的本地内存访问**。
*   **异步流水线**：
    通过 NPU 的 `aclrtLaunchCallback` 机制，在 NPU 计算 Attention 的同时，异步触发 CPU 开始计算 MoE 专家层，实现异构硬件的并行工作。

---

## 🚀 性能表现 (Performance)

**测试环境**: Huawei Ascend 910B + Kunpeng 920 (64 Cores, 4 NUMA Nodes)
**测试模型**: Qwen3-MoE (AWQ Quantized)

### 1. NPU Kernel 性能 (Decoding Phase)
对比 PyTorch 原生实现 (Ref) 与本项目自定义 Vector Kernel (Custom)。

| Batch Size | Ref Latency (us) | **Custom Latency (us)** | Ref Bandwidth (GB/s) | **Custom Bandwidth (GB/s)** | **加速比** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 204.59 | **45.21** | 98.02 | **443.62** | **4.53x** |
| **2** | 275.07 | **62.27** | 145.81 | **644.11** | **4.42x** |
| **4** | 503.63 | **97.84** | 159.27 | **819.86** | **5.15x** |

### 2. CPU Kernel 性能 (Int8 Fused MoE)
基于 NUMA 优化的 ARM Int8 算子实测性能。
*配置: H=2048, I=768, TopK=8, TP=4(2 CPU)*

| Num Tokens | Avg Time (ms) | **Weight Bandwidth (GB/s)** | **Compute (GFLOPS)** | 说明 |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.53 | **75.74** | 142.57 | 小负载受调度开销影响 |
| **4** | 1.05 | **152.23** | 286.56 | 接近 PCIe 带宽的 6 倍 |
| **8** | 1.72 | **186.50** | 351.06 | 达NPU算子带宽的1/5 |
| **32** | 3.06 | **209.70** | 789.47 | 高吞吐场景 |

> **结论**: 优化后的 CPU 算子能提供 **~180-210 GB/s** 的有效带宽，远超 PCIe 传输带宽。在 NPU 显存受限时，直接在 CPU 上计算是更优解。

---

## 🗺️ 路线图 (Roadmap)

### 第一阶段：高性能异构算子库 (Current Focus)
- [x] **NPU**: 实现 Vector Core 加速的 W4A16 Fused MoE 算子。
- [x] **CPU**: 实现基于 NUMA 感知的 ARM Int8 Fused MoE 算子。
- [ ] **CPU**: 升级算子至 **Int4** 精度，进一步降低内存占用并对齐 NPU 量化格式。

### 第二阶段：动态弹性卸载机制 (Elastic Offloading)
- [ ] **KV Cache 驱动的弹性伸缩**:
    - 建立显存竞争模型。随着 KV Cache（长文本）占用增加，动态将 NPU 上的专家权重“挤出”到 CPU。
    - CPU 充当“Shadow Experts”角色，承接被驱逐专家的计算任务。
- [ ] **SGLang 图执行适配**:
    - 在 CUDA/NPU Graph 中引入“虚拟专家节点”。
    - 实现对上层透明的 Dispatcher，在图执行过程中根据标志位动态将 Token 分发至 NPU Stream 或 CPU Callback 线程。
- [ ] **基于激活频率的热点管理**:
    - 实时统计专家路由频率。
    - **Hot Experts** 常驻 NPU HBM。
    - **Cold Experts** 卸载至 CPU RAM。
---

## 📂 项目结构

```text
MoE-Ascend-Opt
├── sglang/                 # 修改适配后的 SGLang 框架
│   ├── python/sglang/srt/layers/moe/arm_ep_wrapper.py  # 异构分发调度器 (NPU/CPU Router)
│   └── ...
├── sgl-kernel-npu/         # 自定义异构算子库
│   ├── MoE-Ascend-Opt-main/
│   │   ├── Int8-gemm/      # CPU ARM 优化核心代码
│   │   │   ├── q8_gemm.cpp       # ARM NEON Int8 微内核
│   │   │   ├── numa_threadpool.h # NUMA 绑核线程池实现
│   │   │   ├── nanovllm_ops.cpp  # NPU Stream 回调与 PyTorch 绑定
│   │   │   └── moe_infer.cpp     # 异构推理逻辑封装
│   │   ├── ...
│   └── csrc/grouped_gemv/  # NPU Ascend C 核心代码 (Vector Core)
└── README.md
```

---

## 🛠️ 使用指南

### 1. 编译自定义算子库
```bash
cd MoE-Ascend-Opt-main/Int8-gemm
# 编译 CPU 与 NPU 混合扩展
python3 setup.py install
```

### 2. 安装 SGLang
```bash
cd sglang/python
pip install -e .
```

### 3. 启动异构推理服务
通过参数开启 ARM Expert Parallelism (EP)：

```bash
# 示例：将所有 MoE 层卸载至 CPU 运行 (模拟显存极度受限场景)
# 设置 TP 线程数与 NUMA 节点匹配
export NANOVLLM_TP_THREADS_PER_NODE=16 

python3 -m sglang.launch_server \
  --model /path/to/Qwen3-MoE-AWQ \
  --sampling-backend ascend \
  --dtype float16 \
  --enable-arm-ep \
  --arm-ep-start-layer 0 \
  --tp-size 1
```

---

## ⚠️ 限制说明

1.  **硬件依赖**: 必须运行在 **Huawei Ascend 910B** (支持 AIV 指令) 和 **ARMv8.2+ CPU** (支持 dotprod 指令) 环境下。x86 架构无法运行本项目代码。
2.  **量化格式**: 目前针对 **AWQ** (W4A16) 和 **Int8** 权重格式优化。
3.  **场景限制**: 极致优化主要针对 **Decoding 阶段 (Small Batch)**。Prefill 阶段的大 Batch 计算目前仍回退到原生实现以保证吞吐量。