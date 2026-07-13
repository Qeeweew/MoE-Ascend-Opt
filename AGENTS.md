# MoE-Ascend-Opt

昇腾 NPU + 鲲鹏 CPU 国产异构平台的 MoE 推理优化系统。

## 核心规则

1. **不修改 sglang / sgl-kernel-npu 源码**——通过 `moe_ascend_npu/patches/` monkey patch + `.pth` 自动注入
2. **不含硬编码的 batch size 限制**——kernel 不限制 batch size，"小 batch"是私有化部署场景决定的
3. cuda graph 是 decode 加速关键，**不要随意 `--disable-cuda-graph`**
4. 不要对未读原文的论文做任何实现/架构断言

## 使用

```bash
pip install -e moe_ascend_npu/            # 安装包
python -m moe_ascend_npu._install_pth install  # 注册 .pth 自动注入
cd moe_ascend_npu && bash build_kernels.sh     # 编译 NPU kernel
bash tests/run_all.sh                           # 运行测试
pip install -e Int8-gemm/ --no-build-isolation  # CPU 引擎（可选）
```

## 文档

- 包架构：`docs/architecture.md`
- 专家缓存调度方案：`docs/design/dynamic_offload.md`
