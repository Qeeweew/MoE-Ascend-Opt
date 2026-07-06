#!/usr/bin/env python
"""Setup script for moe-ascend-npu package.

The Ascend C NPU kernels are built separately by ``build_kernels.sh`` (CMake +
CANN ascendc toolchain) and produce ``moe_ascend_npu/lib/libmoe_ascend_npu_kernels.so``.
This setup script ships the prebuilt shared library together with the Python
package. The ``.pth`` that auto-applies the SGLang monkey patches is installed
separately by ``python -m moe_ascend_npu._install_pth`` (also run by
``build_kernels.sh``), because setuptools wheels cannot install a ``.pth`` with
an absolute site-packages path directly.

The CPU offload extension (``nanovllm_ext``) is built from the sibling
``Int8-gemm/`` directory and is a runtime dependency, not built here.
"""

from setuptools import setup

setup(
    name="moe-ascend-npu",
    version="0.1.0",
    description="MoE Ascend NPU: W4A16 fused-MoE NPU kernels + CPU offload for SGLang",
    package_data={"moe_ascend_npu": ["lib/*.so"]},
)
