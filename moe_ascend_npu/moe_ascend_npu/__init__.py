"""MoE Ascend NPU: standalone W4A16 fused-MoE NPU kernels + CPU offload.

This package provides:
  * Ascend C Vector-Core kernels for W4A16 fused MoE (decoding / small-batch),
    compiled to ``libmoe_ascend_npu_kernels.so`` and exposed as
    ``torch.ops.moe_ascend_npu.*``.
  * A Triton ``repack_int4_npu`` weight-repack kernel.
  * NPU W4A16 linear / MoE layer schemes used when monkey-patching SGLang.

The SGLang integration (server args, MoE offload, scheme routing, ...) lives in
``moe_ascend_npu.patches`` and is applied at process start-up via a ``.pth``
auto-import (see ``moe_ascend_npu._bootstrap``).
"""

__version__ = "0.1.0"
