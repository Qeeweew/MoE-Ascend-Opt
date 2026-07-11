"""NPU kernel interface: load the compiled ``.so`` and expose Python helpers."""

import os
import pathlib

import torch

_so_loaded = False


def _load_kernels():
    """Load ``libmoe_ascend_npu_kernels.so`` once.

    The shared library registers three ops under the ``moe_ascend_npu``
    torch namespace::

        torch.ops.moe_ascend_npu.grouped_gemv_w4a16_moe
        torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs
        torch.ops.moe_ascend_npu.fused_moe_w4a16_cached
        torch.ops.moe_ascend_npu.batch_gemm_w4a16_small_bs
    """
    global _so_loaded
    if _so_loaded:
        return
    # moe_ascend_npu/ (this file is moe_ascend_npu/kernels/__init__.py)
    pkg_dir = pathlib.Path(__file__).parents[1]
    candidates = [
        os.path.join(pkg_dir, "lib", "libmoe_ascend_npu_kernels.so"),
        os.path.join(pkg_dir, "libmoe_ascend_npu_kernels.so"),
    ]
    for so_path in candidates:
        if os.path.exists(so_path):
            torch.ops.load_library(so_path)
            _so_loaded = True
            return
    raise RuntimeError(
        "moe_ascend_npu kernels .so not found. "
        "Run `bash build_kernels.sh` first, then reinstall the package."
    )


def ensure_kernels_loaded():
    """Idempotently load the native kernel library."""
    _load_kernels()


from .repack import repack_int4_npu

__all__ = ["repack_int4_npu", "ensure_kernels_loaded"]
