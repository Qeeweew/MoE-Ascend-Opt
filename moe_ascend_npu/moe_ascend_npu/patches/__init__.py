"""Monkey patches that add W4A16 NPU kernels + MoE CPU offload to SGLang.

Each submodule exposes ``apply()`` which installs one logical patch by direct
``setattr`` on the official SGLang modules/classes (there is no hook registry in
upstream SGLang, so we patch in place). ``apply_patches()`` runs them all.

Patching is triggered at interpreter start-up by ``moe_ascend_npu._bootstrap``
(which a ``.pth`` file auto-imports), but only after SGLang is imported and only
on NPU.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_patches():
    """Apply all SGLang monkey patches. Idempotent."""
    global _applied
    if _applied:
        return
    _applied = True

    from sglang.srt.utils import is_npu

    if not is_npu():
        logger.info("moe_ascend_npu: not on NPU, skipping patches.")
        return

    # Import lazily so importing this package never pulls in SGLang.
    from moe_ascend_npu.patches import (
        compressed_tensors,
        fused_moe_method,
        minimax_m2,
        moe_layer,
        server_args,
        wna16_moe,
    )

    logger.info("moe_ascend_npu: applying SGLang monkey patches...")
    server_args.apply()
    moe_layer.apply()
    compressed_tensors.apply()
    wna16_moe.apply()
    fused_moe_method.apply()
    minimax_m2.apply()
    logger.info("moe_ascend_npu: all SGLang monkey patches applied.")


__all__ = ["apply_patches"]
