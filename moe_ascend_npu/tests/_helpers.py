"""Shared helpers for moe_ascend_npu kernel tests.

These helpers construct int4 (uint4b8) weights in the compressed-tensors layout
that the SGLang integration expects, repack them with the same
``repack_int4_npu`` flow used by ``process_weights_after_loading``, and provide a
pure-PyTorch W4A16 reference for correctness checking.
"""

import math

import torch

from moe_ascend_npu.kernels import ensure_kernels_loaded, repack_int4_npu

GROUP_SIZE = 32  # kernel constant (see grouped_gemv_w4a16_moe.cpp)


def setup():
    """Load the native kernel library (idempotent)."""
    ensure_kernels_loaded()


def pack_uint4b8(raw_signed: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 weights (-8..7) into uint4b8 int32 along the last dim.

    Input:  ``[..., K]`` int (values in -8..7).
    Output: ``[..., K//8]`` int32, 8 nibbles per element, low nibble = K[0].
    The stored nibble is ``value + 8`` (uint4b8 bias-of-8 convention), matching
    the compressed-tensors ``uint4b8`` scalar type.
    """
    assert raw_signed.dtype in (torch.int32, torch.int64)
    *leading, k = raw_signed.shape
    assert k % 8 == 0, f"last dim {k} must be divisible by 8"
    nibble = (raw_signed + 8) & 0xF  # uint4b8
    nibble = nibble.view(*leading, k // 8, 8)
    packed = torch.zeros(*leading, k // 8, dtype=torch.int32, device=raw_signed.device)
    for i in range(8):
        packed |= (nibble[..., i].to(torch.int32) << (4 * i))
    return packed


def repack_expert_weight(weight_packed_ct: torch.Tensor) -> torch.Tensor:
    """Repack ``[E, N, K//8]`` compressed-tensors weights to ``[E, K, N//8]``.

    Mirrors ``NPUW4A16Int4DynamicMoEMethod._transpose_and_repack_int4`` /
    ``process_weights_after_loading``: transpose N<->K//8, flatten experts, run
    the Triton ``repack_int4_npu`` kernel (which also converts uint4b8 -> int4
    two's-complement), reshape back.
    """
    e, n, k8 = weight_packed_ct.shape
    k = k8 * 8
    weight_t = weight_packed_ct.transpose(1, 2).contiguous()  # [E, K//8, N]
    weight_t_flat = weight_t.view(e * k8, n)  # [E*K//8, N]
    repacked = repack_int4_npu(weight_t_flat)  # [E*K, N//8]
    return repacked.view(e, k, n // 8).contiguous()


def repack_single_weight(weight_packed_ct: torch.Tensor) -> torch.Tensor:
    """Repack a single ``[N, K//8]`` weight to ``[K, N//8]`` (no expert dim)."""
    n, k8 = weight_packed_ct.shape
    k = k8 * 8
    weight_t = weight_packed_ct.transpose(0, 1).contiguous()  # [K//8, N]
    return repack_int4_npu(weight_t).view(k, n // 8).contiguous()


def transpose_scale(scale_ct: torch.Tensor) -> torch.Tensor:
    """Transpose scale from ``[E, N, num_groups]`` to ``[E, num_groups, N]``."""
    return scale_ct.transpose(-1, -2).contiguous()


def w4a16_matvec_ref(
    w_signed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Pure-PyTorch W4A16 matvec reference.

    Args:
        w_signed: ``[N, K]`` signed int4 weights (-8..7).
        scale:    ``[num_groups, N]`` per-group scale (fp16/bf16/fp32).
        x:        ``[K]`` activation (fp16/bf16/fp32).

    Returns:
        ``[N]`` output in fp32 (ground truth).
    """
    n, k = w_signed.shape
    num_groups = k // group_size
    assert scale.shape == (num_groups, n), f"scale {scale.shape} != ({num_groups},{n})"
    # Expand per-group scale to per-K: [num_groups, N] -> [K, N]
    scale_expanded = scale.repeat_interleave(group_size, dim=0)  # [K, N]
    # ws[n, k] = w_signed[n, k] * scale_expanded[k, n]
    ws = w_signed.float() * scale_expanded.t().float()  # [N, K]
    return ws @ x.float()  # [N]


def w4a16_matmul_ref(
    w_signed: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Batched W4A16 reference: ``x`` is ``[B, K]``, returns ``[B, N]``."""
    n, k = w_signed.shape
    num_groups = k // group_size
    scale_expanded = scale.repeat_interleave(group_size, dim=0)  # [K, N]
    ws = w_signed.float() * scale_expanded.t().float()  # [N, K]
    return x.float() @ ws.t()  # [B, N]
