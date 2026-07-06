"""Patch minimax_m2 RMSNorm helpers to drop the is_cuda assertions.

The upstream ``rms_sumsq_serial`` / ``rms_apply_serial`` assert ``x.is_cuda``,
which fails on NPU. The triton kernels themselves run fine on NPU, so we replace
the two Python wrappers with assertion-free copies that call the same upstream
triton kernels.
"""

import logging

import torch
import triton

logger = logging.getLogger(__name__)


def apply():
    from sglang.srt.models import minimax_m2 as mm

    def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, D1 = x1.shape
        B2, D2 = x2.shape
        assert B == B2

        stride_x1 = x1.stride(0)
        stride_x2 = x2.stride(0)

        B_padded = (B + B2 + 3) // 4 * 4
        sum_sq = torch.empty(B_padded, device=x1.device, dtype=torch.float32)

        BLOCK_SIZE1 = triton.next_power_of_2(D1)
        BLOCK_SIZE2 = triton.next_power_of_2(D2)
        grid = (B,)

        mm.rmsnorm_sumsq_kernel_serial[grid](
            x1,
            x2,
            stride_x1,
            stride_x2,
            sum_sq,
            B,
            D1,
            D2,
            BLOCK_SIZE1,
            BLOCK_SIZE2,
        )
        return sum_sq

    def rms_apply_serial(
        x1: torch.Tensor,
        x2: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        sum_sq: torch.Tensor,
        tp_world: int = 1,
        eps: float = 1e-5,
    ):
        B, D1 = x1.shape
        B2, D2 = x2.shape
        assert B == B2

        stride_x1 = x1.stride(0)
        stride_x2 = x2.stride(0)
        out1 = torch.empty(B, D1, device=x1.device, dtype=x1.dtype)
        out2 = torch.empty(B, D2, device=x2.device, dtype=x2.dtype)

        BLOCK_SIZE1 = triton.next_power_of_2(D1)
        BLOCK_SIZE2 = triton.next_power_of_2(D2)
        grid = (B,)

        mm.rmsnorm_apply_kernel_serial[grid](
            x1,
            x2,
            w1,
            w2,
            sum_sq,
            out1,
            out2,
            B,
            D1,
            D2,
            stride_x1,
            stride_x2,
            tp_world,
            eps,
            BLOCK_SIZE1,
            BLOCK_SIZE2,
        )
        return out1, out2

    mm.rms_sumsq_serial = rms_sumsq_serial
    mm.rms_apply_serial = rms_apply_serial
    logger.info("minimax_m2: removed is_cuda assertions from rms_sumsq/rms_apply_serial")
