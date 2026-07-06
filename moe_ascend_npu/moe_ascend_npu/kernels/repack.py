"""Triton int4 weight repack kernel for Ascend NPU.

Fuses unpack + transpose + repack of AWQ int4 weights in one kernel on the NPU
Vector Cores. Input is ``[K//8, N]`` (int32, 8 nibbles packed per element,
already transposed); output is ``[K, N//8]`` (int32, the layout expected by the
W4A16 GEMV kernels).
"""

import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def int4_repack_kernel(
    in_ptr,
    out_ptr,
    N,
    K,
    stride_in_n,
    stride_in_k8,
    stride_out_k,
    stride_out_n8,
    NUM_CORES: tl.constexpr,
    BLOCK_N8: tl.constexpr,
):
    """Persistent kernel: each Vector Core owns a contiguous range of K//8 rows.

    1. NUM_CORES binding maintains the Persistent Kernel pattern.
    2. Parallelise directly along the K dimension: split K//8 rows across cores.
    3. N dimension handled internally with 1D vectorisation.
    """
    pid = tl.program_id(0)

    total_k8_rows = K // 8

    rows_per_core = (total_k8_rows + NUM_CORES - 1) // NUM_CORES
    start_row = pid * rows_per_core

    if start_row >= total_k8_rows:
        return

    end_row = tl.minimum(start_row + rows_per_core, total_k8_rows)

    for k8_idx in range(start_row, end_row):
        num_n8 = N // 8
        for n8_base in range(0, num_n8, BLOCK_N8):
            n8_idx = n8_base + tl.arange(0, BLOCK_N8)
            mask_out_n = n8_idx < num_n8

            out_0 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_1 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_2 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_3 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_4 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_5 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_6 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_7 = tl.zeros([BLOCK_N8], dtype=tl.uint32)

            # Statically unroll 8 sub-extractions
            for i in tl.static_range(8):
                n_idx = n8_idx * 8 + i
                mask_n = n_idx < N

                in_ptrs = in_ptr + n_idx * stride_in_n + k8_idx * stride_in_k8
                packed_in = tl.load(in_ptrs, mask=mask_n, other=0).to(tl.uint32)

                shift_n = i * 4
                out_0 |= ((((packed_in >> 0) & 0xF) - 8) & 0xF) << shift_n
                out_1 |= ((((packed_in >> 4) & 0xF) - 8) & 0xF) << shift_n
                out_2 |= ((((packed_in >> 8) & 0xF) - 8) & 0xF) << shift_n
                out_3 |= ((((packed_in >> 12) & 0xF) - 8) & 0xF) << shift_n
                out_4 |= ((((packed_in >> 16) & 0xF) - 8) & 0xF) << shift_n
                out_5 |= ((((packed_in >> 20) & 0xF) - 8) & 0xF) << shift_n
                out_6 |= ((((packed_in >> 24) & 0xF) - 8) & 0xF) << shift_n
                out_7 |= ((((packed_in >> 28) & 0xF) - 8) & 0xF) << shift_n

            k_idx_base = k8_idx * 8
            tl.store(
                out_ptr + (k_idx_base + 0) * stride_out_k + n8_idx * stride_out_n8,
                out_0.to(tl.int32),
                mask=mask_out_n,
            )
            tl.store(
                out_ptr + (k_idx_base + 1) * stride_out_k + n8_idx * stride_out_n8,
                out_1.to(tl.int32),
                mask=mask_out_n,
            )
            tl.store(
                out_ptr + (k_idx_base + 2) * stride_out_k + n8_idx * stride_out_n8,
                out_2.to(tl.int32),
                mask=mask_out_n,
            )
            tl.store(
                out_ptr + (k_idx_base + 3) * stride_out_k + n8_idx * stride_out_n8,
                out_3.to(tl.int32),
                mask=mask_out_n,
            )
            tl.store(
                out_ptr + (k_idx_base + 4) * stride_out_k + n8_idx * stride_out_n8,
                out_4.to(tl.int32),
                mask=mask_out_n,
            )
            tl.store(
                out_ptr + (k_idx_base + 5) * stride_out_k + n8_idx * stride_out_n8,
                out_5.to(tl.int32),
                mask=mask_out_n,
            )
            tl.store(
                out_ptr + (k_idx_base + 6) * stride_out_k + n8_idx * stride_out_n8,
                out_6.to(tl.int32),
                mask=mask_out_n,
            )
            tl.store(
                out_ptr + (k_idx_base + 7) * stride_out_k + n8_idx * stride_out_n8,
                out_7.to(tl.int32),
                mask=mask_out_n,
            )


def repack_int4_npu(weight_packed_t: torch.Tensor) -> torch.Tensor:
    """Repack int4 weights ``[K//8, N]`` -> ``[K, N//8]`` on NPU.

    Args:
        weight_packed_t: int32 tensor of shape ``[K//8, N]`` (8 nibbles per
            element, already transposed so the K dimension is the packed one).

    Returns:
        int32 tensor of shape ``[K, N//8]`` ready for the W4A16 GEMV kernels.
    """
    K_8, N = weight_packed_t.shape
    K = K_8 * 8

    out = torch.empty(
        (K, N // 8), device=weight_packed_t.device, dtype=torch.int32
    )

    # In pure-1D mode we can push BLOCK_N8 to maximise Vector Core throughput.
    BLOCK_N8 = 256

    try:
        _, num_vectorcore = get_device_properties()
    except Exception:
        num_vectorcore = 32

    # Grid locked to num_vectorcore (persistent kernel).
    int4_repack_kernel[(num_vectorcore,)](
        weight_packed_t,
        out,
        N,
        K,
        weight_packed_t.stride(1),
        weight_packed_t.stride(0),
        out.stride(0),
        out.stride(1),
        NUM_CORES=num_vectorcore,
        BLOCK_N8=BLOCK_N8,
    )

    return out


def torch_reference_implementation(
    weight_t: torch.Tensor, K: int, N: int
) -> torch.Tensor:
    """Pure-PyTorch reference for validation.

    Input is already ``[K//8, N]``, so the logic simplifies and there is no
    global transpose.
    """
    # 1. Unpack & offset along dim 0 (K_8) -> result is directly [K, N]
    unpacked_weight = torch.zeros(
        (K, N), device=weight_t.device, dtype=torch.int32
    )
    for i in range(8):
        unpacked_weight[i::8, :] = (weight_t >> (4 * i)) & 0xF
    unpacked_weight = (unpacked_weight - 8).to(torch.int8)

    # 2. Repack along dim 1 (N) -> [K, N // 8]
    repacked_weight = torch.zeros(
        (K, N // 8), device=weight_t.device, dtype=torch.int32
    )
    for i in range(8):
        val = (unpacked_weight[:, i::8].to(torch.int32)) & 0xF
        repacked_weight |= (val << (4 * i))

    return repacked_weight
