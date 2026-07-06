"""Correctness test for the Triton ``repack_int4_npu`` weight-repack kernel.

Validates that the NPU Triton kernel produces the same ``[K, N//8]`` int32
layout as the pure-PyTorch ``torch_reference_implementation`` (which fuses
uint4b8 unpack + transpose + int4 two's-complement repack).
"""

import torch
import torch_npu  # noqa: F401  (registers NPU device)

from _helpers import setup
from moe_ascend_npu.kernels.repack import (
    repack_int4_npu,
    torch_reference_implementation,
)


def test_repack_int4_npu():
    setup()
    device = "npu:0"

    for (N, K) in [(4096, 4096), (1536, 2048), (2048, 768)]:
        torch.manual_seed(0)
        weight_packed_t = torch.randint(
            -2147483648,
            2147483647,
            (K // 8, N),
            dtype=torch.int32,
            device=device,
        )

        out_ref = torch_reference_implementation(weight_packed_t, K, N)
        out_triton = repack_int4_npu(weight_packed_t)
        torch.npu.synchronize()

        match = torch.equal(out_ref, out_triton)
        max_diff = (out_ref.int() - out_triton.int()).abs().max().item()
        print(
            f"[repack_int4_npu] N={N} K={K} -> match={match} "
            f"max_diff={max_diff} shape={tuple(out_triton.shape)}"
        )
        assert match, f"repack mismatch N={N} K={K} max_diff={max_diff}"


if __name__ == "__main__":
    test_repack_int4_npu()
    print("\nRepack test done.")
