"""Correctness test for the fused W4A16 MoE kernel (decoding / small batch).

Tests ``torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs`` against a
step-by-step PyTorch reference that reuses the same uint4b8 weights and
per-group scales, repacked through ``repack_int4_npu`` exactly as
``process_weights_after_loading`` does.

The fused kernel performs: W13 (broadcast x) -> SwiGLU -> W2 (weighted reduce)
in a single launch, so the reference mirrors that data flow.
"""

import math

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers NPU device)

from _helpers import (
    GROUP_SIZE,
    pack_uint4b8,
    repack_expert_weight,
    setup,
    transpose_scale,
    w4a16_matvec_ref,
)


def fused_moe_w4a16_ref(
    x: torch.Tensor,
    raw_w13: torch.Tensor,
    scale13: torch.Tensor,
    raw_w2: torch.Tensor,
    scale2: torch.Tensor,
    expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for the fused W4A16 MoE kernel.

    Args:
        x:            ``[BS, Hidden]``
        raw_w13:      ``[E, 2*Inter, Hidden]`` signed int4
        scale13:      ``[E, num_groups_w13, 2*Inter]``
        raw_w2:       ``[E, Hidden, Inter]`` signed int4
        scale2:       ``[E, num_groups_w2, Hidden]``
        expert_ids:   ``[BS, TopK]`` int32
        topk_weights: ``[BS, TopK]`` float

    Returns:
        ``[BS, Hidden]`` fp32 output.
    """
    bs, hidden = x.shape
    top_k = expert_ids.shape[1]
    inter = raw_w2.shape[2]
    inter2 = 2 * inter

    y = torch.zeros(bs, hidden, dtype=torch.float32, device=x.device)
    for b in range(bs):
        for t in range(top_k):
            e = int(expert_ids[b, t].item())
            # W13: [2*Inter]
            h13 = w4a16_matvec_ref(raw_w13[e], scale13[e], x[b])
            # SwiGLU
            gate, val = h13[:inter], h13[inter:]
            h_act = F.silu(gate) * val
            # W2: [Hidden]
            h2 = w4a16_matvec_ref(raw_w2[e], scale2[e], h_act.to(x.dtype))
            y[b] += float(topk_weights[b, t].item()) * h2
    return y


def _run_one(batch_size):
    device = "npu:0"
    dtype = torch.float16

    TOP_K = 8
    NUM_EXPERTS = 16
    HIDDEN_SIZE = 2048  # divisible by BLOCK_SIZE=128
    INTER_SIZE = 768  # divisible by 128
    GROUP_SIZE_ = GROUP_SIZE  # 32

    torch.manual_seed(42)

    x = torch.randn(batch_size, HIDDEN_SIZE, dtype=dtype, device=device) * (
        1.0 / math.sqrt(HIDDEN_SIZE)
    )
    expert_ids = torch.randint(
        0, NUM_EXPERTS, (batch_size, TOP_K), dtype=torch.int32, device=device
    )
    topk_weights = torch.randn(batch_size, TOP_K, dtype=torch.float32, device=device)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    # W13: raw signed int4 [E, N=2*Inter, K=Hidden]
    raw_w13 = torch.randint(
        -8, 8, (NUM_EXPERTS, 2 * INTER_SIZE, HIDDEN_SIZE), dtype=torch.int32, device=device
    )
    w13_packed_ct = pack_uint4b8(raw_w13)  # [E, 2*Inter, Hidden//8]
    w13_weight = repack_expert_weight(w13_packed_ct)  # [E, Hidden, 2*Inter//8]

    num_groups_w13 = HIDDEN_SIZE // GROUP_SIZE_
    scale13_ct = torch.randn(
        NUM_EXPERTS, 2 * INTER_SIZE, num_groups_w13, dtype=dtype, device=device
    ) * (1.0 / 8.0)
    w13_scale = transpose_scale(scale13_ct)  # [E, num_groups, 2*Inter]

    # W2: raw signed int4 [E, N=Hidden, K=Inter]
    raw_w2 = torch.randint(
        -8, 8, (NUM_EXPERTS, HIDDEN_SIZE, INTER_SIZE), dtype=torch.int32, device=device
    )
    w2_packed_ct = pack_uint4b8(raw_w2)  # [E, Hidden, Inter//8]
    w2_weight = repack_expert_weight(w2_packed_ct)  # [E, Inter, Hidden//8]

    num_groups_w2 = INTER_SIZE // GROUP_SIZE_
    scale2_ct = torch.randn(
        NUM_EXPERTS, HIDDEN_SIZE, num_groups_w2, dtype=dtype, device=device
    ) * (1.0 / 8.0)
    w2_scale = transpose_scale(scale2_ct)  # [E, num_groups, Hidden]

    # Run custom fused kernel
    _ = torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs(
        x, w13_weight, w13_scale, w2_weight, w2_scale, expert_ids, topk_weights
    )  # warmup
    y_custom = torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs(
        x, w13_weight, w13_scale, w2_weight, w2_scale, expert_ids, topk_weights
    )
    torch.npu.synchronize()

    # Reference
    y_ref = fused_moe_w4a16_ref(
        x, raw_w13, w13_scale, raw_w2, w2_scale, expert_ids, topk_weights
    )

    y_custom_f = y_custom.float()
    diff = (y_custom_f - y_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel = max_diff / (y_ref.abs().max().item() + 1e-6)

    print(
        f"[fused_moe_w4a16_small_bs] BS={batch_size} TopK={TOP_K} E={NUM_EXPERTS} "
        f"H={HIDDEN_SIZE} I={INTER_SIZE}\n"
        f"  max_diff={max_diff:.4f} mean_diff={mean_diff:.4f} rel={rel:.4f}"
    )
    print(f"  ref    : {y_ref[0, :5].tolist()}")
    print(f"  custom : {y_custom_f[0, :5].tolist()}")

    # Two stacked W4A16 matvecs + SwiGLU in fp16: allow a loose absolute tol.
    assert max_diff < 1.0, f"fused_moe (BS={batch_size}) max_diff {max_diff} too large"


def test_fused_moe_w4a16_small_bs():
    setup()
    for bs in [1, 2, 4]:
        _run_one(bs)


if __name__ == "__main__":
    test_fused_moe_w4a16_small_bs()
    print("\nFused MoE test done.")
