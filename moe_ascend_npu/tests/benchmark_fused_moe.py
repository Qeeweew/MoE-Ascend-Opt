"""Benchmark: fused MoE W4A16 small-batch kernel vs official npu_fused_experts.

Compares the custom ``torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs`` kernel
(used for batch_size <= NPU_W4A16_SMALL_BS_THRESHOLD in the decode path) against
the official ``npu_fused_experts`` W4A16 antiquant path (used for large batches).

Both paths run inside an NPU graph that captures 16 routing blocks per replay,
and are timed with NPU events.

Migrated from sgl-kernel-npu/tests/python/sgl_kernel_npu/benchmark_fused_moe.py;
the only change is the custom kernel namespace (``npu`` -> ``moe_ascend_npu``).
"""

import math
import sys

import torch
import torch_npu

try:
    import sgl_kernel_npu  # noqa: F401  (registers npu.* ops used by the reference path)
except ImportError:
    print("Warning: sgl_kernel_npu not found.")
    sys.exit(1)

from moe_ascend_npu.kernels import ensure_kernels_loaded

ensure_kernels_loaded()

# ------------------------------------------------------------
# Fixed constants (DO NOT CHANGE)
# ------------------------------------------------------------
TOP_K = 8
NUM_EXPERTS = 128
HIDDEN_SIZE = 2048
INTER_SIZE = 768
GROUP_SIZE = 32
ROUTING_BLOCKS = NUM_EXPERTS // TOP_K  # 16

# Timing configuration
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

device = torch.device("npu:0")
dtype = torch.float16


# ------------------------------------------------------------
# NPU timing with events
# ------------------------------------------------------------
def measure_time_with_events(fn, num_iters: int = 100) -> float:
    """Measure NPU execution time using events."""
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        fn()
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    torch.npu.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    return elapsed_ms


# ------------------------------------------------------------
# Reference implementation (official npu_fused_experts W4A16 path)
# ------------------------------------------------------------
def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    w13_offset = kwargs["w13_offset"]
    w2_offset = kwargs["w2_offset"]
    row_idx = kwargs["row_idx"]

    original_dtype = hidden_states.dtype
    scale_dtype = torch.float32

    # ---- init routing (row_idx MUST be 2D) ----
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=hidden_states.shape[0],
        )
    )

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, w13.shape[0]
    ).to(torch.int64)

    # ---- W13 ----
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        antiquant_scale=[w13_scale],
        antiquant_offset=[w13_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    hidden_states = torch_npu.npu_swiglu(hidden_states)

    # ---- W2 ----
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # ---- finalize ----
    return torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights.to(hidden_states.dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def pack_int4_weights(raw, device):
    flat = raw.view(-1, raw.shape[-1]).to(device)
    packed = torch_npu.npu_convert_weight_to_int4pack(flat)
    return packed.view(*raw.shape[:-1], raw.shape[-1] // 8)


def build_row_idx(B):
    row_idx_len = B * TOP_K
    return (
        torch.arange(row_idx_len, dtype=torch.int32, device=device)
        .view(TOP_K, -1)
        .permute(1, 0)
        .contiguous()
    )


def build_routing_blocks():
    experts = torch.arange(NUM_EXPERTS, device=device, dtype=torch.int32)
    return experts.view(ROUTING_BLOCKS, TOP_K)


def build_weight_blocks():
    w = torch.randn((ROUTING_BLOCKS, TOP_K), device=device)
    return torch.softmax(w, dim=-1)


# ------------------------------------------------------------
# Capture graph: 16 forwards inside ONE graph
# ------------------------------------------------------------
def capture_16_forward_graph(fn, x, row_idx, topk_ids_all, topk_w_all):
    # warmup
    for i in range(ROUTING_BLOCKS):
        fn(x, row_idx, topk_ids_all[i], topk_w_all[i])
    torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for i in range(ROUTING_BLOCKS):
            fn(x, row_idx, topk_ids_all[i], topk_w_all[i])
    torch.npu.synchronize()
    return g


# ------------------------------------------------------------
# Bytes accounting (weight traffic)
# ------------------------------------------------------------
def bytes_per_expert(w13, w13_scale, w2, w2_scale):
    # Symmetric quantization: no offset tensors
    return (
        w13[0].numel() * w13.element_size()
        + w13_scale[0].numel() * w13_scale.element_size()
        + w2[0].numel() * w2.element_size()
        + w2_scale[0].numel() * w2_scale.element_size()
    )


# ------------------------------------------------------------
# Benchmark per batch size
# ------------------------------------------------------------
def benchmark_bs(B, w13, w13_scale, w13_offset, w2, w2_scale, w2_offset):
    x = torch.randn((B, HIDDEN_SIZE), device=device, dtype=dtype)
    row_idx = build_row_idx(B)

    blocks = build_routing_blocks()
    weights = build_weight_blocks()

    # build 16 routing sets
    topk_ids_all, topk_w_all = [], []
    for i in range(ROUTING_BLOCKS):
        ids = torch.empty((B, TOP_K), device=device, dtype=torch.int32)
        wts = torch.empty((B, TOP_K), device=device)
        for b in range(B):
            blk = (i + b) % ROUTING_BLOCKS
            ids[b].copy_(blocks[blk])
            wts[b].copy_(weights[blk])
        topk_ids_all.append(ids)
        topk_w_all.append(wts)

    # reference graph (official npu_fused_experts W4A16 antiquant path)
    g_ref = capture_16_forward_graph(
        lambda x, row_idx, ids, w: npu_fused_experts(
            hidden_states=x,
            w13=w13, w13_scale=w13_scale,
            w2=w2, w2_scale=w2_scale,
            topk_weights=w,
            topk_ids=ids,
            top_k=TOP_K,
            w13_offset=w13_offset,
            w2_offset=w2_offset,
            row_idx=row_idx,
        ),
        x, row_idx, topk_ids_all, topk_w_all,
    )

    # custom fused graph (moe_ascend_npu.fused_moe_w4a16_small_bs)
    g_custom = capture_16_forward_graph(
        lambda x, row_idx, ids, w: torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs(
            x,
            w13, w13_scale,
            w2, w2_scale,
            ids, w,
        ),
        x, row_idx, topk_ids_all, topk_w_all,
    )

    def run_ref():
        g_ref.replay()

    def run_custom():
        g_custom.replay()

    # Event-based timing
    t_ref_ms = measure_time_with_events(run_ref, BENCHMARK_ITERATIONS)
    t_custom_ms = measure_time_with_events(run_custom, BENCHMARK_ITERATIONS)

    bytes_total = (
        B * TOP_K * ROUTING_BLOCKS *
        bytes_per_expert(w13, w13_scale, w2, w2_scale)
    )

    return {
        "B": B,
        "ref_us": t_ref_ms * 1e3 / ROUTING_BLOCKS,
        "custom_us": t_custom_ms * 1e3 / ROUTING_BLOCKS,
        "ref_GBs": bytes_total / (t_ref_ms * 1e-3) / 1e9,
        "custom_GBs": bytes_total / (t_custom_ms * 1e-3) / 1e9,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    torch.manual_seed(0)

    raw_w13 = torch.randint(
        -8, 8, (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTER_SIZE), dtype=torch.int32
    )
    raw_w2 = torch.randint(
        -8, 8, (NUM_EXPERTS, INTER_SIZE, HIDDEN_SIZE), dtype=torch.int32
    )

    w13 = pack_int4_weights(raw_w13, device)
    w2 = pack_int4_weights(raw_w2, device)

    w13_scale = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE // GROUP_SIZE, 2 * INTER_SIZE),
        device=device, dtype=dtype
    )
    w13_offset = torch.zeros_like(w13_scale)

    w2_scale = torch.randn(
        (NUM_EXPERTS, INTER_SIZE // GROUP_SIZE, HIDDEN_SIZE),
        device=device, dtype=dtype
    )
    w2_offset = torch.zeros_like(w2_scale)

    print("BS | Ref(us) | Custom(us) | Ref GB/s | Custom GB/s")
    print("-" * 56)

    for B in [1, 2, 3, 4, 5, 6, 7, 8]:
        r = benchmark_bs(B, w13, w13_scale, w13_offset, w2, w2_scale, w2_offset)
        print(
            f"{B:>2} | {r['ref_us']:>7.2f} | {r['custom_us']:>10.2f} | "
            f"{r['ref_GBs']:>8.2f} | {r['custom_GBs']:>10.2f}"
        )


if __name__ == "__main__":
    main()
