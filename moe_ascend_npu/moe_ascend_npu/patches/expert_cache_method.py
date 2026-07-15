"""Int4 CPU/NPU expert-cache MoE method for compressed-tensors models."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Tuple

import torch

from moe_ascend_npu.cache import ExpertCacheConfig, get_expert_cache_manager
from moe_ascend_npu.patches.offload import (
    MoEOffloadInt4FusedMoEMethod,
    _get_or_create_global_callback_manager,
)

logger = logging.getLogger(__name__)

_SIDE_STREAM_LOCK = threading.Lock()
_CPU_SIDE_STREAMS: Dict[int, Any] = {}


def _npu_grouped_moe_cached_hits(
    hidden_states: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    """Run cache-hit prefill routes with the original NPU grouped GEMM."""
    hit_mask = slot_ids >= 0
    token_ids = (
        torch.arange(
            hidden_states.shape[0], dtype=torch.int64,
            device=hidden_states.device,
        )
        .view(-1, 1)
        .expand_as(slot_ids)[hit_mask]
    )
    hit_slot_ids = slot_ids[hit_mask].reshape(-1, 1)
    if hit_slot_ids.numel() == 0:
        return torch.zeros_like(hidden_states)

    routed_hidden = hidden_states.index_select(0, token_ids)
    routed_weights = topk_weights[hit_mask].reshape(-1, 1).to(hidden_states.dtype)
    num_routes = routed_hidden.shape[0]
    row_idx = torch.arange(
        num_routes, dtype=torch.int32, device=hidden_states.device
    ).view(-1, 1)
    routed_hidden, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            routed_hidden,
            row_idx=row_idx,
            expert_idx=hit_slot_ids,
            active_num=num_routes,
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, w13.shape[0]
    ).to(torch.int64)

    routed_hidden = torch.ops.npu.npu_grouped_matmul(
        x=[routed_hidden],
        weight=[w13],
        antiquant_scale=[s13],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=hidden_states.dtype,
    )[0]
    routed_hidden = torch.ops.npu.npu_swiglu(routed_hidden)
    routed_hidden = torch.ops.npu.npu_grouped_matmul(
        x=[routed_hidden],
        weight=[w2],
        antiquant_scale=[s2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=hidden_states.dtype,
    )[0]
    routed_hidden = torch.ops.npu.npu_moe_finalize_routing(
        routed_hidden,
        skip1=None,
        skip2=None,
        bias=None,
        scales=routed_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=hit_slot_ids,
    )

    output = torch.zeros_like(hidden_states)
    output.index_add_(0, token_ids, routed_hidden)
    return output


def _get_cpu_side_stream():
    """One CPU-callback submission stream per NPU device.

    Layers are data-dependent and join before returning, so additional streams
    cannot overlap different layers.  Sharing one stream avoids dozens of ACL
    report threads and gives graph capture a single stable fork target.
    """
    import torch_npu

    device = int(torch_npu.npu.current_device())
    with _SIDE_STREAM_LOCK:
        stream = _CPU_SIDE_STREAMS.get(device)
        if stream is None:
            stream = torch_npu.npu.Stream(device=device)
            _CPU_SIDE_STREAMS[device] = stream
        return stream


class ExpertCacheFusedMoEMethod(MoEOffloadInt4FusedMoEMethod):
    def __init__(self, cache_config: ExpertCacheConfig):
        super().__init__(group_size=32)
        self.cache_manager = get_expert_cache_manager(cache_config)
        self.graph_contexts: Dict[Tuple[int, int, int], Any] = {}
        self.cpu_stream = None
        self.input_ready = None
        self.cpu_done = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.offload_config is None or self.moe_infer_handle is None:
            return
        # Use loader tensors only to initialise MoEInfer.  MoEInfer's NUMA Q4_0
        # buffers become the sole persistent CPU weight copy; cache replacement
        # exports individual experts from that storage into pinned staging.
        w13 = layer.w13_weight_packed.data.cpu().contiguous()
        s13 = layer.w13_weight_scale.data.cpu().contiguous()
        w2 = layer.w2_weight_packed.data.cpu().contiguous()
        s2 = layer.w2_weight_scale.data.cpu().contiguous()
        self.moe_infer_handle.store_quantized_repack(w13, s13, w2, s2)
        self.cache_manager.register_layer(
            self.layer_idx,
            self.moe_infer_handle,
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
            s13.dtype,
        )
        # Create stream/event identities before graph capture.  Reusing these
        # objects is required for deterministic graph replay dependencies.
        import torch_npu

        self.cpu_stream = _get_cpu_side_stream()
        self.input_ready = torch_npu.npu.Event()
        self.cpu_done = torch_npu.npu.Event()
        _get_or_create_global_callback_manager(int(self.cpu_stream.npu_stream))
        for name in (
            "w13_weight_packed", "w13_weight_scale",
            "w2_weight_packed", "w2_weight_scale",
            "w13_weight_shape", "w2_weight_shape",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
        logger.info("[ExpertCache] registered layer %d (%d experts)", self.layer_idx, self.num_experts)

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.dp_attention import get_is_extend_in_batch
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
        import torch_npu

        from moe_ascend_npu.kernels import ensure_kernels_loaded

        ensure_kernels_loaded()
        self.cache_manager.ensure_allocated()
        x = dispatch_output.hidden_states
        topk_weights = dispatch_output.topk_output.topk_weights.to(torch.float32)
        topk_ids = dispatch_output.topk_output.topk_ids.to(torch.int32)
        table = self.cache_manager.slot_table[self.layer_idx]
        slot_ids = table.gather(0, topk_ids.flatten().to(torch.int64)).view_as(topk_ids)
        cpu_ids = torch.where(slot_ids >= 0, torch.full_like(topk_ids, -1), topk_ids)

        main_stream = torch_npu.npu.current_stream()
        if get_is_extend_in_batch():
            # Prefill consumes the existing cache without training the decode
            # policy: hits run on NPU and misses run on CPU, while telemetry
            # and replacement remain disabled.
            _get_or_create_global_callback_manager(int(main_stream.npu_stream))
            num_tokens, top_k = int(x.shape[0]), int(topk_ids.shape[1])
            dtype_int = 1 if x.dtype == torch.bfloat16 else 0
            ctx = torch.classes.nanovllm.MoEGraphContext(
                self.moe_infer_handle, num_tokens, top_k, dtype_int
            )
            torch.ops.nanovllm.moe_forward_npu_graph_partial_start(
                x, cpu_ids, topk_ids, topk_weights,
                self.moe_infer_handle, ctx, False,
            )
            npu_out = _npu_grouped_moe_cached_hits(
                x, slot_ids, topk_weights,
                self.cache_manager.w13_cache, self.cache_manager.s13_cache,
                self.cache_manager.w2_cache, self.cache_manager.s2_cache,
            )
            cpu_out = torch.empty_like(x)
            torch.ops.nanovllm.moe_forward_npu_eager_partial_wait_out(cpu_out, ctx)
            return StandardCombineInput(
                hidden_states=torch.add(npu_out, cpu_out)
            )

        if get_is_capture_mode():
            # Decode graph path: keep everything on one stream.  The first
            # callback only dispatches CPU work to a dedicated coordinator and
            # returns; the NPU cached kernel then runs while the CPU workers are
            # active.  The second callback joins CPU completion, after which the
            # pre-enqueued H2D copy publishes cpu_out.
            _get_or_create_global_callback_manager(int(main_stream.npu_stream))
            num_tokens, top_k = int(x.shape[0]), int(topk_ids.shape[1])
            dtype_int = 1 if x.dtype == torch.bfloat16 else 0
            key = (num_tokens, top_k, dtype_int)
            ctx = self.graph_contexts.get(key)
            if ctx is None:
                ctx = torch.classes.nanovllm.MoEGraphContext(
                    self.moe_infer_handle, num_tokens, top_k, dtype_int
                )
                self.graph_contexts[key] = ctx
            cpu_out = torch.empty_like(x)
            torch.ops.nanovllm.moe_forward_npu_graph_partial_start(
                x, cpu_ids, topk_ids, topk_weights,
                self.moe_infer_handle, ctx, True,
            )
            npu_out = torch.ops.moe_ascend_npu.fused_moe_w4a16_cached(
                x,
                self.cache_manager.w13_cache, self.cache_manager.s13_cache,
                self.cache_manager.w2_cache, self.cache_manager.s2_cache,
                slot_ids, topk_weights,
            )
            torch.ops.nanovllm.moe_forward_npu_graph_partial_wait_out(cpu_out, ctx)
            return StandardCombineInput(
                hidden_states=torch.add(npu_out, cpu_out)
            )

        if self.cpu_stream is None:
            # Defensive eager-only fallback for unusual loaders that skip
            # process_weights_after_loading.
            self.cpu_stream = _get_cpu_side_stream()
            self.input_ready = torch_npu.npu.Event()
            self.cpu_done = torch_npu.npu.Event()
            _get_or_create_global_callback_manager(int(self.cpu_stream.npu_stream))

        if self.layer_idx == min(self.cache_manager.sources):
            # This branch is decode without graph replay.  Drive the same
            # decode-only controller here; Prefill returned above and never
            # enters routing telemetry or replacement.
            self.cache_manager.before_replay(int(x.shape[0]))

        self.input_ready.record(main_stream)
        with torch_npu.npu.stream(self.cpu_stream):
            self.cpu_stream.wait_event(self.input_ready)
            cpu_out = torch.ops.nanovllm.moe_forward_npu_stream_partial(
                x, cpu_ids, topk_ids, topk_weights, self.moe_infer_handle
            )
            self.cpu_done.record(self.cpu_stream)

        # The NPU hot path is intentionally submitted after the CPU fork and
        # before the join.  It is much faster, so the join normally waits only
        # for the CPU cold remainder rather than adding both latencies.
        npu_out = torch.ops.moe_ascend_npu.fused_moe_w4a16_cached(
            x,
            self.cache_manager.w13_cache, self.cache_manager.s13_cache,
            self.cache_manager.w2_cache, self.cache_manager.s2_cache,
            slot_ids, topk_weights,
        )
        main_stream.wait_event(self.cpu_done)
        return StandardCombineInput(
            hidden_states=torch.add(npu_out, cpu_out)
        )
