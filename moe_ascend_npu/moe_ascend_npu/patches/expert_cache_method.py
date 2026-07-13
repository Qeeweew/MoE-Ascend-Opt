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
        if self.cpu_stream is None:
            # Defensive eager-only fallback for unusual loaders that skip
            # process_weights_after_loading.
            self.cpu_stream = _get_cpu_side_stream()
            self.input_ready = torch_npu.npu.Event()
            self.cpu_done = torch_npu.npu.Event()
            _get_or_create_global_callback_manager(int(self.cpu_stream.npu_stream))

        self.input_ready.record(main_stream)
        with torch_npu.npu.stream(self.cpu_stream):
            self.cpu_stream.wait_event(self.input_ready)
            if not get_is_capture_mode():
                cpu_out = torch.ops.nanovllm.moe_forward_npu_stream_partial(
                    x, cpu_ids, topk_ids, topk_weights, self.moe_infer_handle
                )
            else:
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
                torch.ops.nanovllm.moe_forward_npu_graph_partial_out(
                    x, cpu_ids, topk_ids, topk_weights,
                    self.moe_infer_handle, ctx, cpu_out,
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
