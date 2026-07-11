"""Int4 CPU/NPU expert-cache MoE method for compressed-tensors models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch

from moe_ascend_npu.cache import ExpertCacheConfig, get_expert_cache_manager
from moe_ascend_npu.patches.offload import (
    MoEOffloadInt4FusedMoEMethod,
    _get_or_create_global_callback_manager,
)

logger = logging.getLogger(__name__)


class ExpertCacheFusedMoEMethod(MoEOffloadInt4FusedMoEMethod):
    def __init__(self, cache_config: ExpertCacheConfig):
        super().__init__(group_size=32)
        self.cache_manager = get_expert_cache_manager(cache_config)
        self.graph_contexts: Dict[Tuple[int, int, int], Any] = {}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.offload_config is None or self.moe_infer_handle is None:
            return
        # Materialise one authoritative CPU copy.  SGLang's loader may place
        # parameters created with device="cpu" onto NPU while loading; keeping
        # the layer Parameters alive would therefore defeat the HBM-saving
        # purpose of the cache.
        w13 = layer.w13_weight_packed.data.cpu().contiguous()
        s13 = layer.w13_weight_scale.data.cpu().contiguous()
        w2 = layer.w2_weight_packed.data.cpu().contiguous()
        s2 = layer.w2_weight_scale.data.cpu().contiguous()
        self.moe_infer_handle.store_quantized_repack(w13, s13, w2, s2)
        self.cache_manager.register_layer(
            self.layer_idx, self.moe_infer_handle, w13, s13, w2, s2
        )
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

        npu_out = torch.ops.moe_ascend_npu.fused_moe_w4a16_cached(
            x,
            self.cache_manager.w13_cache, self.cache_manager.s13_cache,
            self.cache_manager.w2_cache, self.cache_manager.s2_cache,
            slot_ids, topk_weights,
        )

        stream_ptr = int(torch_npu.npu.current_stream().npu_stream)
        _get_or_create_global_callback_manager(stream_ptr)
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
        return StandardCombineInput(hidden_states=npu_out + cpu_out)
