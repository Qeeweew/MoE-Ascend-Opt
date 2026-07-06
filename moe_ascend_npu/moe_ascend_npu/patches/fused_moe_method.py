"""Patch NPUW4A16Int4DynamicMoEMethod to use the optimised W4A16 kernels.

Replaces:
  * ``process_weights_after_loading`` - the official path unpacks int4 to int8,
    transposes, then repacks via ``npu_convert_weight_to_int4pack`` (slow). We
    instead fuse unpack+transpose+repack with the Triton ``repack_int4_npu``
    kernel. The packed source weight lives in ``w13_weight_packed`` /
    ``w2_weight_packed`` (registered by our ``create_weights`` patch); the
    repacked ``[E, K, N//8]`` result is stored under the clean name
    ``w13_weight`` / ``w2_weight`` and the ``*_packed`` params are dropped.
  * ``apply`` - the official path always uses ``npu_fused_experts``. We add a
    small-batch fast path (``batch_size <= NPU_W4A16_SMALL_BS_THRESHOLD``,
    default 8) that calls our fused Ascend C kernel
    ``torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs``.
  * ``apply_without_routing_weights`` - the official path uses W4A4 dynamic
    quantisation (``npu_dynamic_quant``), which is incompatible with our W4A16
    weights. We replace it with the W4A16 antiquant ``npu_grouped_matmul`` flow.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_SMALL_BS_THRESHOLD = int(os.environ.get("NPU_W4A16_SMALL_BS_THRESHOLD", "8"))


def _transpose_and_repack_int4(weight_packed: torch.Tensor) -> torch.Tensor:
    """``[E, N, K//8]`` compressed-tensors weight -> ``[E, K, N//8]`` kernel layout."""
    from moe_ascend_npu.kernels import repack_int4_npu

    E, N, K_div_8 = weight_packed.shape
    K = K_div_8 * 8
    # Step 1: Transpose from [E, N, K//8] to [E, K//8, N]
    weight_t = weight_packed.transpose(1, 2).contiguous()
    # Step 2: Flatten to [E*K//8, N] for the Triton kernel
    weight_t_flat = weight_t.view(E * K_div_8, N)
    # Step 3: Use Triton kernel, output is [E*K, N//8]
    weight_repacked_flat = repack_int4_npu(weight_t_flat)
    # Step 4: Reshape back to [E, K, N//8]
    return weight_repacked_flat.view(E, K, N // 8)


def apply():
    from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
        NPUW4A16Int4DynamicMoEMethod,
        npu_fused_experts,
    )
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    from moe_ascend_npu.kernels import ensure_kernels_loaded

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w13_weight_scale = layer.w13_weight_scale.data.transpose(-1, -2).contiguous()
        w2_weight_scale = layer.w2_weight_scale.data.transpose(-1, -2).contiguous()
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)

        layer.w13_weight_offset = torch.nn.Parameter(
            layer.w13_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_offset = torch.nn.Parameter(
            layer.w2_weight_offset.data.transpose(-1, -2).contiguous(),
            requires_grad=False,
        )

        w13_weight = _transpose_and_repack_int4(layer.w13_weight_packed.data)
        w2_weight = _transpose_and_repack_int4(layer.w2_weight_packed.data)

        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        delattr(layer, "w13_weight_packed")
        delattr(layer, "w2_weight_packed")

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

    def apply_method(self, layer, dispatch_output):
        ensure_kernels_loaded()

        combine_input = self._maybe_apply_deepep(layer, dispatch_output)
        if combine_input is not None:
            return combine_input

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, _ = topk_output
        topk_ids = topk_ids.to(torch.int32)

        batch_size, _ = x.shape

        if _SMALL_BS_THRESHOLD > 0 and batch_size <= _SMALL_BS_THRESHOLD:
            topk_weights = topk_weights.to(torch.float)
            output = torch.ops.moe_ascend_npu.fused_moe_w4a16_small_bs(
                x,
                layer.w13_weight,
                layer.w13_weight_scale,
                layer.w2_weight,
                layer.w2_weight_scale,
                topk_ids,
                topk_weights,
            )
        else:
            topk_weights = topk_weights.to(x.dtype)
            output = npu_fused_experts(
                hidden_states=x,
                w13=layer.w13_weight,
                w13_scale=layer.w13_weight_scale,
                w13_offset=layer.w13_weight_offset,
                w2=layer.w2_weight,
                w2_scale=layer.w2_weight_scale,
                w2_offset=layer.w2_weight_offset,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=topk_ids.shape[1],
                use_wna16=True,
            )
        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        if hidden_states_scale is None:
            # gmm1: gate_up_proj
            hidden_states = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w13_weight],
                antiquant_scale=[layer.w13_weight_scale],
                antiquant_offset=[layer.w13_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]

            # act_fn: swiglu
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

            # gmm2: down_proj
            out_hidden = torch.ops.npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[layer.w2_weight],
                antiquant_scale=[layer.w2_weight_scale],
                antiquant_offset=[layer.w2_weight_offset],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=output_dtype,
            )[0]
        else:
            raise ValueError(
                "when weight is int4, hidden_states only supports non-quant dtype!"
            )

        return out_hidden

    NPUW4A16Int4DynamicMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    NPUW4A16Int4DynamicMoEMethod.apply = apply_method
    NPUW4A16Int4DynamicMoEMethod.apply_without_routing_weights = (
        apply_without_routing_weights
    )
    logger.info(
        "NPUW4A16Int4DynamicMoEMethod: patched process_weights_after_loading + apply "
        "+ apply_without_routing_weights (small_bs threshold=%d)",
        _SMALL_BS_THRESHOLD,
    )
