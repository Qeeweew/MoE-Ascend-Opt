"""Patch NPUCompressedTensorsW4A16Int4DynamicMoE for custom kernel compatibility.

Replaces ``__init__`` and ``create_weights``:
  * Read ``strategy`` from the quantization config (instead of hardcoding).
  * Use ``params_dtype`` for the weight scales/offsets (instead of hardcoded
    ``bfloat16``) so fp16 models work with our small-batch kernel.
  * Register the packed int4 weights as ``w13_weight_packed`` / ``w2_weight_packed``
    (instead of the official ``w13_weight`` / ``w2_weight``). The repacked
    ``[E, K, N//8]`` tensor produced by ``process_weights_after_loading`` then
    takes the clean ``w13_weight`` / ``w2_weight`` name, avoiding in-place
    overwrite of the source buffer.
"""

import logging

import torch

from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


def apply():
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
        NPUCompressedTensorsW4A16Int4DynamicMoE,
        NPUW4A16Int4DynamicMoEMethod,
    )

    def init_replace(self, quantization_config) -> None:
        self.pack_factor = 8  # weight dtype is int4, but use int32 to create
        target = (
            "MoEGMM" if "MoEGMM" in quantization_config.target_scheme_map else "Linear"
        )
        if target in quantization_config.target_scheme_map:
            self.group_size = quantization_config.target_scheme_map[target][
                "weights"
            ].group_size
        else:
            self.group_size = 128

        self.kernel = NPUW4A16Int4DynamicMoEMethod()
        config = quantization_config.target_scheme_map["Linear"].get("weights")
        self.strategy = config.strategy

    def create_weights_replace(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.num_experts = num_experts

        extra_weight_attrs.update({"quant_method": self.strategy})

        # weight - register as *_packed so process_weights_after_loading can
        # store the repacked tensor under the clean name (w13_weight / w2_weight)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale - use params_dtype instead of hardcoded bfloat16
        weight_scale_dtype = params_dtype
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.int64, device="cpu"),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.int64, device="cpu"),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)

    NPUCompressedTensorsW4A16Int4DynamicMoE.__init__ = init_replace
    NPUCompressedTensorsW4A16Int4DynamicMoE.create_weights = create_weights_replace
    logger.info(
        "NPUCompressedTensorsW4A16Int4DynamicMoE: patched __init__ + create_weights "
        "(w13_weight_packed/w2_weight_packed, params_dtype scales)"
    )
