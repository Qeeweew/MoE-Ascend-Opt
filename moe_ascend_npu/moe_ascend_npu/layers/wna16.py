"""NPU-compatible W4A16 linear scheme using Ascend int4 operations.

This scheme converts weights from compressed-tensors format to the Ascend NPU
int4pack layout and uses ``npu_weight_quant_batchmatmul`` for computation. It is
selected (on NPU) by patching ``CompressedTensorsConfig._get_scheme_from_parts``.
"""

import logging
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import ActivationOrdering

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_npu

from moe_ascend_npu.layers.linear_method import NPUW4A16LinearMethod

_is_npu = is_npu()

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)

__all__ = ["NPUCompressedTensorsW4A16"]

WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128,
}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class NPUCompressedTensorsW4A16(CompressedTensorsLinearScheme):
    """NPU-compatible W4A16 linear scheme using Ascend int4 operations."""

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: Optional[int] = None,
        symmetric: Optional[bool] = True,
        actorder: Optional[ActivationOrdering] = None,
    ):
        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "NPU W4A16 kernels require group quantization or "
                "channelwise quantization, but found no group "
                "size and strategy is not channelwise."
            )

        if num_bits != 4:
            raise ValueError(
                f"NPUCompressedTensorsW4A16 only supports 4-bit quantization, "
                f"but got {num_bits} bits."
            )

        if not self.symmetric:
            raise ValueError(
                "NPUCompressedTensorsW4A16 only supports symmetric quantization."
            )

        # Initialise the NPU kernel method.
        self.kernel = NPUW4A16LinearMethod(group_size=self.group_size)

    @classmethod
    def get_min_capability(cls) -> int:
        # NPU doesn't use CUDA capability.
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        """Create weights for NPU W4A16 linear layer.

        Weight is packed int4 stored in int32 format.
        """
        output_size_per_partition = sum(output_partition_sizes)

        # If group_size is -1, we are in the channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size

        # For TP sharding, use input_size_per_partition to calculate scales.
        scales_and_zp_size = input_size_per_partition // group_size

        # Weight packed in int32, shape: [output_size, input_size // pack_factor]
        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }

        if self.strategy == "channel":
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0, **weight_scale_args
            )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )

        # A 2D array defining the original shape of the weights before packing.
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64),
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        # Group index (for activation reordering).
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(
                    input_size_per_partition,
                    dtype=torch.int32,
                ),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_g_idx", weight_g_idx)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading for NPU computation."""
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply W4A16 linear transformation using NPU ops."""
        return self.kernel.apply(layer, x, bias)
