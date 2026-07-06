"""NPU W4A16 Linear Method for Ascend NPU.

Uses ``npu_weight_quant_batchmatmul`` for W4A16 computation. The int4 weight is
repacked into the Ascend ``[K, N//8]`` int32 layout by the Triton
``repack_int4_npu`` kernel during ``process_weights_after_loading``.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.quantization.base_config import LinearMethodBase

from moe_ascend_npu.kernels import repack_int4_npu

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class _NPULinearMethodBase(LinearMethodBase):
    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW4A16LinearMethod(_NPULinearMethodBase):
    """Linear method for Ascend NPU W4A16 quantization.

    Weight is packed int4 stored in int32 format. After loading it is repacked
    to ``[K, N//8]`` and run through ``npu_weight_quant_batchmatmul``.
    """

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
        group_size: int = 128,
    ):
        super().__init__(quant_config)
        self.num_bits = 4
        self.pack_factor = 8  # 32 // 4 = 8
        self.group_size = group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert from compressed-tensors format to Ascend NPU format.

        Symmetric quantization only. The op expects weight in ``[K, N//8]``
        (input dim as rows, packed output dim as cols).
        """
        # Current weight shape: [N, K//8] (int32 packed)
        # Step 1: Transpose to [K//8, N] for the optimised kernel.
        weight_t = layer.weight_packed.data.transpose(0, 1).contiguous()

        # Step 2: Fuse unpack + transpose + repack with the Triton kernel.
        # Input: [K//8, N], Output: [K, N//8]
        layer.weight_packed.data = repack_int4_npu(weight_t)

        # Transpose scale from [N, num_groups] to [K//G, N] for op format.
        layer.weight_scale.data = layer.weight_scale.data.transpose(
            0, 1
        ).contiguous()

        # Symmetric quantization: create a zero offset (antiquant_offset must
        # not be None for the op).
        layer.weight_offset = torch.nn.Parameter(
            torch.zeros_like(layer.weight_scale.data), requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply W4A16 linear transformation using npu_weight_quant_batchmatmul."""
        return torch.ops.npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=layer.weight_packed,
            antiquant_scale=layer.weight_scale,
            antiquant_offset=layer.weight_offset,
            antiquant_group_size=self.group_size,
            bias=bias,
        )
