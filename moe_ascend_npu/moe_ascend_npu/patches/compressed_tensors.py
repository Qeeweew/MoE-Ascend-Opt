"""Patch CompressedTensorsConfig._get_scheme_from_parts for NPU W4A16 Linear.

On NPU, W4A16 group/channel-quantized Linear layers should use our
``NPUCompressedTensorsW4A16`` scheme (which repacks to the Ascend int4pack
layout and calls ``npu_weight_quant_batchmatmul``) instead of the default
``CompressedTensorsWNA16``. Other cases delegate to the original.
"""

import logging

logger = logging.getLogger(__name__)


def apply():
    from compressed_tensors import CompressionFormat

    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        WNA16_SUPPORTED_BITS,
    )
    from sglang.srt.utils import is_npu

    from moe_ascend_npu.layers.wna16 import NPUCompressedTensorsW4A16

    if not is_npu():
        logger.info("compressed_tensors: not NPU, skipping Linear scheme patch")
        return

    orig = CompressedTensorsConfig._get_scheme_from_parts

    def _get_scheme_from_parts(self, weight_quant, input_quant):
        if (
            self._is_wNa16_group_channel(weight_quant, input_quant)
            and self.quant_format == CompressionFormat.pack_quantized.value
            and weight_quant.num_bits in WNA16_SUPPORTED_BITS
        ):
            logger.info_once("Using NPUCompressedTensorsW4A16 (Linear)")
            return NPUCompressedTensorsW4A16(
                num_bits=weight_quant.num_bits,
                strategy=weight_quant.strategy,
                group_size=weight_quant.group_size,
                symmetric=weight_quant.symmetric,
                actorder=weight_quant.actorder,
            )
        return orig(self, weight_quant, input_quant)

    CompressedTensorsConfig._get_scheme_from_parts = _get_scheme_from_parts
    logger.info("compressed_tensors: routed NPU W4A16 Linear -> NPUCompressedTensorsW4A16")
