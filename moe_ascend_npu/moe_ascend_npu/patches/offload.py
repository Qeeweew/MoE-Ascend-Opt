# SPDX-License-Identifier: Apache-2.0
"""
MoE Computation Offload Implementation for SGLang.

This module implements MoE computation offload to CPU using Int8/Int4 quantization,
leveraging the nanovllm_ext infrastructure. Supports both eager mode and NPU graph mode.

Features:
- Int8 online quantization for non-quantized models (Q8_0)
- Int4 pre-quantized weights from compressed-tensors format (Q4_0)
- Support for both eager and NPU graph modes
- Asynchronous NPU-CPU data transfers

Note: nanovllm_ext is required for this module to work.
"""

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.utils import is_npu, logger

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.server_args import ServerArgs


# nanovllm_ext (built from the sibling Int8-gemm/ directory) is a hard
# dependency of MoE CPU offload. Importing this module fails fast if it is
# not installed; the module is only imported lazily when offload is actually
# enabled (see moe_ascend_npu.patches.moe_layer), so normal NPU serving does
# not require it.
import nanovllm_ext  # noqa: F401

@dataclass
class MoEOffloadConfig:
    """Configuration for MoE computation offload."""

    enabled: bool = False
    layer_idx: int = -1
    quant_type: str = "q8_0"  # "q8_0" or "q4_0"


def _is_compressed_tensors_int4(quant_config: Any) -> bool:
    """Detect whether *quant_config* is a compressed-tensors W4A16 config.

    Q8_0 (online int8 quantization) requires fp16/bf16 weights; int4-packed
    compressed-tensors weights cannot be re-quantized online and must use the
    Q4_0 (pre-quantized int4) path instead.
    """
    if quant_config is None:
        return False
    try:
        name = quant_config.get_name()
    except Exception:
        return False
    if "compressed" not in str(name).lower():
        return False
    tsm = getattr(quant_config, "target_scheme_map", None) or {}
    for target in ("MoEGMM", "Linear"):
        scheme = tsm.get(target, {})
        weights = scheme.get("weights") if isinstance(scheme, dict) else None
        if weights is not None and getattr(weights, "num_bits", None) == 4:
            return True
    return False


def create_moe_offload_config(
    layer_idx: int,
    server_args: "ServerArgs",
    quant_config: Optional[Any] = None,
) -> Optional[MoEOffloadConfig]:
    """
    Determine if the current layer should be offloaded based on server arguments.

    Args:
        layer_idx: Index of the current MoE layer
        server_args: Global server arguments
        quant_config: Quantization config if the model has pre-quantized weights

    Returns:
        MoEOffloadConfig if offload is enabled for this layer, None otherwise
    """
    if not getattr(server_args, "enable_moe_offload", False):
        return None

    start_layer = getattr(server_args, "moe_offload_start_layer", 0)

    if layer_idx < start_layer:
        return None

    quant_type = getattr(server_args, "moe_offload_quant_type", "q8_0")

    # Check compatibility with existing quantization config
    if quant_config is not None:
        quant_name = quant_config.get_name() if hasattr(quant_config, "get_name") else str(quant_config)

        # Q8_0 (online int8 quantization) needs fp16/bf16 weights; a
        # compressed-tensors int4 (W4A16) model ships int4-packed weights that
        # cannot be re-quantized online -> fall back to Q4_0 (pre-quantized
        # int4) automatically.
        if quant_type == "q8_0" and _is_compressed_tensors_int4(quant_config):
            logger.warning(
                f"[MoE Offload] Layer {layer_idx}: Q8_0 (online int8 quantization) "
                f"is incompatible with compressed-tensors int4 (W4A16) weights "
                f"(quant_name={quant_name}). The weights are int4-packed and "
                f"cannot be re-quantized online. Automatically switching to "
                f"Q4_0 (pre-quantized int4)."
            )
            quant_type = "q4_0"

        # AWQ, GPTQ, and similar pre-quantized formats are not compatible with Q4_0
        # because they use different packing formats
        incompatible_formats = ["awq", "gptq", "gptq_marlin", "awq_marlin"]
        if quant_type == "q4_0" and any(fmt in quant_name.lower() for fmt in incompatible_formats):
            logger.warning(
                f"[MoE Offload] Layer {layer_idx}: Q4_0 is not compatible with {quant_name} "
                f"quantization. Falling back to Q8_0 (online quantization)."
            )
            quant_type = "q8_0"

        # For Q4_0, we need compressed-tensors format or similar
        # If the model has pre-quantized weights but not in the right format,
        # fall back to Q8_0
        if quant_type == "q4_0" and quant_name not in ["compressed_tensors", "compressed-tensors"]:
            logger.warning(
                f"[MoE Offload] Layer {layer_idx}: Q4_0 requires compressed-tensors format, "
                f"but got {quant_name}. Falling back to Q8_0 (online quantization)."
            )
            quant_type = "q8_0"

    logger.info(
        f"[MoE Offload] Layer {layer_idx} will be offloaded to CPU "
        f"(Start Layer: {start_layer}, Quant Type: {quant_type})"
    )

    return MoEOffloadConfig(
        enabled=True,
        layer_idx=layer_idx,
        quant_type=quant_type,
    )


# Global callback manager cache: keyed by (device_id, stream_ptr)
_GLOBAL_CB_LOCK = threading.Lock()
_GLOBAL_CB_MANAGERS: Dict[Tuple[int, int], Any] = {}


def _get_or_create_global_callback_manager(stream_ptr: int) -> Any:
    """
    Get or create a global NPU callback manager for the given stream.

    Each (device_id, stream_ptr) combination gets its own manager to handle
    asynchronous callbacks for MoE computation offload.

    Args:
        stream_ptr: Pointer to the NPU stream

    Returns:
        NpuCallbackManager instance
    """
    import torch_npu

    device_id = int(torch_npu.npu.current_device())
    key = (device_id, int(stream_ptr))

    with _GLOBAL_CB_LOCK:
        mgr = _GLOBAL_CB_MANAGERS.get(key)
        if mgr is not None:
            return mgr

        mgr = torch.classes.nanovllm.NpuCallbackManager(int(stream_ptr), device_id)
        _GLOBAL_CB_MANAGERS[key] = mgr
        logger.info(
            f"[MoE Offload] Created NpuCallbackManager. "
            f"device_id={device_id}, stream_ptr={stream_ptr}"
        )
        return mgr


class MoEOffloadFusedMoEMethod(FusedMoEMethodBase):
    """
    MoE computation offload method using Int8 quantization on CPU.

    This method provides computation offload to CPU using the nanovllm_ext infrastructure.

    Features:
    - Int8 quantization for expert weights (online quantization)
    - NUMA-aware memory placement
    - Support for both eager and NPU graph modes
    - Asynchronous NPU-CPU data transfers

    Note: This class requires nanovllm_ext to be installed; importing this
    module fails fast if it is not (see module docstring).
    """

    def __init__(self):
        super().__init__()

        self.offload_config: Optional[MoEOffloadConfig] = None
        self.moe_infer_handle: Optional[Any] = None

        # Graph context cache: key=(num_tokens, top_k, dtype_int)
        self.graph_contexts: Dict[Tuple[int, int, int], Any] = {}

        # Metadata
        self.num_experts: Optional[int] = None
        self.hidden_size: Optional[int] = None
        self.intermediate_size: Optional[int] = None
        self.params_dtype: Optional[torch.dtype] = None
        self.layer_idx: int = -1

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Create weight placeholders and initialize MoE infer handle.

        This implementation creates lightweight dummy parameters and initializes
        the CPU-side MoE inference engine with quantized weight storage.
        """
        # Initialize metadata
        tp_rank = get_tensor_model_parallel_rank()
        if tp_rank != 0:
            # Only rank 0 handles weight storage for now
            logger.debug(f"[MoE Offload] Rank {tp_rank} skipping weight storage")
            return

        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size_per_partition)
        self.params_dtype = params_dtype

        # Create CPU MoE inference handle with Int8 quantization (type=0)
        if self.moe_infer_handle is None:
            self.moe_infer_handle = torch.classes.nanovllm.MoEInfer(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
                0,  # Int8 Quantization (Q8_0)
            )
            logger.info(
                f"[MoE Offload] Created MoEInfer handle for layer {self.layer_idx}: "
                f"experts={self.num_experts}, hidden_size={self.hidden_size}, "
                f"intermediate_size={self.intermediate_size}, quant_type=Q8_0"
            )

        # Create lightweight dummy parameters for weight loading
        # These are minimal placeholders to satisfy the weight loader interface
        layer.w13_weight = torch.nn.Parameter(
            torch.empty(1, device="cpu", dtype=params_dtype), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.empty(1, device="cpu", dtype=params_dtype), requires_grad=False
        )

        # Register weight loader for on-the-fly quantization
        layer.w13_weight.weight_loader = (
            lambda param, loaded_weight, weight_name, shard_id, expert_id: self._stream_quant_weight(
                layer=layer,
                loaded_weight=loaded_weight,
                shard_id=shard_id,
                expert_id=int(expert_id),
            )
        )
        layer.w2_weight.weight_loader = layer.w13_weight.weight_loader

    def _stream_quant_weight(
        self,
        layer: torch.nn.Module,
        loaded_weight: torch.Tensor,
        shard_id: str,
        expert_id: int,
    ) -> None:
        """
        Quantize and store expert weight in CPU memory.

        This method is called during weight loading to quantize each expert
        weight and store it in the CPU-side MoE inference engine.

        Args:
            layer: The MoE layer
            loaded_weight: The loaded weight tensor (typically on CPU)
            shard_id: Which shard this weight belongs to ("w1", "w3", or "w2")
            expert_id: The expert index
        """
        assert self.moe_infer_handle is not None
        assert self.hidden_size is not None

        # Ensure weight is on CPU
        w = loaded_weight.detach()
        if w.device.type != "cpu":
            w = w.cpu()

        # Map shard_id to projection name
        proj_map = {
            "w1": "gate_proj",
            "w3": "up_proj",
            "w2": "down_proj",
        }

        if shard_id not in proj_map:
            raise ValueError(f"Unsupported shard_id={shard_id}")

        proj_name = proj_map[shard_id]

        # Quantize and store in CPU memory
        self.moe_infer_handle.quantize_and_store_expert(expert_id, proj_name, w)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Post-processing after all weights are loaded.

        For the offload implementation, weights are already quantized and stored
        during loading, so minimal post-processing is needed.
        """
        if self.offload_config is None:
            return

        # Weights are already quantized and stored during loading
        # Optionally clean up any temporary allocations
        try:
            import torch_npu

            torch_npu.npu.empty_cache()
        except Exception:
            pass

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        """
        Create MoE runner for the offload method.

        For CPU offload, computation is handled directly in apply() method
        using nanovllm_ext, so no additional runner is needed.
        """
        # No runner needed for CPU offload - computation is done in apply()
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> StandardCombineInput:
        """
        Apply MoE computation with offload to CPU.

        This method uses the CPU offload implementation for MoE computation.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results wrapped in StandardCombineInput
        """
        # Check if offload is enabled for this layer
        if (
            self.offload_config is None
            or not self.offload_config.enabled
            or self.moe_infer_handle is None
        ):
            raise RuntimeError(
                f"MoE offload is not enabled for layer {self.layer_idx}. "
                "Please check your configuration."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        # Setup callback manager for async NPU-CPU transfers
        import torch_npu

        stream_ptr = int(torch_npu.npu.current_stream().npu_stream)
        _get_or_create_global_callback_manager(stream_ptr)

        num_tokens = int(x.shape[0])
        top_k = int(topk_ids.shape[1])

        # Normalize dtypes
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        # Check if we're in graph capture mode
        is_capture_mode = get_is_capture_mode()

        if not is_capture_mode:
            # Eager mode: use stream-based implementation
            out = torch.ops.nanovllm.moe_forward_npu_stream(
                x,
                topk_ids,
                topk_weights,
                self.moe_infer_handle,
            )
            return StandardCombineInput(hidden_states=out)

        # Graph mode: use pre-allocated buffers
        dtype_int = 1 if x.dtype == torch.bfloat16 else 0  # 0: fp16, 1: bf16
        ctx_key = (num_tokens, top_k, dtype_int)
        ctx = self.graph_contexts.get(ctx_key)

        if ctx is None:
            ctx = torch.classes.nanovllm.MoEGraphContext(
                self.moe_infer_handle,
                num_tokens,
                top_k,
                dtype_int,
            )
            self.graph_contexts[ctx_key] = ctx
            logger.debug(
                f"[MoE Offload] Created graph context for "
                f"tokens={num_tokens}, top_k={top_k}, dtype_int={dtype_int}"
            )

        out = torch.empty_like(x)
        torch.ops.nanovllm.moe_forward_npu_graph_out(
            x, topk_ids, topk_weights, self.moe_infer_handle, ctx, out
        )
        return StandardCombineInput(hidden_states=out)


class MoEOffloadInt4FusedMoEMethod(FusedMoEMethodBase):
    """
    MoE computation offload method using Int4 pre-quantized weights on CPU.

    This method provides computation offload to CPU for models with pre-quantized
    int4 weights in compressed-tensors format.

    Features:
    - Supports pre-quantized int4 weights from compressed-tensors format
    - Group size fixed to 32 for symmetric quantization
    - Weights stored as uint32 (packed 4-bit values)
    - Scales in FP16 or BF16 format
    - NUMA-aware memory placement
    - Support for both eager and NPU graph modes
    - Asynchronous NPU-CPU data transfers

    Note: This class requires nanovllm_ext to be installed; importing this
    module fails fast if it is not (see module docstring).
    """

    def __init__(self, group_size: int = 32):
        super().__init__()

        self.offload_config: Optional[MoEOffloadConfig] = None
        self.moe_infer_handle: Optional[Any] = None

        # Graph context cache: key=(num_tokens, top_k, dtype_int)
        self.graph_contexts: Dict[Tuple[int, int, int], Any] = {}

        # Metadata
        self.num_experts: Optional[int] = None
        self.hidden_size: Optional[int] = None
        self.intermediate_size: Optional[int] = None
        self.params_dtype: Optional[torch.dtype] = None
        self.layer_idx: int = -1
        self.group_size = group_size

        # Int4 quantization parameters
        self.num_bits = 4
        self.packed_factor = 32 // self.num_bits  # 8

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Create weight parameters for pre-quantized int4 weights.

        This implementation creates parameters that match the compressed-tensors
        format for int4 quantization with group size 32.
        """
        tp_rank = get_tensor_model_parallel_rank()
        if tp_rank != 0:
            # Only rank 0 handles weight storage for now
            logger.debug(f"[MoE Offload] Rank {tp_rank} skipping weight storage")
            return

        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size_per_partition)
        self.params_dtype = params_dtype

        # Create CPU MoE inference handle with Int4 quantization (type=1)
        if self.moe_infer_handle is None:
            self.moe_infer_handle = torch.classes.nanovllm.MoEInfer(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
                1,  # Int4 Quantization (Q4_0)
            )
            logger.info(
                f"[MoE Offload] Created MoEInfer handle for layer {self.layer_idx}: "
                f"experts={self.num_experts}, hidden_size={self.hidden_size}, "
                f"intermediate_size={self.intermediate_size}, quant_type=Q4_0, "
                f"group_size={self.group_size}"
            )

        # Create weight parameters matching compressed-tensors format
        # Weights are packed as uint32 (8 x 4-bit values per uint32)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.packed_factor,
                dtype=torch.int32,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.packed_factor,
                dtype=torch.int32,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)

        # Calculate number of groups for scales
        num_groups_w13 = hidden_size // self.group_size
        num_groups_w2 = intermediate_size_per_partition // self.group_size

        # Scales: [num_experts, output_dim, num_groups]
        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                num_groups_w13,
                dtype=params_dtype,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)

        w2_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                num_groups_w2,
                dtype=params_dtype,
                device="cpu",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)

        # Weight shape metadata for validation (following vllm-ascend pattern)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.int32, device="cpu"),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_shape", w13_weight_shape)

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2, dtype=torch.int32, device="cpu"),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)

        # Set weight loader and quant_method attributes for all parameters
        for param in [
            w13_weight, w2_weight, w13_scale, w2_scale,
            w13_weight_shape, w2_weight_shape
        ]:
            if "weight_loader" in extra_weight_attrs:
                param.weight_loader = extra_weight_attrs["weight_loader"]
            # Set quant_method attribute for group quantization
            param.quant_method = "group"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Post-processing after all weights are loaded.

        This method repacks the pre-quantized int4 weights and stores them
        in the CPU-side MoE inference engine using store_quantized_repack.
        """
        if self.offload_config is None or self.moe_infer_handle is None:
            return

        tp_rank = get_tensor_model_parallel_rank()
        if tp_rank != 0:
            return

        # Repack and store pre-quantized weights
        self.moe_infer_handle.store_quantized_repack(
            layer.w13_weight_packed.data.cpu(),
            layer.w13_weight_scale.data.cpu(),
            layer.w2_weight_packed.data.cpu(),
            layer.w2_weight_scale.data.cpu(),
        )

        logger.info(
            f"[MoE Offload] Layer {self.layer_idx}: Stored pre-quantized int4 weights "
            f"in CPU memory (experts={self.num_experts})"
        )

        # Clean up the layer parameters to free memory
        del layer.w13_weight_packed
        del layer.w2_weight_packed
        del layer.w13_weight_scale
        del layer.w2_weight_scale
        del layer.w13_weight_shape
        del layer.w2_weight_shape

        # Optionally clean up cache
        try:
            import torch_npu

            torch_npu.npu.empty_cache()
        except Exception:
            pass

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        """
        Create MoE runner for the offload method.

        For CPU offload, computation is handled directly in apply() method
        using nanovllm_ext, so no additional runner is needed.
        """
        # No runner needed for CPU offload - computation is done in apply()
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> StandardCombineInput:
        """
        Apply MoE computation with offload to CPU using pre-quantized int4 weights.

        This method uses the CPU offload implementation for MoE computation.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results wrapped in StandardCombineInput
        """
        # Check if offload is enabled for this layer
        if (
            self.offload_config is None
            or not self.offload_config.enabled
            or self.moe_infer_handle is None
        ):
            raise RuntimeError(
                f"MoE offload is not enabled for layer {self.layer_idx}. "
                "Please check your configuration."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        # Setup callback manager for async NPU-CPU transfers
        import torch_npu

        stream_ptr = int(torch_npu.npu.current_stream().npu_stream)
        _get_or_create_global_callback_manager(stream_ptr)

        num_tokens = int(x.shape[0])
        top_k = int(topk_ids.shape[1])

        # Normalize dtypes
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        # Check if we're in graph capture mode
        is_capture_mode = get_is_capture_mode()

        if not is_capture_mode:
            # Eager mode: use stream-based implementation
            out = torch.ops.nanovllm.moe_forward_npu_stream(
                x,
                topk_ids,
                topk_weights,
                self.moe_infer_handle,
            )
            return StandardCombineInput(hidden_states=out)

        # Graph mode: use pre-allocated buffers
        dtype_int = 1 if x.dtype == torch.bfloat16 else 0  # 0: fp16, 1: bf16
        ctx_key = (num_tokens, top_k, dtype_int)
        ctx = self.graph_contexts.get(ctx_key)

        if ctx is None:
            ctx = torch.classes.nanovllm.MoEGraphContext(
                self.moe_infer_handle,
                num_tokens,
                top_k,
                dtype_int,
            )
            self.graph_contexts[ctx_key] = ctx
            logger.debug(
                f"[MoE Offload] Created graph context for "
                f"tokens={num_tokens}, top_k={top_k}, dtype_int={dtype_int}"
            )

        out = torch.empty_like(x)
        torch.ops.nanovllm.moe_forward_npu_graph_out(
            x, topk_ids, topk_weights, self.moe_infer_handle, ctx, out
        )
        return StandardCombineInput(hidden_states=out)
