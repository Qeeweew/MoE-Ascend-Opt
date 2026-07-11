"""Patch FusedMoE.__init__ to route selected layers to CPU offload.

When ``--enable-moe-offload`` is set and ``layer_id >= moe_offload_start_layer``,
the layer's ``quant_method`` is replaced with ``MoEOffloadFusedMoEMethod`` (or
the Int4 variant) *before* ``create_weights`` runs, so the offload method owns
weight creation. This mirrors the fork's in-tree branch but is done by wrapping
``__init__``.

The official ``__init__`` chooses ``self.quant_method`` via
``quant_config.get_quant_method(self, prefix)`` (when ``quant_config`` is not
None). We temporarily redirect that call to return our offload method for the
duration of ``__init__``. For the ``quant_config is None`` (unquantized model)
case we inject a minimal shim so the same code path is taken, then restore
``self.quant_config`` to None afterwards.

``moe_ascend_npu.patches.offload`` hard-imports ``nanovllm_ext``. It is imported
lazily *inside* ``_init`` and only when offload is actually enabled for this
layer, so normal NPU serving (no offload) does not require ``nanovllm_ext``;
enabling offload without it raises ``ImportError`` directly at layer
construction.
"""

import logging

logger = logging.getLogger(__name__)


class _OffloadQuantShim:
    """Minimal stand-in quant_config whose get_quant_method returns the offload method."""

    def __init__(self, method):
        self._method = method

    def get_quant_method(self, layer, prefix, *args, **kwargs):
        return self._method

    def get_name(self):
        return "moe_offload"


def apply():
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.server_args import get_global_server_args

    orig_init = FusedMoE.__init__

    def _init(self, *args, **kwargs):
        layer_id = kwargs.get("layer_id", args[3] if len(args) > 3 else None)
        quant_config = kwargs.get("quant_config", None)

        try:
            server_args = get_global_server_args()
        except Exception:
            server_args = None

        cache_enabled = bool(
            server_args is not None
            and getattr(server_args, "enable_moe_expert_cache", False)
        )

        # Only consider offload/cache when a feature is enabled; otherwise stay on
        # the fast path and never import the nanovllm_ext-dependent module.
        if (
            server_args is None
            or not (getattr(server_args, "enable_moe_offload", False) or cache_enabled)
            or layer_id is None
        ):
            return orig_init(self, *args, **kwargs)

        # Offload enabled: import the offload impl (hard-requires nanovllm_ext).
        from moe_ascend_npu.patches.offload import (
            MoEOffloadConfig,
            MoEOffloadFusedMoEMethod,
            MoEOffloadInt4FusedMoEMethod,
            _is_compressed_tensors_int4,
            create_moe_offload_config,
        )

        if cache_enabled:
            from moe_ascend_npu.cache import ExpertCacheConfig
            from moe_ascend_npu.patches.expert_cache_method import ExpertCacheFusedMoEMethod

            if not _is_compressed_tensors_int4(quant_config):
                raise ValueError(
                    "--enable-moe-expert-cache currently requires a symmetric "
                    "compressed-tensors Int4 MoE model"
                )
            cache_cfg = ExpertCacheConfig(
                size=getattr(server_args, "moe_expert_cache_size", 256),
                swap_per_update=getattr(server_args, "moe_expert_cache_swap_per_update", 8),
                update_interval=getattr(server_args, "moe_expert_cache_update_interval", 32),
                warmup_steps=getattr(server_args, "moe_expert_cache_warmup_steps", 16),
                decay=getattr(server_args, "moe_expert_cache_decay", 0.95),
            )
            cache_cfg.validate()
            offload_cfg = MoEOffloadConfig(enabled=True, layer_idx=layer_id, quant_type="q4_0")
            method = ExpertCacheFusedMoEMethod(cache_cfg)
        else:
            offload_cfg = create_moe_offload_config(layer_id, server_args, quant_config)
        if offload_cfg is None:
            return orig_init(self, *args, **kwargs)

        # Build the offload method and make the official __init__ select it.
        if not cache_enabled:
            if offload_cfg.quant_type == "q4_0":
                method = MoEOffloadInt4FusedMoEMethod()
            else:
                method = MoEOffloadFusedMoEMethod()
        method.offload_config = offload_cfg
        method.layer_idx = layer_id

        if quant_config is not None:
            orig_gqm = quant_config.get_quant_method
            quant_config.get_quant_method = (
                lambda layer, prefix, *a, **k: method
            )
            try:
                return orig_init(self, *args, **kwargs)
            finally:
                quant_config.get_quant_method = orig_gqm
        else:
            shim = _OffloadQuantShim(method)
            kwargs["quant_config"] = shim
            result = orig_init(self, *args, **kwargs)
            # Restore: downstream code expects quant_config to be None here.
            self.quant_config = None
            return result

    FusedMoE.__init__ = _init
    logger.info("FusedMoE.__init__: wired MoE CPU offload interception")
