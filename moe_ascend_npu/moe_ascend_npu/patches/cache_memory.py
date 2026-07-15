"""Reserve fixed expert-cache HBM before SGLang sizes the KV cache.

The cache tensors must exist before ``ModelRunner.init_memory_pool`` measures
available HBM.  Deferring allocation to the first MoE forward lets SGLang give
that capacity to the KV pool, then fails during graph capture when the cache is
first touched.  The cache is also populated here with a deterministic uniform
per-layer seed, so requests never pay the cost of filling an empty cache.
"""

from sglang.srt.utils import logger


def apply():
    """Patch the upstream memory-pool boundary without modifying SGLang."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    original_init_memory_pool = ModelRunner.init_memory_pool

    def init_memory_pool(self, *args, **kwargs):
        if getattr(self.server_args, "enable_moe_expert_cache", False):
            from moe_ascend_npu.cache import get_expert_cache_manager

            manager = get_expert_cache_manager()
            if manager.enabled:
                # All MoE layers have completed process_weights_after_loading
                # when ModelRunner reaches init_memory_pool.  Allocate every
                # graph-address-stable tensor now, before SGLang samples free
                # HBM and commits the KV pool size.
                manager.ensure_allocated()
                manager.initialize_uniform()
                logger.info(
                    "[ExpertCache] fixed HBM reservation and uniform seed completed "
                    "before KV Cache allocation"
                )
        return original_init_memory_pool(self, *args, **kwargs)

    ModelRunner.init_memory_pool = init_memory_pool
    logger.info("ModelRunner.init_memory_pool: installed expert-cache HBM reservation")
