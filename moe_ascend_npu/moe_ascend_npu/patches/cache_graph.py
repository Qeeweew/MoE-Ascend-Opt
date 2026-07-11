"""Drive expert-cache maintenance immediately before NPU graph replay."""

from sglang.srt.utils import logger


def apply():
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

    def drive_cache(runner, forward_batch):
        server_args = runner.model_runner.server_args
        if getattr(server_args, "enable_moe_expert_cache", False):
            from moe_ascend_npu.cache import get_expert_cache_manager

            valid_tokens = int(forward_batch.batch_size) * int(runner.num_tokens_per_bs)
            get_expert_cache_manager().before_replay(valid_tokens)

    original_replay = CudaGraphRunner.replay

    def replay(self, forward_batch, *args, **kwargs):
        drive_cache(self, forward_batch)
        return original_replay(self, forward_batch, *args, **kwargs)

    CudaGraphRunner.replay = replay

    # Ascend overrides replay rather than delegating to CudaGraphRunner, so it
    # needs its own wrapper.  This import is still an upstream read-only class;
    # only the method object is monkey patched in this package.
    from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
        NPUGraphRunner,
    )

    original_npu_replay = NPUGraphRunner.replay

    def npu_replay(self, forward_batch, *args, **kwargs):
        drive_cache(self, forward_batch)
        return original_npu_replay(self, forward_batch, *args, **kwargs)

    NPUGraphRunner.replay = npu_replay
    logger.info("CUDA/NPU graph replay: installed expert-cache step hooks")
