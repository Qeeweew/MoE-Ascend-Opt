"""Patch ServerArgs to add MoE-offload CLI arguments.

Adds ``--enable-moe-offload``, ``--moe-offload-start-layer`` and
``--moe-offload-quant-type``. Because ``ServerArgs`` is a dataclass and its
``from_cli_args`` only forwards dataclass fields to ``__init__``, the new flags
are attached as plain instance attributes after construction (no dataclass
modification needed).
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def _add_moe_offload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-moe-expert-cache", action="store_true",
        help="Enable graph-compatible dynamic Int4 expert caching with CPU fallback.",
    )
    parser.add_argument("--moe-expert-cache-size", type=int, default=64)
    parser.add_argument(
        "--moe-expert-cache-swap-per-update", type=int, default=8,
        help="Maximum bootstrap fills per update; steady replacement is capped at 8.",
    )
    parser.add_argument("--moe-expert-cache-update-interval", type=int, default=32)
    parser.add_argument("--moe-expert-cache-warmup-steps", type=int, default=16)
    parser.add_argument("--moe-expert-cache-decay", type=float, default=0.95)
    parser.add_argument(
        "--enable-moe-offload",
        action="store_true",
        help="Enable MoE computation offload to CPU using Int8/Int4 quantization. "
        "Requires nanovllm_ext to be installed.",
    )
    parser.add_argument(
        "--moe-offload-start-layer",
        type=int,
        default=0,
        help="Layer index from which to start offloading MoE computation to CPU. "
        "Only layers with index >= this value are offloaded.",
    )
    parser.add_argument(
        "--moe-offload-quant-type",
        type=str,
        default="q8_0",
        choices=["q8_0", "q4_0"],
        help='Quantization type for MoE offload. "q8_0" = online Int8; '
        '"q4_0" = pre-quantized Int4 (compressed-tensors format).',
    )


def apply():
    from sglang.srt.server_args import ServerArgs

    orig_add_cli_args = ServerArgs.add_cli_args  # staticmethod

    def _add_cli_args(parser, *args, **kwargs):
        orig_add_cli_args(parser, *args, **kwargs)
        _add_moe_offload_args(parser)

    ServerArgs.add_cli_args = staticmethod(_add_cli_args)

    orig_from_cli_args = ServerArgs.from_cli_args  # classmethod (bound)

    def _from_cli_args(cls, args):
        server_args = orig_from_cli_args(args)
        server_args.enable_moe_offload = getattr(args, "enable_moe_offload", False)
        server_args.moe_offload_start_layer = getattr(
            args, "moe_offload_start_layer", 0
        )
        server_args.moe_offload_quant_type = getattr(
            args, "moe_offload_quant_type", "q8_0"
        )
        server_args.enable_moe_expert_cache = getattr(args, "enable_moe_expert_cache", False)
        server_args.moe_expert_cache_size = getattr(args, "moe_expert_cache_size", 64)
        server_args.moe_expert_cache_swap_per_update = getattr(
            args, "moe_expert_cache_swap_per_update", 8
        )
        server_args.moe_expert_cache_update_interval = getattr(
            args, "moe_expert_cache_update_interval", 32
        )
        server_args.moe_expert_cache_warmup_steps = getattr(
            args, "moe_expert_cache_warmup_steps", 16
        )
        server_args.moe_expert_cache_decay = getattr(args, "moe_expert_cache_decay", 0.95)
        if server_args.enable_moe_expert_cache and server_args.enable_moe_offload:
            raise ValueError(
                "--enable-moe-expert-cache and --enable-moe-offload are mutually exclusive"
            )
        return server_args

    ServerArgs.from_cli_args = classmethod(_from_cli_args)
    logger.info("server_args: added --enable-moe-offload CLI args")
