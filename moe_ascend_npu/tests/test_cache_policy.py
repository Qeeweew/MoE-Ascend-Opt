"""CPU-only tests for LFU cache ownership decisions."""

import argparse
import inspect
import re

import torch

from moe_ascend_npu.cache import ExpertCacheConfig, ExpertCacheManager
from moe_ascend_npu.patches.server_args import _add_moe_offload_args


def test_default_small_cache_matches_measured_k256_config():
    config = ExpertCacheConfig()
    assert config.size == 256
    assert config.swap_per_update == 8
    assert config.update_interval == 16
    assert config.warmup_steps == 16


def test_cli_defaults_match_measured_k256_config():
    parser = argparse.ArgumentParser()
    _add_moe_offload_args(parser)
    args = parser.parse_args([])
    config = ExpertCacheConfig()

    assert args.moe_expert_cache_size == config.size
    assert args.moe_expert_cache_swap_per_update == config.swap_per_update
    assert args.moe_expert_cache_update_interval == config.update_interval
    assert args.moe_expert_cache_warmup_steps == config.warmup_steps
    assert args.moe_expert_cache_decay == config.decay

    source = inspect.getsource(_add_moe_offload_args.__globals__["apply"])
    assert 'getattr(args, "moe_expert_cache_size", 256)' in source
    assert re.search(
        r'getattr\(\s*args,\s*"moe_expert_cache_update_interval",\s*16\s*\)',
        source,
    )


def test_fill_then_replace_with_hysteresis():
    manager = ExpertCacheManager(
        ExpertCacheConfig(size=2, swap_per_update=1, update_interval=1, warmup_steps=0)
    )
    manager.freq = {0: torch.tensor([10.0, 8.0, 1.0], dtype=torch.float64)}
    manager.slot_table = torch.full((1, 3), -1, dtype=torch.int32)
    manager.slot_owner = [None, None, None]
    manager._load_owner_into_slot = lambda owner, slot: None

    manager._rebalance()
    manager._rebalance()
    assert set(manager.owner_slot) == {(0, 0), (0, 1)}

    manager.freq[0] = torch.tensor([10.0, 8.0, 20.0], dtype=torch.float64)
    manager._rebalance()
    assert (0, 2) in manager.owner_slot
    assert (0, 1) not in manager.owner_slot
    assert int(manager.slot_table[0, 2]) >= 0
    assert int(manager.slot_table[0, 1]) == -1


def test_config_validation():
    for config in (
        ExpertCacheConfig(size=-1),
        ExpertCacheConfig(swap_per_update=0),
        ExpertCacheConfig(update_interval=0),
        ExpertCacheConfig(decay=1.0),
    ):
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {config}")


def test_large_fill_batch_is_capped_after_full():
    manager = ExpertCacheManager(
        ExpertCacheConfig(size=2, swap_per_update=64, update_interval=1, warmup_steps=0)
    )
    manager.freq = {0: torch.arange(20, dtype=torch.float64)}
    manager.slot_table = torch.full((1, 20), -1, dtype=torch.int32)
    manager.slot_owner = [None] * 66
    manager._load_owner_into_slot = lambda owner, slot: None
    manager._rebalance()
    assert len(manager.owner_slot) == 2

    before = manager.total_swaps
    manager.freq[0].add_(100)
    manager._rebalance()
    assert manager.total_swaps - before <= 8


def test_global_lfu_competes_across_layers():
    manager = ExpertCacheManager(
        ExpertCacheConfig(
            size=3, swap_per_update=3, update_interval=1, warmup_steps=0,
        )
    )
    manager.freq = {
        0: torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64),
        1: torch.tensor([9.0, 1.0, 1.0], dtype=torch.float64),
    }
    manager.slot_table = torch.full((2, 3), -1, dtype=torch.int32)
    manager.slot_owner = [None] * 6
    manager._load_owner_into_slot = lambda owner, slot: None
    manager._rebalance()
    assert set(manager.owner_slot) == {(1, 0), (0, 2), (0, 1)}


def test_steady_interval_backoff_and_reset():
    manager = ExpertCacheManager(
        ExpertCacheConfig(size=2, swap_per_update=1, update_interval=16, warmup_steps=0)
    )
    manager.owner_slot = {(0, 0): 0, (0, 1): 1}

    assert manager._steady_interval_multiplier == 8
    manager._update_interval_backoff(hit_rate=0.25, swaps=0)
    assert manager._steady_interval_multiplier == 8
    manager._update_interval_backoff(hit_rate=0.26, swaps=0)
    assert manager._steady_interval_multiplier == 16

    manager._update_interval_backoff(hit_rate=0.27, swaps=1)
    assert manager._steady_interval_multiplier == 8

    manager._update_interval_backoff(hit_rate=0.30, swaps=0)
    manager._update_interval_backoff(hit_rate=0.31, swaps=0)
    assert manager._steady_interval_multiplier == 16

    manager._update_interval_backoff(hit_rate=0.20, swaps=0)
    assert manager._steady_interval_multiplier == 8


if __name__ == "__main__":
    test_fill_then_replace_with_hysteresis()
    test_config_validation()
    test_large_fill_batch_is_capped_after_full()
    test_global_lfu_competes_across_layers()
    test_steady_interval_backoff_and_reset()
    print("Expert cache policy tests passed.")
