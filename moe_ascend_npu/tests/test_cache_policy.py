"""CPU-only tests for LFU cache ownership decisions."""

import torch

from moe_ascend_npu.cache import ExpertCacheConfig, ExpertCacheManager


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


if __name__ == "__main__":
    test_fill_then_replace_with_hysteresis()
    test_config_validation()
    test_large_fill_batch_is_capped_after_full()
    test_global_lfu_competes_across_layers()
    print("Expert cache policy tests passed.")
