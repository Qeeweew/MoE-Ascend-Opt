"""Graph-outside expert cache shared by all MoE layers in one NPU process."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    # Use SGLang's configured worker logger when running in the server.  The
    # standard-library root logger is not configured in scheduler subprocesses.
    from sglang.srt.utils import logger
except ImportError:  # CPU-only policy tests do not require SGLang.
    import logging

    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpertCacheConfig:
    # Cache a small hot working set across all layers.  K=128 is the smallest
    # verified default for Qwen3-30B-A3B: K=64 is too sparse to beat CPU-only
    # offload on the measured decode workload.
    size: int = 128
    swap_per_update: int = 8
    update_interval: int = 32
    warmup_steps: int = 16
    decay: float = 0.95

    def validate(self) -> None:
        if self.size < 0:
            raise ValueError("moe expert cache size must be >= 0")
        if self.swap_per_update <= 0:
            raise ValueError("moe expert cache swap-per-update must be > 0")
        if self.update_interval <= 0 or self.warmup_steps < 0:
            raise ValueError("cache update interval must be > 0 and warmup must be >= 0")
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("cache decay must be in [0, 1)")


@dataclass
class _LayerSource:
    handle: object
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


class ExpertCacheManager:
    """Own fixed cache tensors, routing EMA, and graph-boundary replacement."""

    _BASE_STEADY_INTERVAL_MULTIPLIER = 8
    _MAX_STEADY_INTERVAL_MULTIPLIER = 32
    _STABLE_WINDOWS_TO_GROW = 2
    _HIT_RATE_DROP_RESET = 0.05

    def __init__(self, config: ExpertCacheConfig):
        config.validate()
        self.config = config
        self.sources: Dict[int, _LayerSource] = {}
        self.slot_table: Optional[torch.Tensor] = None
        self.w13_cache: Optional[torch.Tensor] = None
        self.s13_cache: Optional[torch.Tensor] = None
        self.w2_cache: Optional[torch.Tensor] = None
        self.s2_cache: Optional[torch.Tensor] = None
        self.freq: Dict[int, torch.Tensor] = {}
        self.slot_owner: list[Optional[Tuple[int, int]]] = []
        self.owner_slot: Dict[Tuple[int, int], int] = {}
        self.cooldown: Dict[Tuple[int, int], int] = {}
        self.replay_steps = 0
        self.started = False
        self.total_routes = 0
        self.total_misses = 0
        self.total_swaps = 0
        self._steady_interval_multiplier = self._BASE_STEADY_INTERVAL_MULTIPLIER
        self._stable_no_swap_windows = 0
        self._last_window_hit_rate = 0.0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.config.size > 0

    def register_layer(
        self,
        layer_idx: int,
        handle,
        w13: torch.Tensor,
        s13: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
    ) -> None:
        with self._lock:
            if self.sources:
                first = self.sources[min(self.sources)]
                shapes = (w13.shape[1:], w2.shape[1:])
                expected = (first.w13.shape[1:], first.w2.shape[1:])
                if shapes != expected:
                    raise ValueError(
                        "all layers sharing the expert cache must have identical expert shapes: "
                        f"layer {layer_idx} has {shapes}, expected {expected}"
                    )
            self.sources[layer_idx] = _LayerSource(
                handle=handle,
                w13=w13,
                s13=s13,
                w2=w2,
                s2=s2,
            )
            self.freq[layer_idx] = torch.zeros(
                w13.shape[0], dtype=torch.float64
            )

    def ensure_allocated(self) -> None:
        if self.slot_table is not None:
            return
        with self._lock:
            if self.slot_table is not None:
                return
            if not self.sources:
                raise RuntimeError("expert cache has no registered MoE layers")
            first = self.sources[min(self.sources)]
            num_experts = int(first.w13.shape[0])
            num_layers = max(self.sources) + 1
            physical = self.config.size + self.config.swap_per_update
            device = torch.device("npu", torch.npu.current_device())

            # compressed-tensors source [E,N,K/8] -> kernel [slot,K,N/8]
            self.w13_cache = torch.empty(
                physical, first.w13.shape[2] * 8, first.w13.shape[1] // 8,
                dtype=torch.int32, device=device,
            )
            self.s13_cache = torch.empty(
                physical, first.s13.shape[2], first.s13.shape[1],
                dtype=first.s13.dtype, device=device,
            )
            self.w2_cache = torch.empty(
                physical, first.w2.shape[2] * 8, first.w2.shape[1] // 8,
                dtype=torch.int32, device=device,
            )
            self.s2_cache = torch.empty(
                physical, first.s2.shape[2], first.s2.shape[1],
                dtype=first.s2.dtype, device=device,
            )
            self.slot_table = torch.full(
                (num_layers, num_experts), -1, dtype=torch.int32, device=device
            )
            self.slot_owner = [None] * physical
            slot_bytes = (
                self.w13_cache[0].nbytes + self.s13_cache[0].nbytes
                + self.w2_cache[0].nbytes + self.s2_cache[0].nbytes
            )
            logger.info(
                "[ExpertCache] allocated active=%d spare=%d slot=%.2f MiB total=%.2f GiB",
                self.config.size, self.config.swap_per_update,
                slot_bytes / 2**20, physical * slot_bytes / 2**30,
            )

    def reset_capture_stats(self) -> None:
        for source in self.sources.values():
            source.handle.reset_routing_stats()

    def before_replay(self, valid_tokens: int) -> None:
        """Called outside the captured graph immediately before each replay."""
        if not self.enabled:
            return
        self.ensure_allocated()
        with self._lock:
            if not self.started:
                self.reset_capture_stats()
                self.started = True
            for source in self.sources.values():
                source.handle.set_valid_tokens(int(valid_tokens))
            self.replay_steps += 1
            if self.replay_steps <= self.config.warmup_steps:
                return
            # Bootstrap quickly, then make replacement a genuinely low-rate
            # control-plane event.  The steady-state multiplier avoids paying
            # a stream synchronization every few decode tokens.
            interval = self.config.update_interval
            if len(self.owner_slot) >= self.config.size:
                interval *= self._steady_interval_multiplier
            if (self.replay_steps - self.config.warmup_steps) % interval:
                return

            # Routing counters are protected inside MoEInfer.  Taking a
            # snapshot concurrently with a callback is safe: a callback that
            # finishes after this snapshot is simply counted in the next
            # window.  Do not globally synchronize here; on decode that pause
            # is substantially more expensive than the small-cache compute it
            # is intended to save.  Cache copies/table publication below are
            # enqueued on the main stream before the next replay and therefore
            # remain stream ordered.
            hit_rate = self._collect_stats()
            swaps = self._rebalance()
            self._update_interval_backoff(hit_rate, swaps)

    def _collect_stats(self) -> float:
        window_total = window_miss = 0
        for layer_idx, source in self.sources.items():
            stats = source.handle.take_routing_stats()
            values = stats.tolist()
            counts = torch.tensor(values[:-3], dtype=torch.float64)
            self.freq[layer_idx].mul_(self.config.decay).add_(counts)
            window_total += int(values[-3])
            window_miss += int(values[-2])
        self.total_routes += window_total
        self.total_misses += window_miss
        hit_rate = 0.0 if window_total == 0 else 1.0 - window_miss / window_total
        logger.info(
            "[ExpertCache] step=%d active=%d window_hit=%.2f%% swaps=%d",
            self.replay_steps, len(self.owner_slot), hit_rate * 100, self.total_swaps,
        )
        return hit_rate

    def _rebalance(self) -> int:
        for owner in list(self.cooldown):
            self.cooldown[owner] -= 1
            if self.cooldown[owner] <= 0:
                del self.cooldown[owner]

        candidates = sorted(
            ((float(scores[e]), (layer, e))
             for layer, scores in self.freq.items()
             for e in range(scores.numel())
             if (layer, e) not in self.owner_slot and scores[e] > 0),
            reverse=True,
        )
        swaps = 0
        # A large configured batch is useful for bootstrapping an empty cache,
        # but allowing the same volume after it is full causes route-window
        # noise to trigger expensive churn.  Steady state is deliberately
        # capped at eight replacements per update.
        if len(self.owner_slot) < self.config.size:
            # Filling and eviction are separate update phases.  Do not begin
            # evicting entries that were inserted earlier in this same call.
            swap_limit = min(
                self.config.swap_per_update,
                self.config.size - len(self.owner_slot),
            )
        else:
            swap_limit = min(self.config.swap_per_update, 8)
        for score, owner in candidates:
            if swaps >= swap_limit:
                break
            if len(self.owner_slot) < self.config.size:
                slot = self._find_spare_slot()
                victim = None
            else:
                victim_score, victim = self._coldest_owner()
                if victim is None or owner in self.cooldown:
                    break
                if score <= victim_score * 1.10:
                    break
                slot = self._find_spare_slot()
            self._load_owner_into_slot(owner, slot)
            if victim is not None:
                old_slot = self.owner_slot.pop(victim)
                self.slot_table[victim[0], victim[1]].fill_(-1)
                self.slot_owner[old_slot] = None
                self.cooldown[victim] = 2
            self.owner_slot[owner] = slot
            self.slot_owner[slot] = owner
            self.slot_table[owner[0], owner[1]].fill_(slot)
            swaps += 1
            self.total_swaps += 1
        return swaps

    def _update_interval_backoff(self, hit_rate: float, swaps: int) -> None:
        if len(self.owner_slot) < self.config.size:
            self._steady_interval_multiplier = self._BASE_STEADY_INTERVAL_MULTIPLIER
            self._stable_no_swap_windows = 0
            self._last_window_hit_rate = hit_rate
            return

        hit_rate_dropped = (
            hit_rate + self._HIT_RATE_DROP_RESET < self._last_window_hit_rate
        )
        if swaps > 0 or hit_rate_dropped:
            self._steady_interval_multiplier = self._BASE_STEADY_INTERVAL_MULTIPLIER
            self._stable_no_swap_windows = 0
            self._last_window_hit_rate = hit_rate
            return

        self._stable_no_swap_windows += 1
        if (
            self._stable_no_swap_windows >= self._STABLE_WINDOWS_TO_GROW
            and self._steady_interval_multiplier < self._MAX_STEADY_INTERVAL_MULTIPLIER
        ):
            self._steady_interval_multiplier = min(
                self._steady_interval_multiplier * 2,
                self._MAX_STEADY_INTERVAL_MULTIPLIER,
            )
            self._stable_no_swap_windows = 0
            logger.info(
                "[ExpertCache] steady update interval backed off to %d steps",
                self.config.update_interval * self._steady_interval_multiplier,
            )
        self._last_window_hit_rate = hit_rate

    def _find_spare_slot(self) -> int:
        for slot, owner in enumerate(self.slot_owner):
            if owner is None:
                return slot
        raise RuntimeError("expert cache has no spare slot")

    def _coldest_owner(self):
        eligible = [o for o in self.owner_slot if o not in self.cooldown]
        if not eligible:
            return 0.0, None
        victim = min(eligible, key=lambda o: float(self.freq[o[0]][o[1]]))
        return float(self.freq[victim[0]][victim[1]]), victim

    def _load_owner_into_slot(self, owner: Tuple[int, int], slot: int) -> None:
        from moe_ascend_npu.patches.fused_moe_method import _transpose_and_repack_int4

        layer_idx, expert_idx = owner
        src = self.sources[layer_idx]
        w13 = _transpose_and_repack_int4(src.w13[expert_idx:expert_idx + 1].npu())
        w2 = _transpose_and_repack_int4(src.w2[expert_idx:expert_idx + 1].npu())
        self.w13_cache[slot].copy_(w13[0])
        self.w2_cache[slot].copy_(w2[0])
        self.s13_cache[slot].copy_(src.s13[expert_idx].npu().transpose(-1, -2).contiguous())
        self.s2_cache[slot].copy_(src.s2[expert_idx].npu().transpose(-1, -2).contiguous())

    def metrics(self) -> dict:
        hit_rate = 0.0 if self.total_routes == 0 else 1.0 - self.total_misses / self.total_routes
        return {
            "steps": self.replay_steps,
            "active_slots": len(self.owner_slot),
            "swaps": self.total_swaps,
            "routes": self.total_routes,
            "hit_rate": hit_rate,
        }


_manager: Optional[ExpertCacheManager] = None


def get_expert_cache_manager(config: Optional[ExpertCacheConfig] = None) -> ExpertCacheManager:
    global _manager
    if _manager is None:
        if config is None:
            raise RuntimeError("expert cache manager has not been configured")
        _manager = ExpertCacheManager(config)
    elif config is not None and _manager.config != config:
        raise RuntimeError("inconsistent expert cache configuration in one process")
    return _manager
