"""Graph-outside expert cache shared by all MoE layers in one NPU process."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
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
    # Cache a small hot working set across all layers.  K=256 keeps 4.17% of
    # Qwen3 expert instances and is the current experimental default; matched
    # ShareGPT tests show that this capacity does not yet beat CPU-only Q4.
    size: int = 256
    swap_per_update: int = 8
    update_interval: int = 16
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
    num_experts: int
    hidden_size: int
    intermediate_size: int
    scale_dtype: torch.dtype


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
        self._staging_buffers: list[Tuple[torch.Tensor, ...]] = []
        self._staging_events: list[Optional[object]] = []
        self._staging_cursor = 0
        self.replay_steps = 0
        self.started = False
        self.total_routes = 0
        self.total_misses = 0
        self.total_swaps = 0
        self._steady_interval_multiplier = self._BASE_STEADY_INTERVAL_MULTIPLIER
        self._stable_no_swap_windows = 0
        self._last_window_hit_rate = 0.0
        self._lock = threading.RLock()
        self._trace_path = os.environ.get("MOE_EXPERT_CACHE_TRACE_PATH")
        self._control_us = deque(maxlen=2048)
        self._collect_us = deque(maxlen=2048)
        self._rebalance_us = deque(maxlen=2048)
        self._export_us = deque(maxlen=2048)
        self._staging_wait_us = deque(maxlen=2048)
        self._copy_submit_us = deque(maxlen=2048)
        self._pending_trace_record: Optional[dict] = None

    @property
    def enabled(self) -> bool:
        return self.config.size > 0

    def register_layer(
        self,
        layer_idx: int,
        handle,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        scale_dtype: torch.dtype,
    ) -> None:
        with self._lock:
            if self.sources:
                first = self.sources[min(self.sources)]
                shape = (hidden_size, intermediate_size, num_experts)
                expected = (
                    first.hidden_size, first.intermediate_size, first.num_experts
                )
                if shape != expected:
                    raise ValueError(
                        "all layers sharing the expert cache must have identical expert shapes: "
                        f"layer {layer_idx} has {shape}, expected {expected}"
                    )
            self.sources[layer_idx] = _LayerSource(
                handle=handle,
                num_experts=int(num_experts),
                hidden_size=int(hidden_size),
                intermediate_size=int(intermediate_size),
                scale_dtype=scale_dtype,
            )
            self.freq[layer_idx] = torch.zeros(
                num_experts, dtype=torch.float64
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
            num_experts = first.num_experts
            num_layers = max(self.sources) + 1
            physical = self.config.size + self.config.swap_per_update
            device = torch.device("npu", torch.npu.current_device())
            hidden = first.hidden_size
            intermediate = first.intermediate_size

            self.w13_cache = torch.empty(
                physical, hidden, 2 * intermediate // 8,
                dtype=torch.int32, device=device,
            )
            self.s13_cache = torch.empty(
                physical, hidden // 32, 2 * intermediate,
                dtype=first.scale_dtype, device=device,
            )
            self.w2_cache = torch.empty(
                physical, intermediate, hidden // 8,
                dtype=torch.int32, device=device,
            )
            self.s2_cache = torch.empty(
                physical, intermediate // 32, hidden,
                dtype=first.scale_dtype, device=device,
            )
            self.slot_table = torch.full(
                (num_layers, num_experts), -1, dtype=torch.int32, device=device
            )
            self.slot_owner = [None] * physical
            # One pinned staging set per maximum replacement in an update.
            # They are reused through the C++ _out API, avoiding allocation on
            # the cache-control hot path while retaining asynchronous H2D.
            for _ in range(self.config.swap_per_update):
                self._staging_buffers.append((
                    torch.empty(hidden, 2 * intermediate // 8, dtype=torch.int32,
                                device="cpu", pin_memory=True),
                    torch.empty(hidden // 32, 2 * intermediate,
                                dtype=first.scale_dtype, device="cpu", pin_memory=True),
                    torch.empty(intermediate, hidden // 8, dtype=torch.int32,
                                device="cpu", pin_memory=True),
                    torch.empty(intermediate // 32, hidden,
                                dtype=first.scale_dtype, device="cpu", pin_memory=True),
                ))
                self._staging_events.append(None)
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
            control_begin = time.perf_counter_ns()
            collect_begin = time.perf_counter_ns()
            hit_rate = self._collect_stats()
            self._collect_us.append((time.perf_counter_ns() - collect_begin) / 1e3)
            export_before = len(self._export_us)
            wait_before = len(self._staging_wait_us)
            copy_before = len(self._copy_submit_us)
            rebalance_begin = time.perf_counter_ns()
            swaps = self._rebalance()
            self._rebalance_us.append((time.perf_counter_ns() - rebalance_begin) / 1e3)
            self._update_interval_backoff(hit_rate, swaps)
            self._control_us.append((time.perf_counter_ns() - control_begin) / 1e3)
            if self._pending_trace_record is not None:
                self._pending_trace_record.update({
                    "swaps": swaps,
                    "control_us": self._control_us[-1],
                    "collect_us": self._collect_us[-1],
                    "rebalance_us": self._rebalance_us[-1],
                    "replacement_timing_us": {
                        "exports": list(self._export_us)[export_before:],
                        "staging_waits": list(self._staging_wait_us)[wait_before:],
                        "copy_submits": list(self._copy_submit_us)[copy_before:],
                    },
                })
                self._append_trace(self._pending_trace_record)
                self._pending_trace_record = None

    def _collect_stats(self) -> float:
        window_total = window_miss = 0
        trace_layers = []
        for layer_idx, source in self.sources.items():
            stats = source.handle.take_routing_stats()
            values = stats.tolist()
            counts = torch.tensor(values[:-3], dtype=torch.float64)
            self.freq[layer_idx].mul_(self.config.decay).add_(counts)
            window_total += int(values[-3])
            window_miss += int(values[-2])
            if self._trace_path:
                trace_layers.append({
                    "layer": layer_idx,
                    "counts": values[:-3],
                    "routes": int(values[-3]),
                    "misses": int(values[-2]),
                    "calls": int(values[-1]),
                })
        self.total_routes += window_total
        self.total_misses += window_miss
        hit_rate = 0.0 if window_total == 0 else 1.0 - window_miss / window_total
        logger.info(
            "[ExpertCache] step=%d active=%d window_hit=%.2f%% swaps=%d",
            self.replay_steps, len(self.owner_slot), hit_rate * 100, self.total_swaps,
        )
        if self._trace_path:
            self._pending_trace_record = {
                "schema": "moe_route_window_v1",
                "time_ns": time.time_ns(),
                "step": self.replay_steps,
                "window_routes": window_total,
                "window_misses": window_miss,
                "window_hit_rate": hit_rate,
                "active_slots": len(self.owner_slot),
                "layers": trace_layers,
            }
        return hit_rate

    def _append_trace(self, record: dict) -> None:
        path = Path(self._trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _rebalance(self) -> int:
        for owner in list(self.cooldown):
            self.cooldown[owner] -= 1
            if self.cooldown[owner] <= 0:
                del self.cooldown[owner]

        # Do not index CPU torch tensors one scalar at a time here.  The old
        # loop performed more than 12k Python->PyTorch dispatches for Qwen3's
        # 48x128 expert instances and cost ~123 ms per update.  NumPy views are
        # zero-copy and make scalar filtering a normal CPU operation.
        score_views = {
            layer: scores.numpy() for layer, scores in self.freq.items()
        }
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

        layers = sorted(score_views)
        experts_per_layer = score_views[layers[0]].size
        if any(score_views[layer].size != experts_per_layer for layer in layers):
            raise RuntimeError("expert cache frequency tensors have inconsistent sizes")
        layer_position = {layer: pos for pos, layer in enumerate(layers)}
        candidate_scores = np.concatenate(
            [score_views[layer] for layer in layers]
        ).copy()
        for layer, expert_idx in self.owner_slot:
            candidate_scores[
                layer_position[layer] * experts_per_layer + expert_idx
            ] = -np.inf
        flat_indices = np.arange(candidate_scores.size, dtype=np.int64)
        # Match the old ``sorted((score, (layer, expert)), reverse=True)`` tie
        # break exactly: score descending, then layer/expert (flat id)
        # descending.  This keeps policy semantics unchanged.
        candidate_order = np.lexsort((-flat_indices, -candidate_scores))

        for flat_idx in candidate_order:
            if swaps >= swap_limit:
                break
            score = float(candidate_scores[flat_idx])
            if not np.isfinite(score) or score <= 0:
                break
            layer_pos, expert_idx = divmod(int(flat_idx), experts_per_layer)
            owner = (layers[layer_pos], expert_idx)
            if len(self.owner_slot) < self.config.size:
                slot = self._find_spare_slot()
                victim = None
            else:
                victim_score, victim = self._coldest_owner(score_views)
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

    def _coldest_owner(self, score_views=None):
        eligible = [o for o in self.owner_slot if o not in self.cooldown]
        if not eligible:
            return 0.0, None
        if score_views is None:
            score_views = {
                layer: scores.numpy() for layer, scores in self.freq.items()
            }
        victim = min(eligible, key=lambda o: float(score_views[o[0]][o[1]]))
        return float(score_views[victim[0]][victim[1]]), victim

    def _load_owner_into_slot(self, owner: Tuple[int, int], slot: int) -> None:
        layer_idx, expert_idx = owner
        src = self.sources[layer_idx]
        staging_idx = self._staging_cursor
        self._staging_cursor = (self._staging_cursor + 1) % len(self._staging_buffers)
        prior_copy = self._staging_events[staging_idx]
        if prior_copy is not None:
            wait_begin = time.perf_counter_ns()
            prior_copy.synchronize()
            self._staging_wait_us.append((time.perf_counter_ns() - wait_begin) / 1e3)
        w13, s13, w2, s2 = self._staging_buffers[staging_idx]

        export_begin = time.perf_counter_ns()
        src.handle.export_expert_npu_layout_out(expert_idx, w13, s13, w2, s2)
        self._export_us.append((time.perf_counter_ns() - export_begin) / 1e3)
        copy_begin = time.perf_counter_ns()
        self.w13_cache[slot].copy_(w13, non_blocking=True)
        self.s13_cache[slot].copy_(s13, non_blocking=True)
        self.w2_cache[slot].copy_(w2, non_blocking=True)
        self.s2_cache[slot].copy_(s2, non_blocking=True)
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        self._staging_events[staging_idx] = event
        self._copy_submit_us.append((time.perf_counter_ns() - copy_begin) / 1e3)

    @staticmethod
    def _timing_summary(values) -> dict:
        if not values:
            return {"samples": 0, "median_us": None, "p95_us": None, "max_us": None}
        ordered = sorted(values)
        p95_idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
        return {
            "samples": len(ordered),
            "median_us": ordered[len(ordered) // 2],
            "p95_us": ordered[p95_idx],
            "max_us": ordered[-1],
        }

    def metrics(self) -> dict:
        hit_rate = 0.0 if self.total_routes == 0 else 1.0 - self.total_misses / self.total_routes
        return {
            "steps": self.replay_steps,
            "active_slots": len(self.owner_slot),
            "swaps": self.total_swaps,
            "routes": self.total_routes,
            "hit_rate": hit_rate,
            "timing": {
                "control": self._timing_summary(self._control_us),
                "collect": self._timing_summary(self._collect_us),
                "rebalance": self._timing_summary(self._rebalance_us),
                "expert_export": self._timing_summary(self._export_us),
                "staging_wait": self._timing_summary(self._staging_wait_us),
                "h2d_submit": self._timing_summary(self._copy_submit_us),
            },
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
