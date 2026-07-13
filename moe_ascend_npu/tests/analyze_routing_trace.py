"""Analyze Qwen3 MoE routing locality from ExpertCache JSONL traces.

The trace contains per-layer expert histograms for each controller window.  It
is intentionally aggregated rather than token-level: that is sufficient for
the global LFU policy implemented by ExpertCacheManager and keeps collection
overhead out of the decode hot path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_trace(path: Path) -> tuple[list[dict], np.ndarray]:
    records = []
    windows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != "moe_route_window_v1":
            continue
        layers = sorted(row["layers"], key=lambda item: int(item["layer"]))
        if not layers:
            continue
        matrix = np.asarray([item["counts"] for item in layers], dtype=np.int64)
        if windows and matrix.shape != windows[0].shape:
            raise ValueError(f"inconsistent trace shape at line {line_no}: {matrix.shape}")
        records.append(row)
        windows.append(matrix)
    if not windows:
        raise ValueError(f"no moe_route_window_v1 records in {path}")
    return records, np.stack(windows)


def topk_hit(counts: np.ndarray, k: int) -> float:
    flat = counts.reshape(-1)
    total = int(flat.sum())
    if total == 0 or k <= 0:
        return 0.0
    k = min(k, flat.size)
    chosen = np.argpartition(flat, -k)[-k:]
    return float(flat[chosen].sum() / total)


def predictive_lfu(windows: np.ndarray, k: int, decay: float) -> dict:
    scores = np.zeros(windows.shape[1:], dtype=np.float64)
    hits = routes = 0
    evaluated = 0
    per_window = []
    for window in windows:
        total = int(window.sum())
        if scores.sum() > 0 and total > 0:
            flat_scores = scores.reshape(-1)
            chosen = np.argpartition(flat_scores, -min(k, flat_scores.size))[-min(k, flat_scores.size):]
            current = window.reshape(-1)
            window_hits = int(current[chosen].sum())
            hits += window_hits
            routes += total
            evaluated += 1
            per_window.append(window_hits / total)
        scores *= decay
        scores += window
    return {
        "hit_rate": None if routes == 0 else hits / routes,
        "evaluated_windows": evaluated,
        "window_hit_p50": None if not per_window else float(np.percentile(per_window, 50)),
        "window_hit_p10": None if not per_window else float(np.percentile(per_window, 10)),
    }


def bounded_lfu(
    windows: np.ndarray, k: int, decay: float, swap_limit: int, hysteresis: float = 1.10
) -> dict:
    """Replay a bounded global LFU controller over aggregated windows."""
    scores = np.zeros(windows.shape[1] * windows.shape[2], dtype=np.float64)
    cache: set[int] = set()
    hits = routes = swaps = 0
    for window in windows:
        current = window.reshape(-1)
        total = int(current.sum())
        if cache and total:
            chosen = np.fromiter(cache, dtype=np.int64)
            hits += int(current[chosen].sum())
        routes += total
        scores *= decay
        scores += current

        changed = 0
        candidates = np.argsort(scores)[::-1]
        for candidate in candidates:
            candidate = int(candidate)
            if changed >= swap_limit or scores[candidate] <= 0:
                break
            if candidate in cache:
                continue
            if len(cache) < k:
                cache.add(candidate)
            else:
                victim = min(cache, key=lambda idx: scores[idx])
                if scores[candidate] <= scores[victim] * hysteresis:
                    break
                cache.remove(victim)
                cache.add(candidate)
            changed += 1
            swaps += 1
    return {
        "hit_rate": None if routes == 0 else hits / routes,
        "swaps": swaps,
        "final_active": len(cache),
    }


def static_prefix_hit(windows: np.ndarray, k: int, train_fraction: float) -> float | None:
    split = max(1, min(len(windows) - 1, math.ceil(len(windows) * train_fraction)))
    if split >= len(windows):
        return None
    train = windows[:split].sum(axis=0).reshape(-1)
    test = windows[split:].sum(axis=0).reshape(-1)
    if test.sum() == 0:
        return None
    chosen = np.argpartition(train, -min(k, train.size))[-min(k, train.size):]
    return float(test[chosen].sum() / test.sum())


def working_set(total: np.ndarray, targets=(0.5, 0.8, 0.9, 0.95)) -> dict:
    ordered = np.sort(total.reshape(-1))[::-1]
    cumulative = np.cumsum(ordered)
    routes = int(cumulative[-1])
    if routes == 0:
        return {str(target): None for target in targets}
    return {
        str(target): int(np.searchsorted(cumulative, target * routes, side="left") + 1)
        for target in targets
    }


def expected_cpu_latency(hit_rate: float, latency_path: Path, top_k: int) -> dict:
    bench = json.loads(latency_path.read_text(encoding="utf-8"))
    values = {int(row["cpu_routes"]): float(row["compute_median_ms"])
              for row in bench["results"]}
    # Independent-route binomial model.  This is a theoretical translation of
    # routing hit rate into CPU remainder time, not an end-to-end prediction.
    miss = 1.0 - hit_rate
    expected = 0.0
    for routes in range(top_k + 1):
        probability = math.comb(top_k, routes) * miss**routes * hit_rate**(top_k - routes)
        expected += probability * values[routes]
    full = values[top_k]
    return {
        "expected_compute_ms": expected,
        "full_miss_compute_ms": full,
        "reduction_vs_full_percent": (full - expected) / full * 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--cache-sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument(
        "--decays", nargs="+", type=float,
        default=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        help="EMA decays to compare for previous-window predictive LFU.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.2)
    parser.add_argument("--slot-mib", type=float, default=2.53125)
    parser.add_argument("--cpu-latency-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    records, windows = load_trace(args.trace)
    total = windows.sum(axis=0)
    num_windows, num_layers, num_experts = windows.shape
    total_instances = num_layers * num_experts
    total_routes = int(total.sum())
    observed_routes = sum(int(row.get("window_routes", 0)) for row in records)
    observed_misses = sum(int(row.get("window_misses", 0)) for row in records)

    cache_rows = []
    for k in args.cache_sizes:
        oracle = topk_hit(total, k)
        row = {
            "k": k,
            "expert_instance_fraction": k / total_instances,
            "active_cache_gib": k * args.slot_mib / 1024.0,
            "uniform_hit_rate": min(1.0, k / total_instances),
            "oracle_global_lfu_hit_rate": oracle,
            "static_prefix_hit_rate": static_prefix_hit(windows, k, args.train_fraction),
            "predictive_ema_lfu": {
                str(decay): predictive_lfu(windows, k, decay)
                for decay in args.decays
            },
            "bounded_lfu": {
                f"decay={decay},swaps={limit}": bounded_lfu(
                    windows, k, decay, limit
                )
                for decay in (0.0, 0.5, 0.95)
                for limit in (8, 16, 32, 64)
            },
        }
        if args.cpu_latency_json:
            row["cpu_remainder_oracle"] = expected_cpu_latency(
                oracle, args.cpu_latency_json, top_k=8
            )
        cache_rows.append(row)

    layer_entropy = []
    for layer in total:
        probs = layer / max(1, layer.sum())
        nonzero = probs[probs > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum())
        layer_entropy.append(entropy / math.log2(num_experts))

    control = [float(row.get("control_us", 0.0)) for row in records if row.get("control_us") is not None]
    result = {
        "trace": str(args.trace),
        "windows": num_windows,
        "layers": num_layers,
        "experts_per_layer": num_experts,
        "expert_instances": total_instances,
        "routes": total_routes,
        "observed_cache_hit_rate": (
            None if observed_routes == 0 else 1.0 - observed_misses / observed_routes
        ),
        "steps": [int(records[0]["step"]), int(records[-1]["step"])],
        "working_set_expert_instances": working_set(total),
        "normalized_layer_entropy": {
            "mean": float(np.mean(layer_entropy)),
            "min": float(np.min(layer_entropy)),
            "max": float(np.max(layer_entropy)),
        },
        "controller_control_us": {
            "samples": len(control),
            "median": None if not control else float(np.median(control)),
            "p95": None if not control else float(np.percentile(control, 95)),
            "max": None if not control else max(control),
        },
        "cache_sizes": cache_rows,
    }
    if args.cpu_latency_json and result["observed_cache_hit_rate"] is not None:
        result["observed_cpu_remainder"] = expected_cpu_latency(
            result["observed_cache_hit_rate"], args.cpu_latency_json, top_k=8
        )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
