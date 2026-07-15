"""Decode benchmark for the CPU remainder of NPU expert caching.

The cache marks hit routes as -1.  This benchmark keeps TopK fixed while
varying how many routes remain on CPU, which is the workload that the ordinary
full-MoE benchmark cannot represent.
"""

import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import nanovllm_ext  # noqa: F401 - registers torch custom classes and ops


def random_q4(rows: int, cols: int):
    qs = torch.from_numpy(
        np.random.randint(0, 2**32, size=(rows, cols // 8), dtype=np.uint32)
    ).contiguous()
    scales = (torch.rand(rows, cols // 32) / np.sqrt(cols)).to(torch.float16)
    return qs, scales.contiguous()


def make_handle(experts: int, hidden: int, intermediate: int):
    gate_qs, gate_d = random_q4(experts * 2 * intermediate, hidden)
    down_qs, down_d = random_q4(experts * hidden, intermediate)
    handle = torch.classes.nanovllm.MoEInfer(experts, hidden, intermediate, 1)
    handle.store_quantized_repack(
        gate_qs.view(experts, 2 * intermediate, hidden // 8),
        gate_d.view(experts, 2 * intermediate, hidden // 32),
        down_qs.view(experts, hidden, intermediate // 8),
        down_d.view(experts, hidden, intermediate // 32),
    )
    return handle


def make_case(
    hidden: int,
    experts: int,
    top_k: int,
    tokens: int,
    cpu_routes: int,
    variant: int,
    routing_experts: int,
):
    x = torch.randn(tokens, hidden, dtype=torch.float16).contiguous()
    ids = torch.full((tokens, top_k), -1, dtype=torch.int32)
    # Rotate through the real expert pool instead of repeatedly benchmarking
    # the same eight hot experts.  A real decode step visits 48 different MoE
    # layers, so keeping a tiny eight-expert handle unrealistically favours the
    # CPU cache hierarchy.
    for token in range(tokens):
        base = ((variant * tokens + token) * top_k) % routing_experts
        routed = (
            torch.arange(top_k, dtype=torch.int32) + base
        ) % routing_experts
        ids[token, :cpu_routes] = routed[:cpu_routes]
    weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32)
    return x, ids, weights


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def bootstrap_ci(values, seed: int, samples: int = 2000) -> tuple[float, float]:
    """Deterministic bootstrap CI for the median."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(arr, size=(samples, arr.size), replace=True), axis=1)
    return percentile(medians, 2.5), percentile(medians, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument(
        "--routing-experts",
        type=int,
        help="expert-id pool used by routes; smaller values create cross-token collisions",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--cpu-routes",
        type=int,
        nargs="+",
        help="CPU miss routes per token (default: every value from 0 to top-k)",
    )
    parser.add_argument("--variants", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if args.experts < args.top_k:
        parser.error("--experts must be >= --top-k")
    routing_experts = args.routing_experts or args.experts
    if routing_experts < args.top_k or routing_experts > args.experts:
        parser.error("--routing-experts must be between top-k and experts")
    if args.variants <= 0 or args.runs <= 0:
        parser.error("--variants and --runs must be positive")
    if any(tokens <= 0 for tokens in args.tokens):
        parser.error("--tokens values must be positive")
    route_counts = args.cpu_routes or list(range(0, args.top_k + 1))
    if any(routes < 0 or routes > args.top_k for routes in route_counts):
        parser.error("--cpu-routes values must be between 0 and top-k")

    handle = make_handle(args.experts, args.hidden, args.intermediate)
    cases = {
        (tokens, cpu_routes): [
            make_case(
                args.hidden,
                args.experts,
                args.top_k,
                tokens,
                cpu_routes,
                variant,
                routing_experts,
            )
            for variant in range(args.variants)
        ]
        for tokens in args.tokens
        for cpu_routes in route_counts
    }

    # Warm every shape before timing, then interleave route counts in a
    # deterministic shuffled order. This avoids measuring 0..TopK under
    # systematically different thread-pool/cache states.
    for key in cases:
        for x, ids, weights in cases[key]:
            for _ in range(args.warmup):
                torch.ops.nanovllm.moe_forward(x, ids, weights, handle)

    wall_samples = {key: [] for key in cases}
    compute_samples = {key: [] for key in cases}
    order = list(cases)
    for run_idx in range(args.runs):
        random.shuffle(order)
        for key in order:
            x, ids, weights = cases[key][run_idx % args.variants]
            begin = time.perf_counter_ns()
            torch.ops.nanovllm.moe_forward(x, ids, weights, handle)
            wall_samples[key].append((time.perf_counter_ns() - begin) / 1e6)
            compute_samples[key].append(handle.get_last_run_time_ms())

    results = {}
    rows = []
    print("tokens  cpu_routes/token  compute_p50_ms  compute_p95_ms  wall_p50_ms  median_95%_ci")
    for tokens in args.tokens:
        for cpu_routes in route_counts:
            key = (tokens, cpu_routes)
            compute = compute_samples[key]
            wall = wall_samples[key]
            median = statistics.median(compute)
            ci_low, ci_high = bootstrap_ci(compute, seed=tokens * 100 + cpu_routes)
            results[key] = median
            print(
                f"{tokens:>6}  {cpu_routes:>16}  {median:>14.4f}  "
                f"{percentile(compute, 95):>14.4f}  "
                f"{statistics.median(wall):>11.4f}  [{ci_low:.4f}, {ci_high:.4f}]"
            )
            rows.append({
                "tokens": tokens,
                "cpu_routes_per_token": cpu_routes,
                "compute_median_ms": median,
                "compute_mean_ms": statistics.mean(compute),
                "compute_p95_ms": percentile(compute, 95),
                "compute_median_ci95_ms": [ci_low, ci_high],
                "wall_median_ms": statistics.median(wall),
                "wall_p95_ms": percentile(wall, 95),
            })
    drop = None
    if len(args.tokens) == 1 and args.top_k in route_counts and args.top_k - 1 in route_counts:
        tokens = args.tokens[0]
        full = results[(tokens, args.top_k)]
        minus_one = results[(tokens, args.top_k - 1)]
        drop = (full - minus_one) / full * 100.0
        print(
            f"\nTopK->{args.top_k - 1} drop: {drop:.2f}% "
            f"(ideal one-route share: {100.0 / args.top_k:.2f}%)"
        )

    first_tokens = args.tokens[0]
    fixed = results.get((first_tokens, 0), 0.0)
    incremental = {
        routes: max(0.0, results[(first_tokens, routes)] - fixed)
        for routes in route_counts
    }
    incremental_drop = None
    if args.top_k in incremental and args.top_k - 1 in incremental and incremental[args.top_k] > 0:
        incremental_drop = (
            (incremental[args.top_k] - incremental[args.top_k - 1])
            / incremental[args.top_k] * 100.0
        )
        print(
            f"TopK->{args.top_k - 1} incremental-compute drop after subtracting "
            f"0-route fixed cost: {incremental_drop:.2f}%"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "cpu_partial_moe_decode_q4_0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_shape": {
                "source": "Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32",
                "hidden_size": args.hidden,
                "moe_intermediate_size": args.intermediate,
                "num_experts": args.experts,
                "top_k": args.top_k,
            },
            "environment": {
                "nanovllm_tp_size": os.environ.get("NANOVLLM_TP_SIZE"),
                "nanovllm_tp_threads_per_node": os.environ.get("NANOVLLM_TP_THREADS_PER_NODE"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "torch_version": torch.__version__,
            },
            "settings": {
                "warmup_per_variant": args.warmup,
                "runs": args.runs,
                "route_variants": args.variants,
                "timing": "C++ compute timer plus Python wall timer; shuffled route counts",
            },
            "topk_minus_one_drop_percent": drop,
            "topk_minus_one_incremental_drop_percent": incremental_drop,
            "zero_route_fixed_compute_ms": fixed,
            "results": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON result: {args.json_out}")


if __name__ == "__main__":
    main()
