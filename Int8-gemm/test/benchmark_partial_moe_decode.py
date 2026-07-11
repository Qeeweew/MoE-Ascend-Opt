"""Decode benchmark for the CPU remainder of NPU expert caching.

The cache marks hit routes as -1.  This benchmark keeps TopK fixed while
varying how many routes remain on CPU, which is the workload that the ordinary
full-MoE benchmark cannot represent.
"""

import argparse
import statistics
import time

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


def measure(handle, hidden: int, top_k: int, cpu_routes: int, warmup: int, runs: int):
    x = torch.randn(1, hidden, dtype=torch.float16).contiguous()
    ids = torch.full((1, top_k), -1, dtype=torch.int32)
    ids[0, :cpu_routes] = torch.arange(cpu_routes, dtype=torch.int32)
    weights = torch.full((1, top_k), 1.0 / top_k, dtype=torch.float32)
    for _ in range(warmup):
        torch.ops.nanovllm.moe_forward(x, ids, weights, handle)
    samples = []
    for _ in range(runs):
        begin = time.perf_counter_ns()
        torch.ops.nanovllm.moe_forward(x, ids, weights, handle)
        samples.append((time.perf_counter_ns() - begin) / 1e6)
    return statistics.median(samples), statistics.mean(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    handle = make_handle(args.top_k, args.hidden, args.intermediate)
    results = {}
    print("cpu_routes  median_ms  mean_ms")
    for cpu_routes in range(0, args.top_k + 1):
        median, mean = measure(
            handle, args.hidden, args.top_k, cpu_routes, args.warmup, args.runs
        )
        results[cpu_routes] = median
        print(f"{cpu_routes:>10}  {median:>9.3f}  {mean:>7.3f}")
    if args.top_k >= 1 and results.get(args.top_k, 0.0) > 0:
        full = results[args.top_k]
        minus_one = results[args.top_k - 1]
        drop = (full - minus_one) / full * 100.0
        print(
            f"\nTopK->{args.top_k - 1} drop: {drop:.2f}% "
            f"(ideal one-route share: {100.0 / args.top_k:.2f}%)"
        )


if __name__ == "__main__":
    main()
