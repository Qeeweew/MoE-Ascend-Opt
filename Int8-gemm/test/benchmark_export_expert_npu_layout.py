"""Benchmark Q4_0 CPU-layout to Ascend W4A16 expert repacking."""

from __future__ import annotations

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

import nanovllm_ext  # noqa: F401


def random_q4(rows: int, cols: int, rng: np.random.Generator):
    qs = torch.from_numpy(
        rng.integers(0, 2**32, size=(rows, cols // 8), dtype=np.uint32)
    ).contiguous()
    scales = torch.ones(rows, cols // 32, dtype=torch.float16)
    return qs, scales


def summarize(samples_us: list[float], output_bytes: int) -> dict:
    median = statistics.median(samples_us)
    return {
        "samples": len(samples_us),
        "median_us": median,
        "p95_us": float(np.percentile(samples_us, 95)),
        "max_us": max(samples_us),
        "output_gbps": output_bytes / median / 1e3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--hot-runs", type=int, default=64)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(1)
    q13, s13 = random_q4(args.experts * 2 * args.intermediate, args.hidden, rng)
    q2, s2 = random_q4(args.experts * args.hidden, args.intermediate, rng)
    handle = torch.classes.nanovllm.MoEInfer(
        args.experts, args.hidden, args.intermediate, 1
    )
    handle.store_quantized_repack(
        q13.view(args.experts, 2 * args.intermediate, args.hidden // 8),
        s13.view(args.experts, 2 * args.intermediate, args.hidden // 32),
        q2.view(args.experts, args.hidden, args.intermediate // 8),
        s2.view(args.experts, args.hidden, args.intermediate // 32),
    )
    outputs = (
        torch.empty(args.hidden, 2 * args.intermediate // 8, dtype=torch.int32, pin_memory=True),
        torch.empty(args.hidden // 32, 2 * args.intermediate, dtype=torch.float16, pin_memory=True),
        torch.empty(args.intermediate, args.hidden // 8, dtype=torch.int32, pin_memory=True),
        torch.empty(args.intermediate // 32, args.hidden, dtype=torch.float16, pin_memory=True),
    )
    output_bytes = sum(t.nbytes for t in outputs)
    order = list(range(args.experts))
    random.Random(0).shuffle(order)

    def run(experts: list[int]) -> list[float]:
        samples = []
        for expert in experts:
            begin = time.perf_counter_ns()
            handle.export_expert_npu_layout_out(expert, *outputs)
            samples.append((time.perf_counter_ns() - begin) / 1e3)
        return samples

    results = {
        "cold_random": summarize(run(order), output_bytes),
        "hot_same": summarize(run([order[-1]] * args.hot_runs), output_bytes),
        "second_random": summarize(run(order), output_bytes),
    }
    for name, row in results.items():
        print(
            f"{name}: median={row['median_us']:.3f} us, "
            f"p95={row['p95_us']:.3f} us, output={row['output_gbps']:.2f} GB/s"
        )

    if args.json_out:
        payload = {
            "benchmark": "export_expert_npu_layout_q4_0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "shape": {
                "experts": args.experts,
                "hidden_size": args.hidden,
                "intermediate_size": args.intermediate,
                "output_bytes_per_expert": output_bytes,
            },
            "environment": {
                "nanovllm_tp_size": os.environ.get("NANOVLLM_TP_SIZE"),
                "nanovllm_tp_threads_per_node": os.environ.get(
                    "NANOVLLM_TP_THREADS_PER_NODE"
                ),
                "torch_version": torch.__version__,
            },
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
