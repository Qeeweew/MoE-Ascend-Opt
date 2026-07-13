#!/usr/bin/env python3
"""Measure whether cached-NPU MoE is hidden behind the CPU miss remainder.

By default the benchmark reproduces the original side-stream implementation.
``--single-stream`` instead uses two callbacks on the main stream: the first
dispatches CPU work, the NPU cached MoE runs next, and the second joins the CPU
work before H2D and Add. Both paths keep Qwen3-30B-A3B TP2 CPU thread counts
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu

import nanovllm_ext  # noqa: F401 - register nanovllm ops/classes
from moe_ascend_npu.kernels import ensure_kernels_loaded


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def random_q4(rows: int, cols: int, scale_dtype: torch.dtype):
    qs = torch.from_numpy(
        np.random.randint(0, 2**32, size=(rows, cols // 8), dtype=np.uint32)
    ).contiguous()
    scales = (torch.rand(rows, cols // 32) / np.sqrt(cols)).to(scale_dtype)
    return qs, scales.contiguous()


def make_cpu_handle(
    experts: int,
    hidden: int,
    intermediate: int,
    scale_dtype: torch.dtype,
):
    gate_qs, gate_d = random_q4(experts * 2 * intermediate, hidden, scale_dtype)
    down_qs, down_d = random_q4(experts * hidden, intermediate, scale_dtype)
    handle = torch.classes.nanovllm.MoEInfer(experts, hidden, intermediate, 1)
    handle.store_quantized_repack(
        gate_qs.view(experts, 2 * intermediate, hidden // 8),
        gate_d.view(experts, 2 * intermediate, hidden // 32),
        down_qs.view(experts, hidden, intermediate // 8),
        down_d.view(experts, hidden, intermediate // 32),
    )
    return handle


def make_npu_cache(
    slots: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    device: torch.device,
):
    # The latency benchmark only requires valid packed-layout buffers.  Random
    # int32 words have exactly the same memory traffic as repacked int4 data.
    w13 = torch.randint(
        -(2**31), 2**31 - 1,
        (slots, hidden, 2 * intermediate // 8),
        dtype=torch.int32,
        device=device,
    )
    s13 = torch.randn(
        slots, hidden // 32, 2 * intermediate,
        dtype=dtype,
        device=device,
    ) * 0.01
    w2 = torch.randint(
        -(2**31), 2**31 - 1,
        (slots, intermediate, hidden // 8),
        dtype=torch.int32,
        device=device,
    )
    s2 = torch.randn(
        slots, intermediate // 32, hidden,
        dtype=dtype,
        device=device,
    ) * 0.01
    return w13, s13, w2, s2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--variants", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--hits", type=int, nargs="+", default=list(range(0, 9)))
    parser.add_argument(
        "--profile-once",
        action="store_true",
        help="Warm overlap once, then execute exactly one MSTX-marked overlap call.",
    )
    parser.add_argument(
        "--single-stream",
        action="store_true",
        help="Use async-start/join callbacks around NPU MoE on the main stream.",
    )
    parser.add_argument("--check-stream-equivalence", action="store_true")
    parser.add_argument(
        "--graph-replay-smoke",
        type=int,
        default=0,
        help="Capture the single-stream path and compare this many replays with eager.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if any(hit < 0 or hit > args.top_k for hit in args.hits):
        parser.error("--hits values must be between 0 and top-k")

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.npu.set_device(args.device)
    ensure_kernels_loaded()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(f"npu:{args.device}")
    handle = make_cpu_handle(args.experts, args.hidden, args.intermediate, dtype)
    cache = make_npu_cache(args.top_k, args.hidden, args.intermediate, dtype, device)

    side_stream = torch_npu.npu.Stream(device=args.device)
    manager_keepalive = torch.classes.nanovllm.NpuCallbackManager(
        int(side_stream.npu_stream), args.device
    )
    input_ready = torch_npu.npu.Event()
    cpu_done = torch_npu.npu.Event()
    graph_ctx = torch.classes.nanovllm.MoEGraphContext(
        handle, 1, args.top_k, 0 if dtype == torch.float16 else 1
    )
    cpu_out = torch.empty((1, args.hidden), dtype=dtype, device=device)
    zero_out = torch.zeros_like(cpu_out)

    cases: dict[int, list[tuple[torch.Tensor, ...]]] = {}
    for hits in args.hits:
        variants = []
        for variant in range(args.variants):
            x = torch.randn(1, args.hidden, dtype=dtype, device=device)
            base = (variant * args.top_k) % args.experts
            routing_ids = (
                (torch.arange(args.top_k, dtype=torch.int32, device=device) + base)
                % args.experts
            ).view(1, args.top_k)
            cpu_ids = routing_ids.clone()
            cpu_ids[:, :hits] = -1
            slot_ids = torch.full_like(routing_ids, -1)
            if hits:
                slot_ids[:, :hits] = torch.arange(hits, dtype=torch.int32, device=device)
            weights = torch.full(
                (1, args.top_k), 1.0 / args.top_k,
                dtype=torch.float32,
                device=device,
            )
            variants.append((x, routing_ids, cpu_ids, slot_ids, weights))
        cases[hits] = variants

    main_stream = torch_npu.npu.current_stream()
    main_manager_keepalive = None
    if args.single_stream or args.check_stream_equivalence or args.graph_replay_smoke:
        main_manager_keepalive = torch.classes.nanovllm.NpuCallbackManager(
            int(main_stream.npu_stream), args.device
        )

    def submit_cpu(x, routing_ids, cpu_ids, weights):
        input_ready.record(main_stream)
        with torch_npu.npu.stream(side_stream):
            side_stream.wait_event(input_ready)
            torch.ops.nanovllm.moe_forward_npu_graph_partial_out(
                x, cpu_ids, routing_ids, weights, handle, graph_ctx, cpu_out
            )
            cpu_done.record(side_stream)

    def cpu_path(case):
        x, routing_ids, cpu_ids, _slot_ids, weights = case
        submit_cpu(x, routing_ids, cpu_ids, weights)
        main_stream.wait_event(cpu_done)
        return torch.add(cpu_out, zero_out)

    def npu_path(case):
        x, _routing_ids, _cpu_ids, slot_ids, weights = case
        npu_out = torch.ops.moe_ascend_npu.fused_moe_w4a16_cached(
            x, *cache, slot_ids, weights
        )
        return torch.add(npu_out, zero_out)

    def overlap_path(case):
        x, routing_ids, cpu_ids, slot_ids, weights = case
        submit_cpu(x, routing_ids, cpu_ids, weights)
        npu_out = torch.ops.moe_ascend_npu.fused_moe_w4a16_cached(
            x, *cache, slot_ids, weights
        )
        main_stream.wait_event(cpu_done)
        return torch.add(npu_out, cpu_out)

    def single_stream_overlap_path(case):
        x, routing_ids, cpu_ids, slot_ids, weights = case
        torch.ops.nanovllm.moe_forward_npu_graph_partial_start(
            x, cpu_ids, routing_ids, weights, handle, graph_ctx
        )
        npu_out = torch.ops.moe_ascend_npu.fused_moe_w4a16_cached(
            x, *cache, slot_ids, weights
        )
        torch.ops.nanovllm.moe_forward_npu_graph_partial_wait_out(
            cpu_out, graph_ctx
        )
        return torch.add(npu_out, cpu_out)

    selected_overlap_path = (
        single_stream_overlap_path if args.single_stream else overlap_path
    )
    modes = {"cpu": cpu_path, "npu": npu_path, "overlap": selected_overlap_path}
    if args.check_stream_equivalence:
        for hits in args.hits:
            case = cases[hits][0]
            old_out = overlap_path(case)
            torch.npu.synchronize()
            new_out = single_stream_overlap_path(case)
            torch.npu.synchronize()
            max_diff = float((old_out.float() - new_out.float()).abs().max().cpu())
            print(f"STREAM_EQUIVALENCE hits={hits} max_diff={max_diff:.8f}")
            if max_diff != 0.0:
                raise AssertionError(
                    f"single-stream output mismatch for hits={hits}: {max_diff}"
                )
    if args.graph_replay_smoke:
        if len(args.hits) != 1:
            parser.error("--graph-replay-smoke requires exactly one --hits value")
        hits = args.hits[0]
        case = cases[hits][0]
        single_stream_overlap_path(case)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        capture_manager_keepalive = None
        with torch.npu.graph(graph):
            capture_stream = torch_npu.npu.current_stream()
            if int(capture_stream.npu_stream) != int(main_stream.npu_stream):
                capture_manager_keepalive = torch.classes.nanovllm.NpuCallbackManager(
                    int(capture_stream.npu_stream), args.device
                )
            graph_result = single_stream_overlap_path(case)
        torch.npu.synchronize()

        max_seen = 0.0
        x = case[0]
        for replay_idx in range(args.graph_replay_smoke):
            x.add_(torch.tensor(0.0001, dtype=dtype, device=device))
            eager_result = single_stream_overlap_path(case)
            torch.npu.synchronize()
            eager_saved = eager_result.clone()
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            max_diff = float(
                (eager_saved.float() - graph_result.float()).abs().max().cpu()
            )
            max_seen = max(max_seen, max_diff)
            if max_diff > 0.125:
                torch.npu.synchronize()
                if capture_manager_keepalive is not None:
                    del capture_manager_keepalive
                    capture_manager_keepalive = None
                if main_manager_keepalive is not None:
                    del main_manager_keepalive
                    main_manager_keepalive = None
                del manager_keepalive
                raise AssertionError(
                    f"graph replay mismatch at {replay_idx}: {max_diff}"
                )
        print(
            f"GRAPH_REPLAY_SMOKE hits={hits} replays={args.graph_replay_smoke} "
            f"max_diff={max_seen:.8f}"
        )
        if capture_manager_keepalive is not None:
            del capture_manager_keepalive
        del manager_keepalive
        if main_manager_keepalive is not None:
            del main_manager_keepalive
        return 0
    if args.profile_once:
        if len(args.hits) != 1:
            parser.error("--profile-once requires exactly one --hits value")
        hits = args.hits[0]
        case = cases[hits][0]
        selected_overlap_path(case)
        torch.npu.synchronize()
        print(f"PROFILE_WARMUP_DONE hits={hits} misses={args.top_k - hits}", flush=True)
        time.sleep(0.5)
        range_id = torch_npu.npu.mstx.range_start(
            f"MOE_CPU_NPU_OVERLAP_HITS_{hits}",
            stream=main_stream,
            domain="moe_overlap",
        )
        wall_begin = time.perf_counter_ns()
        result = selected_overlap_path(case)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter_ns() - wall_begin) / 1e6
        torch_npu.npu.mstx.range_end(range_id, domain="moe_overlap")
        # Materialize one value after synchronization so the output remains a
        # live result and the target cannot be removed as dead work.
        checksum = float(result[0, 0].float().cpu())
        print(
            f"PROFILE_TARGET_DONE hits={hits} misses={args.top_k - hits} "
            f"wall_ms={elapsed_ms:.6f} checksum={checksum:.6f}",
            flush=True,
        )
        del manager_keepalive
        if main_manager_keepalive is not None:
            del main_manager_keepalive
        return 0

    for hits in args.hits:
        for case in cases[hits]:
            for fn in modes.values():
                for _ in range(args.warmup):
                    fn(case)
    torch.npu.synchronize()

    samples = {
        hits: {mode: [] for mode in modes}
        for hits in args.hits
    }
    wall_samples = {
        hits: {mode: [] for mode in modes}
        for hits in args.hits
    }
    schedule = [(hits, mode) for hits in args.hits for mode in modes]
    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)
    for run_idx in range(args.runs):
        random.shuffle(schedule)
        for hits, mode in schedule:
            case = cases[hits][run_idx % args.variants]
            wall_begin = time.perf_counter_ns()
            start_event.record(main_stream)
            modes[mode](case)
            end_event.record(main_stream)
            torch.npu.synchronize()
            wall_samples[hits][mode].append(
                (time.perf_counter_ns() - wall_begin) / 1e6
            )
            samples[hits][mode].append(start_event.elapsed_time(end_event))

    rows = []
    print(
        "hits misses  cpu_p50_ms npu_p50_ms overlap_p50_ms "
        "residual_vs_cpu_ms efficiency hidden"
    )
    for hits in sorted(args.hits):
        primary = samples[hits]
        cpu = statistics.median(primary["cpu"])
        npu = statistics.median(primary["npu"])
        overlap = statistics.median(primary["overlap"])
        residual = overlap - cpu
        serial_excess = cpu + npu - max(cpu, npu)
        overlap_saved = cpu + npu - overlap
        efficiency = overlap_saved / serial_excess if serial_excess > 0 else 1.0
        # Five microseconds is below one decode layer's practical timing noise.
        hidden = residual <= 0.005
        print(
            f"{hits:>4} {args.top_k - hits:>6}  {cpu:>10.4f} {npu:>10.4f} "
            f"{overlap:>14.4f} {residual:>18.4f} {efficiency:>10.3f} {str(hidden):>6}"
        )
        rows.append({
            "hits": hits,
            "misses": args.top_k - hits,
            "cpu_p50_ms": cpu,
            "cpu_p95_ms": percentile(primary["cpu"], 95),
            "npu_p50_ms": npu,
            "npu_p95_ms": percentile(primary["npu"], 95),
            "overlap_p50_ms": overlap,
            "overlap_p95_ms": percentile(primary["overlap"], 95),
            "residual_vs_cpu_p50_ms": residual,
            "overlap_efficiency": efficiency,
            "npu_hidden_within_5us": hidden,
            "event_p50_ms": {
                mode: statistics.median(samples[hits][mode]) for mode in modes
            },
            "wall_p50_ms": {
                mode: statistics.median(wall_samples[hits][mode]) for mode in modes
            },
        })

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "qwen3_cpu_npu_cached_moe_overlap",
            "shape": {
                "hidden": args.hidden,
                "intermediate_tp2": args.intermediate,
                "experts": args.experts,
                "top_k": args.top_k,
                "dtype": args.dtype,
            },
            "environment": {
                "nanovllm_tp_size": os.environ.get("NANOVLLM_TP_SIZE"),
                "nanovllm_tp_threads_per_node": os.environ.get(
                    "NANOVLLM_TP_THREADS_PER_NODE"
                ),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            },
            "settings": {
                "warmup": args.warmup,
                "runs": args.runs,
                "variants": args.variants,
                "timing": (
                    "Persistent graph-safe CPU callback context executed directly; "
                    "one joined layer per NPU-event sample; interleaved modes and hit counts"
                ),
            },
            "results": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON result: {args.json_out}")

    # Keep the report subscription alive until all callbacks are complete.
    del manager_keepalive
    if main_manager_keepalive is not None:
        del main_manager_keepalive
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
