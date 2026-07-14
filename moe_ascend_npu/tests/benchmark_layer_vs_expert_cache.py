#!/usr/bin/env python3
"""Matched ShareGPT comparison: fixed NPU layers versus dynamic expert cache.

For Qwen3-30B-A3B every MoE layer has 128 experts, so the default pairs use
the same persistent NPU MoE weight capacity:

* 2 fixed NPU layers  versus K=256 dynamic expert slots;
* 4 fixed NPU layers  versus K=512;
* 8 fixed NPU layers  versus K=1024;
* 16 fixed NPU layers versus K=2048.

The fixed-layer case has no cache and therefore no same-layer CPU/NPU overlap:
layers before ``--moe-offload-start-layer`` run wholly on NPU and the remaining
layers run wholly on CPU. The cache case always uses the production overlap
path: cache hits run on NPU while the CPU computes the miss remainder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    layer_count: int | None = None
    cache_size: int | None = None


def wait_ready(proc: subprocess.Popen, base_url: str, timeout_s: float, log_path: Path) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode}; see {log_path}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5.0):
                return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2.0)
    raise TimeoutError(f"server readiness timed out: {last_error}; see {log_path}")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=30.0)
        return
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10.0)


def server_command(args: argparse.Namespace, case: Case) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model-path",
        args.model_path,
        "--trust-remote-code",
        "--tp-size",
        "1",
        "--attention-backend",
        "ascend",
        "--cuda-graph-bs",
        "1",
        "2",
        "4",
        "8",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
    ]
    if case.mode == "cpu":
        cmd += [
            "--enable-moe-offload",
            "--moe-offload-start-layer",
            "0",
            "--moe-offload-quant-type",
            "q4_0",
        ]
    elif case.mode == "layer":
        if case.layer_count is None:
            raise ValueError("fixed-layer case requires layer_count")
        # Layers [0, layer_count) retain the normal NPU fused-MoE method;
        # layers [layer_count, num_layers) use the Q4 CPU implementation.
        cmd += [
            "--enable-moe-offload",
            "--moe-offload-start-layer",
            str(case.layer_count),
            "--moe-offload-quant-type",
            "q4_0",
        ]
    elif case.mode == "cache":
        if case.cache_size is None:
            raise ValueError("cache case requires cache_size")
        cmd += [
            "--enable-moe-expert-cache",
            "--moe-expert-cache-size",
            str(case.cache_size),
            "--moe-expert-cache-swap-per-update",
            str(args.swap_per_update),
            "--moe-expert-cache-update-interval",
            str(args.update_interval),
            "--moe-expert-cache-warmup-steps",
            str(args.warmup_steps),
            "--moe-expert-cache-decay",
            str(args.decay),
        ]
    else:
        raise ValueError(f"unknown case mode: {case.mode}")
    return cmd


def benchmark_command(
    args: argparse.Namespace,
    output_path: Path,
    *,
    num_prompts: int,
    output_len: int,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://{args.host}:{args.port}",
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        args.dataset_path,
        "--num-prompts",
        str(num_prompts),
        "--sharegpt-output-len",
        str(output_len),
        "--max-concurrency",
        str(args.max_concurrency),
        "--warmup-requests",
        "0",
        "--seed",
        str(seed),
        "--output-file",
        str(output_path),
    ]


def read_last_json(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"no JSON object found in {path}")


def parse_cache_log(
    path: Path,
    cache_size: int | None,
    *,
    start_offset: int = 0,
) -> dict[str, Any]:
    if cache_size is None:
        return {}
    text = path.read_bytes()[start_offset:].decode("utf-8", errors="replace")
    rows = [
        (int(active), float(hit) / 100.0, int(swaps))
        for active, hit, swaps in re.findall(
            r"active=(\d+) window_hit=([0-9.]+)% swaps=(\d+)", text
        )
    ]
    full = [hit for active, hit, _ in rows if active >= cache_size]
    return {
        "windows": len(rows),
        "full_windows": len(full),
        "full_mean_hit_rate": sum(full) / len(full) if full else None,
        "full_tail32_hit_rate": sum(full[-32:]) / len(full[-32:]) if full else None,
        "total_swaps": rows[-1][2] if rows else None,
        "max_active": max((active for active, _, _ in rows), default=0),
    }


def equivalent_slots(case: Case, experts_per_layer: int) -> int:
    if case.cache_size is not None:
        return case.cache_size
    if case.layer_count is not None:
        return case.layer_count * experts_per_layer
    return 0


def warmup_prompt_count(args: argparse.Namespace, case: Case) -> int:
    slots = equivalent_slots(case, args.experts_per_layer)
    if slots <= 0:
        return 1
    fill_updates = (slots + args.swap_per_update - 1) // args.swap_per_update
    fill_steps = args.warmup_steps + fill_updates * args.update_interval
    return max(
        1,
        (fill_steps + args.warmup_output_len - 1) // args.warmup_output_len
        + args.warmup_margin_prompts,
    )


def select_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "duration",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "output_throughput",
        "median_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "p99_itl_ms",
    )
    return {key: raw.get(key) for key in keys}


def write_summary(
    result_dir: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "model": args.model_path,
        "dataset": args.dataset_path,
        "seed": args.seed,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "max_concurrency": args.max_concurrency,
        "num_moe_layers": args.num_moe_layers,
        "experts_per_layer": args.experts_per_layer,
        "nanovllm_tp_size": args.nanovllm_tp_size,
        "threads_per_node": args.threads_per_node,
        "results": results,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    by_name = {row["case"]["name"]: row for row in results}
    lines = [
        "# Qwen3 fixed-layer offload vs dynamic expert cache",
        "",
        "同一行使用相同的 NPU MoE 权重容量。固定层方案没有 cache，也没有同层 CPU/NPU overlap；",
        "动态缓存方案始终使用当前 overlap 路径。",
        "",
        "| NPU 容量 | 固定层 tok/s | Cache tok/s | Cache 相对提升 | 固定层 NPU 路由占比 | Cache 填满后命中率 | 固定层 TPOT | Cache TPOT |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layers, cache_size in zip(args.layer_counts, args.cache_sizes):
        fixed = by_name.get(f"layer_l{layers}")
        cache = by_name.get(f"cache_k{cache_size}")
        if fixed is None or cache is None:
            continue
        fixed_tps = float(fixed["metrics"]["output_throughput"])
        cache_tps = float(cache["metrics"]["output_throughput"])
        speedup = cache_tps / fixed_tps - 1.0
        hit_rate = cache["cache"].get("full_mean_hit_rate")
        hit_text = "N/A" if hit_rate is None else f"{hit_rate:.2%}"
        lines.append(
            f"| {cache_size} experts = {layers} layers | {fixed_tps:.3f} | "
            f"{cache_tps:.3f} | {speedup:+.2%} | "
            f"{layers / args.num_moe_layers:.2%} | {hit_text} | "
            f"{fixed['metrics']['median_tpot_ms']:.3f} ms | "
            f"{cache['metrics']['median_tpot_ms']:.3f} ms |"
        )
    lines += [
        "",
        "解释口径：固定层方案每个 token 在选中的完整层上全部走 NPU，因此总体 NPU 路由占比",
        "等于 `N / num_moe_layers`；动态缓存若填满后命中率高于该比例，说明全局替换策略捕获了",
        "跨层热点。最终吞吐差异同时反映更高有效命中率和命中/未命中的 CPU+NPU overlap。",
        "",
    ]
    (result_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32",
    )
    parser.add_argument(
        "--dataset-path",
        default="/home/xwj/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument(
        "--result-dir",
        default="docs/bench_results/raw/qwen3_layer_vs_cache_seed1",
    )
    parser.add_argument("--layer-counts", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--cache-sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--num-moe-layers", type=int, default=48)
    parser.add_argument("--experts-per-layer", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=320)
    parser.add_argument("--warmup-output-len", type=int, default=320)
    parser.add_argument("--warmup-seed", type=int, default=0)
    parser.add_argument("--warmup-margin-prompts", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--nanovllm-tp-size", type=int, default=2)
    parser.add_argument("--threads-per-node", type=int, default=20)
    parser.add_argument("--mem-fraction-static", type=float, default=0.793)
    parser.add_argument("--swap-per-update", type=int, default=8)
    parser.add_argument("--update-interval", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--decay", type=float, default=0.95)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.layer_counts) != len(args.cache_sizes):
        parser.error("--layer-counts and --cache-sizes must have equal lengths")
    for layers, cache_size in zip(args.layer_counts, args.cache_sizes):
        if cache_size != layers * args.experts_per_layer:
            parser.error(
                f"K={cache_size} is not equal to {layers} layers * "
                f"{args.experts_per_layer} experts"
            )

    cases: list[Case] = []
    if not args.skip_cpu:
        cases.append(Case(name="cpu_q4", mode="cpu"))
    for layers, cache_size in zip(args.layer_counts, args.cache_sizes):
        cases.append(Case(name=f"layer_l{layers}", mode="layer", layer_count=layers))
        cases.append(Case(name=f"cache_k{cache_size}", mode="cache", cache_size=cache_size))

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    env["NANOVLLM_TP_SIZE"] = str(args.nanovllm_tp_size)
    env["NANOVLLM_TP_THREADS_PER_NODE"] = str(args.threads_per_node)
    env["OMP_NUM_THREADS"] = str(args.threads_per_node)
    env.setdefault("GLOO_SOCKET_IFNAME", "lo")

    if args.dry_run:
        for case in cases:
            output_path = result_dir / f"{case.name}.jsonl"
            warmup_path = result_dir / f"{case.name}.warmup.jsonl"
            print(json.dumps({
                "case": asdict(case),
                "server": server_command(args, case),
                "warmup_prompts": warmup_prompt_count(args, case),
                "warmup": benchmark_command(
                    args,
                    warmup_path,
                    num_prompts=warmup_prompt_count(args, case),
                    output_len=args.warmup_output_len,
                    seed=args.warmup_seed,
                ),
                "benchmark": benchmark_command(
                    args,
                    output_path,
                    num_prompts=args.num_prompts,
                    output_len=args.output_len,
                    seed=args.seed,
                ),
            }, ensure_ascii=False))
        return 0

    results: list[dict[str, Any]] = []
    for case in cases:
        output_path = result_dir / f"{case.name}.jsonl"
        warmup_path = result_dir / f"{case.name}.warmup.jsonl"
        server_log = result_dir / f"{case.name}.server.log"
        warmup_log = result_dir / f"{case.name}.warmup.log"
        bench_log = result_dir / f"{case.name}.bench.log"
        case_path = result_dir / f"{case.name}.summary.json"
        if case_path.exists() and not args.force:
            results.append(json.loads(case_path.read_text(encoding="utf-8")))
            print(f"==> reuse {case.name}", flush=True)
            continue

        for stale in (output_path, warmup_path):
            if stale.exists():
                stale.unlink()
        server_cmd = server_command(args, case)
        num_warmup_prompts = warmup_prompt_count(args, case)
        warmup_cmd = benchmark_command(
            args,
            warmup_path,
            num_prompts=num_warmup_prompts,
            output_len=args.warmup_output_len,
            seed=args.warmup_seed,
        )
        bench_cmd = benchmark_command(
            args,
            output_path,
            num_prompts=args.num_prompts,
            output_len=args.output_len,
            seed=args.seed,
        )
        print(f"==> launch {case.name}", flush=True)
        with server_log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(
                server_cmd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        try:
            wait_ready(
                proc,
                f"http://{args.host}:{args.port}",
                args.ready_timeout,
                server_log,
            )
            print(
                f"==> warmup {case.name}: {num_warmup_prompts}x"
                f"{args.warmup_output_len} tokens",
                flush=True,
            )
            with warmup_log.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    warmup_cmd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True,
                )
            warmup_cache = parse_cache_log(server_log, case.cache_size)
            if (
                case.cache_size is not None
                and warmup_cache["max_active"] < case.cache_size
            ):
                raise RuntimeError(
                    f"cache warmup incomplete for {case.name}: "
                    f"active={warmup_cache['max_active']} < K={case.cache_size}; "
                    f"increase --warmup-margin-prompts"
                )
            measurement_log_offset = server_log.stat().st_size
            print(f"==> measure {case.name}", flush=True)
            with bench_log.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    bench_cmd,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=True,
                )
            raw = read_last_json(output_path)
            row = {
                "case": asdict(case),
                "equivalent_expert_slots": equivalent_slots(
                    case, args.experts_per_layer
                ),
                "fixed_layer_npu_route_fraction": (
                    case.layer_count / args.num_moe_layers
                    if case.layer_count is not None else None
                ),
                "metrics": select_metrics(raw),
                "warmup": {
                    "prompts": num_warmup_prompts,
                    "output_len": args.warmup_output_len,
                    "seed": args.warmup_seed,
                    "cache": warmup_cache,
                },
                "cache": parse_cache_log(
                    server_log,
                    case.cache_size,
                    start_offset=measurement_log_offset,
                ),
                "server_command": server_cmd,
                "warmup_command": warmup_cmd,
                "benchmark_command": bench_cmd,
            }
            case_path.write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(row)
            write_summary(result_dir, args, results)
        finally:
            stop_server(proc)

    write_summary(result_dir, args, results)
    print(f"results: {result_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
