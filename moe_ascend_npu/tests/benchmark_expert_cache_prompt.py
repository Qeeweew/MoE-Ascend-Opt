#!/usr/bin/env python3
"""Fixed-prompt expert-cache sweep for Qwen3 small-cache experiments.

This script reproduces the measurement used in
the dynamic expert-cache evaluation workflow:

* launch one SGLang server per mode;
* keep NPU graph enabled with BS 1/2/4/8;
* issue the same deterministic prompt repeatedly;
* record E2E latency, output hash, cache hit windows, backoff events, and
  local decode throughput parsed from server logs.

It intentionally uses HTTP requests instead of ``bench_one_batch_server`` so
the output is directly comparable to an online serving workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Explain the practical tradeoffs of CPU and NPU cooperation for "
    "small-batch MoE inference. "
) * 12


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else None


def wait_ready(proc: subprocess.Popen, base_url: str, timeout_s: float, log_path: Path) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited before becoming ready with code {proc.returncode}; "
                f"see {log_path}"
            )
        try:
            _http_json("GET", f"{base_url}/model_info", timeout=5.0)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2.0)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s: {last_error}")


def launch_server(args: argparse.Namespace, mode: str, cache_size: int | None, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    env["NANOVLLM_TP_SIZE"] = str(args.nanovllm_tp_size)
    env.setdefault("GLOO_SOCKET_IFNAME", args.gloo_socket_ifname)

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
    ]
    if mode == "cpu_q4_0":
        cmd += ["--enable-moe-offload", "--moe-offload-quant-type", "q4_0"]
    elif mode == "cache":
        if cache_size is None:
            raise ValueError("cache_size is required for cache mode")
        cmd += [
            "--enable-moe-expert-cache",
            "--moe-expert-cache-size",
            str(cache_size),
            "--moe-expert-cache-swap-per-update",
            str(args.swap_per_update),
            "--moe-expert-cache-update-interval",
            str(args.update_interval),
            "--moe-expert-cache-warmup-steps",
            str(args.warmup_steps),
        ]
    else:
        raise ValueError(f"unknown mode {mode}")

    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    log_file.close()
    return proc


def stop_server(proc: subprocess.Popen, timeout_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10.0)


def run_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    prompt = args.prompt.strip()
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
        },
    }
    rows: list[dict[str, Any]] = []
    for run_idx in range(1, args.requests + 1):
        t0 = time.perf_counter()
        resp = _http_json("POST", f"http://{args.host}:{args.port}/generate", payload, timeout=args.request_timeout)
        latency = time.perf_counter() - t0
        text = str(resp.get("text", ""))
        rows.append(
            {
                "run": run_idx,
                "latency_s": round(latency, 6),
                "throughput_tps": round(args.output_tokens / latency, 3),
                "chars": len(text),
                "sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    return rows


def parse_server_log(log_path: Path, hit_tail: int = 8) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    hit_rates = [float(x) / 100.0 for x in re.findall(r"window_hit=([0-9.]+)%", text)]
    swaps = [int(x) for x in re.findall(r"swaps=(\d+)", text)]
    backoffs = [int(x) for x in re.findall(r"backed off to (\d+) steps", text)]
    local_tps = [float(x) for x in re.findall(r"gen throughput \(token/s\): ([0-9.]+)", text)]
    active_spare = re.search(r"allocated active=(\d+) spare=(\d+) slot=([0-9.]+) MiB total=([0-9.]+) GiB", text)
    parsed: dict[str, Any] = {
        "hit_rate_samples": hit_rates,
        "hit_rate_last": hit_rates[-1] if hit_rates else None,
        "hit_rate_median_tail": statistics.median(hit_rates[-hit_tail:]) if len(hit_rates) >= hit_tail else None,
        "swap_last": swaps[-1] if swaps else None,
        "observed_backoff_steps": backoffs,
        "local_decode_tps_samples": local_tps,
        "local_decode_tps_median_tail16": statistics.median(local_tps[-16:]) if len(local_tps) >= 16 else None,
    }
    if active_spare:
        parsed["allocation"] = {
            "active": int(active_spare.group(1)),
            "spare": int(active_spare.group(2)),
            "slot_mib": float(active_spare.group(3)),
            "total_gib": float(active_spare.group(4)),
        }
    return parsed


def summarize(
    mode: str,
    cache_size: int | None,
    output_tokens: int,
    request_rows: list[dict[str, Any]],
    log_stats: dict[str, Any],
) -> dict[str, Any]:
    latencies = [float(row["latency_s"]) for row in request_rows]
    steady = latencies[1:] if len(latencies) > 1 else latencies
    hashes = sorted({row["sha256_12"] for row in request_rows})
    summary = {
        "mode": mode,
        "k": cache_size,
        "latency_s_samples": latencies,
        "latency_s_median_drop_first": statistics.median(steady) if steady else None,
        "throughput_tps_median_drop_first": (
            round(output_tokens / statistics.median(steady), 3) if steady else None
        ),
        "output_hashes": hashes,
        "output_hash_stable": len(hashes) == 1,
        "log": log_stats,
    }
    return summary


def enrich_results(
    results: list[dict[str, Any]],
    *,
    output_tokens: int,
    slot_mib: float,
    num_moe_layers: int,
    num_experts: int,
) -> dict[str, Any]:
    """Add cross-run metrics used by the thesis tables."""

    cpu = next((row for row in results if row["mode"] == "cpu_q4_0"), None)
    cpu_latency = (
        float(cpu["latency_s_median_drop_first"])
        if cpu and cpu.get("latency_s_median_drop_first")
        else None
    )
    cpu_local = None
    if cpu:
        cpu_local = cpu.get("log", {}).get("local_decode_tps_median_tail16")

    total_instances = num_moe_layers * num_experts
    all_hash_sets = [
        tuple(row.get("output_hashes", []))
        for row in results
        if row.get("output_hashes")
    ]
    if len(all_hash_sets) >= 2:
        exact_hash_match = len(set(all_hash_sets)) == 1
    else:
        exact_hash_match = None

    for row in results:
        latency = row.get("latency_s_median_drop_first")
        if latency:
            row["throughput_tps_median_drop_first"] = round(output_tokens / float(latency), 3)
        if cpu_latency and latency and row["mode"] != "cpu_q4_0":
            row["speedup_vs_cpu_e2e"] = round(cpu_latency / float(latency), 4)

        local_tps = row.get("log", {}).get("local_decode_tps_median_tail16")
        if cpu_local and local_tps and row["mode"] != "cpu_q4_0":
            row["speedup_vs_cpu_local_decode"] = round(float(local_tps) / float(cpu_local), 4)

        cache_size = row.get("k")
        if cache_size:
            row["expert_instance_fraction"] = round(float(cache_size) / total_instances, 10)
            allocation = row.get("log", {}).get("allocation") or {}
            physical_slots = int(allocation.get("active", cache_size)) + int(allocation.get("spare", 0))
            row["physical_cache_gib"] = round(physical_slots * slot_mib / 1024.0, 10)

    return {
        "exact_text_hash_match_across_modes": exact_hash_match,
        "reference_output_hashes": list(all_hash_sets[0]) if all_hash_sets else [],
        "cpu_latency_s_median_drop_first": cpu_latency,
        "cpu_local_decode_tps_median_tail16": cpu_local,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32")
    parser.add_argument("--result-dir", default="expert_cache_prompt_results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--device", default=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0"))
    parser.add_argument("--nanovllm-tp-size", type=int, default=int(os.environ.get("NANOVLLM_TP_SIZE", "2")))
    parser.add_argument("--gloo-socket-ifname", default=os.environ.get("GLOO_SOCKET_IFNAME", "lo"))
    parser.add_argument("--cache-sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU Q4_0 baseline.")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=320)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-moe-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--slot-mib", type=float, default=2.53125)
    parser.add_argument("--swap-per-update", type=int, default=8)
    parser.add_argument("--update-interval", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"

    plan: list[tuple[str, int | None]] = []
    if not args.skip_cpu:
        plan.append(("cpu_q4_0", None))
    plan.extend(("cache", k) for k in args.cache_sizes)

    all_results: list[dict[str, Any]] = []
    for mode, cache_size in plan:
        name = mode if cache_size is None else f"cache_k{cache_size}"
        log_path = result_dir / f"{name}.log"
        print(f"==> launching {name}", flush=True)
        proc = launch_server(args, mode, cache_size, log_path)
        try:
            wait_ready(proc, base_url, args.ready_timeout, log_path)
            rows = run_requests(args)
            log_stats = parse_server_log(log_path)
            summary = summarize(mode, cache_size, args.output_tokens, rows, log_stats)
            summary["request_rows"] = rows
            (result_dir / f"{name}.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            all_results.append(summary)
        finally:
            stop_server(proc)

    derived = enrich_results(
        all_results,
        output_tokens=args.output_tokens,
        slot_mib=args.slot_mib,
        num_moe_layers=args.num_moe_layers,
        num_experts=args.num_experts,
    )
    output = {
        "model": args.model_path,
        "policy": "global_lfu_expert_pool",
        "requests": args.requests,
        "output_tokens": args.output_tokens,
        "prompt_chars": len(args.prompt.strip()),
        "num_moe_layers": args.num_moe_layers,
        "num_experts": args.num_experts,
        "slot_mib": args.slot_mib,
        "derived": derived,
        "results": all_results,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
