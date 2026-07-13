"""Benchmark: W4A16 linear kernel vs official op vs torch.bmm.

Compares three kernels for W4A16 grouped linear (int4 symmetric, group_size=32):

  * ``torch.bmm`` (bfloat16)                 - dense reference
  * ``torch.ops.npu.npu_weight_quant_batchmatmul`` - official NPU W4A16 op
  * ``torch.ops.moe_ascend_npu.batch_gemm_w4a16_small_bs`` - our custom kernel

Each graph captures NUM_WEIGHT_COPIES kernel executions with different weights
(to avoid cache effects), and is timed with NPU events.

Migrated from sgl-kernel-npu/tests/python/sgl_kernel_npu/benchmark_w4a16_linear.py;
the only change is the custom kernel namespace (``npu`` -> ``moe_ascend_npu``).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_npu

try:
    import sgl_kernel_npu  # noqa: F401  (registers npu.* ops used by the reference path)
except ImportError:
    print("Warning: sgl_kernel_npu not found.")
    sys.exit(1)

from moe_ascend_npu.kernels import ensure_kernels_loaded

ensure_kernels_loaded()

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
device = torch.device("npu:0")
dtype = torch.bfloat16  # Native bmm uses bfloat16

# Model dimensions (typical for MoE models)
HIDDEN_SIZE = 8192
INTERMEDIATE_SIZE = 8192
GROUP_SIZE = 32

# Number of weight copies to avoid cache effects in graph mode
NUM_WEIGHT_COPIES = 4
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# ------------------------------------------------------------
# Utils: unpack int4 from int32
# ------------------------------------------------------------
def unpack_from_int32(
    weight: torch.Tensor,
    shape: torch.Size,
    num_bits: int,
    packed_dim: int = 1,
) -> torch.Tensor:
    """Unpacks quantized weights from int32 format back to original bits."""
    assert weight.dtype == torch.int32
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked_weight = torch.zeros(
            (weight.shape[0], weight.shape[1] * pack_factor),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[1])
        unpacked_weight = unpacked_weight[:, :original_row_size]
    else:
        unpacked_weight = torch.zeros(
            (weight.shape[0] * pack_factor, weight.shape[1]),
            device=weight.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked_weight[i::pack_factor, :] = (weight >> (num_bits * i)) & mask
        original_row_size = int(shape[0])
        unpacked_weight = unpacked_weight[:original_row_size, :]

    offset = pow(2, num_bits) // 2
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)
    return unpacked_weight


# ------------------------------------------------------------
# Create quantized weights
# ------------------------------------------------------------
def create_quantized_weights(
    output_size: int,
    input_size: int,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create quantized int4 weights for NPU.

    Returns:
        Tuple of (weight_packed, weight_scale, weight_offset, weight_bf16_T)
        - weight_packed: [input_size, output_size // 8]
        - weight_scale: [num_groups, output_size]
        - weight_offset: [num_groups, output_size] (zeros for symmetric)
        - weight_bf16_T: [input_size, output_size] (transposed for bmm, contiguous)
    """
    num_groups = input_size // group_size
    pack_factor = 8

    # Generate random int4 weights [output_size, input_size]
    weight_int4 = torch.randint(-8, 8, (output_size, input_size), dtype=torch.int32, device="cpu")

    # Compute per-group scales [num_groups, output_size]
    weight_scale = torch.zeros((num_groups, output_size), dtype=dtype, device="cpu")
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size
        group_weight = weight_int4[:, start_idx:end_idx].to(torch.float32)
        max_val = group_weight.abs().max(dim=1)[0]
        scale = max_val / 7.0
        weight_scale[g, :] = scale.to(dtype)

    # Quantize
    weight_quantized = torch.zeros_like(weight_int4)
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size
        scale = weight_scale[g, :].to(torch.float32).unsqueeze(1)
        group_weight = weight_int4[:, start_idx:end_idx].to(torch.float32)
        quantized = torch.clamp(torch.round(group_weight / scale), -8, 7).to(torch.int32)
        weight_quantized[:, start_idx:end_idx] = quantized

    # Create bf16 reference weight (transposed for bmm: [input_size, output_size])
    weight_bf16_ref = torch.zeros((output_size, input_size), dtype=dtype, device="cpu")
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size
        scale = weight_scale[g, :].unsqueeze(1).to(dtype)
        quantized = weight_quantized[:, start_idx:end_idx].to(dtype)
        weight_bf16_ref[:, start_idx:end_idx] = quantized * scale

    # Transpose and make contiguous for bmm: [input_size, output_size]
    weight_bf16_T = weight_bf16_ref.transpose(0, 1).contiguous().to(device)

    # Pack to int32 [output_size, input_size // 8]
    weight_packed_cpu = torch.zeros((output_size, input_size // pack_factor), dtype=torch.int32)
    for i in range(pack_factor):
        weight_packed_cpu += weight_quantized[:, i::pack_factor] << (4 * i)

    # Move to NPU, unpack, transpose, repack
    weight_packed_npu = weight_packed_cpu.to(device)
    unpacked = unpack_from_int32(weight_packed_npu, torch.Size([output_size, input_size]), 4, packed_dim=1)
    unpacked_T = unpacked.transpose(0, 1).contiguous().int()
    weight_packed_final = torch.ops.npu.npu_convert_weight_to_int4pack(unpacked_T)
    # weight_packed_final: [input_size, output_size // 8]

    weight_scale_npu = weight_scale.to(device)
    weight_offset_npu = torch.zeros_like(weight_scale_npu)

    return weight_packed_final, weight_scale_npu, weight_offset_npu, weight_bf16_T


# ------------------------------------------------------------
# Create multiple quantized weights
# ------------------------------------------------------------
def create_multiple_quantized_weights(
    output_size: int,
    input_size: int,
    group_size: int = 32,
    num_copies: int = 10,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Create multiple copies of quantized int4 weights for NPU.
    This ensures weight data isn't cached during graph replay.
    """
    weight_packed_list = []
    weight_scale_list = []
    weight_offset_list = []
    weight_bf16_T_list = []

    for _ in range(num_copies):
        packed, scale, offset, bf16_T = create_quantized_weights(output_size, input_size, group_size)
        weight_packed_list.append(packed)
        weight_scale_list.append(scale)
        weight_offset_list.append(offset)
        weight_bf16_T_list.append(bf16_T)

    return weight_packed_list, weight_scale_list, weight_offset_list, weight_bf16_T_list


# ------------------------------------------------------------
# NPU timing with events
# ------------------------------------------------------------
def measure_time_with_events(fn, num_iters: int = 100) -> float:
    """Measure NPU execution time using events."""
    # Warmup
    for _ in range(10):
        fn()
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    torch.npu.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    return elapsed_ms


# ------------------------------------------------------------
# Bytes accounting (weight traffic only)
# ------------------------------------------------------------
def bytes_weight_only(output_size, input_size, group_size=32):
    """Calculate weight bytes only for bandwidth calculation."""
    # BF16 weight: [output_size, input_size]
    bytes_bf16 = output_size * input_size * 2

    # W4A16: packed int4 + bf16 scale
    bytes_w4a16 = (
        output_size * input_size // 2 +  # packed int4: 4 bits per element
        output_size * (input_size // group_size) * 2  # bf16 scale per group
    )

    return bytes_bf16, bytes_w4a16


# ------------------------------------------------------------
# Benchmark per batch size
# ------------------------------------------------------------
def benchmark_bs(batch_size, input_size, output_size, weight_bundle, group_size=32):
    """Benchmark npu_weight_quant_batchmatmul and batch_gemm_w4a16_small_bs vs torch.bmm.

    Each graph captures NUM_WEIGHT_COPIES kernel executions with different weights
    to simulate real forward pass memory traffic.
    """

    # Reuse the same cold-weight pool across batch sizes. Recreating 8192x8192
    # weights for every batch size measures Python setup rather than kernels.
    weight_packed_list, weight_scale_list, weight_offset_list, weight_bf16_T_list = weight_bundle

    # Input tensor [batch_size, input_size]
    x = torch.randn((batch_size, input_size), device=device, dtype=dtype)

    # Build functions that run all weights in sequence
    def make_bmm_all_fn():
        def fn():
            for i in range(NUM_WEIGHT_COPIES):
                w = weight_bf16_T_list[i]
                x_3d = x.unsqueeze(1)
                weight_3d = w.unsqueeze(0)
                _ = torch.bmm(x_3d, weight_3d)
        return fn

    def make_w4a16_all_fn():
        def fn():
            for i in range(NUM_WEIGHT_COPIES):
                wp = weight_packed_list[i]
                ws = weight_scale_list[i]
                wo = weight_offset_list[i]
                _ = torch.ops.npu.npu_weight_quant_batchmatmul(
                    x=x, weight=wp, antiquant_scale=ws,
                    antiquant_offset=wo, antiquant_group_size=group_size, bias=None)
        return fn

    def make_custom_all_fn():
        def fn():
            for i in range(NUM_WEIGHT_COPIES):
                wp = weight_packed_list[i]
                ws = weight_scale_list[i]
                _ = torch.ops.moe_ascend_npu.batch_gemm_w4a16_small_bs(x, wp, ws)
        return fn

    fn_bmm_all = make_bmm_all_fn()
    fn_w4a16_all = make_w4a16_all_fn()
    fn_custom_all = make_custom_all_fn()

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        fn_bmm_all()
    torch.npu.synchronize()

    for _ in range(WARMUP_ITERATIONS):
        fn_w4a16_all()
    torch.npu.synchronize()

    for _ in range(WARMUP_ITERATIONS):
        fn_custom_all()
    torch.npu.synchronize()

    # Capture graphs - each graph contains NUM_WEIGHT_COPIES kernel executions
    g_bmm = torch.npu.NPUGraph()
    with torch.npu.graph(g_bmm):
        fn_bmm_all()
    torch.npu.synchronize()

    g_w4a16 = torch.npu.NPUGraph()
    with torch.npu.graph(g_w4a16):
        fn_w4a16_all()
    torch.npu.synchronize()

    g_custom = torch.npu.NPUGraph()
    with torch.npu.graph(g_custom):
        fn_custom_all()
    torch.npu.synchronize()

    # Event-based timing
    def run_bmm_graph():
        g_bmm.replay()

    def run_w4a16_graph():
        g_w4a16.replay()

    def run_custom_graph():
        g_custom.replay()

    # Time per graph replay (which runs NUM_WEIGHT_COPIES kernels)
    t_bmm_ms_per_graph = measure_time_with_events(run_bmm_graph, BENCHMARK_ITERATIONS)
    t_w4a16_ms_per_graph = measure_time_with_events(run_w4a16_graph, BENCHMARK_ITERATIONS)
    t_custom_ms_per_graph = measure_time_with_events(run_custom_graph, BENCHMARK_ITERATIONS)

    # Convert to per-kernel time
    t_bmm_ms = t_bmm_ms_per_graph / NUM_WEIGHT_COPIES
    t_w4a16_ms = t_w4a16_ms_per_graph / NUM_WEIGHT_COPIES
    t_custom_ms = t_custom_ms_per_graph / NUM_WEIGHT_COPIES

    # Bytes per kernel (weight only)
    bytes_bf16, bytes_w4a16 = bytes_weight_only(output_size, input_size, group_size)

    # FLOPs per kernel: 2 * M * N * K
    flops = 2 * batch_size * input_size * output_size

    return {
        "batch_size": batch_size,
        "input_size": input_size,
        "output_size": output_size,
        "bmm_us": t_bmm_ms * 1e3,  # ms to us
        "w4a16_us": t_w4a16_ms * 1e3,
        "custom_w4a16_us": t_custom_ms * 1e3,
        "bmm_GBs": bytes_bf16 / (t_bmm_ms * 1e-3) / 1e9,
        "w4a16_GBs": bytes_w4a16 / (t_w4a16_ms * 1e-3) / 1e9,
        "custom_w4a16_GBs": bytes_w4a16 / (t_custom_ms * 1e-3) / 1e9,
        "bmm_TFlops": flops / (t_bmm_ms * 1e-3) / 1e12,
        "w4a16_TFlops": flops / (t_w4a16_ms * 1e-3) / 1e12,
        "custom_w4a16_TFlops": flops / (t_custom_ms * 1e-3) / 1e12,
        "speedup_vs_bmm": t_bmm_ms / t_custom_ms,
        "speedup_vs_npu": t_w4a16_ms / t_custom_ms,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)

    print("=" * 100)
    print("Benchmark: W4A16 (custom vs npu_native) vs torch.bmm (bfloat16)")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Intermediate Size: {INTERMEDIATE_SIZE}")
    print(f"  Group Size: {GROUP_SIZE}")
    print(f"  Quantization: int4 symmetric")
    print("=" * 100)

    test_cases = [("Generic 8192x8192 Linear", HIDDEN_SIZE, INTERMEDIATE_SIZE)]

    all_results = []
    for case_name, input_size, output_size in test_cases:
        print(f"Preparing {NUM_WEIGHT_COPIES} independent 8192x8192 weight copies...")
        weight_bundle = create_multiple_quantized_weights(
            output_size, input_size, GROUP_SIZE, NUM_WEIGHT_COPIES
        )
        print(f"\n{case_name}: [{input_size}] -> [{output_size}]")
        print("-" * 120)
        print(f"{'BS':>3} | {'BMM(us)':>10} | {'NPU(us)':>10} | {'Custom(us)':>11} | "
              f"{'Speedup(vs BMM)':>15} | {'Speedup(vs NPU)':>15} | "
              f"{'Custom(GB/s)':>12} | {'Custom(TF/s)':>12}")
        print("-" * 120)

        for batch_size in args.batch_sizes:
            r = benchmark_bs(batch_size, input_size, output_size, weight_bundle, GROUP_SIZE)
            r["case"] = case_name
            all_results.append(r)
            print(
                f"{r['batch_size']:>3} | "
                f"{r['bmm_us']:>10.2f} | "
                f"{r['w4a16_us']:>10.2f} | "
                f"{r['custom_w4a16_us']:>11.2f} | "
                f"{r['speedup_vs_bmm']:>15.2f} | "
                f"{r['speedup_vs_npu']:>15.2f} | "
                f"{r['custom_w4a16_GBs']:>12.2f} | "
                f"{r['custom_w4a16_TFlops']:>12.2f}"
            )

    print("\n" + "=" * 100)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "npu_w4a16_linear_8192",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "shape": {
                "input_size": HIDDEN_SIZE,
                "output_size": INTERMEDIATE_SIZE,
                "group_size": GROUP_SIZE,
            },
            "environment": {
                "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "torch_version": torch.__version__,
                "torch_npu_version": torch_npu.__version__,
            },
            "settings": {
                "num_weight_copies": NUM_WEIGHT_COPIES,
                "warmup_iterations": WARMUP_ITERATIONS,
                "benchmark_iterations": BENCHMARK_ITERATIONS,
            },
            "results": all_results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON result: {args.json_out}")


if __name__ == "__main__":
    main()
