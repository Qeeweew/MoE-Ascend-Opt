# benchmark_fused_moe_int4.py
import argparse
import json
import os
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import nanovllm_ext

# =============================================================================
# 1. Test Parameters
# =============================================================================
# Qwen3-30B-A3B-Instruct-2507 MoE dimensions.
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
TOP_K = 8

RENORMALIZE = True

MIN_RUN_TIME_S = 2.0
NUM_WEIGHT_SETS = 4
WARMUP_RUNS = 5          # Warmup iterations
BENCHMARK_RUNS = 50      # Number of benchmark iterations to average
TOKEN_COUNTS_TO_TEST = list(range(1, 9)) + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Q4_0 block size
QK4_0 = 32

assert HIDDEN_SIZE % 32 == 0
assert INTERMEDIATE_SIZE % 32 == 0

torch.set_grad_enabled(False)
torch.set_num_interop_threads(1)


# =============================================================================
# 2. Q4_0 Weight Generation Helpers (from fused_moe_cpu_int4.py)
# =============================================================================
def pack_int4_to_uint32(int4_values):
    """
    Pack 8 int4 values into a uint32.
    Packing format: [q0, q4, q1, q5, q2, q6, q3, q7] (interleaved)
    Each value is 4 bits, stored as uint4 (0-15).

    For the kernel format [q0, q4, q1, q5, q2, q6, q3, q7]:
    - Bits 0-3: q0
    - Bits 4-7: q4
    - Bits 8-11: q1
    - Bits 12-15: q5
    - Bits 16-19: q2
    - Bits 20-23: q6
    - Bits 24-27: q3
    - Bits 28-31: q7
    """
    assert len(int4_values) == 8
    # Convert to uint4 (0-15 range)
    q = [(v & 0xF) for v in int4_values]
    packed = (q[0] << 0) | (q[2] << 8) | (q[4] << 16) | (q[6] << 24) | \
             (q[1] << 4) | (q[3] << 12) | (q[5] << 20) | (q[7] << 28)
    return np.uint32(packed)


def uint4_to_int4_vec(values):
    """Vectorized conversion: uint4 (0-15) with zero_point=8 -> int4 (-8 to 7)."""
    # zero_point = 8, so: 0->-8, 8->0, 15->7
    return (values - 8).astype(np.int8)


def unpack_uint32_to_int4_vec(packed_array):
    """
    Vectorized unpack uint32 array to int4 values.
    Input: packed_array of shape (..., N) uint32
    Output: int4 values of shape (..., N*8) int8
    """
    # Extract 8 nibbles per uint32 using bit operations
    # Format: [q0, q4, q1, q5, q2, q6, q3, q7]
    q0 = (packed_array >> 0) & 0xF
    q1 = (packed_array >> 4) & 0xF
    q2 = (packed_array >> 8) & 0xF
    q3 = (packed_array >> 12) & 0xF
    q4 = (packed_array >> 16) & 0xF
    q5 = (packed_array >> 20) & 0xF
    q6 = (packed_array >> 24) & 0xF
    q7 = (packed_array >> 28) & 0xF

    # Stack and convert to int4 in correct order [q0, q1, q2, q3, q4, q5, q6, q7]
    stacked = np.stack([q0, q1, q2, q3, q4, q5, q6, q7], axis=-1)
    return uint4_to_int4_vec(stacked)


def generate_random_q4_0_weight(rows, cols):
    """
    Generate random Q4_0 quantized weight matrix using vectorized operations.
    Returns:
        qs: uint32 tensor of shape [rows, cols//8] (packed int4)
        d: float16 tensor of shape [rows, cols//32] (scales, one per 32 elements)
    """
    assert cols % 32 == 0, "cols must be multiple of 32"

    num_blocks_per_row = cols // QK4_0  # Number of 32-element blocks per row
    num_uint32_per_block = QK4_0 // 8    # 4 uint32 values per 32-element block

    # Generate random scales (one per QK4_0=32 elements)
    d = torch.rand(rows, num_blocks_per_row, dtype=torch.float32) * 1.0 / 8 * np.sqrt(1.0 / cols)

    # Total number of uint32 values needed
    total_uint32 = rows * num_blocks_per_row * num_uint32_per_block

    # Generate random uint32 values directly (each contains 8 random int4 values)
    qs_np = np.random.randint(0, 2**32, size=total_uint32, dtype=np.uint32)

    # Reshape to [rows, num_blocks_per_row, num_uint32_per_block]
    qs_3d = qs_np.reshape(rows, num_blocks_per_row, num_uint32_per_block)

    # Reshape qs to [rows, cols//8] for storage
    qs = torch.from_numpy(qs_np.reshape(rows, cols // 8)).contiguous()
    d = d.to(torch.float16).contiguous()

    return qs, d


def create_dummy_q4_0_weights():
    """
    Create Q4_0 quantized weights for all experts.
    gate_up: [E, 2I, H/8] uint32 (qs), [E, 2I, H/32] fp16 (d)
    down:   [E, H, I/8] uint32 (qs), [E, H, I/32] fp16 (d)
    """
    w_gate_qs_list = []
    w_gate_d_list = []
    w_up_qs_list = []
    w_up_d_list = []
    w_down_qs_list = []
    w_down_d_list = []

    for e in range(NUM_EXPERTS):
        # gate_proj: [INTERMEDIATE_SIZE, HIDDEN_SIZE]
        qs_g, d_g = generate_random_q4_0_weight(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        w_gate_qs_list.append(qs_g)
        w_gate_d_list.append(d_g)

        # up_proj: [INTERMEDIATE_SIZE, HIDDEN_SIZE]
        qs_u, d_u = generate_random_q4_0_weight(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        w_up_qs_list.append(qs_u)
        w_up_d_list.append(d_u)

        # down_proj: [HIDDEN_SIZE, INTERMEDIATE_SIZE]
        qs_d, d_d = generate_random_q4_0_weight(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        w_down_qs_list.append(qs_d)
        w_down_d_list.append(d_d)

    # Stack to create [E, ...] tensors
    w_gate_qs = torch.stack(w_gate_qs_list, dim=0)  # [E, I, H/8] uint32
    w_gate_d = torch.stack(w_gate_d_list, dim=0)    # [E, I, H/32] fp16
    w_up_qs = torch.stack(w_up_qs_list, dim=0)
    w_up_d = torch.stack(w_up_d_list, dim=0)
    w_down_qs = torch.stack(w_down_qs_list, dim=0)
    w_down_d = torch.stack(w_down_d_list, dim=0)

    # Concatenate gate and up for storage: [E, 2*I, H/8]
    gate_up_qs = torch.cat([w_gate_qs, w_up_qs], dim=1).contiguous()
    gate_up_d = torch.cat([w_gate_d, w_up_d], dim=1).contiguous()

    return gate_up_qs, gate_up_d, w_down_qs, w_down_d


# =============================================================================
# 3. Performance Calculation Helpers
# =============================================================================
def calculate_flops_per_layer(num_tokens: int):
    """
    Calculate FLOPs for MoE layer.
    gate+up fused matmul: [T*K, H] x [H, 2I] => 2 * (T*K) * H * (2I)
    down matmul:          [T*K, I] x [I, H] => 2 * (T*K) * I * H
    """
    total_expert_tokens = num_tokens * TOP_K
    flops_gate_up = 2 * total_expert_tokens * HIDDEN_SIZE * (2 * INTERMEDIATE_SIZE)
    flops_down = 2 * total_expert_tokens * INTERMEDIATE_SIZE * HIDDEN_SIZE
    return flops_gate_up + flops_down


def bytes_per_expert_weights_q4_0():
    """
    Calculate bytes per expert for Q4_0 weights:
      gate_up_qs: (2I*H/8)*uint32 = (2I*H/8)*4 bytes
      gate_up_d : (2I*(H/32))*fp16
      down_qs   : (H*I/8)*uint32 = (H*I/8)*4 bytes
      down_d    : (H*(I/32))*fp16
    """
    sizeof_uint32 = 4
    sizeof_half = 2

    # gate_up: qs is uint32 per 8 elements, d is fp16 per 32 elements
    gate_up_qs_bytes = (2 * INTERMEDIATE_SIZE) * (HIDDEN_SIZE // 8) * sizeof_uint32
    gate_up_d_bytes = (2 * INTERMEDIATE_SIZE) * (HIDDEN_SIZE // 32) * sizeof_half

    # down: qs is uint32 per 8 elements, d is fp16 per 32 elements
    down_qs_bytes = HIDDEN_SIZE * (INTERMEDIATE_SIZE // 8) * sizeof_uint32
    down_d_bytes = HIDDEN_SIZE * (INTERMEDIATE_SIZE // 32) * sizeof_half

    return gate_up_qs_bytes + gate_up_d_bytes + down_qs_bytes + down_d_bytes


def get_weight_bytes_for_routing(selected_experts_i32: torch.Tensor):
    """
    Calculate bytes from weights for unique experts in routing.
    """
    active_experts = int(torch.unique(selected_experts_i32).numel())
    return active_experts * bytes_per_expert_weights_q4_0(), active_experts


# =============================================================================
# 4. Benchmark Runner Wrapper Class
# =============================================================================
class BenchmarkRunner:
    def __init__(self, handles):
        self.handles = handles
        self.num_handles = len(handles)
        self.iter_idx = 0

    def __call__(self, hidden_states, topk_weights, topk_ids):
        h = self.handles[self.iter_idx % self.num_handles]
        result = torch.ops.nanovllm.moe_forward(hidden_states, topk_ids, topk_weights, h)
        self.iter_idx += 1
        return result

    def reset(self):
        self.iter_idx = 0

    def get_last_time_ms(self):
        """Get the last run time from the most recently used handle."""
        h = self.handles[(self.iter_idx - 1) % self.num_handles]
        return h.get_last_run_time_ms()


# =============================================================================
# 5. Main Program
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKEN_COUNTS_TO_TEST)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--weight-sets", type=int, default=NUM_WEIGHT_SETS)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    print("=" * 80)
    print("MoE CPU INT4 (Q4_0) Performance Test (nanovllm_ext)")
    print("=" * 80)
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not Set')}")
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"Params: H={HIDDEN_SIZE}, I={INTERMEDIATE_SIZE}, E={NUM_EXPERTS}, K={TOP_K}")
    print(f"Weight sets: {args.weight_sets}, Benchmark runs: {args.runs}")

    # ---- init handles & load weights ----
    print("\nInitializing handles & loading Q4_0 quantized weights...")
    moe_handles = []
    for i in range(args.weight_sets):
        # quant_type=1 for Q4_0
        print("  Generating weights for set", i + 1)
        handle = torch.classes.nanovllm.MoEInfer(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, 1)
        gate_up_qs, gate_up_d, down_qs, down_d = create_dummy_q4_0_weights()

        handle.store_quantized_repack(gate_up_qs, gate_up_d, down_qs, down_d)
        moe_handles.append(handle)

    print(f"Setup finished. Quant type: {moe_handles[0].get_quant_type()}")
    print(f"Weight bytes per expert: {bytes_per_expert_weights_q4_0()}")

    runner = BenchmarkRunner(moe_handles)

    # ---- run benchmark ----
    print("\n--- Performance Results (using get_last_run_time_ms()) ---")
    header = (
        f"{'Num Tokens':<12} "
        f"{'ActiveE':<8} "
        f"{'Avg Time (ms)':<16} "
        f"{'Weight BW (GB/s)':<18} "
        f"{'Compute (GFLOPS)':<18}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for num_tokens in args.tokens:
        # inputs on CPU
        hidden_states = torch.randn((num_tokens, HIDDEN_SIZE), dtype=torch.float16).contiguous()

        # Create routing with uniform distribution across experts
        ids = torch.empty((num_tokens, TOP_K), dtype=torch.int32)
        base = torch.arange(num_tokens, dtype=torch.int64) * TOP_K
        for k in range(TOP_K):
            ids[:, k] = ((base + k) % NUM_EXPERTS).to(torch.int32)
        ids = ids.contiguous()

        # routing weights: float32 [T, K]
        topk_w = torch.full((num_tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32).contiguous()

        # bytes only from weights (unique experts)
        weight_bytes, active_e = get_weight_bytes_for_routing(ids)

        # Warmup runs
        for _ in range(args.warmup):
            runner(hidden_states, topk_w, ids)

        # Benchmark runs with weight cycling
        runner.reset()
        samples_ms = []
        for i in range(args.runs):
            runner(hidden_states, topk_w, ids)
            samples_ms.append(runner.get_last_time_ms())

        avg_time_ms = float(np.mean(samples_ms))
        p50_time_ms = float(np.median(samples_ms))
        p95_time_ms = float(np.percentile(samples_ms, 95))
        avg_time_s = avg_time_ms / 1000.0

        # throughput metrics
        bw_gb_s = (weight_bytes / 1e9) / avg_time_s
        gflops = (calculate_flops_per_layer(num_tokens) / 1e9) / avg_time_s

        print(f"{num_tokens:<12} {active_e:<8} {avg_time_ms:<16.4f} {bw_gb_s:<18.2f} {gflops:<18.2f}")
        results.append({
            "num_tokens": num_tokens,
            "active_experts": active_e,
            "mean_ms": avg_time_ms,
            "p50_ms": p50_time_ms,
            "p95_ms": p95_time_ms,
            "weight_bandwidth_gb_s": bw_gb_s,
            "compute_gflops": gflops,
        })

    print("-" * len(header))
    print("Test complete.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "cpu_fused_moe_q4_0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_shape": {
                "source": "Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32",
                "hidden_size": HIDDEN_SIZE,
                "moe_intermediate_size": INTERMEDIATE_SIZE,
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "group_size": QK4_0,
            },
            "environment": {
                "nanovllm_tp_size": os.environ.get("NANOVLLM_TP_SIZE"),
                "nanovllm_tp_threads_per_node": os.environ.get("NANOVLLM_TP_THREADS_PER_NODE"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "torch_version": torch.__version__,
            },
            "settings": {
                "warmup": args.warmup,
                "runs": args.runs,
                "weight_sets": args.weight_sets,
            },
            "results": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON result: {args.json_out}")


if __name__ == "__main__":
    main()
