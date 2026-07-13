# bench_moe_cpu_bandwidth.py
import argparse
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.utils.benchmark as benchmark
import nanovllm_ext
import time

# =============================================================================
# 1. 测试参数定义
# =============================================================================
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
TOP_K = 8

RENORMALIZE = True  # 此测试只用均匀权重，不影响性能；保留参数语义

MIN_RUN_TIME_S = 2.0
NUM_WEIGHT_SETS = 4
WARMUP_RUNS = 5          # Warmup iterations
BENCHMARK_RUNS = 50      # Number of benchmark iterations to average
TOKEN_COUNTS_TO_TEST = list(range(1, 9)) + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

assert HIDDEN_SIZE % 32 == 0
assert INTERMEDIATE_SIZE % 32 == 0

torch.set_grad_enabled(False)

# =============================================================================
# 2. 权重创建和性能计算辅助函数
# =============================================================================
def create_dummy_quantized_weights_rowmajor():
    """
    创建 row-major 的“伪量化权重”(qs, d)，交给 C++ 侧 repack。
    gate_up: [E, 2I, H]
      qs: int8
      d : fp16, [E, 2I, H/32]
    down:   [E, H, I]
      qs: int8
      d : fp16, [E, H, I/32]
    """
    k_blocks_gate_up = HIDDEN_SIZE // 32
    k_blocks_down = INTERMEDIATE_SIZE // 32

    gate_up_qs = torch.randint(
        -128, 127, (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=torch.int8
    )
    gate_up_d = torch.randn(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, k_blocks_gate_up),
        dtype=torch.float16
    )

    down_qs = torch.randint(
        -128, 127, (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.int8
    )
    down_d = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE, k_blocks_down),
        dtype=torch.float16
    )
    return gate_up_qs.contiguous(), gate_up_d.contiguous(), down_qs.contiguous(), down_d.contiguous()


def calculate_flops_per_layer(num_tokens: int):
    """
    按 routed token 数 = num_tokens * TOP_K（每个 token 都路由到 TOP_K 个 expert）
    gate+up fused matmul: [T*K, H] x [H, 2I] => 2 * (T*K) * H * (2I)
    down matmul:          [T*K, I] x [I, H] => 2 * (T*K) * I * H
    """
    total_expert_tokens = num_tokens * TOP_K
    flops_gate_up = 2 * total_expert_tokens * HIDDEN_SIZE * (2 * INTERMEDIATE_SIZE)
    flops_down = 2 * total_expert_tokens * INTERMEDIATE_SIZE * HIDDEN_SIZE
    return flops_gate_up + flops_down


def bytes_per_expert_weights():
    """
    只算权重 bytes（不含 activation）：
      gate_up_qs: (2I*H)*int8
      gate_up_d : (2I*(H/32))*fp16
      down_qs   : (H*I)*int8
      down_d    : (H*(I/32))*fp16
    """
    sizeof_half = 2
    gate_up_qs_bytes = (2 * INTERMEDIATE_SIZE) * HIDDEN_SIZE
    gate_up_d_bytes = (2 * INTERMEDIATE_SIZE) * (HIDDEN_SIZE // 32) * sizeof_half
    down_qs_bytes = HIDDEN_SIZE * INTERMEDIATE_SIZE
    down_d_bytes = HIDDEN_SIZE * (INTERMEDIATE_SIZE // 32) * sizeof_half
    return gate_up_qs_bytes + gate_up_d_bytes + down_qs_bytes + down_d_bytes


def get_weight_bytes_for_routing(selected_experts_i32: torch.Tensor):
    """
    更准确：从 routing 里统计本次实际访问到多少个 unique expert
    """
    active_experts = int(torch.unique(selected_experts_i32).numel())
    return active_experts * bytes_per_expert_weights(), active_experts


# =============================================================================
# 3. Benchmark Runner Wrapper Class
# =============================================================================
class BenchmarkRunner:
    def __init__(self, handles):
        self.handles = handles
        self.num_handles = len(handles)
        self.iter_idx = 0

    def __call__(self, hidden_states, topk_weights, topk_ids):
        h = self.handles[self.iter_idx % self.num_handles]
        # 执行 forward（返回新 tensor，但 benchmark stmt 会丢弃返回值）
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
# 4. 主程序
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
    print("MoE CPU fused performance test (nanovllm_ext) - weight bandwidth only")
    print("=" * 80)
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not Set')}")
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"Params: H={HIDDEN_SIZE}, I={INTERMEDIATE_SIZE}, E={NUM_EXPERTS}, K={TOP_K}")
    print(f"Weight sets: {args.weight_sets}, benchmark runs={args.runs}")

    # ---- init handles & load weights ----
    print("\nInitializing handles & loading dummy quantized weights (row-major -> repack in C++) ...")
    moe_handles = []
    for i in range(args.weight_sets):
        handle = torch.classes.nanovllm.MoEInfer(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0)
        gate_up_qs, gate_up_d, down_qs, down_d = create_dummy_quantized_weights_rowmajor()

        handle.store_quantized_repack(gate_up_qs, gate_up_d, down_qs, down_d)

        moe_handles.append(handle)
    print("Setup finished.\n")

    runner = BenchmarkRunner(moe_handles)

    # ---- run benchmark ----
    print("--- Performance Results (using get_last_run_time_ms()) ---")
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

        # 构造一个"尽量均匀覆盖 expert"的 routing，避免 active_experts 偏小
        # shape: [T, K]
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

        samples = torch.tensor(samples_ms, dtype=torch.float64)
        avg_time_ms = float(samples.mean())
        p50_time_ms = float(samples.median())
        p95_time_ms = float(torch.quantile(samples, 0.95))
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
            "benchmark": "cpu_fused_moe_q8_0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_shape": {
                "source": "Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32",
                "hidden_size": HIDDEN_SIZE,
                "moe_intermediate_size": INTERMEDIATE_SIZE,
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "group_size": 32,
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
