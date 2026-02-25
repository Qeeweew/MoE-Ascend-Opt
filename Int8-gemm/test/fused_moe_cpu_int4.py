import os
import math
import time
import numpy as np
import torch

import nanovllm_ext

HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
TOP_K = 8
RENORMALIZE = True

# Q4_0 block size
QK4_0 = 32

torch.set_num_threads(min(32, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


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
        fp16: float16 tensor of shape [rows, cols] (reference FP16 values)
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

    # Vectorized unpack to get int4 values and convert to fp16 reference
    # Unpack each uint32 to 8 int4 values
    int4_vals = unpack_uint32_to_int4_vec(qs_3d)  # [rows, num_blocks, 4, 8] -> [rows, num_blocks, 32]
    int4_vals = int4_vals.reshape(rows, cols)

    # Convert to fp16: multiply by corresponding scale
    # Scale is per 32 elements: d[row, block_idx] applies to elements [block_idx*32 : (block_idx+1)*32]
    d_np = d.numpy()
    scales_expanded = np.repeat(d_np, QK4_0, axis=1)  # [rows, cols]
    fp16_vals = (int4_vals * scales_expanded).astype(np.float16)

    # Reshape qs to [rows, cols//8] for storage
    qs = torch.from_numpy(qs_np.reshape(rows, cols // 8)).contiguous()
    d = d.to(torch.float16).contiguous()
    fp16 = torch.from_numpy(fp16_vals).contiguous()

    return qs, d, fp16


def make_topk_from_logits_cpu(logits_fp32: torch.Tensor, top_k: int, renormalize: bool):
    """
    logits_fp32: [T, E] float32 on CPU
    Returns:
        topk_ids: [T, K] int32
        topk_w:  [T, K] float32
    """
    topk_val, topk_ids = torch.topk(logits_fp32, k=top_k, dim=-1, largest=True, sorted=True)
    if renormalize:
        topk_w = torch.softmax(topk_val, dim=-1)
    else:
        w_all = torch.softmax(logits_fp32, dim=-1)
        topk_w = torch.gather(w_all, dim=-1, index=topk_ids)

    return topk_ids.to(torch.int32).contiguous(), topk_w.to(torch.float32).contiguous()


def reference_moe_fp16(
    x_fp16: torch.Tensor,
    topk_ids_i32: torch.Tensor,
    topk_w_f32: torch.Tensor,
    w_gate_fp16: torch.Tensor,
    w_up_fp16: torch.Tensor,
    w_down_fp16: torch.Tensor,
):
    """
    FP16 reference implementation.
    """
    T, H = x_fp16.shape
    K = topk_ids_i32.shape[1]

    out_acc = torch.zeros((T, H), dtype=torch.float32)
    x_fp16 = x_fp16.contiguous()

    for k in range(K):
        ids_k = topk_ids_i32[:, k]
        w_k = topk_w_f32[:, k]

        for e in range(NUM_EXPERTS):
            mask = (ids_k == e)
            if not mask.any():
                continue
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

            x_e = x_fp16.index_select(0, idx)

            gate = (x_e @ w_gate_fp16[e].t()).to(torch.float16)
            up = (x_e @ w_up_fp16[e].t()).to(torch.float16)

            act = torch.nn.functional.silu(gate) * up

            y_e = (act @ w_down_fp16[e].t()).to(torch.float16)

            out_acc.index_add_(0, idx, y_e.to(torch.float32) * w_k[idx].unsqueeze(1))

    return out_acc.to(torch.float16).contiguous()


@torch.no_grad()
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("=== Testing INT4 (Q4_0) MoE ===")

    # Generate weights using Q4_0 format
    print("Generating Q4_0 weights...")

    w_gate_qs_list = []
    w_gate_d_list = []
    w_gate_fp16_list = []

    w_up_qs_list = []
    w_up_d_list = []
    w_up_fp16_list = []

    w_down_qs_list = []
    w_down_d_list = []
    w_down_fp16_list = []

    for e in range(NUM_EXPERTS):
        # gate_proj: [INTERMEDIATE_SIZE, HIDDEN_SIZE]
        qs_g, d_g, fp16_g = generate_random_q4_0_weight(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        w_gate_qs_list.append(qs_g)
        w_gate_d_list.append(d_g)
        w_gate_fp16_list.append(fp16_g)

        # up_proj: [INTERMEDIATE_SIZE, HIDDEN_SIZE]
        qs_u, d_u, fp16_u = generate_random_q4_0_weight(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        w_up_qs_list.append(qs_u)
        w_up_d_list.append(d_u)
        w_up_fp16_list.append(fp16_u)

        # down_proj: [HIDDEN_SIZE, INTERMEDIATE_SIZE]
        qs_d, d_d, fp16_d = generate_random_q4_0_weight(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        w_down_qs_list.append(qs_d)
        w_down_d_list.append(d_d)
        w_down_fp16_list.append(fp16_d)

    # Stack to create [E, ...] tensors
    w_gate_qs = torch.stack(w_gate_qs_list, dim=0)  # [E, I, H/8] uint32
    w_gate_d = torch.stack(w_gate_d_list, dim=0)    # [E, I, H/32] fp16
    w_gate_fp16 = torch.stack(w_gate_fp16_list, dim=0)  # [E, I, H] fp16

    w_up_qs = torch.stack(w_up_qs_list, dim=0)
    w_up_d = torch.stack(w_up_d_list, dim=0)
    w_up_fp16 = torch.stack(w_up_fp16_list, dim=0)

    w_down_qs = torch.stack(w_down_qs_list, dim=0)
    w_down_d = torch.stack(w_down_d_list, dim=0)
    w_down_fp16 = torch.stack(w_down_fp16_list, dim=0)

    # Concatenate gate and up for storage: [E, 2*I, H/8]
    gate_up_qs = torch.cat([w_gate_qs, w_up_qs], dim=1).contiguous()
    gate_up_d = torch.cat([w_gate_d, w_up_d], dim=1).contiguous()

    print(f"gate_up_qs shape: {gate_up_qs.shape}, dtype: {gate_up_qs.dtype}")
    print(f"gate_up_d shape: {gate_up_d.shape}, dtype: {gate_up_d.dtype}")
    print(f"down_proj_qs shape: {w_down_qs.shape}, dtype: {w_down_qs.dtype}")
    print(f"down_proj_d shape: {w_down_d.shape}, dtype: {w_down_d.dtype}")

    # Create Q4_0 MoEInfer handle (quant_type=1)
    handle = torch.classes.nanovllm.MoEInfer(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, 1)
    print(f"Quant type: {handle.get_quant_type()}")

    # Store pre-quantized weights
    t0 = time.time()
    handle.store_quantized_repack(gate_up_qs, gate_up_d, w_down_qs, w_down_d)
    t1 = time.time()
    print(f"[store_quantized_repack] done in {(t1 - t0):.3f}s")

    # Test multiple token sizes
    token_sizes = [1, 8, 32, 128, 256]

    for T in token_sizes:
        print(f"\n=== Test T={T} ===")

        # Input hidden
        x = torch.randn(T, HIDDEN_SIZE, dtype=torch.float16).contiguous()

        # Router logits
        logits = torch.randn(T, NUM_EXPERTS, dtype=torch.float32).contiguous()
        topk_ids, topk_w = make_topk_from_logits_cpu(logits, TOP_K, RENORMALIZE)

        # Run extension CPU routed
        t0 = time.time()
        y_ext = torch.ops.nanovllm.moe_forward(x, topk_ids, topk_w, handle)
        t1 = time.time()

        # Run FP16 reference
        t2 = time.time()
        y_ref = reference_moe_fp16(x, topk_ids, topk_w, w_gate_fp16, w_up_fp16, w_down_fp16)
        t3 = time.time()

        # Error metrics
        diff = (y_ext - y_ref).to(torch.float32)
        ref_f = y_ref.to(torch.float32)

        abs_max = diff.abs().max().item()
        abs_mean = diff.abs().mean().item()
        rmse = torch.sqrt((diff * diff).mean()).item()
        rel_rmse = rmse / (torch.sqrt((ref_f * ref_f).mean()).item() + 1e-12)

        print(f"[time] ext={(t1 - t0)*1000:.2f} ms, ref={(t3 - t2)*1000:.2f} ms")
        print(f"[err ] abs_max={abs_max:.6f}, abs_mean={abs_mean:.6f}, rmse={rmse:.6f}, rel_rmse={rel_rmse:.6f}")

        assert y_ext.shape == y_ref.shape
        assert y_ext.dtype == torch.float16

        # Sanity check: relative RMSE should be small (allowing for some quantization error)
        if rel_rmse > 0.1:
            print(f"WARNING: rel_rmse {rel_rmse:.6f} is higher than expected!")

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()
