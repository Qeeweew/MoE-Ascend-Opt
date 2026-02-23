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

torch.set_num_threads(min(32, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


def make_topk_from_logits_cpu(logits_fp32: torch.Tensor, top_k: int, renormalize: bool):
    """
    logits_fp32: [T, E] float32 on CPU
    Returns:
        topk_ids: [T, K] int32
        topk_w:  [T, K] float32
    """
    # topk indices
    topk_val, topk_ids = torch.topk(logits_fp32, k=top_k, dim=-1, largest=True, sorted=True)
    # softmax over top-k only if renormalize=True (common MoE)
    if renormalize:
        topk_w = torch.softmax(topk_val, dim=-1)
    else:
        # softmax over all experts then gather top-k weights
        w_all = torch.softmax(logits_fp32, dim=-1)
        topk_w = torch.gather(w_all, dim=-1, index=topk_ids)

    return topk_ids.to(torch.int32).contiguous(), topk_w.to(torch.float32).contiguous()


def reference_moe_fp16(
    x_fp16: torch.Tensor,
    topk_ids_i32: torch.Tensor,
    topk_w_f32: torch.Tensor,
    w_gate_fp16: torch.Tensor,  # [E, I, H]
    w_up_fp16: torch.Tensor,    # [E, I, H]
    w_down_fp16: torch.Tensor,  # [E, H, I]
):
    """
    A straightforward fp16 reference:
      y[t] = sum_{k} w[t,k] * down(e) ( silu( gate(e)(x[t]) ) * up(e)(x[t]) )

    Compute in fp16, accumulate in fp32 for stability, then cast back to fp16 output.
    """
    T, H = x_fp16.shape
    K = topk_ids_i32.shape[1]
    assert H == HIDDEN_SIZE
    assert w_gate_fp16.shape == (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert w_up_fp16.shape == (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert w_down_fp16.shape == (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)

    # accumulate in fp32
    out_acc = torch.zeros((T, H), dtype=torch.float32)

    # For simplicity: loop over k then over experts by masking tokens belonging to each expert.
    # This avoids doing E matmuls. Complexity is OK for small tests.
    x_fp16 = x_fp16.contiguous()

    for k in range(K):
        ids_k = topk_ids_i32[:, k]  # [T]
        w_k = topk_w_f32[:, k]      # [T] float32

        # group tokens per expert
        for e in range(NUM_EXPERTS):
            mask = (ids_k == e)
            if not mask.any():
                continue
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

            x_e = x_fp16.index_select(0, idx)  # [Te, H] fp16

            # gate/up: [Te, H] @ [H, I] -> [Te, I]
            # w_gate[e]: [I, H] so use transpose
            gate = (x_e @ w_gate_fp16[e].t()).to(torch.float16)  # [Te, I]
            up = (x_e @ w_up_fp16[e].t()).to(torch.float16)      # [Te, I]

            # silu(gate) * up
            act = torch.nn.functional.silu(gate) * up            # [Te, I] fp16

            # down: [Te, I] @ [I, H] -> [Te, H]
            y_e = (act @ w_down_fp16[e].t()).to(torch.float16)   # [Te, H]

            # weighted add: out[idx] += w_k[idx] * y_e
            out_acc.index_add_(0, idx, y_e.to(torch.float32) * w_k[idx].unsqueeze(1))

    return out_acc.to(torch.float16).contiguous()


@torch.no_grad()
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # --------- init weights (fp16 for reference) ----------
    # Use a moderate std to keep outputs in a reasonable range.
    # For MoE FFN, a common heuristic is std ~ 1/sqrt(H)
    std = 1.0 / math.sqrt(HIDDEN_SIZE)

    w_gate = torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.float16) * std
    w_up = torch.randn(NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.float16) * std
    w_down = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.float16) * std

    # --------- create MoEInfer handle and online quantize ----------
    handle = torch.classes.nanovllm.MoEInfer(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0)

    t0 = time.time()
    # Quantization expects CPU float32 weights
    # gate_proj / up_proj: [I, H]
    # down_proj: [H, I]
    for e in range(NUM_EXPERTS):
        handle.quantize_and_store_expert(e, "gate_proj", w_gate[e].to(torch.float32).contiguous())
        handle.quantize_and_store_expert(e, "up_proj", w_up[e].to(torch.float32).contiguous())
        handle.quantize_and_store_expert(e, "down_proj", w_down[e].to(torch.float32).contiguous())
    t1 = time.time()
    print(f"[quantize] done in {(t1 - t0):.3f}s")

    # --------- test multiple token sizes ----------
    # Keep sizes reasonable for reference loop; you can add more.
    token_sizes = [1, 8, 32, 128, 256]

    for T in token_sizes:
        print(f"\n=== Test T={T} ===")

        # input hidden
        x = (torch.randn(T, HIDDEN_SIZE, dtype=torch.float16) * 0.5).contiguous()

        # router logits for generating topk on CPU (float32)
        logits = torch.randn(T, NUM_EXPERTS, dtype=torch.float32).contiguous()

        topk_ids, topk_w = make_topk_from_logits_cpu(logits, TOP_K, RENORMALIZE)

        # --------- run extension CPU routed (returns new tensor) ----------
        t0 = time.time()
        y_ext = torch.ops.nanovllm.moe_forward(x, topk_ids, topk_w, handle)
        t1 = time.time()

        # --------- run fp16 reference ----------
        t2 = time.time()
        y_ref = reference_moe_fp16(x, topk_ids, topk_w, w_gate, w_up, w_down)
        t3 = time.time()

        # --------- error metrics ----------
        diff = (y_ext - y_ref).to(torch.float32)
        ref_f = y_ref.to(torch.float32)

        abs_max = diff.abs().max().item()
        abs_mean = diff.abs().mean().item()
        rmse = torch.sqrt((diff * diff).mean()).item()
        rel_rmse = rmse / (torch.sqrt((ref_f * ref_f).mean()).item() + 1e-12)

        print(f"[time] ext={(t1 - t0)*1000:.2f} ms, ref={(t3 - t2)*1000:.2f} ms")
        print(f"[err ] abs_max={abs_max:.6f}, abs_mean={abs_mean:.6f}, rmse={rmse:.6f}, rel_rmse={rel_rmse:.6f}")

        # basic sanity
        assert y_ext.shape == y_ref.shape
        assert y_ext.dtype == torch.float16

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()
