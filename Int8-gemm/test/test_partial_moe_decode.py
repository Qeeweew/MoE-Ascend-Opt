import os

os.environ.setdefault("NANOVLLM_TP_SIZE", "2")
os.environ.setdefault("NANOVLLM_TP_THREADS_PER_NODE", "4")

import numpy as np
import pytest
import torch

import nanovllm_ext  # noqa: F401


def _random_q4(rows: int, cols: int):
    qs = torch.from_numpy(
        np.random.randint(0, 2**32, size=(rows, cols // 8), dtype=np.uint32)
    ).contiguous()
    scales = (torch.rand(rows, cols // 32) / np.sqrt(cols)).to(torch.float16)
    return qs, scales.contiguous()


def _make_handle(experts: int, hidden: int, intermediate: int):
    gate_qs, gate_d = _random_q4(experts * 2 * intermediate, hidden)
    down_qs, down_d = _random_q4(experts * hidden, intermediate)
    handle = torch.classes.nanovllm.MoEInfer(experts, hidden, intermediate, 1)
    handle.store_quantized_repack(
        gate_qs.view(experts, 2 * intermediate, hidden // 8),
        gate_d.view(experts, 2 * intermediate, hidden // 32),
        down_qs.view(experts, hidden, intermediate // 8),
        down_d.view(experts, hidden, intermediate // 32),
    )
    return handle


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [4, 8])
def test_small_batch_matches_independent_tokens(dtype, tokens):
    np.random.seed(0)
    torch.manual_seed(0)
    experts, hidden, intermediate, top_k = 16, 128, 128, 8
    handle = _make_handle(experts, hidden, intermediate)
    hidden_states = torch.randn(tokens, hidden, dtype=dtype).contiguous()
    # Include expert collisions across tokens: the batched path groups these
    # routes into a single expert task before the small-batch engine expands
    # them back into independently tiled routes.
    ids = torch.stack(
        [(torch.arange(top_k, dtype=torch.int32) + token * 3) % experts
         for token in range(tokens)]
    ).contiguous()
    weights = torch.rand(tokens, top_k, dtype=torch.float32).contiguous()

    actual = torch.ops.nanovllm.moe_forward(hidden_states, ids, weights, handle)
    expected = torch.cat([
        torch.ops.nanovllm.moe_forward(
            hidden_states[token:token + 1],
            ids[token:token + 1],
            weights[token:token + 1],
            handle,
        )
        for token in range(tokens)
    ])

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
