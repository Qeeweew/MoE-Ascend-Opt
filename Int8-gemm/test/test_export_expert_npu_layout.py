"""Bit-exact test for MoEInfer's pinned NPU-cache export."""

import os

os.environ.setdefault("NANOVLLM_TP_SIZE", "2")

import torch
import torch_npu  # noqa: F401

import nanovllm_ext  # noqa: F401


def pack_uint4b8(raw: torch.Tensor) -> torch.Tensor:
    values = (raw + 8).to(torch.int32) & 0xF
    values = values.view(*raw.shape[:-1], raw.shape[-1] // 8, 8)
    out = torch.zeros(values.shape[:-1], dtype=torch.int32)
    for i in range(8):
        out |= values[..., i] << (4 * i)
    return out


def pack_npu_signed(raw_nk: torch.Tensor) -> torch.Tensor:
    values = raw_nk.t().contiguous().to(torch.int32) & 0xF
    values = values.view(values.shape[0], values.shape[1] // 8, 8)
    out = torch.zeros(values.shape[:-1], dtype=torch.int32)
    for i in range(8):
        out |= values[..., i] << (4 * i)
    return out


def test_export_expert_npu_layout():
    torch.manual_seed(7)
    experts, hidden, intermediate = 3, 64, 128
    raw_w13 = torch.randint(-8, 8, (experts, 2 * intermediate, hidden))
    raw_w2 = torch.randint(-8, 8, (experts, hidden, intermediate))
    w13 = pack_uint4b8(raw_w13)
    w2 = pack_uint4b8(raw_w2)
    s13 = torch.randn(experts, 2 * intermediate, hidden // 32, dtype=torch.float16)
    s2 = torch.randn(experts, hidden, intermediate // 32, dtype=torch.float16)

    handle = torch.classes.nanovllm.MoEInfer(experts, hidden, intermediate, 1)
    handle.store_quantized_repack(w13, s13, w2, s2)

    for expert in range(experts):
        got_w13, got_s13, got_w2, got_s2 = handle.export_expert_npu_layout(expert)
        assert all(t.is_pinned() for t in (got_w13, got_s13, got_w2, got_s2))
        assert torch.equal(got_w13, pack_npu_signed(raw_w13[expert]))
        assert torch.equal(got_s13, s13[expert].t().contiguous())
        assert torch.equal(got_w2, pack_npu_signed(raw_w2[expert]))
        assert torch.equal(got_s2, s2[expert].t().contiguous())

        out = tuple(torch.empty_like(t, pin_memory=True) for t in (got_w13, got_s13, got_w2, got_s2))
        handle.export_expert_npu_layout_out(expert, *out)
        assert all(torch.equal(a, b) for a, b in zip(out, (got_w13, got_s13, got_w2, got_s2)))

        copied = out[0].npu(non_blocking=True)
        torch.npu.synchronize()
        assert torch.equal(copied.cpu(), got_w13)

    order = torch.tensor([2, 0, 1], dtype=torch.int64)
    refs = [handle.export_expert_npu_layout(int(expert)) for expert in order]
    batch_out = tuple(
        torch.empty((len(order), *ref.shape), dtype=ref.dtype, pin_memory=True)
        for ref in refs[0]
    )
    handle.export_experts_npu_layout_out(order, *batch_out)
    for row, ref in enumerate(refs):
        assert all(torch.equal(out[row], expected) for out, expected in zip(batch_out, ref))


if __name__ == "__main__":
    test_export_expert_npu_layout()
    print("MoEInfer pinned NPU-layout export test passed.")
