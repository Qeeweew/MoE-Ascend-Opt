# moe_ascend_npu

Standalone W4A16 fused-MoE NPU kernels + MoE CPU offload for SGLang on Ascend NPU.

This package extracts the W4A16 fused-MoE Ascend C kernels (formerly maintained
inside a `sgl-kernel-npu` fork) and the Triton int4 repack kernel into one
independently-buildable package, so the **upstream** `sglang` / `sgl-kernel-npu`
can be used unmodified. SGLang integration is applied at runtime via monkey
patch, auto-activated by a `.pth` file at interpreter start-up.

## Quick start

```bash
cd moe_ascend_npu
bash build_kernels.sh Ascend910_9382   # builds .so + installs the .pth hook
pip install -e .                       # ships the .so + Python package
pip install -e ../Int8-gemm            # OPTIONAL: enables MoE CPU offload (nanovllm_ext)
```

After this, every SGLang launch on NPU is automatically patched — no code
changes, no extra CLI flags:

```bash
sglang launch-server --model-path /path/to/Qwen3-MoE-AWQ --attention-backend ascend ...
# enable MoE CPU offload:
sglang launch-server --model-path ... --enable-moe-offload --moe-offload-quant-type q4_0 ...
```

Dynamic expert caching (compressed-tensors Int4, TP=1):

```bash
export NANOVLLM_TP_SIZE=2
sglang launch-server \
  --model-path /mnt/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit-gs32 \
  --attention-backend ascend \
  --enable-moe-expert-cache \
  --moe-expert-cache-size 64 \
  --moe-expert-cache-swap-per-update 8 \
  --moe-expert-cache-update-interval 32 \
  --moe-expert-cache-warmup-steps 16
```

Do not pass ``--disable-cuda-graph``. The fixed cache tensors and slot table are
captured once; LFU replacement is driven by a hook immediately before graph
replay and does not recapture the graph. ``--enable-moe-expert-cache`` and
``--enable-moe-offload`` are mutually exclusive. The configured update interval
is used while filling; after the cache is full, the controller automatically
uses an 8x longer steady-state interval.

## What the package provides

### 1. W4A16 fused-MoE NPU kernels (decoding / small batch)

Built with CMake + the CANN ascendc toolchain (`build_kernels.sh`), producing
`moe_ascend_npu/lib/libmoe_ascend_npu_kernels.so` and registering three ops
under the `moe_ascend_npu` torch namespace:

| Op | Signature |
| :-- | :-- |
| `grouped_gemv_w4a16_moe` | `(x[?,K], w[E,K,N//8], scales[E,K//g,N], expert_ids) -> y` |
| `fused_moe_w4a16_small_bs` | `(x[BS,K], w13, s13, w2, s2, expert_ids[BS,TopK], topk_w[BS,TopK]) -> y[BS,K]` |
| `fused_moe_w4a16_cached` | Same computation with cache `slot_ids`; `-1` routes contribute zero |
| `batch_gemm_w4a16_small_bs` | `(x[BS<=4,K], w[K,N//8], scales[K//g,N]) -> y[BS,N]` |

Plus the Triton `repack_int4_npu` weight-repack kernel (vendored; upstream
`sgl-kernel-npu` does not ship it).

### 2. MoE CPU offload (Int8/Int4)

Provided by the sibling `Int8-gemm/` extension (`nanovllm_ext`), kept as a
separate build. When `--enable-moe-offload` is set, selected MoE layers are
routed to CPU using NUMA-aware ARM NEON Int8/Int4 kernels. Install it with
`pip install -e ../Int8-gemm`.

## SGLang monkey patches

`moe_ascend_npu/patches/` applies (on NPU only) via `apply_patches()`:

| Patch | Target in upstream SGLang | Effect |
| :-- | :-- | :-- |
| `server_args.py` | `ServerArgs.add_cli_args` / `from_cli_args` | adds `--enable-moe-offload*` flags |
| `moe_layer.py` | `FusedMoE.__init__` | routes offloaded layers to the CPU offload method *before* `create_weights` |
| `offload.py` | (new module) | `MoEOffloadFusedMoEMethod` / `MoEOffloadInt4FusedMoEMethod` (uses `nanovllm_ext`) |
| `compressed_tensors.py` | `CompressedTensorsConfig._get_scheme_from_parts` | NPU W4A16 Linear -> `NPUCompressedTensorsW4A16` |
| `wna16_moe.py` | `NPUCompressedTensorsW4A16Int4DynamicMoE.create_weights` | cast scales/offsets to `params_dtype` (upstream hardcodes bf16) |
| `fused_moe_method.py` | `NPUW4A16Int4DynamicMoEMethod.process_weights_after_loading` / `apply` | fast `repack_int4_npu` + small-batch fused kernel (`BS<=8`) |
| `minimax_m2.py` | `rms_sumsq_serial` / `rms_apply_serial` | drop `is_cuda` assertions for NPU |

TopK is **not** patched — upstream `fused_topk_npu` is already correct.

### Auto-activation (`.pth`)

`moe_ascend_npu.pth` (installed by `build_kernels.sh` / `python -m
moe_ascend_npu._install_pth`) contains `import moe_ascend_npu._bootstrap`, which
Python executes at every interpreter start-up. `_bootstrap` installs a cheap
`builtins.__import__` wrapper that fires `apply_patches()` only after SGLang has
been imported (import depth back to 0) and only on NPU. Non-SGLang processes pay
just one dict-lookup per import; the wrapper unwraps itself after firing.

## Tests

```bash
cd moe_ascend_npu/tests && bash run_all.sh
```

| Test | Validates |
| :-- | :-- |
| `test_repack.py` | Triton `repack_int4_npu` vs PyTorch reference (bit-exact) |
| `test_gemv_w4a16.py` | `grouped_gemv_w4a16_moe` + `batch_gemm_w4a16_small_bs` vs PyTorch W4A16 ref |
| `test_fused_moe.py` | `fused_moe_w4a16_small_bs` (BS=1/2/4) vs step-by-step PyTorch MoE ref |

All run on NPU and share uint4b8 weights repacked through `repack_int4_npu` (the
exact flow used by `process_weights_after_loading`).

## Layout

```
moe_ascend_npu/
├── build_kernels.sh              # CMake build for Ascend C kernels + installs .pth
├── CMakeLists.txt / cmake/       # ascendc build config
├── csrc/                         # Ascend C host + kernel sources (namespace moe_ascend_npu)
│   ├── pytorch_extensions.cpp    # registers torch.ops.moe_ascend_npu.*
│   ├── op_host/grouped_gemv_w4a16_moe.cpp
│   ├── op_kernel/grouped_gemv_w4a16_moe.cpp
│   └── utils/
├── include/moe_ascend_npu_ops.h
├── moe_ascend_npu.pth            # auto-import hook (-> site-packages)
├── moe_ascend_npu/
│   ├── __init__.py
│   ├── _bootstrap.py             # lazy __import__ hook -> apply_patches()
│   ├── _install_pth.py           # python -m moe_ascend_npu._install_pth
│   ├── kernels/                  # .so loader + Triton repack_int4_npu
│   ├── layers/                   # NPU W4A16 linear scheme (used by patches)
│   └── patches/                  # SGLang monkey patches + offload impl
└── tests/                        # NPU correctness tests
```
