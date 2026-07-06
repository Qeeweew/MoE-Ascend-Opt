#include "version.h"

#include "torch_helper.h"
#include "moe_ascend_npu_ops.h"

namespace {
TORCH_LIBRARY_FRAGMENT(moe_ascend_npu, m)
{
    m.def(
        "grouped_gemv_w4a16_moe(Tensor x_in, Tensor weight, Tensor scales, Tensor expert_ids) -> Tensor");

    m.def(
        "fused_moe_w4a16_small_bs(Tensor x_in, Tensor w13_weight, Tensor w13_scales, "
        "Tensor w2_weight, Tensor w2_scales, Tensor expert_ids, Tensor topk_weights) -> Tensor");

    m.def(
        "batch_gemm_w4a16_small_bs(Tensor x_in, Tensor weight, Tensor scales) -> Tensor");
}
}  // namespace

namespace {
TORCH_LIBRARY_IMPL(moe_ascend_npu, PrivateUse1, m)
{
    m.impl("grouped_gemv_w4a16_moe", TORCH_FN(moe_ascend_npu::npu_kernel::grouped_gemv_w4a16_moe));

    m.impl("fused_moe_w4a16_small_bs", TORCH_FN(moe_ascend_npu::npu_kernel::fused_moe_w4a16_small_bs));

    m.impl("batch_gemm_w4a16_small_bs", TORCH_FN(moe_ascend_npu::npu_kernel::batch_gemm_w4a16_small_bs));
}
}  // namespace
