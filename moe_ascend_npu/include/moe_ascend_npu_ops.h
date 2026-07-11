#ifndef MOE_ASCEND_NPU_OPS_H
#define MOE_ASCEND_NPU_OPS_H

#include <ATen/ATen.h>

namespace moe_ascend_npu {
namespace npu_kernel {

at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y);

at::Tensor grouped_gemv_w4a16_moe(const at::Tensor &x_in, const at::Tensor &weight,
                                   const at::Tensor &scales, const at::Tensor &expert_ids);

at::Tensor fused_moe_w4a16_small_bs(
    const at::Tensor &x_in,
    const at::Tensor &w13_weight, const at::Tensor &w13_scales,
    const at::Tensor &w2_weight, const at::Tensor &w2_scales,
    const at::Tensor &expert_ids, const at::Tensor &topk_weights);

// Cache variant: ``slot_ids`` indexes the first dimension of the cache
// tensors; -1 denotes a miss and contributes zero.
at::Tensor fused_moe_w4a16_cached(
    const at::Tensor &x_in,
    const at::Tensor &w13_weight, const at::Tensor &w13_scales,
    const at::Tensor &w2_weight, const at::Tensor &w2_scales,
    const at::Tensor &slot_ids, const at::Tensor &topk_weights);

at::Tensor batch_gemm_w4a16_small_bs(const at::Tensor &x_in, const at::Tensor &weight,
                                      const at::Tensor &scales);

} // namespace npu_kernel
} // namespace moe_ascend_npu

#endif // MOE_ASCEND_NPU_OPS_H
