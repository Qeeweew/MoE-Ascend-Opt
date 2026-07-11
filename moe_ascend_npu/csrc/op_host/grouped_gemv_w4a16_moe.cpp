#include "defines.h"
#include "tiling/platform/platform_ascendc.h"
// 包含生成的 Kernel Launch 头文件 (根据 Kernel 导出名)
#include "aclrtlaunch_grouped_gemv_w4a16_moe_fp16.h"
#include "aclrtlaunch_grouped_gemv_w4a16_moe_bf16.h"
#include "aclrtlaunch_fused_moe_small_bs_w4a16_fp16.h"
#include "aclrtlaunch_fused_moe_small_bs_w4a16_bf16.h"
#include "aclrtlaunch_batch_gemm_w4a16_small_bs_fp16.h"
#include "aclrtlaunch_batch_gemm_w4a16_small_bs_bf16.h"
#include "moe_ascend_npu_ops.h"
#include "torch_helper.h"

namespace moe_ascend_npu {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;

// -----------------------------------------------------------------------------
// Standalone GEMV Host API
// -----------------------------------------------------------------------------
// Note: offsets parameter removed for symmetric quantization
HOST_API at::Tensor grouped_gemv_w4a16_moe(const at::Tensor &x_in, const at::Tensor &weight,
                                           const at::Tensor &scales,
                                           const at::Tensor &expert_ids)
{
    // 1. Shape & Type Checks
    TORCH_CHECK(weight.dtype() == at::kInt, "weight must be int32 (packed int4)");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "expert_ids must be int32");
    TORCH_CHECK(x_in.dtype() == scales.dtype(), "x and scales must have the same dtype");

    // Weight Shape: [NumExperts, InDim, OutDim_Packed]
    int32_t num_experts = weight.size(0);
    int32_t in_dim = weight.size(1);
    int32_t out_dim = weight.size(2) * 8;

    // 处理 expert_ids (支持 1D 或 2D)
    // Kernel 需要 total_tokens 和 top_k
    int32_t total_tokens = expert_ids.numel();
    int32_t top_k = 0;

    at::Tensor expert_ids_flat = expert_ids.flatten().contiguous();

    if (expert_ids.dim() == 2) {
        // [BatchSize, TopK]
        top_k = expert_ids.size(1);
    } else if (expert_ids.dim() == 1) {
        // [TopK] (BatchSize=1)
        top_k = expert_ids.size(0);
    } else {
        TORCH_CHECK(false, "expert_ids must be 1D or 2D");
    }

    // Preprocess Input X
    // 如果 X 是 [BS, InDim]，需要扩展为 [TotalTokens, InDim] 供 GEMV 使用
    // 但通常 grouped_gemv 的输入 X 已经是 gather 后的 [TotalTokens, InDim] 或者需要广播
    // 这里假设 grouped_gemv 的 X 需要与 expert_ids 对应。
    // 为了兼容性，如果 X 是 [BS, InDim]，我们做 view/expand。
    at::Tensor x_contiguous;
    int32_t batch_size = total_tokens / top_k;

    if (x_in.size(0) == batch_size && x_in.size(1) == in_dim) {
        // X: [BS, InDim] -> expand to [BS, TopK, InDim] -> flatten to [TotalTokens, InDim]
        x_contiguous = x_in.unsqueeze(1).expand({batch_size, top_k, in_dim}).reshape({-1, in_dim}).contiguous();
    } else {
        TORCH_CHECK(x_in.size(0) == total_tokens, "X dim 0 must match total_tokens or batch_size");
        x_contiguous = x_in.contiguous();
    }

    at::Tensor y = at::zeros({total_tokens, out_dim}, x_in.options());

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    auto acl_stream = c10_npu::getCurrentNPUStream();

    if (x_in.dtype() == at::kHalf) {
        ACLRT_LAUNCH_KERNEL(grouped_gemv_w4a16_moe_fp16)(
            block_dim, acl_stream,
            const_cast<void *>(x_contiguous.data_ptr()),
            const_cast<void *>(weight.data_ptr()),
            const_cast<void *>(scales.data_ptr()),
            const_cast<void *>(expert_ids_flat.data_ptr()),
            y.data_ptr(),
            total_tokens, in_dim, out_dim, num_experts, top_k
        );
    } else if (x_in.dtype() == at::kBFloat16) {
        ACLRT_LAUNCH_KERNEL(grouped_gemv_w4a16_moe_bf16)(
            block_dim, acl_stream,
            const_cast<void *>(x_contiguous.data_ptr()),
            const_cast<void *>(weight.data_ptr()),
            const_cast<void *>(scales.data_ptr()),
            const_cast<void *>(expert_ids_flat.data_ptr()),
            y.data_ptr(),
            total_tokens, in_dim, out_dim, num_experts, top_k
        );
    } else {
        TORCH_CHECK(false, "FP16 or BF16 support only!");
    }

    return y;
}

// -----------------------------------------------------------------------------
// Fused MoE Host API (BS <= 4)
// -----------------------------------------------------------------------------
// Note: offsets parameters removed for symmetric quantization
HOST_API at::Tensor fused_moe_w4a16_small_bs(
    const at::Tensor &x_in,
    const at::Tensor &w13_weight, const at::Tensor &w13_scales,
    const at::Tensor &w2_weight, const at::Tensor &w2_scales,
    const at::Tensor &expert_ids, const at::Tensor &topk_weights)
{
    // 1. 维度检查与预处理
    // expert_ids: [BatchSize, TopK]
    TORCH_CHECK(expert_ids.dim() == 2, "expert_ids must be 2D [BatchSize, TopK]");
    TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2D [BatchSize, TopK]");
    TORCH_CHECK(expert_ids.sizes() == topk_weights.sizes(), "expert_ids and topk_weights shape mismatch");

    int32_t batch_size = expert_ids.size(0);
    int32_t top_k = expert_ids.size(1);
    int32_t total_tokens = expert_ids.numel(); // BS * TopK

    // Flatten expert_ids and topk_weights for the kernel
    at::Tensor expert_ids_flat = expert_ids.view({-1}).contiguous();
    at::Tensor topk_weights_flat = topk_weights.view({-1}).contiguous();

    // Input X: [BatchSize, InDim]
    TORCH_CHECK(x_in.dim() == 2, "x_in must be 2D [BatchSize, InDim]");
    TORCH_CHECK(x_in.size(0) == batch_size, "x_in batch size mismatch");
    int32_t in_dim = x_in.size(1);

    // Weights Check
    // W13: [NumExperts, InDim, OutDim_Packed]
    int32_t num_experts = w13_weight.size(0);
    int32_t w13_packed_out = w13_weight.size(2);
    int32_t inter_dim_2x = w13_packed_out * 8;
    int32_t inter_dim = inter_dim_2x / 2;

    // W2: [NumExperts, InterDim, OutDim_Packed]
    int32_t out_dim = w2_weight.size(2) * 8;
    TORCH_CHECK(in_dim == out_dim, "in_dim and out_dim must be equal for MoE");

    // Workspace Calculation (in T elements)
    // W13 Output (FP32): [TotalTokens, 2 * InterDim] -> 2x space
    // SwiGLU Output (T): [TotalTokens, InterDim]
    // W2 Output (FP32): [BatchSize, OutDim] -> 2x space
    int64_t w13_out_elems = (int64_t)total_tokens * inter_dim * 2 * 2;  // *2 for FP32
    int64_t swiglu_out_elems = (int64_t)total_tokens * inter_dim;
    int64_t w2_out_elems = (int64_t)batch_size * out_dim * 2;  // *2 for FP32
    int64_t total_workspace_elems = w13_out_elems + swiglu_out_elems + w2_out_elems;

    // y is the final output tensor
    at::Tensor y = at::empty({batch_size, out_dim}, x_in.options());

    // workspace for intermediate results (mix of FP32 and T)
    at::Tensor workspace = at::empty({total_workspace_elems}, x_in.options());

    // 2. Kernel Launch
    auto acl_stream = c10_npu::getCurrentNPUStream();
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());

    if (x_in.dtype() == at::kHalf) {
        ACLRT_LAUNCH_KERNEL(fused_moe_small_bs_w4a16_fp16)(
            block_dim, acl_stream,
            const_cast<void *>(x_in.data_ptr()),
            const_cast<void *>(w13_weight.data_ptr()),
            const_cast<void *>(w13_scales.data_ptr()),
            const_cast<void *>(w2_weight.data_ptr()),
            const_cast<void *>(w2_scales.data_ptr()),
            const_cast<void *>(expert_ids_flat.data_ptr()),
            const_cast<void *>(topk_weights_flat.data_ptr()),
            y.data_ptr(),
            workspace.data_ptr(),
            batch_size, in_dim, inter_dim, num_experts, top_k
        );
    } else if (x_in.dtype() == at::kBFloat16) {
        ACLRT_LAUNCH_KERNEL(fused_moe_small_bs_w4a16_bf16)(
            block_dim, acl_stream,
            const_cast<void *>(x_in.data_ptr()),
            const_cast<void *>(w13_weight.data_ptr()),
            const_cast<void *>(w13_scales.data_ptr()),
            const_cast<void *>(w2_weight.data_ptr()),
            const_cast<void *>(w2_scales.data_ptr()),
            const_cast<void *>(expert_ids_flat.data_ptr()),
            const_cast<void *>(topk_weights_flat.data_ptr()),
            y.data_ptr(),
            workspace.data_ptr(),
            batch_size, in_dim, inter_dim, num_experts, top_k
        );
    } else {
        TORCH_CHECK(false, "FP16 or BF16 support only!");
    }

    return y;
}

HOST_API at::Tensor fused_moe_w4a16_cached(
    const at::Tensor &x_in,
    const at::Tensor &w13_weight, const at::Tensor &w13_scales,
    const at::Tensor &w2_weight, const at::Tensor &w2_scales,
    const at::Tensor &slot_ids, const at::Tensor &topk_weights)
{
    // The device implementation treats the first weight dimension as an
    // opaque index space.  For the cache it is the slot count rather than the
    // model expert count; negative ids are skipped in ProcessBlock().
    return fused_moe_w4a16_small_bs(
        x_in, w13_weight, w13_scales, w2_weight, w2_scales,
        slot_ids, topk_weights);
}

// -----------------------------------------------------------------------------
// Small Batch GEMM Host API (Replacement for npu_weight_quant_batchmatmul)
// -----------------------------------------------------------------------------
HOST_API at::Tensor batch_gemm_w4a16_small_bs(const at::Tensor &x_in, const at::Tensor &weight,
                                              const at::Tensor &scales)
{
    // 1. Shape & Type Checks
    TORCH_CHECK(weight.dtype() == at::kInt, "weight must be int32 (packed int4)");
    TORCH_CHECK(x_in.dtype() == scales.dtype(), "x and scales must have the same dtype");
    TORCH_CHECK(x_in.dim() == 2, "X must be 2D [BatchSize, InDim]");

    int32_t batch_size = x_in.size(0);
    int32_t in_dim = x_in.size(1);

    // Weight Shape: [InDim, OutDim_Packed]
    int32_t out_dim = weight.size(1) * 8;

    TORCH_CHECK(batch_size <= 4, "batch_gemm_w4a16_small_bs only supports batch_size <= 4");
    TORCH_CHECK(weight.size(0) == in_dim, "Weight dim 0 must match InDim");

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    auto acl_stream = c10_npu::getCurrentNPUStream();

    // y is the final output tensor
    at::Tensor y = at::empty({batch_size, out_dim}, x_in.options());

    // workspace for FP32 accumulation (size = batch_size * out_dim * sizeof(float))
    at::Tensor workspace = at::empty({batch_size * out_dim}, at::dtype(at::kFloat).device(x_in.device()));

    // 2. Kernel Launch - split_k logic acts independently within the AIV cores
    if (x_in.dtype() == at::kHalf) {
        ACLRT_LAUNCH_KERNEL(batch_gemm_w4a16_small_bs_fp16)(
            block_dim, acl_stream,
            const_cast<void *>(x_in.data_ptr()),
            const_cast<void *>(weight.data_ptr()),
            const_cast<void *>(scales.data_ptr()),
            y.data_ptr(),
            workspace.data_ptr(),
            batch_size, in_dim, out_dim
        );
    } else if (x_in.dtype() == at::kBFloat16) {
        ACLRT_LAUNCH_KERNEL(batch_gemm_w4a16_small_bs_bf16)(
            block_dim, acl_stream,
            const_cast<void *>(x_in.data_ptr()),
            const_cast<void *>(weight.data_ptr()),
            const_cast<void *>(scales.data_ptr()),
            y.data_ptr(),
            workspace.data_ptr(),
            batch_size, in_dim, out_dim
        );
    } else {
        TORCH_CHECK(false, "FP16 or BF16 support only!");
    }

    return y;
}

} // namespace npu_kernel
} // namespace moe_ascend_npu
