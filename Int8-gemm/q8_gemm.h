#pragma once
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include "numa_threadpool.h"
#include "gemm_kernels.h"
#include "quant_traits.h"

// Quantize weights (Q8_0 only, for backward compatibility)
std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_float);

// Legacy constants (use quant::QuantTraits instead)
constexpr int QK8_0 = 32;
constexpr int MR = 4;
constexpr int NR = 8;


// Unified MoE forward implementation (supports both Q8_0 and Q4_0)
// QT: QuantType (Q8_0 or Q4_0)
// T: data type (at::Half or at::BFloat16)
// B_SCALE_TYPE: scale type (at::Half or at::BFloat16)
template <quant::QuantType QT, typename T, typename B_SCALE_TYPE>
void moe_forward_ptr_impl(
    const T* x_in_ptr,
    T* y_out_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,

    // per-tp weights (void* for unified interface)
    const void* const* gate_up_qs_tp,
    const B_SCALE_TYPE* const* gate_up_d_tp,
    const void* const* down_proj_qs_tp,
    const B_SCALE_TYPE* const* down_proj_d_tp,

    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size, int64_t tp_size, int64_t top_k);

