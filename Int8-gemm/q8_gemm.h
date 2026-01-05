#pragma once
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include "numa_threadpool.h"

std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_float);

template<typename D_TYPE>
void repack_B_q8_0_from_ptr(
    int64_t N, int64_t K,
    const int8_t* src_qs, const D_TYPE* src_d,
    int8_t* dest_qs_packed, D_TYPE* dest_d_packed);

template <typename T>
void moe_q8_forward_ptr_impl(
    const T* x_in_ptr,
    T* y_out_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,

    // per-tp weights
    const int8_t* const* gate_up_qs_tp,
    const at::Half* const* gate_up_d_tp,
    const int8_t* const* down_proj_qs_tp,
    const at::Half* const* down_proj_d_tp,

    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size, int64_t tp_size, int64_t top_k);
