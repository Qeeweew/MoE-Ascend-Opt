#pragma once
#include <torch/extension.h>
#include <vector>

// Quantize a 2D weight (CPU float32) -> (qs int8 [N,K], d half [N,K/32])
std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_float);

// Repack row-major (qs,d) into packed layout expected by kernel
template<typename D_TYPE>
void repack_B_q8_0_from_ptr(
    int64_t N, int64_t K,
    const int8_t* src_qs, const D_TYPE* src_d,
    int8_t* dest_qs_packed, D_TYPE* dest_d_packed
);

// MoE core (ARM CPU): input x_in -> output y_out
template <typename T>
void moe_q8_forward_ptr_impl(
    const T* x_in_ptr,
    T* y_out_ptr,
    const float* routing_weights_ptr,   // [num_tokens, top_k], float32
    const int32_t* selected_experts_ptr,// [num_tokens, top_k], int32
    const int8_t* gate_up_qs_packed,    // packed
    const at::Half* gate_up_d_packed,   // packed scales
    const int8_t* down_proj_qs_packed,  // packed
    const at::Half* down_proj_d_packed, // packed scales
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size,
    int64_t top_k
);
