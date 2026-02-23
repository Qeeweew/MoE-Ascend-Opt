#pragma once
#include "quant_traits.h"
#include <torch/extension.h>
#include <cstdint>

namespace gemm {

// ==================== Quantization (Q8_0 only) ====================

// Quantize a block of floats to Q8_0 format
// Note: Q4_0 should NOT be used for online quantization due to precision loss
template<typename D_TYPE>
void quantize_block_q8_0(
    const float* x,
    D_TYPE* y_d,
    int8_t* y_qs);

// ==================== Repacking ====================

// Repack B matrix from quantized format to packed format (Q8_0)
template <typename D_TYPE>
void repack_B_q8_0(
    int64_t N, int64_t K,
    const int8_t* src_qs,
    const D_TYPE* src_d,
    int8_t* dest_qs_packed,
    D_TYPE* dest_d_packed);

// Repack B matrix from quantized format to packed format (Q4_0)
// Note: src_qs is uint32_t (already packed 4-bit values)
template <typename D_TYPE>
void repack_B_q4_0(
    int64_t N, int64_t K,
    const uint32_t* src_qs,
    const D_TYPE* src_d,
    uint32_t* dest_qs_packed,
    D_TYPE* dest_d_packed);

// ==================== Microkernels ====================

// Main microkernel for Q8_0
template <int MR_T, typename B_SCALE_TYPE>
void gemm_q8_0_microkernel_specialized(
    int kc_size,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const int8_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate);

// Dispatch wrapper for Q8_0 microkernel
template<typename B_SCALE_TYPE>
void gemm_q8_0_microkernel(
    int kc_size, int mr,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const int8_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate);

// Main microkernel for Q4_0
template <int MR_T, typename B_SCALE_TYPE>
void gemm_q4_0_microkernel_specialized(
    int kc_size,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const uint32_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate);

// Dispatch wrapper for Q4_0 microkernel
template<typename B_SCALE_TYPE>
void gemm_q4_0_microkernel(
    int kc_size, int mr,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const uint32_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate);

// ==================== Packing ====================

// Pack A matrix after quantization (Q8_0)
void pack_A_q8_0(
    int M, int K,
    const float* A, int lda,
    int8_t* A_qs_packed,
    float* A_d_packed);

// Pack A matrix from already quantized data (Q8_0)
void pack_A_q8_0_from_quantized_indirect(
    int M, int K,
    const int8_t* x_qs_base,
    const float* x_d_base,
    const struct MoETokenInfo* token_map,
    int token_offset,
    int8_t* A_qs_packed,
    float* A_d_packed);

// ==================== GEMM Compute ====================

// Main GEMM computation function (Q8_0)
void gemm_q8_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const int8_t* B_qs_packed,
    const at::Half* B_d_packed_f16,
    float* C,
    int ldc);

// Main GEMM computation function (Q4_0)
void gemm_q4_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const uint32_t* B_qs_packed,
    const at::Half* B_d_packed_f16,
    float* C,
    int ldc);

// ==================== Utilities ====================

// Token info for MoE routing
struct MoETokenInfo {
    int32_t token_id;
    int32_t expert_idx_in_tok;
};

// Activation function: SiLU(x) * x
void silu_and_mul(float* inout, int64_t rows, int64_t cols);

} // namespace gemm
