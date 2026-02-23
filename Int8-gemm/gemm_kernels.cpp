#include "gemm_kernels.h"
#include "utils.h"
#include "vec_simd.h"

#include <ATen/Parallel.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

using ggml_half = at::Half;

namespace gemm {

// ==================== Q8_0 Quantization ====================
// Note: Q4_0 should use pre-quantized weights only, not online quantization

template<typename D_TYPE>
void quantize_block_q8_0(
    const float* x,
    D_TYPE* y_d,
    int8_t* y_qs)
{
    float d;

    float32x4_t srcv[8];
    float32x4_t asrcv[8];
    float32x4_t amaxv[4];

    for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + 4*j);
    for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);
    for (int j = 0; j < 4; j++) amaxv[j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
    for (int j = 0; j < 2; j++) amaxv[j] = vmaxq_f32(amaxv[2*j], amaxv[2*j+1]);
    const float amax = vmaxvq_f32(vmaxq_f32(amaxv[0], amaxv[1]));

    d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    for (int j = 0; j < 8; j+=2) {
        const float32x4_t v0  = vmulq_n_f32(srcv[j], id);
        const float32x4_t v1  = vmulq_n_f32(srcv[j + 1], id);
        const int32x4_t   v0_i32 = vcvtnq_s32_f32(v0);
        const int32x4_t   v1_i32 = vcvtnq_s32_f32(v1);
        const int16x4_t   v0_i16 = vqmovn_s32(v0_i32);
        const int16x4_t   v1_i16 = vqmovn_s32(v1_i32);
        const int8x8_t    vi8  = vqmovn_s16(vcombine_s16(v0_i16, v1_i16));
        vst1_s8(y_qs + 4*j, vi8);
    }

    *y_d = static_cast<D_TYPE>(d);
}

// Template instantiations for Q8_0 only
template void quantize_block_q8_0<at::Half>(const float*, at::Half*, int8_t*);
template void quantize_block_q8_0<float>(const float*, float*, int8_t*);

// ==================== Q8_0 Microkernel ====================

template <int MR_T, typename B_SCALE_TYPE>
void gemm_q8_0_microkernel_specialized(
    int kc_size,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const int8_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate)
{
    static_assert(MR_T > 0 && MR_T <= 4, "MR_T invalid");
    constexpr int QK8_0 = 32;
    constexpr int MR = 4;
    constexpr int NR = 8;
    const int KC_BLOCKS = kc_size / QK8_0;

    float32x4_t c_v[MR_T][2];

    if (accumulate) {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i][0] = vld1q_f32(C + i * ldc + 0);
            c_v[i][1] = vld1q_f32(C + i * ldc + 4);
        }
    } else {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i][0] = vdupq_n_f32(0.0f);
            c_v[i][1] = vdupq_n_f32(0.0f);
        }
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const B_SCALE_TYPE* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        int32x4_t sum_v[MR_T][2];
        for (int i = 0; i < MR_T; ++i) {
            sum_v[i][0] = vdupq_n_s32(0);
            sum_v[i][1] = vdupq_n_s32(0);
        }

        for (int k4_step = 0; k4_step < QK8_0 / 4; ++k4_step) {
            int8x16_t a_vec = vld1q_s8(a_ptr);
            a_ptr += 16;

            int8x16_t b_vec_0 = vld1q_s8(b_ptr);
            int8x16_t b_vec_1 = vld1q_s8(b_ptr + 16);
            b_ptr += 32;

            if constexpr (MR_T > 0) {
                sum_v[0][0] = vdotq_laneq_s32(sum_v[0][0], b_vec_0, a_vec, 0);
                sum_v[0][1] = vdotq_laneq_s32(sum_v[0][1], b_vec_1, a_vec, 0);
            }
            if constexpr (MR_T > 1) {
                sum_v[1][0] = vdotq_laneq_s32(sum_v[1][0], b_vec_0, a_vec, 1);
                sum_v[1][1] = vdotq_laneq_s32(sum_v[1][1], b_vec_1, a_vec, 1);
            }
            if constexpr (MR_T > 2) {
                sum_v[2][0] = vdotq_laneq_s32(sum_v[2][0], b_vec_0, a_vec, 2);
                sum_v[2][1] = vdotq_laneq_s32(sum_v[2][1], b_vec_1, a_vec, 2);
            }
            if constexpr (MR_T > 3) {
                sum_v[3][0] = vdotq_laneq_s32(sum_v[3][0], b_vec_0, a_vec, 3);
                sum_v[3][1] = vdotq_laneq_s32(sum_v[3][1], b_vec_1, a_vec, 3);
            }
        }

        float32x4_t d_b_v0, d_b_v1;
        if constexpr (std::is_same_v<B_SCALE_TYPE, at::BFloat16>) {
            // BF16 path: use bit manipulation for fast conversion
            uint16x8_t bf16_vec = vld1q_u16(reinterpret_cast<const uint16_t*>(bd_ptr));
            // Split into low and high halves
            uint16x4_t low_half = vget_low_u16(bf16_vec);
            uint16x4_t high_half = vget_high_u16(bf16_vec);
            // Shift left by 16 
            d_b_v0 = vreinterpretq_f32_u32(vshll_n_u16(low_half, 16));
            d_b_v1 = vreinterpretq_f32_u32(vshll_n_u16(high_half, 16));
        } else {
            // FP16 path
            float16x8_t b_scales_f16 = vld1q_f16((const __fp16 *) bd_ptr);
            d_b_v0 = vcvt_f32_f16(vget_low_f16(b_scales_f16));
            d_b_v1 = vcvt_f32_f16(vget_high_f16(b_scales_f16));
        }
        bd_ptr += NR;

        float32x4_t d_a_v = vld1q_f32(ad_ptr);
        ad_ptr += MR;

        if constexpr (MR_T > 0) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[0][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[0][1]);
            c_v[0][0] = vmlaq_laneq_f32(c_v[0][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 0);
            c_v[0][1] = vmlaq_laneq_f32(c_v[0][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 0);
        }
        if constexpr (MR_T > 1) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[1][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[1][1]);
            c_v[1][0] = vmlaq_laneq_f32(c_v[1][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 1);
            c_v[1][1] = vmlaq_laneq_f32(c_v[1][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 1);
        }
        if constexpr (MR_T > 2) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[2][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[2][1]);
            c_v[2][0] = vmlaq_laneq_f32(c_v[2][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 2);
            c_v[2][1] = vmlaq_laneq_f32(c_v[2][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 2);
        }
        if constexpr (MR_T > 3) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[3][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[3][1]);
            c_v[3][0] = vmlaq_laneq_f32(c_v[3][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 3);
            c_v[3][1] = vmlaq_laneq_f32(c_v[3][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 3);
        }
    }

    for (int i = 0; i < MR_T; ++i) {
        vst1q_f32(C + i * ldc + 0, c_v[i][0]);
        vst1q_f32(C + i * ldc + 4, c_v[i][1]);
    }
}

template<typename B_SCALE_TYPE>
void gemm_q8_0_microkernel(
    int kc_size, int mr,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const int8_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate)
{
    assert(mr > 0 && mr <= 4);
    switch (mr) {
        case 1: gemm_q8_0_microkernel_specialized<1>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 2: gemm_q8_0_microkernel_specialized<2>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 3: gemm_q8_0_microkernel_specialized<3>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 4: gemm_q8_0_microkernel_specialized<4>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
    }
}

// Template instantiations
template void gemm_q8_0_microkernel<at::Half>(int, int, const int8_t*, const float*, const int8_t*, const at::Half*, float*, int, bool);
template void gemm_q8_0_microkernel<at::BFloat16>(int, int, const int8_t*, const float*, const int8_t*, const at::BFloat16*, float*, int, bool);

// ==================== Q4_0 Microkernel ====================

template <int MR_T, typename B_SCALE_TYPE>
void gemm_q4_0_microkernel_specialized(
    int kc_size,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const uint32_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate)
{
    static_assert(MR_T > 0 && MR_T <= 4, "MR_T invalid");
    constexpr int QK4_0 = 32;
    constexpr int MR = 4;
    constexpr int NR = 8;
    const int KC_BLOCKS = kc_size / QK4_0;

    float32x4_t c_v[MR_T][2];

    if (accumulate) {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i][0] = vld1q_f32(C + i * ldc + 0);
            c_v[i][1] = vld1q_f32(C + i * ldc + 4);
        }
    } else {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i][0] = vdupq_n_f32(0.0f);
            c_v[i][1] = vdupq_n_f32(0.0f);
        }
    }

    const int8_t* a_ptr = A_qs_packed;
    const uint32_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const B_SCALE_TYPE* bd_ptr = B_d_packed;

    // Constants for Q4_0 unpacking
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        int32x4_t sum_v[MR_T][2];
        for (int i = 0; i < MR_T; ++i) {
            sum_v[i][0] = vdupq_n_s32(0);
            sum_v[i][1] = vdupq_n_s32(0);
        }

        // Q4_0: 8 K steps per block (each processes 4 bytes = 32 q4 elements)
        for (int k8_step = 0; k8_step < QK4_0 / 8; ++k8_step) {
            // Load A: MR=4 rows, 8 elements per row = 32 bytes
            // We need to load 2 sets of 16 bytes for the interleaved K dimensions
            int8x16_t a_vec_0 = vld1q_s8(a_ptr);      // First 16 bytes (rows 0-3, first 4 K elements)
            int8x16_t a_vec_1 = vld1q_s8(a_ptr + 16); // Second 16 bytes (rows 0-3, second 4 K elements)
            a_ptr += 32;

            // Load B: 32 bytes (NR=8 cols x 8 K-elements = 64 x 4-bit = 32 bytes)
            uint8x16_t b_bytes_0 = vld1q_u8(reinterpret_cast<const uint8_t*>(b_ptr)); // Cols 0..3
            uint8x16_t b_bytes_1 = vld1q_u8(reinterpret_cast<const uint8_t*>(b_ptr + 4));
            b_ptr += 8;

            // Unpack B (cols 0..3): [q0,q4,q1,q5,q2,q6,q3,q7] format
            // Lower nibble: q0,q1,q2,q3 -> maps to k=0..3
            // Upper nibble: q4,q5,q6,q7 -> maps to k=4..7
            int8x16_t b_v0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b_bytes_0, m4b)), s8b);
            int8x16_t b_v0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b_bytes_0, 4)), s8b);

            // Unpack B (cols 4..7)
            int8x16_t b_v1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b_bytes_1, m4b)), s8b);
            int8x16_t b_v1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b_bytes_1, 4)), s8b);

            // Dual FMA accumulation (Summing k=0..3 and k=4..7)
            if constexpr (MR_T > 0) {
                sum_v[0][0] = vdotq_laneq_s32(sum_v[0][0], b_v0_l, a_vec_0, 0);
                sum_v[0][1] = vdotq_laneq_s32(sum_v[0][1], b_v1_l, a_vec_0, 0);
                sum_v[0][0] = vdotq_laneq_s32(sum_v[0][0], b_v0_h, a_vec_1, 0);
                sum_v[0][1] = vdotq_laneq_s32(sum_v[0][1], b_v1_h, a_vec_1, 0);
            }
            if constexpr (MR_T > 1) {
                sum_v[1][0] = vdotq_laneq_s32(sum_v[1][0], b_v0_l, a_vec_0, 1);
                sum_v[1][1] = vdotq_laneq_s32(sum_v[1][1], b_v1_l, a_vec_0, 1);
                sum_v[1][0] = vdotq_laneq_s32(sum_v[1][0], b_v0_h, a_vec_1, 1);
                sum_v[1][1] = vdotq_laneq_s32(sum_v[1][1], b_v1_h, a_vec_1, 1);
            }
            if constexpr (MR_T > 2) {
                sum_v[2][0] = vdotq_laneq_s32(sum_v[2][0], b_v0_l, a_vec_0, 2);
                sum_v[2][1] = vdotq_laneq_s32(sum_v[2][1], b_v1_l, a_vec_0, 2);
                sum_v[2][0] = vdotq_laneq_s32(sum_v[2][0], b_v0_h, a_vec_1, 2);
                sum_v[2][1] = vdotq_laneq_s32(sum_v[2][1], b_v1_h, a_vec_1, 2);
            }
            if constexpr (MR_T > 3) {
                sum_v[3][0] = vdotq_laneq_s32(sum_v[3][0], b_v0_l, a_vec_0, 3);
                sum_v[3][1] = vdotq_laneq_s32(sum_v[3][1], b_v1_l, a_vec_0, 3);
                sum_v[3][0] = vdotq_laneq_s32(sum_v[3][0], b_v0_h, a_vec_1, 3);
                sum_v[3][1] = vdotq_laneq_s32(sum_v[3][1], b_v1_h, a_vec_1, 3);
            }
        }

        // Load B scales
        float32x4_t d_b_v0, d_b_v1;
        if constexpr (std::is_same_v<B_SCALE_TYPE, at::BFloat16>) {
            // Fast BF16 to FP32 conversion using NEON bit operations
            uint16x8_t bf16_vec = vld1q_u16(reinterpret_cast<const uint16_t*>(bd_ptr));
            // Split into low and high halves
            uint16x4_t low_half = vget_low_u16(bf16_vec);
            uint16x4_t high_half = vget_high_u16(bf16_vec);
            // Shift left by 16 
            d_b_v0 = vreinterpretq_f32_u32(vshll_n_u16(low_half, 16));
            d_b_v1 = vreinterpretq_f32_u32(vshll_n_u16(high_half, 16));
        } else {
            // FP16 path
            float16x8_t b_scales_f16 = vld1q_f16((const __fp16 *) bd_ptr);
            d_b_v0 = vcvt_f32_f16(vget_low_f16(b_scales_f16));
            d_b_v1 = vcvt_f32_f16(vget_high_f16(b_scales_f16));
        }
        bd_ptr += NR;

        // Load A scales
        float32x4_t d_a_v = vld1q_f32(ad_ptr);
        ad_ptr += MR;

        // FMA: Accumulate (Sum * B_scale) * A_scale
        if constexpr (MR_T > 0) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[0][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[0][1]);
            c_v[0][0] = vmlaq_laneq_f32(c_v[0][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 0);
            c_v[0][1] = vmlaq_laneq_f32(c_v[0][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 0);
        }
        if constexpr (MR_T > 1) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[1][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[1][1]);
            c_v[1][0] = vmlaq_laneq_f32(c_v[1][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 1);
            c_v[1][1] = vmlaq_laneq_f32(c_v[1][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 1);
        }
        if constexpr (MR_T > 2) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[2][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[2][1]);
            c_v[2][0] = vmlaq_laneq_f32(c_v[2][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 2);
            c_v[2][1] = vmlaq_laneq_f32(c_v[2][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 2);
        }
        if constexpr (MR_T > 3) {
            float32x4_t sum_f_0 = vcvtq_f32_s32(sum_v[3][0]);
            float32x4_t sum_f_1 = vcvtq_f32_s32(sum_v[3][1]);
            c_v[3][0] = vmlaq_laneq_f32(c_v[3][0], vmulq_f32(sum_f_0, d_b_v0), d_a_v, 3);
            c_v[3][1] = vmlaq_laneq_f32(c_v[3][1], vmulq_f32(sum_f_1, d_b_v1), d_a_v, 3);
        }
    }

    // Store results
    for (int i = 0; i < MR_T; ++i) {
        vst1q_f32(C + i * ldc + 0, c_v[i][0]);
        vst1q_f32(C + i * ldc + 4, c_v[i][1]);
    }
}

template<typename B_SCALE_TYPE>
void gemm_q4_0_microkernel(
    int kc_size, int mr,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const uint32_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc,
    bool accumulate)
{
    assert(mr > 0 && mr <= 4);
    switch (mr) {
        case 1: gemm_q4_0_microkernel_specialized<1>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 2: gemm_q4_0_microkernel_specialized<2>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 3: gemm_q4_0_microkernel_specialized<3>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 4: gemm_q4_0_microkernel_specialized<4>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
    }
}

// Template instantiations
template void gemm_q4_0_microkernel<at::Half>(int, int, const int8_t*, const float*, const uint32_t*, const at::Half*, float*, int, bool);
template void gemm_q4_0_microkernel<at::BFloat16>(int, int, const int8_t*, const float*, const uint32_t*, const at::BFloat16*, float*, int, bool);

// ==================== Packing Functions ====================

void pack_A_q8_0(
    int M, int K,
    const float* A, int lda,
    int8_t* A_qs_packed,
    float* A_d_packed)
{
    constexpr int QK8_0 = 32;
    constexpr int MR = 4;
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;

    for (int i_pack = 0; i_pack < num_a_packs; ++ i_pack) {
        int8_t a_qs_buf[MR * QK8_0];
        int i = i_pack * MR;
        for (int j = 0; j < K_BLOCKS; ++j) {
            int M_rem = std::min(MR, M - i);
            float* current_A_d_ptr = A_d_packed + i * K_BLOCKS + j * MR;
            for (int row = 0; row < M_rem; ++row) {
                quantize_block_q8_0(A + (i + row) * lda + j * QK8_0,
                                   &current_A_d_ptr[row],
                                   &a_qs_buf[row * QK8_0]);
            }
            int8_t* current_qs_ptr = A_qs_packed + i * K + j * QK8_0 * MR;
            for (int k = 0; k < QK8_0; k += 4) {
                for (int row = 0; row < MR; ++row) {
                    memcpy(current_qs_ptr, &a_qs_buf[row * QK8_0 + k], 4);
                    current_qs_ptr += 4;
                }
            }
        }
    };
}

void pack_A_q8_0_from_quantized_indirect(
    int M, int K,
    const int8_t* x_qs_base,
    const float* x_d_base,
    const MoETokenInfo* token_map,
    int token_offset,
    int8_t* A_qs_packed,
    float* A_d_packed)
{
    constexpr int QK8_0 = 32;
    constexpr int MR = 4;
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;

    for (int i_pack = 0; i_pack < num_a_packs; ++ i_pack) {
        int row_in_expert = i_pack * MR;
        for (int j = 0; j < K_BLOCKS; ++j) {
            int M_rem = std::min(MR, M - row_in_expert);
            float* current_A_d_ptr = A_d_packed + row_in_expert * K_BLOCKS + j * MR;
            for (int local_row = 0; local_row < M_rem; ++local_row) {
                int global_token_id = token_map[token_offset + row_in_expert + local_row].token_id;
                current_A_d_ptr[local_row] = x_d_base[global_token_id * K_BLOCKS + j];
            }
            int8_t* current_qs_ptr = A_qs_packed + row_in_expert * K + j * QK8_0 * MR;
            for (int k = 0; k < QK8_0; k += 4) {
                for (int local_row = 0; local_row < MR; ++local_row) {
                    if (local_row < M_rem) {
                        int global_token_id = token_map[token_offset + row_in_expert + local_row].token_id;
                        memcpy(current_qs_ptr,
                               x_qs_base + global_token_id * K + j * QK8_0 + k,
                               4);
                    }
                    current_qs_ptr += 4;
                }
            }
        }
    };
}

// ==================== Activation ====================

void silu_and_mul(float* inout, int64_t rows, int64_t cols) {
    ::silu_and_mul(inout, static_cast<int>(rows), static_cast<int>(cols));
}

// ==================== Repack Functions ====================

template <typename D_TYPE>
void repack_B_q8_0(
    int64_t N, int64_t K,
    const int8_t* src_qs,
    const D_TYPE* src_d,
    int8_t* dest_qs_packed,
    D_TYPE* dest_d_packed)
{
    constexpr int QK8_0 = 32;
    constexpr int NR = 8;
    const int K_BLOCKS = K / QK8_0;

    for (int j = 0; j < N; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < N) {
                    *dest_d_packed = src_d[(j + col) * K_BLOCKS + k_block];
                }
                dest_d_packed++;
            }
        }

        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int k_rem = 0; k_rem < QK8_0; k_rem += 4) {
                for (int col = 0; col < NR; ++col) {
                    if (j + col < N) {
                        const int8_t* src_ptr = src_qs + (j + col) * K + k_block * QK8_0 + k_rem;
                        memcpy(dest_qs_packed, src_ptr, 4);
                    }
                    dest_qs_packed += 4;
                }
            }
        }
    }
}

template <typename D_TYPE>
void repack_B_q4_0(
    int64_t N, int64_t K,
    const uint32_t* src_qs,
    const D_TYPE* src_d,
    uint32_t* dest_qs_packed,
    D_TYPE* dest_d_packed)
{
    constexpr int QK4_0 = 32;
    constexpr int NR = 8;
    const int K_BLOCKS = K / QK4_0;

    for (int j = 0; j < N; j += NR) {
        // 1. Repack scales
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < N) {
                    *dest_d_packed = src_d[(j + col) * K_BLOCKS + k_block];
                }
                dest_d_packed++;
            }
        }

        // 2. Repack quantized values (uint32_t aware)
        // Each K block has QK4_0 elements = 4 uint32_t blocks
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int k_rem = 0; k_rem < QK4_0 / 8; ++k_rem) { // 4 iterations
                for (int col = 0; col < NR; ++col) {
                    if (j + col < N) {
                        // (K / 8) is the stride because src_qs is uint32_t*
                        dest_qs_packed[0] = src_qs[(j + col) * (K / 8) + k_block * 4 + k_rem];
                    } else {
                        dest_qs_packed[0] = 0;
                    }
                    dest_qs_packed++;
                }
            }
        }
    }
}

// Template instantiations
// uint16_t for FP16/BF16, float for FP32
template void repack_B_q8_0<float>(int64_t, int64_t, const int8_t*, const float*, int8_t*, float*);
template void repack_B_q8_0<uint16_t>(int64_t, int64_t, const int8_t*, const uint16_t*, int8_t*, uint16_t*);
template void repack_B_q4_0<float>(int64_t, int64_t, const uint32_t*, const float*, uint32_t*, float*);
template void repack_B_q4_0<uint16_t>(int64_t, int64_t, const uint32_t*, const uint16_t*, uint32_t*, uint16_t*);

// ==================== GEMM Compute Functions ====================

template<typename B_SCALE_TYPE>
void gemm_q8_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const int8_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc)
{
    constexpr int QK8_0 = 32;
    constexpr int MR = 4;
    constexpr int NR = 8;
    constexpr int KC = 2048;
    constexpr int NC = 64;

    const int K_BLOCKS = K / QK8_0;
    const int num_nc_blocks = (N + NC - 1) / NC;

    if (M <= MR) {
        for (int jc_idx = 0; jc_idx < num_nc_blocks; ++jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int jr = 0; jr < nc; jr += NR) {
                gemm_q8_0_microkernel(
                    K, M,
                    A_qs_packed,
                    A_d_packed,
                    B_qs_packed + (jc + jr) * K,
                    B_d_packed + (jc + jr) * (K / QK8_0),
                    C + (jc + jr),
                    ldc,
                    false);
            }
        };
    } else {
        for (int jc_idx = 0; jc_idx < num_nc_blocks; ++jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int kc = 0; kc < K; kc += KC) {
                const int kc_size = std::min(KC, K - kc);
                const int k_block_offset = kc / QK8_0;
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < M; ir += MR) {
                        gemm_q8_0_microkernel(
                            kc_size, std::min(MR, M - ir),
                            A_qs_packed + (ir) * K + kc * MR,
                            A_d_packed + (ir) * K_BLOCKS + k_block_offset * MR,
                            B_qs_packed + (jc + jr) * K + kc * NR,
                            B_d_packed + (jc + jr) * K_BLOCKS + k_block_offset * NR,
                            C + (ir) * ldc + (jc + jr), ldc, kc != 0);
                    }
                }
            }
        };
    }
}

// Template instantiations
template void gemm_q8_0_compute_packed<at::Half>(int, int, int, const int8_t*, const float*, const int8_t*, const at::Half*, float*, int);
template void gemm_q8_0_compute_packed<at::BFloat16>(int, int, int, const int8_t*, const float*, const int8_t*, const at::BFloat16*, float*, int);

template<typename B_SCALE_TYPE>
void gemm_q4_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed,
    const float* A_d_packed,
    const uint32_t* B_qs_packed,
    const B_SCALE_TYPE* B_d_packed,
    float* C,
    int ldc)
{
    constexpr int QK4_0 = 32;
    constexpr int MR = 4;
    constexpr int NR = 8;
    constexpr int KC = 2048;
    constexpr int NC = 64;

    constexpr int INT4_PER_UINT32 = 8;
    // 1 个 K_BLOCK 在单列中占用的 uint32_t 数量 (32 / 8 = 4)
    constexpr int UINT32_PER_K_BLOCK_COL = QK4_0 / INT4_PER_UINT32;
    // 1 个 K_BLOCK 在一个 NR 宏块中占用的 uint32_t 数量 (4 * NR)
    constexpr int UINT32_PER_K_BLOCK_MACRO = UINT32_PER_K_BLOCK_COL * NR;

    const int K_BLOCKS = K / QK4_0;
    const int num_nc_blocks = (N + NC - 1) / NC;

    if (M <= MR) {
        for (int jc_idx = 0; jc_idx < num_nc_blocks; ++jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int jr = 0; jr < nc; jr += NR) {
                gemm_q4_0_microkernel(
                    K, M,
                    A_qs_packed,
                    A_d_packed,
                    B_qs_packed + (jc + jr) * K_BLOCKS * UINT32_PER_K_BLOCK_COL,
                    B_d_packed + (jc + jr) * K_BLOCKS,
                    C + (jc + jr),
                    ldc,
                    false);
            }
        };
    } else {
        for (int jc_idx = 0; jc_idx < num_nc_blocks; ++jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int kc = 0; kc < K; kc += KC) {
                const int kc_size = std::min(KC, K - kc);
                const int k_block_offset = kc / QK4_0;
                for (int jr = 0; jr < nc; jr += NR) {
                    for (int ir = 0; ir < M; ir += MR) {
                        gemm_q4_0_microkernel(
                            kc_size, std::min(MR, M - ir),
                            A_qs_packed + (ir) * K + kc * MR,
                            A_d_packed + (ir) * K_BLOCKS + k_block_offset * MR,
                            B_qs_packed + (jc + jr) * K_BLOCKS * UINT32_PER_K_BLOCK_COL
                                        + k_block_offset * UINT32_PER_K_BLOCK_MACRO,
                            B_d_packed + (jc + jr) * K_BLOCKS + k_block_offset * NR,
                            C + (ir) * ldc + (jc + jr), ldc, kc != 0);
                    }
                }
            }
        };
    }
}

// Template instantiations
template void gemm_q4_0_compute_packed<at::Half>(int, int, int, const int8_t*, const float*, const uint32_t*, const at::Half*, float*, int);
template void gemm_q4_0_compute_packed<at::BFloat16>(int, int, int, const int8_t*, const float*, const uint32_t*, const at::BFloat16*, float*, int);

} // namespace gemm
