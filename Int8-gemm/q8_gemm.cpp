#include "q8_gemm.h"
#include "utils.h"
#include "vec_simd.h"

#include <ATen/Parallel.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

using ggml_half = at::Half;

#define QK8_0 32
#define MR 8
#define NR 4

// ------------------------- q8 quant block (ARM NEON) -------------------------
template<typename D_TYPE>
static inline void quantize_block_q8_0(const float* x, D_TYPE* y_d, int8_t* y_qs) {
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

// ------------------------- microkernel (NEON) -------------------------
template <int MR_T, typename B_SCALE_TYPE>
static void gemm_q8_0_microkernel_specialized(
    int kc_size,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const B_SCALE_TYPE* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    static_assert(MR_T > 0 && MR_T <= MR, "MR_T invalid");
    const int KC_BLOCKS = kc_size / QK8_0;

    float32x4_t c_v[MR_T];
    if (accumulate) {
        for (int i = 0; i < MR_T; ++i) c_v[i] = vld1q_f32(C + i * ldc);
    } else {
        for (int i = 0; i < MR_T; ++i) c_v[i] = vdupq_n_f32(0.0f);
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const B_SCALE_TYPE* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        int32x4_t sum_v[MR];
        for (int i = 0; i < MR_T; ++i) sum_v[i] = vdupq_n_s32(0);

        for (int k4_step = 0; k4_step < QK8_0 / 4; ++k4_step) {
            int8x16_t a_vec_0 = vld1q_s8(a_ptr);
            a_ptr += 16;
            int8x16_t a_vec_1;
            if constexpr (MR_T > 4) a_vec_1 = vld1q_s8(a_ptr);
            a_ptr += 16;

            int8x16_t b_vec = vld1q_s8(b_ptr);
            b_ptr += 16;

            if constexpr (MR_T > 0) sum_v[0] = vdotq_laneq_s32(sum_v[0], b_vec, a_vec_0, 0);
            if constexpr (MR_T > 1) sum_v[1] = vdotq_laneq_s32(sum_v[1], b_vec, a_vec_0, 1);
            if constexpr (MR_T > 2) sum_v[2] = vdotq_laneq_s32(sum_v[2], b_vec, a_vec_0, 2);
            if constexpr (MR_T > 3) sum_v[3] = vdotq_laneq_s32(sum_v[3], b_vec, a_vec_0, 3);
            if constexpr (MR_T > 4) sum_v[4] = vdotq_laneq_s32(sum_v[4], b_vec, a_vec_1, 0);
            if constexpr (MR_T > 5) sum_v[5] = vdotq_laneq_s32(sum_v[5], b_vec, a_vec_1, 1);
            if constexpr (MR_T > 6) sum_v[6] = vdotq_laneq_s32(sum_v[6], b_vec, a_vec_1, 2);
            if constexpr (MR_T > 7) sum_v[7] = vdotq_laneq_s32(sum_v[7], b_vec, a_vec_1, 3);
        }

        float32x4_t d_b_v;
        if constexpr (std::is_same_v<B_SCALE_TYPE, float>) {
            d_b_v = vld1q_f32(reinterpret_cast<const float*>(bd_ptr));
        } else {
            d_b_v = vcvt_f32_f16(vld1_f16((const __fp16*)bd_ptr));
        }
        bd_ptr += NR;

        float32x4_t d_a_v0 = vld1q_f32(ad_ptr);
        float32x4_t d_a_v1;
        if constexpr (MR_T > 4) d_a_v1 = vld1q_f32(ad_ptr + 4);
        ad_ptr += MR;

        if constexpr (MR_T > 0) c_v[0] = vmlaq_laneq_f32(c_v[0], vmulq_f32(vcvtq_f32_s32(sum_v[0]), d_b_v), d_a_v0, 0);
        if constexpr (MR_T > 1) c_v[1] = vmlaq_laneq_f32(c_v[1], vmulq_f32(vcvtq_f32_s32(sum_v[1]), d_b_v), d_a_v0, 1);
        if constexpr (MR_T > 2) c_v[2] = vmlaq_laneq_f32(c_v[2], vmulq_f32(vcvtq_f32_s32(sum_v[2]), d_b_v), d_a_v0, 2);
        if constexpr (MR_T > 3) c_v[3] = vmlaq_laneq_f32(c_v[3], vmulq_f32(vcvtq_f32_s32(sum_v[3]), d_b_v), d_a_v0, 3);

        if constexpr (MR_T > 4) c_v[4] = vmlaq_laneq_f32(c_v[4], vmulq_f32(vcvtq_f32_s32(sum_v[4]), d_b_v), d_a_v1, 0);
        if constexpr (MR_T > 5) c_v[5] = vmlaq_laneq_f32(c_v[5], vmulq_f32(vcvtq_f32_s32(sum_v[5]), d_b_v), d_a_v1, 1);
        if constexpr (MR_T > 6) c_v[6] = vmlaq_laneq_f32(c_v[6], vmulq_f32(vcvtq_f32_s32(sum_v[6]), d_b_v), d_a_v1, 2);
        if constexpr (MR_T > 7) c_v[7] = vmlaq_laneq_f32(c_v[7], vmulq_f32(vcvtq_f32_s32(sum_v[7]), d_b_v), d_a_v1, 3);
    }

    for (int i = 0; i < MR_T; ++i) {
        vst1q_f32(C + i * ldc, c_v[i]);
    }
}

template<typename B_SCALE_TYPE>
static void gemm_q8_0_microkernel(
    int kc_size, int mr,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const B_SCALE_TYPE* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    assert(mr > 0 && mr <= MR);
    switch (mr) {
        case 1: gemm_q8_0_microkernel_specialized<1>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 2: gemm_q8_0_microkernel_specialized<2>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 3: gemm_q8_0_microkernel_specialized<3>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 4: gemm_q8_0_microkernel_specialized<4>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 5: gemm_q8_0_microkernel_specialized<5>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 6: gemm_q8_0_microkernel_specialized<6>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 7: gemm_q8_0_microkernel_specialized<7>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 8: gemm_q8_0_microkernel_specialized<8>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
    }
}

// ------------------------- pack/quantize A -------------------------
template <ExecutionPolicy Policy, typename T>
static void quantize_pack_A_q8_0(
    int M, int K, const float* A, int lda,
    int8_t* A_qs_packed, float* A_d_packed)
{
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;

    dispatch_for<Policy>(0, num_a_packs, [&](int64_t i_pack) {
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
    });
}

template <ExecutionPolicy Policy>
static void gemm_q8_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const ggml_half* B_d_packed_f16,
    float* C, int ldc)
{
    const int K_BLOCKS = K / QK8_0;
    constexpr int MC = 32;
    constexpr int KC = 1024;
    constexpr int NC = 32;

    const int num_nc_blocks = (N + NC - 1) / NC;

    if (M <= MR) {
        dispatch_for<Policy>(0, num_nc_blocks, [&](int64_t jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int jr = 0; jr < nc; jr += NR) {
                gemm_q8_0_microkernel(
                    K, M,
                    A_qs_packed,
                    A_d_packed,
                    B_qs_packed + (jc + jr) * K,
                    B_d_packed_f16 + (jc + jr) * (K / QK8_0),
                    C + (jc + jr),
                    ldc,
                    false);
            }
        });
    } else {
        dispatch_for<Policy>(0, num_nc_blocks, [&](int64_t jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int kc = 0; kc < K; kc += KC) {
                const int kc_size = std::min(KC, K - kc);
                const int k_block_offset = kc / QK8_0;
                for (int ic = 0; ic < M; ic += MC) {
                    const int mc = std::min(MC, M - ic);
                    for (int jr = 0; jr < nc; jr += NR) {
                        for (int ir = 0; ir < mc; ir += MR) {
                            gemm_q8_0_microkernel(
                                kc_size, std::min(MR, mc - ir),
                                A_qs_packed + (ic + ir) * K + kc * MR,
                                A_d_packed + (ic + ir) * K_BLOCKS + k_block_offset * MR,
                                B_qs_packed + (jc + jr) * K + kc * NR,
                                B_d_packed_f16 + (jc + jr) * K_BLOCKS + k_block_offset * NR,
                                C + (ic + ir) * ldc + (jc + jr), ldc, kc != 0);
                        }
                    }
                }
            }
        });
    }
}

// ------------------------- quantize weight + repack -------------------------
template<typename D_TYPE>
static void quantize_row_q8_0_no_repack(
    const float * src,
    int8_t* dest_qs,
    D_TYPE* dest_d,
    int64_t K)
{
    const int64_t k_blocks = K / QK8_0;
    for (int i = 0; i < k_blocks; ++i) {
        quantize_block_q8_0(src + i * QK8_0, &dest_d[i], dest_qs + i * QK8_0);
    }
}

template<typename D_TYPE>
void repack_B_q8_0_from_ptr(
    int64_t N, int64_t K,
    const int8_t* src_qs, const D_TYPE* src_d,
    int8_t* dest_qs_packed, D_TYPE* dest_d_packed)
{
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

template void repack_B_q8_0_from_ptr<at::Half>(
    int64_t N, int64_t K,
    const int8_t* src_qs, const at::Half* src_d,
    int8_t* dest_qs_packed, at::Half* dest_d_packed);

std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_float) {
    TORCH_CHECK(B_float.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(B_float.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(B_float.device().is_cpu(), "Weight must be on CPU");
    TORCH_CHECK(B_float.scalar_type() == torch::kFloat32, "Weight must be float32 for quantize_weight_only");

    const auto N = B_float.size(0);
    const auto K = B_float.size(1);
    TORCH_CHECK(K % QK8_0 == 0, "K must be multiple of ", QK8_0);

    const int K_BLOCKS = K / QK8_0;
    auto B_qs_tensor = torch::empty({N, K}, torch::kInt8);
    auto B_d_tensor  = torch::empty({N, K_BLOCKS}, torch::kHalf);

    at::parallel_for(0, N, 0, [&](int64_t s, int64_t e) {
        for (int64_t j = s; j < e; ++j) {
            quantize_row_q8_0_no_repack(
                B_float.data_ptr<float>() + j * K,
                B_qs_tensor.data_ptr<int8_t>() + j * K,
                reinterpret_cast<at::Half*>(B_d_tensor.data_ptr<at::Half>() + j * K_BLOCKS),
                K
            );
        }
    });
    return {B_qs_tensor, B_d_tensor};
}

// ------------------------- MoE routing preprocess (topk=1/8) -------------------------
struct MoETokenInfo {
    int32_t token_id;
    int32_t expert_idx_in_tok;
};

template<int TopK>
static void preprocess_moe_routing_template(
    int num_experts, int num_tokens,
    const int32_t* selected_experts,
    std::vector<int>& expert_counts,
    std::vector<int>& expert_starts,
    std::vector<MoETokenInfo>& token_map,
    std::vector<int32_t>& scatter_map)
{
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < TopK; ++k) {
            int expert_id = selected_experts[t * TopK + k];
            if (expert_id >= 0 && expert_id < num_experts) expert_counts[expert_id]++;
        }
    }

    expert_starts[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
        expert_starts[i + 1] = expert_starts[i] + expert_counts[i];
    }

    const int total_expert_tokens = expert_starts[num_experts];
    token_map.resize(total_expert_tokens);
    scatter_map.resize(num_tokens * TopK);

    std::vector<int> current_expert_counts(num_experts, 0);
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < TopK; ++k) {
            int expert_id = selected_experts[t * TopK + k];
            if (expert_id >= 0 && expert_id < num_experts) {
                int pos = expert_starts[expert_id] + current_expert_counts[expert_id]++;
                token_map[pos] = {t, k};
                scatter_map[t * TopK + k] = pos;
            } else {
                scatter_map[t * TopK + k] = -1;
            }
        }
    }
}

static void preprocess_moe_routing(
    int num_experts, int top_k, int num_tokens,
    const int32_t* selected_experts,
    std::vector<int>& expert_counts,
    std::vector<int>& expert_starts,
    std::vector<MoETokenInfo>& token_map,
    std::vector<int32_t>& scatter_map)
{
    expert_counts.assign(num_experts, 0);
    expert_starts.resize(num_experts + 1);

    switch (top_k) {
        case 1:
            preprocess_moe_routing_template<1>(
                num_experts, num_tokens, selected_experts,
                expert_counts, expert_starts, token_map, scatter_map);
            break;
        case 8:
            preprocess_moe_routing_template<8>(
                num_experts, num_tokens, selected_experts,
                expert_counts, expert_starts, token_map, scatter_map);
            break;
        default:
            TORCH_CHECK(false, "Unsupported top_k. Only 1 and 8 are supported.");
    }
}

// ------------------------- pack A from per-token quant (indirect) -------------------------
template <ExecutionPolicy Policy>
static void pack_A_q8_0_from_quantized_indirect(
    int M, int K,
    const int8_t* x_qs_base,
    const float* x_d_base,
    const MoETokenInfo* token_map,
    int token_offset,
    int8_t* A_qs_packed,
    float* A_d_packed)
{
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;

    dispatch_for<Policy>(0, num_a_packs, [&](int64_t i_pack) {
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
    });
}

static inline size_t align_to_64(size_t n) { return (n + 63) & ~63; }

struct ThreadWorkspace {
    char* memory_pool = nullptr;
    size_t current_size = 0;

    int8_t* A_qs_packed1 = nullptr;
    float*  A_d_packed1 = nullptr;
    float*  expert_intermediate1 = nullptr;
    int8_t* A_qs_packed2 = nullptr;
    float*  A_d_packed2 = nullptr;
    float*  temp_row_buffer = nullptr;

    void ensure_size(int m_ceil, int k_hidden, int k_inter, int k_inter_x2) {
        const int k_hidden_k_blocks = k_hidden / QK8_0;
        const int k_inter_k_blocks  = k_inter / QK8_0;

        const size_t size_A_qs1 = (size_t)m_ceil * k_hidden * sizeof(int8_t);
        const size_t size_A_d1  = (size_t)m_ceil * k_hidden_k_blocks * sizeof(float);
        const size_t size_inter1= (size_t)m_ceil * k_inter_x2 * sizeof(float);
        const size_t size_A_qs2 = (size_t)m_ceil * k_inter * sizeof(int8_t);
        const size_t size_A_d2  = (size_t)m_ceil * k_inter_k_blocks * sizeof(float);
        const size_t size_temp  = (size_t)k_hidden * sizeof(float);

        const size_t off_A_qs1 = 0;
        const size_t off_A_d1  = align_to_64(off_A_qs1 + size_A_qs1);
        const size_t off_inter = align_to_64(off_A_d1  + size_A_d1);
        const size_t off_A_qs2 = align_to_64(off_inter + size_inter1);
        const size_t off_A_d2  = align_to_64(off_A_qs2 + size_A_qs2);
        const size_t off_temp  = align_to_64(off_A_d2  + size_A_d2);

        const size_t required = align_to_64(off_temp + size_temp);

        if (required > current_size) {
            if (memory_pool) std::free(memory_pool);
            memory_pool = (char*)std::aligned_alloc(64, required);
            TORCH_CHECK(memory_pool != nullptr, "Failed to allocate ThreadWorkspace memory.");
            current_size = required;
        }

        A_qs_packed1 = (int8_t*)(memory_pool + off_A_qs1);
        A_d_packed1  = (float*)(memory_pool + off_A_d1);
        expert_intermediate1 = (float*)(memory_pool + off_inter);
        A_qs_packed2 = (int8_t*)(memory_pool + off_A_qs2);
        A_d_packed2  = (float*)(memory_pool + off_A_d2);
        temp_row_buffer = (float*)(memory_pool + off_temp);
    }

    ~ThreadWorkspace() {
        if (memory_pool) std::free(memory_pool);
    }
};

static thread_local ThreadWorkspace ws;

// ------------------------- MoE forward (routing already provided) -------------------------
template <typename T>
void moe_q8_forward_ptr_impl(
    const T* x_in_ptr,
    T* y_out_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const int8_t* gate_up_qs_packed,
    const at::Half* gate_up_d_packed,
    const int8_t* down_proj_qs_packed,
    const at::Half* down_proj_d_packed,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size,
    int64_t top_k)
{
    TORCH_CHECK(hidden_dim % QK8_0 == 0, "hidden_dim must be multiple of 32");
    TORCH_CHECK(intermediate_size % QK8_0 == 0, "intermediate_size must be multiple of 32");
    TORCH_CHECK(top_k == 1 || top_k == 8, "top_k must be 1 or 8");

    const int64_t hidden_dim_k_blocks = hidden_dim / QK8_0;

    std::vector<int> expert_counts(num_experts, 0);
    std::vector<int> expert_starts(num_experts + 1, 0);
    std::vector<MoETokenInfo> token_map;
    std::vector<int32_t> scatter_map;

    preprocess_moe_routing(
        (int)num_experts, (int)top_k, (int)num_tokens, selected_experts_ptr,
        expert_counts, expert_starts, token_map, scatter_map);

    const int total_expert_tokens = expert_starts[num_experts];
    if (total_expert_tokens == 0) {
        std::memset(y_out_ptr, 0, (size_t)num_tokens * hidden_dim * sizeof(T));
        return;
    }

    constexpr int M_BLOCK = 32;
    struct MoeTask { int expert_id, num_tokens, global_token_start_pos; };
    std::vector<MoeTask> tasks;
    tasks.reserve((size_t)total_expert_tokens / M_BLOCK + (size_t)num_experts);

    for (int exp_id = 0; exp_id < (int)num_experts; ++exp_id) {
        const int count = expert_counts[exp_id];
        if (count == 0) continue;
        const int start_pos = expert_starts[exp_id];
        for (int offset = 0; offset < count; offset += M_BLOCK) {
            tasks.push_back({exp_id, std::min(M_BLOCK, count - offset), start_pos + offset});
        }
    }
    const int task_count = (int)tasks.size();

    // Quantize input x_in -> x_qs/x_d
    int8_t* x_qs = (int8_t*)std::aligned_alloc(64, (size_t)num_tokens * hidden_dim * sizeof(int8_t));
    float*  x_d  = (float*) std::aligned_alloc(64, (size_t)num_tokens * hidden_dim_k_blocks * sizeof(float));
    TORCH_CHECK(x_qs && x_d, "Failed to allocate x_qs/x_d");

    // expert_intermediate2: [total_expert_tokens, hidden_dim]
    float* expert_intermediate2 = (float*)std::aligned_alloc(64, (size_t)total_expert_tokens * hidden_dim * sizeof(float));
    TORCH_CHECK(expert_intermediate2, "Failed to allocate expert_intermediate2");

    const int omp_max_threads = omp_get_max_threads();
    std::atomic<int> task_counter;
    task_counter.store(0, std::memory_order_relaxed);

    #pragma omp parallel num_threads(omp_max_threads)
    {
        ws.ensure_size(M_BLOCK, (int)hidden_dim, (int)intermediate_size, (int)(2 * intermediate_size));

        // A) per-token quantize
        #pragma omp for
        for (int64_t i = 0; i < num_tokens; ++i) {
            const T* src_row = x_in_ptr + i * hidden_dim;
            for (int j = 0; j < (int)hidden_dim; ++j) ws.temp_row_buffer[j] = (float)src_row[j];
            for (int k_block = 0; k_block < (int)hidden_dim_k_blocks; ++k_block) {
                quantize_block_q8_0(
                    ws.temp_row_buffer + k_block * QK8_0,
                    x_d + i * hidden_dim_k_blocks + k_block,
                    x_qs + i * hidden_dim + k_block * QK8_0
                );
            }
        }

        #pragma omp barrier

        auto process_task = [&](int task_idx) {
            const auto& task = tasks[task_idx];
            const int exp_id = task.expert_id;
            const int count = task.num_tokens;
            const int global_start_pos = task.global_token_start_pos;

            const int8_t* gate_up_qs_ptr = gate_up_qs_packed + (int64_t)exp_id * (2 * intermediate_size) * hidden_dim;
            const at::Half* gate_up_d_ptr = gate_up_d_packed + (int64_t)exp_id * (2 * intermediate_size) * (hidden_dim / QK8_0);

            const int8_t* down_proj_qs_ptr = down_proj_qs_packed + (int64_t)exp_id * hidden_dim * intermediate_size;
            const at::Half* down_proj_d_ptr = down_proj_d_packed + (int64_t)exp_id * hidden_dim * (intermediate_size / QK8_0);

            pack_A_q8_0_from_quantized_indirect<ExecutionPolicy::Sequential>(
                count, (int)hidden_dim, x_qs, x_d, token_map.data(), global_start_pos,
                ws.A_qs_packed1, ws.A_d_packed1
            );

            gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
                count, (int)(2 * intermediate_size), (int)hidden_dim,
                ws.A_qs_packed1, ws.A_d_packed1,
                gate_up_qs_ptr, reinterpret_cast<const ggml_half*>(gate_up_d_ptr),
                ws.expert_intermediate1, (int)(2 * intermediate_size)
            );

            silu_and_mul<ExecutionPolicy::Sequential>(ws.expert_intermediate1, count, (int)(2 * intermediate_size));

            quantize_pack_A_q8_0<ExecutionPolicy::Sequential, float>(
                count, (int)intermediate_size,
                ws.expert_intermediate1, (int)(2 * intermediate_size),
                ws.A_qs_packed2, ws.A_d_packed2
            );

            gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
                count, (int)hidden_dim, (int)intermediate_size,
                ws.A_qs_packed2, ws.A_d_packed2,
                down_proj_qs_ptr, reinterpret_cast<const ggml_half*>(down_proj_d_ptr),
                expert_intermediate2 + (int64_t)global_start_pos * hidden_dim, (int)hidden_dim
            );
        };

        while (true) {
            int idx = task_counter.fetch_add(1, std::memory_order_relaxed);
            if (idx >= task_count) break;
            process_task(idx);
        }

        #pragma omp barrier

        // Scatter & weight accumulate to y_out
        #pragma omp for
        for (int64_t t = 0; t < num_tokens; ++t) {
            float* acc = ws.temp_row_buffer;
            std::memset(acc, 0, (size_t)hidden_dim * sizeof(float));

            for (int k = 0; k < (int)top_k; ++k) {
                const int scatter_idx = (int)(t * top_k + k);
                const int src_row_idx = scatter_map[scatter_idx];
                if (src_row_idx == -1) continue;

                const float w = routing_weights_ptr[scatter_idx];
                const float* src_row = expert_intermediate2 + (int64_t)src_row_idx * hidden_dim;
                for (int j = 0; j < (int)hidden_dim; ++j) acc[j] += w * src_row[j];
            }

            T* dst = y_out_ptr + t * hidden_dim;
            for (int j = 0; j < (int)hidden_dim; ++j) dst[j] = (T)acc[j];
        }
    }

    std::free(x_qs);
    std::free(x_d);
    std::free(expert_intermediate2);
}

template void moe_q8_forward_ptr_impl<at::Half>(
    const at::Half*, at::Half*,
    const float*, const int32_t*,
    const int8_t*, const at::Half*,
    const int8_t*, const at::Half*,
    int64_t,int64_t,int64_t,int64_t,int64_t);

template void moe_q8_forward_ptr_impl<at::BFloat16>(
    const at::BFloat16*, at::BFloat16*,
    const float*, const int32_t*,
    const int8_t*, const at::Half*,
    const int8_t*, const at::Half*,
    int64_t,int64_t,int64_t,int64_t,int64_t);
