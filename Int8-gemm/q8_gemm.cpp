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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>

using ggml_half = at::Half;

// ========================== Profiling Macros ==========================
// Set to 1 to enable profiling, 0 to disable
#define ENABLE_PROFILING 0

namespace {
    struct ProfilingEntry {
        double stage_pre_ms = 0.0;  // Preprocessing (memory + routing)
        double stage_abc_ms = 0.0;  // Distributed Execution (A: Quantize + B: Expert Compute + C1: Scatter)
        double stage_c2_ms = 0.0;   // Final Reduce
        double total_ms = 0.0;      // Total time
    };

    struct ProfilingStats {
        std::vector<double> stage_pre_times;
        std::vector<double> stage_abc_times;
        std::vector<double> stage_c2_times;
        std::vector<double> total_times;
        int count = 0;

        void add(const ProfilingEntry& entry) {
            stage_pre_times.push_back(entry.stage_pre_ms);
            stage_abc_times.push_back(entry.stage_abc_ms);
            stage_c2_times.push_back(entry.stage_c2_ms);
            total_times.push_back(entry.total_ms);
            count++;
        }

        void print(int64_t num_tokens) const {
            if (count == 0) return;

            auto avg = [](const std::vector<double>& v) -> double {
                if (v.empty()) return 0.0;
                return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            };

            std::cout << "\n========== MoE Profiling [num_tokens=" << num_tokens
                      << ", iterations=" << count << "] ==========\n";
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Stage PRE (Preprocessing):    " << avg(stage_pre_times) << " ms\n";
            std::cout << "  - Memory allocation\n";
            std::cout << "  - Routing preprocessing\n";
            std::cout << "  - Task generation\n";
            std::cout << "Stage ABC (Distributed Exec): " << avg(stage_abc_times) << " ms\n";
            std::cout << "  - A: Quantize Input\n";
            std::cout << "  - B: Expert Compute (GEMM1+Act+GEMM2)\n";
            std::cout << "  - C1: Local Scatter & Accumulate\n";
            std::cout << "Stage C2 (Final Reduce):      " << avg(stage_c2_times) << " ms\n";
            std::cout << "Total:                        " << avg(total_times) << " ms\n";
            std::cout << "============================================================\n";
        }
    };

    static std::map<int64_t, ProfilingStats> g_profiling_data;
    static std::mutex g_profiling_mutex;

    struct ProfilingContext {
        ProfilingEntry entry;
        std::chrono::high_resolution_clock::time_point total_start;
        std::chrono::high_resolution_clock::time_point stage_pre_start;
        std::chrono::high_resolution_clock::time_point stage_abc_start;
        std::chrono::high_resolution_clock::time_point stage_c2_start;

        ProfilingContext() : total_start(std::chrono::high_resolution_clock::now()) {}

        void start_stage_pre() {
            stage_pre_start = std::chrono::high_resolution_clock::now();
        }

        void end_stage_pre() {
            auto end = std::chrono::high_resolution_clock::now();
            entry.stage_pre_ms = std::chrono::duration<double, std::milli>(end - stage_pre_start).count();
        }

        void start_stage_abc() {
            stage_abc_start = std::chrono::high_resolution_clock::now();
        }

        void end_stage_abc() {
            auto end = std::chrono::high_resolution_clock::now();
            entry.stage_abc_ms = std::chrono::duration<double, std::milli>(end - stage_abc_start).count();
        }

        void start_stage_c2() {
            stage_c2_start = std::chrono::high_resolution_clock::now();
        }

        void end_stage_c2() {
            auto end = std::chrono::high_resolution_clock::now();
            entry.stage_c2_ms = std::chrono::duration<double, std::milli>(end - stage_c2_start).count();
        }

        void finish_total() {
            auto total_end = std::chrono::high_resolution_clock::now();
            entry.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        }
    };

    static thread_local ProfilingContext* g_profiling_ctx = nullptr;
}

#if ENABLE_PROFILING
    #define PROFILE_INIT(num_tokens) \
        ProfilingContext _profile_ctx; \
        g_profiling_ctx = &_profile_ctx; \
        int64_t _profile_num_tokens = (num_tokens);

    #define PROFILE_STAGE_BEGIN(name) \
        g_profiling_ctx->start_##name();

    #define PROFILE_STAGE_END(name) \
        g_profiling_ctx->end_##name();

    #define PROFILE_FINISH() \
        do { \
            g_profiling_ctx->finish_total(); \
            std::lock_guard<std::mutex> _profile_lock(g_profiling_mutex); \
            g_profiling_data[_profile_num_tokens].add(g_profiling_ctx->entry); \
            g_profiling_ctx = nullptr; \
        } while(0)

    #define PRINT_PROFILING() \
        do { \
            std::cout << "\n====================== MoE Profiling Summary ======================\n"; \
            for (const auto& kv : g_profiling_data) { \
                kv.second.print(kv.first); \
            } \
            std::cout << "===============================================================\n"; \
        } while(0)
#else
    #define PROFILE_INIT(num_tokens)
    #define PROFILE_STAGE_BEGIN(name)
    #define PROFILE_STAGE_END(name)
    #define PROFILE_FINISH()
    #define PRINT_PROFILING()
#endif

// Auto-print profiling on exit
#if ENABLE_PROFILING
static struct ProfilingCleanup {
    ~ProfilingCleanup() {
        PRINT_PROFILING();
    }
} g_profiling_cleanup;
#endif

// ========================== End Profiling Macros ==========================

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

    // c_v[row][0] covers columns 0-3
    // c_v[row][1] covers columns 4-7
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
        // 32-bit int accumulators for dot products
        // sum_v[row][0] for cols 0-3, sum_v[row][1] for cols 4-7
        int32x4_t sum_v[MR_T][2];
        for (int i = 0; i < MR_T; ++i) {
            sum_v[i][0] = vdupq_n_s32(0);
            sum_v[i][1] = vdupq_n_s32(0);
        }

        // QK8_0 = 32. Step 4. Total 8 iters.
        for (int k4_step = 0; k4_step < QK8_0 / 4; ++k4_step) {
            // Load A: MR=4. pack_A interleaves 4 bytes per row.
            // 4 rows * 4 bytes = 16 bytes. Fits in 1 register.
            int8x16_t a_vec = vld1q_s8(a_ptr);
            a_ptr += 16;

            // Load B: NR=8. pack_B interleaves 4 bytes per col.
            // 8 cols * 4 bytes = 32 bytes. Fits in 2 registers.
            int8x16_t b_vec_0 = vld1q_s8(b_ptr);      // Cols 0-3
            int8x16_t b_vec_1 = vld1q_s8(b_ptr + 16); // Cols 4-7
            b_ptr += 32;

            // Dot Products
            // A lane 'i' corresponds to Row 'i'
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

        // Load Scales
        // B Scales: NR=8. 8 scales.
        float32x4_t d_b_v0, d_b_v1;
        if constexpr (std::is_same_v<B_SCALE_TYPE, float>) {
            d_b_v0 = vld1q_f32(bd_ptr);
            d_b_v1 = vld1q_f32(bd_ptr + 4);
        } else {
            // Load 8 halfs, convert to 2x float32x4
            float16x8_t b_scales_f16 = vld1q_f16((const __fp16 *) bd_ptr);
            d_b_v0 = vcvt_f32_f16(vget_low_f16(b_scales_f16));
            d_b_v1 = vcvt_f32_f16(vget_high_f16(b_scales_f16));
        }
        bd_ptr += NR;

        // A Scales: MR=4. 4 scales. Fits in 1 register.
        float32x4_t d_a_v = vld1q_f32(ad_ptr);
        ad_ptr += MR;

        // FMA: Accumulate (Sum * B_scale) * A_scale
        // We broadcast the specific A scale for the row.
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
    }
}

// ------------------------- pack/quantize A -------------------------
static void quantize_pack_A_q8_0(
    int M, int K, const float* A, int lda,
    int8_t* A_qs_packed, float* A_d_packed)
{
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

static void gemm_q8_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const ggml_half* B_d_packed_f16,
    float* C, int ldc)
{
    const int K_BLOCKS = K / QK8_0;
    constexpr int KC = 2048;
    constexpr int NC = 64;

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
                    B_d_packed_f16 + (jc + jr) * (K / QK8_0),
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
                            B_d_packed_f16 + (jc + jr) * K_BLOCKS + k_block_offset * NR,
                            C + (ir) * ldc + (jc + jr), ldc, kc != 0);
                    }
                }
            }
        };
    }
}

// ------------------------- quantize weight + repack -------------------------
template<typename SRC_T, typename D_TYPE>
static void quantize_row_q8_0_no_repack(
    const SRC_T * src,
    int8_t* dest_qs,
    D_TYPE* dest_d,
    int64_t K)
{
    const int64_t k_blocks = K / QK8_0;
    for (int i = 0; i < k_blocks; ++i) {
        if constexpr (std::is_same_v<SRC_T, float>) {
            quantize_block_q8_0(src + i * QK8_0, &dest_d[i], dest_qs + i * QK8_0);
        } else {
            float buf[QK8_0];
            const SRC_T* blk = src + i * QK8_0;
            for (int j = 0; j < QK8_0; ++j) {
                buf[j] = static_cast<float>(blk[j]);
            }
            quantize_block_q8_0(buf, &dest_d[i], dest_qs + i * QK8_0);
        }
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

std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_input) {
    TORCH_CHECK(B_input.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(B_input.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(B_input.device().is_cpu(), "Weight must be on CPU");
    
    auto val_type = B_input.scalar_type();
    TORCH_CHECK(val_type == torch::kFloat32 || val_type == torch::kHalf || val_type == torch::kBFloat16, 
        "Weight must be float32, float16 or bfloat16 for quantize_weight_only");

    const auto N = B_input.size(0);
    const auto K = B_input.size(1);
    TORCH_CHECK(K % QK8_0 == 0, "K must be multiple of ", QK8_0);

    const int K_BLOCKS = K / QK8_0;
    auto B_qs_tensor = torch::empty({N, K}, torch::kInt8);
    auto B_d_tensor  = torch::empty({N, K_BLOCKS}, torch::kHalf);

    auto launch_quant = [&](auto type_dummy) {
        using T = decltype(type_dummy);
        const T* B_ptr = B_input.data_ptr<T>();
        at::parallel_for(0, N, 0, [&](int64_t s, int64_t e) {
            for (int64_t j = s; j < e; ++j) {
                quantize_row_q8_0_no_repack(
                    B_ptr + j * K,
                    B_qs_tensor.data_ptr<int8_t>() + j * K,
                    reinterpret_cast<at::Half*>(B_d_tensor.data_ptr<at::Half>() + j * K_BLOCKS),
                    K
                );
            }
        });
    };

    if (val_type == torch::kFloat32) {
        launch_quant(float{});
    } else if (val_type == torch::kHalf) {
        launch_quant(at::Half{});
    } else if (val_type == torch::kBFloat16) {
        launch_quant(at::BFloat16{});
    }

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
        case 10:
            preprocess_moe_routing_template<10>(
                num_experts, num_tokens, selected_experts,
                expert_counts, expert_starts, token_map, scatter_map);
            break;
        default:
            TORCH_CHECK(false, "Unsupported top_k. Only 1,8,10 are supported.");
    }
}

// ------------------------- pack A from per-token quant (indirect) -------------------------
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

    void ensure_size(int m_ceil, int k_hidden, int k_inter) {
        const int k_hidden_k_blocks = k_hidden / QK8_0;
        const int k_inter_k_blocks  = k_inter / QK8_0;
        const int k_inter_x2 = k_inter * 2;

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

// ------------------------- Global NUMA Memory Pool -------------------------
struct NumaBufferPool {
    size_t capacity_tokens = 0;
    int64_t hidden_dim = 0;
    int64_t intermediate_shard = 0;
    int64_t tp_size = 0;
    int64_t top_k = 0;

    // Per-TP-rank buffers
    std::vector<int8_t*> x_qs_ptrs;
    std::vector<float*> x_d_ptrs;
    std::vector<float*> expert_out_ptrs;
    std::vector<float*> y_partial_ptrs;
    
    // Shared intermediate buffer for split-task optimization (num_tokens=1)
    std::vector<float*> expert_inter_ptrs;

    void ensure_capacity(int64_t req_tokens, int64_t req_hidden, int64_t req_inter_shard, int64_t req_tp, int64_t req_topk) {
        
        // Check if configuration changed
        bool shape_changed = (req_hidden != hidden_dim) || (req_inter_shard != intermediate_shard) || (req_tp != tp_size) || (req_topk != top_k);
        bool size_grew = (req_tokens > (int64_t)capacity_tokens);

        if (!shape_changed && !size_grew && !x_qs_ptrs.empty()) {
            return;
        }

        if (capacity_tokens > 0) {
            free_buffers_unsafe();
        }

        if (shape_changed) {
            hidden_dim = req_hidden;
            intermediate_shard = req_inter_shard;
            tp_size = req_tp;
            top_k = req_topk;
        }

        // Growth strategy
        size_t new_cap = (capacity_tokens == 0) ? (size_t)req_tokens : capacity_tokens;
        if (size_grew || new_cap < (size_t)req_tokens) {
            if (new_cap == 0) new_cap = 128; 
            while (new_cap < (size_t)req_tokens) {
                new_cap *= 2;
            }
            capacity_tokens = new_cap;
        }

        x_qs_ptrs.resize(tp_size);
        x_d_ptrs.resize(tp_size);
        expert_out_ptrs.resize(tp_size);
        y_partial_ptrs.resize(tp_size);
        expert_inter_ptrs.resize(tp_size);

        const size_t x_qs_sz = capacity_tokens * hidden_dim * sizeof(int8_t);
        const size_t x_d_sz  = capacity_tokens * (hidden_dim / QK8_0) * sizeof(float);
        const size_t out_sz  = capacity_tokens * top_k * hidden_dim * sizeof(float);
        const size_t part_sz = capacity_tokens * hidden_dim * sizeof(float);
        
        // Intermediate buffer size: max_tasks * (2 * intermediate_shard).
        // Max tasks = tokens * top_k.
        const size_t inter_sz = capacity_tokens * top_k * (2 * intermediate_shard) * sizeof(float);

        for (int i = 0; i < tp_size; ++i) {
            x_qs_ptrs[i] = (int8_t*)numa_alloc_onnode(x_qs_sz, i);
            x_d_ptrs[i]  = (float*) numa_alloc_onnode(x_d_sz, i);
            expert_out_ptrs[i] = (float*)numa_alloc_onnode(out_sz, i);
            y_partial_ptrs[i]  = (float*)numa_alloc_onnode(part_sz, i);
            expert_inter_ptrs[i] = (float*)numa_alloc_onnode(inter_sz, i);

            TORCH_CHECK(x_qs_ptrs[i] && x_d_ptrs[i] && expert_out_ptrs[i] && y_partial_ptrs[i] && expert_inter_ptrs[i], 
                        "Numa allocation failed for node ", i);
        }
    }

    void free_buffers_unsafe() {
        if (capacity_tokens == 0) return;
        const size_t x_qs_sz = capacity_tokens * hidden_dim * sizeof(int8_t);
        const size_t x_d_sz  = capacity_tokens * (hidden_dim / QK8_0) * sizeof(float);
        const size_t out_sz  = capacity_tokens * top_k * hidden_dim * sizeof(float);
        const size_t part_sz = capacity_tokens * hidden_dim * sizeof(float);
        const size_t inter_sz = capacity_tokens * top_k * (2 * intermediate_shard) * sizeof(float);

        for (size_t i = 0; i < x_qs_ptrs.size(); ++i) {
            if (x_qs_ptrs[i]) numa_free(x_qs_ptrs[i], x_qs_sz);
            if (x_d_ptrs[i])  numa_free(x_d_ptrs[i], x_d_sz);
            if (expert_out_ptrs[i]) numa_free(expert_out_ptrs[i], out_sz);
            if (y_partial_ptrs[i])  numa_free(y_partial_ptrs[i], part_sz);
            if (expert_inter_ptrs[i]) numa_free(expert_inter_ptrs[i], inter_sz);
        }
        x_qs_ptrs.clear();
        x_d_ptrs.clear();
        expert_out_ptrs.clear();
        y_partial_ptrs.clear();
        expert_inter_ptrs.clear();
    }

    ~NumaBufferPool() {
        free_buffers_unsafe();
    }
};

static NumaBufferPool g_numa_pool;

// ------------------------- MoE forward (routing already provided) -------------------------
template <typename T>
void moe_q8_forward_ptr_impl(
    const T* x_in_ptr,
    T* y_out_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,

    const int8_t* const* gate_up_qs_tp,
    const at::Half* const* gate_up_d_tp,
    const int8_t* const* down_proj_qs_tp,
    const at::Half* const* down_proj_d_tp,

    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size,
    int64_t tp_size,
    int64_t top_k)
{
    PROFILE_INIT(num_tokens);

    TORCH_CHECK(tp_size >= 1, "tp_size must be >= 1");
    TORCH_CHECK(numa_available() != -1, "libnuma not available");

    // H_CHUNK removed - computation too small for chunking to be beneficial

    const int64_t intermediate_shard = intermediate_size / tp_size;

    // 1. Prepare Pool Memory
    PROFILE_STAGE_BEGIN(stage_pre);
    g_numa_pool.ensure_capacity(num_tokens, hidden_dim, intermediate_shard, tp_size, top_k);
    
    const auto& pool_x_qs = g_numa_pool.x_qs_ptrs;
    const auto& pool_x_d  = g_numa_pool.x_d_ptrs;
    const auto& pool_expert_out = g_numa_pool.expert_out_ptrs;
    const auto& pool_y_partial  = g_numa_pool.y_partial_ptrs;
    const auto& pool_expert_inter = g_numa_pool.expert_inter_ptrs;

    const int64_t hidden_dim_k_blocks = hidden_dim / QK8_0;

    // 2. Preprocess Routing (Global)
    std::vector<int> expert_counts(num_experts, 0);
    std::vector<int> expert_starts(num_experts + 1, 0);
    std::vector<MoETokenInfo> token_map;
    std::vector<int32_t> scatter_map;

    preprocess_moe_routing(
        (int)num_experts, (int)top_k, (int)num_tokens, selected_experts_ptr,
        expert_counts, expert_starts, token_map, scatter_map);

    const int total_expert_tokens = expert_starts[num_experts];
    if (total_expert_tokens == 0) {
        std::memset(y_out_ptr, 0, (size_t)num_tokens * (size_t)hidden_dim * sizeof(T));
        PROFILE_FINISH();
        return;
    }

    // 3. Generate Common Compute Tasks
    constexpr int M_BLOCK = 32;
    struct MoeTask { int expert_id, num_tokens, global_token_start_pos; };
    std::vector<MoeTask> global_tasks;
    global_tasks.reserve((size_t)total_expert_tokens / M_BLOCK + (size_t)num_experts);

    for (int exp_id = 0; exp_id < (int)num_experts; ++exp_id) {
        const int count = expert_counts[exp_id];
        if (count == 0) continue;
        const int start_pos = expert_starts[exp_id];
        for (int offset = 0; offset < count; offset += M_BLOCK) {
            global_tasks.push_back({exp_id, std::min(M_BLOCK, count - offset), start_pos + offset});
        }
    }
    PROFILE_STAGE_END(stage_pre);

    // 4. Launch Distributed Execution (Pipeline A -> B -> C1)
    PROFILE_STAGE_BEGIN(stage_abc);
    std::vector<std::future<void>> futures;
    futures.reserve((size_t)tp_size);

    for (int tp = 0; tp < (int)tp_size; ++tp) {
        auto exec = nanovllm::NumaExecutorManager::get(tp);

        // Submit one fused job per TP rank
        futures.emplace_back(exec->launcher->submit([&, tp, exec]() {

            // --- Stage A: Quantize Input (Local Replica) ---
            int8_t* local_x_qs = pool_x_qs[tp];
            float*  local_x_d  = pool_x_d[tp];

            exec->pool->parallel_for_static(0, num_tokens, [&](int64_t i) {
                const T* src_row = x_in_ptr + i * hidden_dim;
                for (int k_block = 0; k_block < (int)hidden_dim_k_blocks; ++k_block) {
                    float y_block[QK8_0];
                    for (int j = 0; j < QK8_0; ++j) {
                        y_block[j] = static_cast<float>(src_row[k_block * QK8_0 + j]);
                    }
                    quantize_block_q8_0(
                        y_block,
                        local_x_d + i * hidden_dim_k_blocks + k_block,
                        local_x_qs + i * hidden_dim + k_block * QK8_0
                    );
                }
            });

            // --- Stage B: Expert Compute ---
            const int pool_num_threads = exec->pool->num_threads();

            if (global_tasks.size() * 2 <= (size_t)pool_num_threads) {
                // Optimization: Direct thread-id based task assignment when tasks are few
                // Each thread directly computes its assigned task based on thread ID
                const int num_active_threads = (int)global_tasks.size() * 2;
                exec->pool->execute_per_thread(num_active_threads, [&](int tid) {
                    // Compute task assignment from thread ID
                    const int task_idx = tid / 2;
                    const int split_idx = tid % 2;  // 0: First half, 1: Second half

                    const MoeTask& task = global_tasks[task_idx];
                    ws.ensure_size(M_BLOCK, (int)hidden_dim, (int)intermediate_shard);

                    const int exp_id = task.expert_id;
                    const int count = task.num_tokens;
                    const int global_start_pos = task.global_token_start_pos;

                    const int64_t H = hidden_dim;
                    const int64_t Ish = intermediate_shard;
                    const int64_t H_BLK = H / QK8_0;

                    // ============ Phase 1: GEMM 1 Split ============
                    // Weights Pointers (GateUp is 2*Ish cols. Split at Ish)
                    const int64_t col_offset = split_idx * Ish;
                    const int8_t* gate_up_qs_ptr = gate_up_qs_tp[tp] + (int64_t)exp_id * (2 * Ish) * H
                                                 + col_offset * H;
                    const at::Half* gate_up_d_ptr = gate_up_d_tp[tp] + (int64_t)exp_id * (2 * Ish) * H_BLK
                                                  + col_offset * H_BLK;

                    // Output buffer in shared memory (indexed by task_idx to avoid conflict)
                    float* shared_inter_out = pool_expert_inter[tp] + (int64_t)task_idx * (2 * Ish) + col_offset;

                    // 1. Pack A (Both threads do it locally, redundancy is cheap)
                    pack_A_q8_0_from_quantized_indirect(
                        count, (int)hidden_dim, local_x_qs, local_x_d, token_map.data(), global_start_pos,
                        ws.A_qs_packed1, ws.A_d_packed1
                    );

                    // 2. GEMM 1 Partial
                    gemm_q8_0_compute_packed(
                        count, (int)Ish, (int)hidden_dim,
                        ws.A_qs_packed1, ws.A_d_packed1,
                        gate_up_qs_ptr, reinterpret_cast<const ggml_half*>(gate_up_d_ptr),
                        shared_inter_out, (int)(2 * Ish) // Stride is full width
                    );

                    // Barrier: Wait for all threads to complete Phase 1
                    exec->pool->barrier();

                    // ============ Phase 2: SiLU + GEMM 2 Split ============
                    const int64_t Ish_BLK = Ish / QK8_0;

                    // Read from shared intermediate
                    float* shared_inter_in = pool_expert_inter[tp] + (int64_t)task_idx * (2 * Ish);

                    // 3. Activation (SiLU) - Copy to local and execute (Redundant but lock-free)
                    std::memcpy(ws.expert_intermediate1, shared_inter_in, count * (2 * Ish) * sizeof(float));
                    silu_and_mul(ws.expert_intermediate1, count, (int)(2 * Ish));

                    // 4. Quantize Intermediate (Redundant)
                    quantize_pack_A_q8_0(
                        count, (int)intermediate_shard,
                        ws.expert_intermediate1, (int)(2 * intermediate_shard),
                        ws.A_qs_packed2, ws.A_d_packed2
                    );

                    // 5. GEMM 2 Partial (Split output dimension H)
                    const int64_t H_split = H / 2;
                    const int64_t N_chunk = (split_idx == 0) ? H_split : (H - H_split);
                    const int64_t col_offset2 = split_idx * H_split;

                    const int8_t* down_qs_ptr = down_proj_qs_tp[tp] + (int64_t)exp_id * H * Ish
                                              + col_offset2 * Ish;
                    const at::Half* down_d_ptr = down_proj_d_tp[tp] + (int64_t)exp_id * H * Ish_BLK
                                               + col_offset2 * Ish_BLK;

                    float* out = pool_expert_out[tp] + (int64_t)global_start_pos * H + col_offset2;

                    gemm_q8_0_compute_packed(
                        count, (int)N_chunk, (int)Ish,
                        ws.A_qs_packed2, ws.A_d_packed2,
                        down_qs_ptr, reinterpret_cast<const ggml_half*>(down_d_ptr),
                        out, (int)H // Stride is full H
                    );
                });

            } else {
                // Original Fused Pipeline for larger batch sizes
                exec->pool->parallel_for(0, (int64_t)global_tasks.size(), [&](int64_t task_idx) {
                    const MoeTask& task = global_tasks[task_idx];
                    
                    // Ensure TLS workspace
                    ws.ensure_size(M_BLOCK, (int)hidden_dim, (int)intermediate_shard);

                    const int exp_id = task.expert_id;
                    const int count = task.num_tokens;
                    const int global_start_pos = task.global_token_start_pos;

                    const int64_t H = hidden_dim;
                    const int64_t Ish = intermediate_shard;
                    const int64_t H_BLK = H / QK8_0;
                    const int64_t Ish_BLK = Ish / QK8_0;

                    // Select TP-shard weights
                    const int8_t* gate_up_qs_ptr = gate_up_qs_tp[tp] + (int64_t)exp_id * (2 * Ish) * H;
                    const at::Half* gate_up_d_ptr = gate_up_d_tp[tp] + (int64_t)exp_id * (2 * Ish) * H_BLK;
                    const int8_t* down_qs_ptr = down_proj_qs_tp[tp] + (int64_t)exp_id * H * Ish;
                    const at::Half* down_d_ptr = down_proj_d_tp[tp] + (int64_t)exp_id * H * Ish_BLK;

                    // 1. Pack A (Indirect)
                    pack_A_q8_0_from_quantized_indirect(
                        count, (int)hidden_dim, local_x_qs, local_x_d, token_map.data(), global_start_pos,
                        ws.A_qs_packed1, ws.A_d_packed1
                    );

                    // 2. GEMM 1 (Gate + Up)
                    gemm_q8_0_compute_packed(
                        count, (int)(2 * intermediate_shard), (int)hidden_dim,
                        ws.A_qs_packed1, ws.A_d_packed1,
                        gate_up_qs_ptr, reinterpret_cast<const ggml_half*>(gate_up_d_ptr),
                        ws.expert_intermediate1, (int)(2 * intermediate_shard)
                    );

                    // 3. Activation
                    silu_and_mul(ws.expert_intermediate1, count, (int)(2 * intermediate_shard));

                    // 4. Quantize Intermediate
                    quantize_pack_A_q8_0(
                        count, (int)intermediate_shard,
                        ws.expert_intermediate1, (int)(2 * intermediate_shard),
                        ws.A_qs_packed2, ws.A_d_packed2
                    );

                    // 5. GEMM 2 (Down)
                    float* out = pool_expert_out[tp] + (int64_t)global_start_pos * (int64_t)hidden_dim;
                    gemm_q8_0_compute_packed(
                        count, (int)hidden_dim, (int)intermediate_shard,
                        ws.A_qs_packed2, ws.A_d_packed2,
                        down_qs_ptr, reinterpret_cast<const ggml_half*>(down_d_ptr),
                        out, (int)hidden_dim
                    );
                });
            }

            // --- Stage C1: Local Scatter & Accumulate ---
            float* local_y_partial = pool_y_partial[tp];
            float* local_expert_out = pool_expert_out[tp];

            exec->pool->parallel_for_static(0, num_tokens, [&](int64_t t) {
                float* acc = local_y_partial + t * hidden_dim;
                std::memset(acc, 0, hidden_dim * sizeof(float));

                for (int k = 0; k < (int)top_k; ++k) {
                    const int scatter_idx = (int)(t * top_k + k);
                    const int src_row_idx = scatter_map[scatter_idx];
                    if (src_row_idx == -1) continue;

                    const float w = routing_weights_ptr[scatter_idx];
                    const float* src_row = local_expert_out + (int64_t)src_row_idx * (int64_t)hidden_dim;

                    for (int j = 0; j < (int)hidden_dim; ++j) {
                        acc[j] += w * src_row[j];
                    }
                }
            });
        }));
    }

    // Wait for all nodes to finish Stages A, B, and C1
    for (auto& f : futures) f.get();
    PROFILE_STAGE_END(stage_abc);

    // 5. Stage C2: Final Reduce on Node 0
    PROFILE_STAGE_BEGIN(stage_c2);
    auto exec0 = nanovllm::NumaExecutorManager::get(0);

    exec0->pool->parallel_for_static(0, num_tokens, [&](int64_t t) {
        float* acc = pool_y_partial[0] + t * hidden_dim;

        // Sum partials from other TP nodes
        for (int tp = 1; tp < (int)tp_size; ++tp) {
            const float* src = pool_y_partial[tp] + t * hidden_dim;
            for (int j = 0; j < (int)hidden_dim; ++j) {
                acc[j] += src[j];
            }
        }

        // Cast to output type
        T* dst = y_out_ptr + t * hidden_dim;
        for (int j = 0; j < (int)hidden_dim; ++j) {
            dst[j] = (T)acc[j];
        }
    });

    PROFILE_STAGE_END(stage_c2);

    PROFILE_FINISH();
}

template void moe_q8_forward_ptr_impl<at::Half>(
    const at::Half*, at::Half*,
    const float*, const int32_t*,
    const int8_t* const*, const at::Half* const*,
    const int8_t* const*, const at::Half* const*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);

template void moe_q8_forward_ptr_impl<at::BFloat16>(
    const at::BFloat16*, at::BFloat16*,
    const float*, const int32_t*,
    const int8_t* const*, const at::Half* const*,
    const int8_t* const*, const at::Half* const*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
