#include "q8_gemm.h"
#include "utils.h"
#include "vec_simd.h"
#include "moe_common.h"

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

// ------------------------- Using Declarations (gemm_kernels.cpp) -------------------------
// Functions defined in gemm_kernels.cpp:
// - quantize_block_q8_0: Quantize a block of floats to Q8_0 format
// - pack_A_q8_0: Pack A matrix after quantization (Q8_0)
// - pack_A_q8_0_from_quantized_indirect: Pack A matrix from already quantized data (Q8_0)
// - silu_and_mul: Activation function wrapper (uses vec_simd.h implementation)
// - gemm_q8_0_microkernel: Q8_0 GEMM microkernel dispatch wrapper
using gemm::quantize_block_q8_0;
using gemm::pack_A_q8_0;
using gemm::pack_A_q8_0_from_quantized_indirect;
using gemm::silu_and_mul;
using gemm::gemm_q8_0_microkernel;

// ------------------------- Using Declarations (moe_common.h) -------------------------
// Types and functions already defined in moe_common.h/cpp:
// - MoETokenInfo: Token information for MoE routing
// - MoeTask: Task description for distributed execution
// - ThreadWorkspace: Thread-local workspace (global instance: moe::g_thread_workspace)
// - NumaBufferPool: NUMA-aware buffer pool (global instance: moe::g_numa_pool)
// - preprocess_moe_routing: Routing preprocessing function


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

// ------------------------- Reference to Global Instances (moe_common.cpp) -------------------------
static thread_local moe::ThreadWorkspace& ws = moe::g_thread_workspace;
static moe::NumaBufferPool& g_numa_pool = moe::g_numa_pool;

// Using declarations for convenience (moe_common.h)
using moe::MoETokenInfo;  // This is now an alias for gemm::MoETokenInfo
using moe::MoeTask;
using moe::preprocess_moe_routing;

// ------------------------- Unified MoE forward (Q8_0 and Q4_0) -------------------------
// Template function that works for both Q8_0 and Q4_0
// QT: QuantType (Q8_0 or Q4_0)
// T: data type (at::Half or at::BFloat16)
template <quant::QuantType QT, typename T>
void moe_forward_ptr_impl(
    const T* x_in_ptr,
    T* y_out_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,

    const void* const* gate_up_qs_tp,
    const at::Half* const* gate_up_d_tp,
    const void* const* down_proj_qs_tp,
    const at::Half* const* down_proj_d_tp,

    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size,
    int64_t tp_size,
    int64_t top_k)
{
    PROFILE_INIT(num_tokens);

    TORCH_CHECK(tp_size >= 1, "tp_size must be >= 1");
    TORCH_CHECK(numa_available() != -1, "libnuma not available");

    const int64_t intermediate_shard = intermediate_size / tp_size;

    // 1. Prepare Pool Memory
    PROFILE_STAGE_BEGIN(stage_pre);
    g_numa_pool.ensure_capacity(num_tokens, hidden_dim, intermediate_shard, tp_size, top_k);

    // Safety check: ensure pool was properly initialized
    if (g_numa_pool.x_qs_ptrs.size() != static_cast<size_t>(tp_size) ||
        g_numa_pool.x_d_ptrs.size() != static_cast<size_t>(tp_size)) {
        throw std::runtime_error("NumaBufferPool initialization failed!");
    }

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
                const int num_active_threads = (int)global_tasks.size() * 2;
                exec->pool->execute_per_thread(num_active_threads, [&](int tid) {
                    const int task_idx = tid / 2;
                    const int split_idx = tid % 2;

                    const MoeTask& task = global_tasks[task_idx];
                    ws.ensure_size(M_BLOCK, (int)hidden_dim, (int)intermediate_shard);

                    const int exp_id = task.expert_id;
                    const int count = task.num_tokens;
                    const int global_start_pos = task.global_token_start_pos;

                    const int64_t H = hidden_dim;
                    const int64_t Ish = intermediate_shard;
                    const int64_t H_BLK = H / QK8_0;
                    const int64_t Ish_BLK = Ish / QK8_0;

                    // ============ Phase 1: GEMM 1 Split ============
                    const int64_t col_offset = split_idx * Ish;

                    // Type-safe weight pointer access based on quantization type
                    using StorageType = typename quant::QuantTraits<QT>::storage_type;

                    // Compile-time type check for Q8_0
                    if constexpr (QT == quant::QuantType::Q8_0) {
                        static_assert(std::is_same_v<StorageType, int8_t>, "Q8_0 storage must be int8_t");
                    } else if constexpr (QT == quant::QuantType::Q4_0) {
                        static_assert(std::is_same_v<StorageType, uint32_t>, "Q4_0 storage must be uint32_t");
                    }

                    // Explicit pointer type conversion
                    const StorageType* const* gate_up_qs_typed = reinterpret_cast<const StorageType* const*>(gate_up_qs_tp);
                    const StorageType* const* down_proj_qs_typed = reinterpret_cast<const StorageType* const*>(down_proj_qs_tp);

                    const StorageType* gate_up_qs_ptr = gate_up_qs_typed[tp] 
                        + (int64_t)exp_id * (2 * Ish) * H / (QT == quant::QuantType::Q4_0 ? 8 : 1)
                        + col_offset * H / (QT == quant::QuantType::Q4_0 ? 8 : 1);
                    const at::Half* gate_up_d_ptr = gate_up_d_tp[tp] + (int64_t)exp_id * (2 * Ish) * H_BLK
                                                      + col_offset * H_BLK;

                    float* shared_inter_out = pool_expert_inter[tp] + (int64_t)task_idx * (2 * Ish) + col_offset;

                    // 1. Pack A
                    gemm::pack_A_q8_0_from_quantized_indirect(
                        count, (int)hidden_dim, local_x_qs, local_x_d,
                        token_map.data(), global_start_pos,
                        ws.A_qs_packed1, ws.A_d_packed1
                    );

                    // 2. GEMM 1 Partial - Dispatch based on quantization type
                    if constexpr (QT == quant::QuantType::Q8_0) {
                        gemm::gemm_q8_0_compute_packed(
                            count, (int)Ish, (int)hidden_dim,
                            ws.A_qs_packed1, ws.A_d_packed1,
                            gate_up_qs_ptr,
                            gate_up_d_ptr,
                            shared_inter_out, (int)(2 * Ish)
                        );
                    } else {
                        gemm::gemm_q4_0_compute_packed(
                            count, (int)Ish, (int)hidden_dim,
                            ws.A_qs_packed1, ws.A_d_packed1,
                            gate_up_qs_ptr,
                            gate_up_d_ptr,
                            shared_inter_out, (int)(2 * Ish)
                        );
                    }

                    // Barrier
                    exec->pool->barrier();

                    // ============ Phase 2: SiLU + GEMM 2 Split ============
                    float* shared_inter_in = pool_expert_inter[tp] + (int64_t)task_idx * (2 * Ish);

                    std::memcpy(ws.expert_intermediate1, shared_inter_in, count * (2 * Ish) * sizeof(float));
                    gemm::silu_and_mul(ws.expert_intermediate1, count, (int)(2 * Ish));

                    gemm::pack_A_q8_0(
                        count, (int)intermediate_shard,
                        ws.expert_intermediate1, (int)(2 * intermediate_shard),
                        ws.A_qs_packed2, ws.A_d_packed2
                    );

                    const int64_t H_split = H / 2;
                    const int64_t N_chunk = (split_idx == 0) ? H_split : (H - H_split);
                    const int64_t col_offset2 = split_idx * H_split;

                    const StorageType* down_qs_ptr = down_proj_qs_typed[tp] 
                        + (int64_t)exp_id * H * Ish / (QT == quant::QuantType::Q4_0 ? 8 : 1)
                        + col_offset2 * Ish / (QT == quant::QuantType::Q4_0 ? 8 : 1);

                    const at::Half* down_d_ptr = down_proj_d_tp[tp] + (int64_t)exp_id * H * Ish_BLK
                                                        + col_offset2 * Ish_BLK;

                    float* out = pool_expert_out[tp] + (int64_t)global_start_pos * H + col_offset2;

                    // 5. GEMM 2 Partial - Dispatch based on quantization type
                    if constexpr (QT == quant::QuantType::Q8_0) {
                        gemm::gemm_q8_0_compute_packed(
                            count, (int)N_chunk, (int)Ish,
                            ws.A_qs_packed2, ws.A_d_packed2,
                            reinterpret_cast<const int8_t*>(down_qs_ptr),
                            reinterpret_cast<const ggml_half*>(down_d_ptr),
                            out, (int)H
                        );
                    } else {
                        gemm::gemm_q4_0_compute_packed(
                            count, (int)N_chunk, (int)Ish,
                            ws.A_qs_packed2, ws.A_d_packed2,
                            reinterpret_cast<const uint32_t*>(down_qs_ptr),
                            reinterpret_cast<const ggml_half*>(down_d_ptr),
                            out, (int)H
                        );
                    }
                });

            } else {
                // Original Fused Pipeline for larger batch sizes
                exec->pool->parallel_for(0, (int64_t)global_tasks.size(), [&](int64_t task_idx) {
                    const MoeTask& task = global_tasks[task_idx];

                    ws.ensure_size(M_BLOCK, (int)hidden_dim, (int)intermediate_shard);

                    const int exp_id = task.expert_id;
                    const int count = task.num_tokens;
                    const int global_start_pos = task.global_token_start_pos;

                    const int64_t H = hidden_dim;
                    const int64_t Ish = intermediate_shard;
                    const int64_t H_BLK = H / QK8_0;
                    const int64_t Ish_BLK = Ish / QK8_0;

                    // Select TP-shard weights (type-safe based on quantization type)
                    using StorageType = typename quant::QuantTraits<QT>::storage_type;

                    // Compile-time type check
                    if constexpr (QT == quant::QuantType::Q8_0) {
                        static_assert(std::is_same_v<StorageType, int8_t>, "Q8_0 storage must be int8_t");
                    } else if constexpr (QT == quant::QuantType::Q4_0) {
                        static_assert(std::is_same_v<StorageType, uint32_t>, "Q4_0 storage must be uint32_t");
                    }

                    // Explicit pointer type conversion
                    const StorageType* const* gate_up_qs_typed = reinterpret_cast<const StorageType* const*>(gate_up_qs_tp);
                    const StorageType* const* down_proj_qs_typed = reinterpret_cast<const StorageType* const*>(down_proj_qs_tp);

                    const StorageType* gate_up_qs_ptr = gate_up_qs_typed[tp] + (int64_t)exp_id * (2 * Ish) * H / (QT == quant::QuantType::Q4_0 ? 8 : 1);
                    const at::Half* gate_up_d_ptr = gate_up_d_tp[tp] + (int64_t)exp_id * (2 * Ish) * H_BLK;
                    const StorageType* down_qs_ptr = down_proj_qs_typed[tp] + (int64_t)exp_id * H * Ish / (QT == quant::QuantType::Q4_0 ? 8 : 1);
                    const at::Half* down_d_ptr = down_proj_d_tp[tp] + (int64_t)exp_id * H * Ish_BLK;

                    // 1. Pack A (Indirect)
                    gemm::pack_A_q8_0_from_quantized_indirect(
                        count, (int)hidden_dim, local_x_qs, local_x_d,
                        token_map.data(), global_start_pos,
                        ws.A_qs_packed1, ws.A_d_packed1
                    );

                    // 2. GEMM 1 (Gate + Up) - Dispatch based on quantization type
                    if constexpr (QT == quant::QuantType::Q8_0) {
                        gemm::gemm_q8_0_compute_packed(
                            count, (int)(2 * intermediate_shard), (int)hidden_dim,
                            ws.A_qs_packed1, ws.A_d_packed1,
                            reinterpret_cast<const int8_t*>(gate_up_qs_ptr),
                            reinterpret_cast<const ggml_half*>(gate_up_d_ptr),
                            ws.expert_intermediate1, (int)(2 * intermediate_shard)
                        );
                    } else {
                        gemm::gemm_q4_0_compute_packed(
                            count, (int)(2 * intermediate_shard), (int)hidden_dim,
                            ws.A_qs_packed1, ws.A_d_packed1,
                            gate_up_qs_ptr,
                            reinterpret_cast<const ggml_half*>(gate_up_d_ptr),
                            ws.expert_intermediate1, (int)(2 * intermediate_shard)
                        );
                    }

                    // 3. Activation
                    gemm::silu_and_mul(ws.expert_intermediate1, count, (int)(2 * intermediate_shard));

                    // 4. Quantize Intermediate
                    gemm::pack_A_q8_0(
                        count, (int)intermediate_shard,
                        ws.expert_intermediate1, (int)(2 * intermediate_shard),
                        ws.A_qs_packed2, ws.A_d_packed2
                    );

                    // 5. GEMM 2 (Down) - Dispatch based on quantization type
                    float* out = pool_expert_out[tp] + (int64_t)global_start_pos * (int64_t)hidden_dim;
                    if constexpr (QT == quant::QuantType::Q8_0) {
                        gemm::gemm_q8_0_compute_packed(
                            count, (int)hidden_dim, (int)intermediate_shard,
                            ws.A_qs_packed2, ws.A_d_packed2,
                            reinterpret_cast<const int8_t*>(down_qs_ptr),
                            reinterpret_cast<const ggml_half*>(down_d_ptr),
                            out, (int)hidden_dim
                        );
                    } else {
                        gemm::gemm_q4_0_compute_packed(
                            count, (int)hidden_dim, (int)intermediate_shard,
                            ws.A_qs_packed2, ws.A_d_packed2,
                            down_qs_ptr,
                            reinterpret_cast<const ggml_half*>(down_d_ptr),
                            out, (int)hidden_dim
                        );
                    }
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

// Template instantiations for unified moe_forward_ptr_impl
template void moe_forward_ptr_impl<quant::QuantType::Q8_0, at::Half>(
    const at::Half*, at::Half*,
    const float*, const int32_t*,
    const void* const*, const at::Half* const*,
    const void* const*, const at::Half* const*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);

template void moe_forward_ptr_impl<quant::QuantType::Q8_0, at::BFloat16>(
    const at::BFloat16*, at::BFloat16*,
    const float*, const int32_t*,
    const void* const*, const at::Half* const*,
    const void* const*, const at::Half* const*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);

template void moe_forward_ptr_impl<quant::QuantType::Q4_0, at::Half>(
    const at::Half*, at::Half*,
    const float*, const int32_t*,
    const void* const*, const at::Half* const*,
    const void* const*, const at::Half* const*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);

template void moe_forward_ptr_impl<quant::QuantType::Q4_0, at::BFloat16>(
    const at::BFloat16*, at::BFloat16*,
    const float*, const int32_t*,
    const void* const*, const at::Half* const*,
    const void* const*, const at::Half* const*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
