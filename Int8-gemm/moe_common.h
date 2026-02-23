#pragma once
#include <cstdint>
#include <torch/extension.h>
#include <vector>
#include <numa.h>
#include "gemm_kernels.h"

namespace moe {

// Use gemm::MoETokenInfo instead of defining our own
using gemm::MoETokenInfo;

// Task descriptor for MoE computation
struct MoeTask {
    int expert_id;
    int num_tokens;
    int global_token_start_pos;
};

// Workspace for thread-local computation
struct ThreadWorkspace {
    char* memory_pool = nullptr;
    size_t current_size = 0;

    int8_t* A_qs_packed1 = nullptr;
    float*  A_d_packed1 = nullptr;
    float*  expert_intermediate1 = nullptr;
    int8_t* A_qs_packed2 = nullptr;
    float*  A_d_packed2 = nullptr;
    float*  temp_row_buffer = nullptr;

    void ensure_size(int m_ceil, int k_hidden, int k_inter);
    ~ThreadWorkspace();
};

// Unified NUMA buffer pool for both Q8_0 and Q4_0
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
    std::vector<float*> expert_inter_ptrs;

    void ensure_capacity(int64_t req_tokens, int64_t req_hidden,
                        int64_t req_inter_shard, int64_t req_tp, int64_t req_topk);
    void free_buffers_unsafe();
    ~NumaBufferPool();
};

// Global pool instance
extern NumaBufferPool g_numa_pool;

// Thread-local workspace
extern thread_local ThreadWorkspace g_thread_workspace;

// Routing preprocessing
void preprocess_moe_routing(
    int num_experts, int top_k, int num_tokens,
    const int32_t* selected_experts,
    std::vector<int>& expert_counts,
    std::vector<int>& expert_starts,
    std::vector<MoETokenInfo>& token_map,
    std::vector<int32_t>& scatter_map);

} // namespace moe
