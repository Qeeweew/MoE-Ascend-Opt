#include "moe_common.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace moe {

static inline size_t align_to_64(size_t n) { return (n + 63) & ~63; }

// ==================== ThreadWorkspace ====================

void ThreadWorkspace::ensure_size(int m_ceil, int k_hidden, int k_inter) {
    constexpr int QK8_0 = 32;
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
        if (!memory_pool) {
            throw std::bad_alloc();
        }
        current_size = required;
    }

    A_qs_packed1 = (int8_t*)(memory_pool + off_A_qs1);
    A_d_packed1  = (float*)(memory_pool + off_A_d1);
    expert_intermediate1 = (float*)(memory_pool + off_inter);
    A_qs_packed2 = (int8_t*)(memory_pool + off_A_qs2);
    A_d_packed2  = (float*)(memory_pool + off_A_d2);
    temp_row_buffer = (float*)(memory_pool + off_temp);
}

ThreadWorkspace::~ThreadWorkspace() {
    if (memory_pool) std::free(memory_pool);
}

// ==================== NumaBufferPool ====================

void NumaBufferPool::ensure_capacity(int64_t req_tokens, int64_t req_hidden,
                                     int64_t req_inter_shard, int64_t req_tp, int64_t req_topk) {
    bool shape_changed = (req_hidden != hidden_dim) || (req_inter_shard != intermediate_shard) ||
                        (req_tp != tp_size) || (req_topk != top_k);
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

    constexpr int QK8_0 = 32;
    const size_t x_qs_sz = capacity_tokens * hidden_dim * sizeof(int8_t);
    const size_t x_d_sz  = capacity_tokens * (hidden_dim / QK8_0) * sizeof(float);
    const size_t out_sz  = capacity_tokens * top_k * hidden_dim * sizeof(float);
    const size_t part_sz = capacity_tokens * hidden_dim * sizeof(float);
    const size_t inter_sz = capacity_tokens * top_k * (2 * intermediate_shard) * sizeof(float);

    for (int i = 0; i < tp_size; ++i) {
        x_qs_ptrs[i] = (int8_t*)numa_alloc_onnode(x_qs_sz, i);
        x_d_ptrs[i]  = (float*) numa_alloc_onnode(x_d_sz, i);
        expert_out_ptrs[i] = (float*)numa_alloc_onnode(out_sz, i);
        y_partial_ptrs[i]  = (float*)numa_alloc_onnode(part_sz, i);
        expert_inter_ptrs[i] = (float*)numa_alloc_onnode(inter_sz, i);

        if (!x_qs_ptrs[i] || !x_d_ptrs[i] || !expert_out_ptrs[i] || !y_partial_ptrs[i] || !expert_inter_ptrs[i]) {
            throw std::runtime_error("Numa allocation failed for node " + std::to_string(i));
        }
    }
}

void NumaBufferPool::free_buffers_unsafe() {
    if (capacity_tokens == 0) return;
    constexpr int QK8_0 = 32;
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

NumaBufferPool::~NumaBufferPool() {
    free_buffers_unsafe();
}

// Global instances
NumaBufferPool g_numa_pool;
thread_local ThreadWorkspace g_thread_workspace;

// ==================== Routing Preprocessing ====================

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

void preprocess_moe_routing(
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
            throw std::runtime_error("Unsupported top_k. Only 1,8,10 are supported.");
    }
}

} // namespace moe
