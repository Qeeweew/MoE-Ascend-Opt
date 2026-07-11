#include "moe_infer.h"
#include "q8_gemm.h"
#include "numa_threadpool.h"

#include <chrono>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <numa.h>

static inline void* numa_alloc_or_throw(size_t bytes, int node) {
    void* p = numa_alloc_onnode(bytes, node);
    if (!p) throw std::runtime_error("numa_alloc_onnode failed");
    std::memset(p, 0, bytes);
    return p;
}

MoEInfer::MoEInfer(int64_t num_experts, int64_t hidden_size, int64_t intermediate_size,
                   quant::QuantType quant_type)
    : num_experts_(num_experts),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      quant_type_(quant_type) {

    routing_counts_.assign((size_t)num_experts_, 0);

    tp_size_ = nanovllm::detail::read_env_int64("NANOVLLM_TP_SIZE", 2);

    TORCH_CHECK(tp_size_ >= 1, "tp_size must be >= 1");
    TORCH_CHECK(hidden_size_ % 32 == 0, "hidden_size must be multiple of 32");
    TORCH_CHECK(intermediate_size_ % tp_size_ == 0, "intermediate_size must be divisible by tp_size");
    intermediate_shard_ = intermediate_size_ / tp_size_;
    TORCH_CHECK(intermediate_shard_ % 32 == 0, "intermediate_size/tp_size must be multiple of 32");

    TORCH_CHECK(numa_available() != -1, "libnuma not available on this system");
    int max_node = numa_max_node();
    TORCH_CHECK(tp_size_ - 1 <= max_node,
                "tp_size=", tp_size_, " exceeds max NUMA node=", max_node);

    // bytes per tp (tp-local buffers contain all experts for that tp)
    const int64_t H = hidden_size_;
    const int64_t Ish = intermediate_shard_;
    const int64_t H_BLK = H / 32;
    const int64_t Ish_BLK = Ish / 32;

    // Calculate storage based on quantization type
    // Q4_0 uses half the storage of Q8_0 (2 elements per byte)
    if (quant_type_ == quant::QuantType::Q4_0) {
        gate_up_qs_bytes_per_tp_ =
            (size_t)num_experts_ * (size_t)(2 * Ish) * (size_t)H * sizeof(uint32_t) / 8;
        down_qs_bytes_per_tp_ =
            (size_t)num_experts_ * (size_t)H * (size_t)Ish * sizeof(uint32_t) / 8;
    } else {
        gate_up_qs_bytes_per_tp_ =
            (size_t)num_experts_ * (size_t)(2 * Ish) * (size_t)H * sizeof(int8_t);
        down_qs_bytes_per_tp_ =
            (size_t)num_experts_ * (size_t)H * (size_t)Ish * sizeof(int8_t);
    }

    gate_up_d_bytes_per_tp_  =
        (size_t)num_experts_ * (size_t)(2 * Ish) * (size_t)H_BLK * sizeof(uint16_t);
    down_d_bytes_per_tp_  =
        (size_t)num_experts_ * (size_t)H * (size_t)Ish_BLK * sizeof(uint16_t);

    gate_up_qs_tp_.resize((size_t)tp_size_, nullptr);
    gate_up_d_tp_.resize((size_t)tp_size_, nullptr);
    down_proj_qs_tp_.resize((size_t)tp_size_, nullptr);
    down_proj_d_tp_.resize((size_t)tp_size_, nullptr);

    for (int tp = 0; tp < (int)tp_size_; ++tp) {
        int node = tp;
        gate_up_qs_tp_[(size_t)tp]   = (int8_t*)numa_alloc_or_throw(gate_up_qs_bytes_per_tp_, node);
        gate_up_d_tp_[(size_t)tp]    = (uint16_t*)numa_alloc_or_throw(gate_up_d_bytes_per_tp_, node);
        down_proj_qs_tp_[(size_t)tp] = (int8_t*)numa_alloc_or_throw(down_qs_bytes_per_tp_, node);
        down_proj_d_tp_[(size_t)tp]  = (uint16_t*)numa_alloc_or_throw(down_d_bytes_per_tp_, node);
    }
}

void MoEInfer::record_routing(const int32_t* routing_ids,
                              const int32_t* compute_ids,
                              int64_t num_tokens, int64_t top_k) {
    int64_t valid = valid_tokens_.load();
    if (valid <= 0 || valid > num_tokens) valid = num_tokens;
    std::lock_guard<std::mutex> guard(routing_stats_mutex_);
    for (int64_t t = 0; t < valid; ++t) {
        for (int64_t k = 0; k < top_k; ++k) {
            const int64_t idx = t * top_k + k;
            const int32_t expert = routing_ids[idx];
            if (expert >= 0 && expert < num_experts_) {
                routing_counts_[(size_t)expert] += 1;
                routing_total_ += 1;
                if (compute_ids[idx] >= 0) routing_miss_ += 1;
            }
        }
    }
    routing_calls_ += 1;
}

torch::Tensor MoEInfer::take_routing_stats() {
    auto result = torch::zeros({num_experts_ + 3}, torch::TensorOptions().dtype(torch::kInt64));
    auto* out = result.data_ptr<int64_t>();
    std::lock_guard<std::mutex> guard(routing_stats_mutex_);
    std::copy(routing_counts_.begin(), routing_counts_.end(), out);
    out[num_experts_] = routing_total_;
    out[num_experts_ + 1] = routing_miss_;
    out[num_experts_ + 2] = routing_calls_;
    std::fill(routing_counts_.begin(), routing_counts_.end(), 0);
    routing_total_ = routing_miss_ = routing_calls_ = 0;
    return result;
}

void MoEInfer::reset_routing_stats() {
    std::lock_guard<std::mutex> guard(routing_stats_mutex_);
    std::fill(routing_counts_.begin(), routing_counts_.end(), 0);
    routing_total_ = routing_miss_ = routing_calls_ = 0;
}

MoEInfer::~MoEInfer() {
    for (int tp = 0; tp < (int)tp_size_; ++tp) {
        if (gate_up_qs_tp_[(size_t)tp])   { numa_free(gate_up_qs_tp_[(size_t)tp], gate_up_qs_bytes_per_tp_); gate_up_qs_tp_[(size_t)tp] = nullptr; }
        if (gate_up_d_tp_[(size_t)tp])    { numa_free(gate_up_d_tp_[(size_t)tp], gate_up_d_bytes_per_tp_); gate_up_d_tp_[(size_t)tp] = nullptr; }
        if (down_proj_qs_tp_[(size_t)tp]) { numa_free(down_proj_qs_tp_[(size_t)tp], down_qs_bytes_per_tp_); down_proj_qs_tp_[(size_t)tp] = nullptr; }
        if (down_proj_d_tp_[(size_t)tp])  { numa_free(down_proj_d_tp_[(size_t)tp], down_d_bytes_per_tp_); down_proj_d_tp_[(size_t)tp] = nullptr; }
    }
}

size_t MoEInfer::calculate_qs_bytes(int64_t rows, int64_t cols) const {
    if (quant_type_ == quant::QuantType::Q4_0) {
        // Q4_0: 2 elements per byte, stored as uint32_t
        return ((size_t)rows * (size_t)cols + 1) / 2;
    } else {
        // Q8_0: 1 element per byte
        return (size_t)rows * (size_t)cols;
    }
}

void MoEInfer::store_quantized_weights_repack(
    const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d,
    const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d) {

    TORCH_CHECK(gate_up_qs.device().is_cpu() && gate_up_d.device().is_cpu(), "gate_up must be CPU");
    TORCH_CHECK(down_proj_qs.device().is_cpu() && down_proj_d.device().is_cpu(), "down_proj must be CPU");
    TORCH_CHECK(gate_up_qs.is_contiguous() && gate_up_d.is_contiguous(), "gate_up must be contiguous");
    TORCH_CHECK(down_proj_qs.is_contiguous() && down_proj_d.is_contiguous(), "down_proj must be contiguous");

    // Support both FP16 and BF16 scales
    TORCH_CHECK(gate_up_d.scalar_type() == torch::kFloat16 || gate_up_d.scalar_type() == torch::kBFloat16,
                "gate_up_d must be float16 or bfloat16");
    TORCH_CHECK(down_proj_d.scalar_type() == torch::kFloat16 || down_proj_d.scalar_type() == torch::kBFloat16,
                "down_proj_d must be float16 or bfloat16");
    TORCH_CHECK(gate_up_d.scalar_type() == down_proj_d.scalar_type(),
                "gate_up_d and down_proj_d must have the same dtype");

    TORCH_CHECK(gate_up_qs.dim() == 3, "gate_up_qs must be [E,2I,H]");
    TORCH_CHECK(down_proj_qs.dim() == 3, "down_proj_qs must be [E,H,I]");

    const int64_t E = gate_up_qs.size(0);
    TORCH_CHECK(E == num_experts_, "E mismatch");

    const int64_t I = intermediate_size_;
    const int64_t Ish = intermediate_shard_;
    const int64_t H = hidden_size_;
    const int64_t H_BLK = H / 32;
    const int64_t Ish_BLK = Ish / 32;
    const int64_t I_BLK = I / 32;

    // Determine scale type and use uint16_t for repack (simplified handling)
    const bool use_bf16 = (gate_up_d.scalar_type() == torch::kBFloat16);
    scale_dtype_ = use_bf16 ? at::kBFloat16 : at::kHalf;

    auto run_tp_parallel = [&](const auto& fn_per_expert) {
        nanovllm::NumaThreadPool* current_pool = nanovllm::NumaThreadPool::current_thread_pool();
        int64_t inline_tp = -1;
        for (int64_t tp = 0; tp < tp_size_; ++tp) {
            auto exec = nanovllm::NumaExecutorManager::get((int)tp);
            if (exec->pool.get() == current_pool) {
                inline_tp = tp;
                break;
            }
        }
        if (inline_tp < 0 && tp_size_ > 0) {
            inline_tp = 0;
        }

        std::vector<std::future<void>> futures;
        futures.reserve((size_t)(tp_size_ > 0 ? tp_size_ - 1 : 0));
        for (int64_t tp = 0; tp < tp_size_; ++tp) {
            auto exec = nanovllm::NumaExecutorManager::get((int)tp);
            if (tp == inline_tp) {
                exec->pool->parallel_for_static(0, E, [&](int64_t exp) {
                    fn_per_expert(tp, exp);
                });
                continue;
            }

            futures.emplace_back(std::async(std::launch::async, [pool = exec->pool, tp, E, &fn_per_expert]() {
                pool->parallel_for_static(0, E, [&](int64_t exp) {
                    fn_per_expert(tp, exp);
                });
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
    };

    // Dispatch based on quantization type
    if (quant_type_ == quant::QuantType::Q4_0) {
        constexpr int64_t Q4_PACK = 8;

        // Q4_0 path: weights are uint32_t (packed 4-bit)
        TORCH_CHECK(gate_up_qs.scalar_type() == torch::kInt32 || gate_up_qs.scalar_type() == torch::kUInt32,
                    "Q4_0 gate_up_qs must be int32 (uint32 storage)");
        TORCH_CHECK(down_proj_qs.scalar_type() == torch::kInt32 || down_proj_qs.scalar_type() == torch::kUInt32,
                    "Q4_0 down_proj_qs must be int32 (uint32 storage)");

        const uint32_t* src_gate_up_qs = static_cast<const uint32_t*>(gate_up_qs.data_ptr());
        const uint32_t* src_down_qs = static_cast<const uint32_t*>(down_proj_qs.data_ptr());

        // Use uint16_t for scale pointers (simplified handling as per requirements)
        const uint16_t* src_gate_up_d = static_cast<const uint16_t*>(gate_up_d.data_ptr());
        const uint16_t* src_down_d = static_cast<const uint16_t*>(down_proj_d.data_ptr());

        auto repack_gate_up = [&](int64_t tp, int64_t exp) {
            uint32_t* dst_qs =
                reinterpret_cast<uint32_t*>(gate_up_qs_tp_[(size_t)tp]) + exp * (2 * Ish * H / Q4_PACK);
            uint16_t* dst_d =
                reinterpret_cast<uint16_t*>(gate_up_d_tp_[(size_t)tp]) + exp * (2 * Ish * H_BLK);

            auto repack_half = [&](int64_t src_row_offset, uint32_t* half_dst_qs, uint16_t* half_dst_d) {
                const int64_t src_qs_off =
                    exp * (2 * I * H / Q4_PACK) + src_row_offset * (H / Q4_PACK) + tp * (Ish * H / Q4_PACK);
                const int64_t src_d_off =
                    exp * (2 * I * H_BLK) + src_row_offset * H_BLK + tp * (Ish * H_BLK);

                gemm::repack_B_q4_0<uint16_t>(
                    Ish, H,
                    src_gate_up_qs + src_qs_off,
                    src_gate_up_d + src_d_off,
                    half_dst_qs, half_dst_d
                );
            };

            repack_half(/*src_row_offset=*/0, dst_qs, dst_d);
            repack_half(/*src_row_offset=*/I, dst_qs + Ish * H / Q4_PACK, dst_d + Ish * H_BLK);
        };

        auto repack_down = [&](int64_t tp, int64_t exp) {
            thread_local std::vector<uint32_t> tmp_qs;
            thread_local std::vector<uint16_t> tmp_d;
            tmp_qs.resize((size_t)H * (size_t)Ish / Q4_PACK);
            tmp_d.resize((size_t)H * (size_t)Ish_BLK);

            for (int64_t row = 0; row < H; ++row) {
                const int64_t src_row_qs_off = exp * (H * I / Q4_PACK) + row * (I / Q4_PACK) + tp * (Ish / Q4_PACK);
                std::memcpy(tmp_qs.data() + row * (Ish / Q4_PACK),
                            src_down_qs + src_row_qs_off,
                            (size_t)(Ish / Q4_PACK) * sizeof(uint32_t));

                const int64_t src_row_d_off = exp * (H * I_BLK) + row * I_BLK + tp * Ish_BLK;
                std::memcpy(tmp_d.data() + row * Ish_BLK,
                            src_down_d + src_row_d_off,
                            (size_t)Ish_BLK * sizeof(uint16_t));
            }

            uint32_t* dst_qs =
                reinterpret_cast<uint32_t*>(down_proj_qs_tp_[(size_t)tp]) + exp * (H * Ish / Q4_PACK);
            uint16_t* dst_d =
                reinterpret_cast<uint16_t*>(down_proj_d_tp_[(size_t)tp]) + exp * (H * Ish_BLK);

            gemm::repack_B_q4_0<uint16_t>(
                /*N=*/H, /*K=*/Ish,
                tmp_qs.data(), tmp_d.data(),
                dst_qs, dst_d
            );
        };

        run_tp_parallel([&](int64_t tp, int64_t exp) {
            repack_gate_up(tp, exp);
            repack_down(tp, exp);
        });
    } else {
        // Q8_0 path: weights are int8_t
        TORCH_CHECK(gate_up_qs.scalar_type() == torch::kInt8, "Q8_0 gate_up_qs must be int8");
        TORCH_CHECK(down_proj_qs.scalar_type() == torch::kInt8, "Q8_0 down_proj_qs must be int8");

        const int8_t* src_gate_up_qs = gate_up_qs.data_ptr<int8_t>();
        const int8_t* src_down_qs = down_proj_qs.data_ptr<int8_t>();

        // Use uint16_t for scale pointers (simplified handling as per requirements)
        const uint16_t* src_gate_up_d = reinterpret_cast<const uint16_t*>(gate_up_d.data_ptr());
        const uint16_t* src_down_d = reinterpret_cast<const uint16_t*>(down_proj_d.data_ptr());

        auto repack_gate_up = [&](int64_t tp, int64_t exp) {
            int8_t* dst_qs = gate_up_qs_tp_[(size_t)tp] + exp * (2 * Ish * H);
            uint16_t* dst_d =
                reinterpret_cast<uint16_t*>(gate_up_d_tp_[(size_t)tp]) + exp * (2 * Ish * H_BLK);

            auto repack_half = [&](int64_t src_row_offset, int8_t* half_dst_qs, uint16_t* half_dst_d) {
                const int64_t src_qs_off = exp * (2 * I * H) + src_row_offset * H + tp * (Ish * H);
                const int64_t src_d_off = exp * (2 * I * H_BLK) + src_row_offset * H_BLK + tp * (Ish * H_BLK);

                gemm::repack_B_q8_0<uint16_t>(
                    Ish, H,
                    src_gate_up_qs + src_qs_off,
                    src_gate_up_d + src_d_off,
                    half_dst_qs, half_dst_d
                );
            };

            repack_half(/*src_row_offset=*/0, dst_qs, dst_d);
            repack_half(/*src_row_offset=*/I, dst_qs + Ish * H, dst_d + Ish * H_BLK);
        };

        auto repack_down = [&](int64_t tp, int64_t exp) {
            thread_local std::vector<int8_t> tmp_qs;
            thread_local std::vector<uint16_t> tmp_d;
            tmp_qs.resize((size_t)H * (size_t)Ish);
            tmp_d.resize((size_t)H * (size_t)Ish_BLK);

            for (int64_t row = 0; row < H; ++row) {
                const int64_t src_row_qs_off = exp * (H * I) + row * I + tp * Ish;
                std::memcpy(tmp_qs.data() + row * Ish,
                            src_down_qs + src_row_qs_off,
                            (size_t)Ish * sizeof(int8_t));

                const int64_t src_row_d_off = exp * (H * I_BLK) + row * I_BLK + tp * Ish_BLK;
                std::memcpy(tmp_d.data() + row * Ish_BLK,
                            src_down_d + src_row_d_off,
                            (size_t)Ish_BLK * sizeof(uint16_t));
            }

            int8_t* dst_qs = down_proj_qs_tp_[(size_t)tp] + exp * (H * Ish);
            uint16_t* dst_d =
                reinterpret_cast<uint16_t*>(down_proj_d_tp_[(size_t)tp]) + exp * (H * Ish_BLK);

            gemm::repack_B_q8_0<uint16_t>(
                /*N=*/H, /*K=*/Ish,
                tmp_qs.data(), tmp_d.data(),
                dst_qs, dst_d
            );
        };

        run_tp_parallel([&](int64_t tp, int64_t exp) {
            repack_gate_up(tp, exp);
            repack_down(tp, exp);
        });
    }
}

void MoEInfer::quantize_and_store_expert(
    int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight_cpu) {

    TORCH_CHECK(weight_cpu.device().is_cpu(), "weight must be CPU");
    TORCH_CHECK(weight_cpu.is_contiguous(), "weight must be contiguous");

    const int64_t I = intermediate_size_;
    const int64_t Ish = intermediate_shard_;
    const int64_t H = hidden_size_;
    const int64_t H_BLK = H / 32;
    const int64_t Ish_BLK = Ish / 32;
    const int64_t I_BLK = I / 32;

    std::vector<torch::Tensor> q = quantize_weight_only(weight_cpu);
    torch::Tensor& qs = q[0];
    torch::Tensor& d  = q[1];

    const int8_t*   qs_ptr = qs.data_ptr<int8_t>();
    const uint16_t* d_ptr  = static_cast<const uint16_t*>(d.data_ptr());

    if (proj_name == "gate_proj" || proj_name == "up_proj") {
        TORCH_CHECK(qs.size(0) == I && qs.size(1) == H, "gate/up weight must be [I,H]");

        for (int64_t tp = 0; tp < tp_size_; ++tp) {
            const int64_t N = Ish;
            const int64_t K = H;

            const int64_t src_base_qs = tp * (Ish * H);
            const int64_t src_base_d  = tp * (Ish * H_BLK);

            const int64_t row_off = (proj_name == "up_proj") ? Ish : 0;

            // per-tp buffer, per-expert offset
            int8_t*   dst_qs = gate_up_qs_tp_[(size_t)tp] + expert_idx * (2 * Ish * H) + row_off * H;
            uint16_t* dst_d  = gate_up_d_tp_[(size_t)tp]  + expert_idx * (2 * Ish * H_BLK) + row_off * H_BLK;

            gemm::repack_B_q8_0<uint16_t>(
                N, K,
                qs_ptr + src_base_qs,
                d_ptr  + src_base_d,
                dst_qs, dst_d
            );
        }
        return;
    }

    if (proj_name == "down_proj") {
        TORCH_CHECK(qs.size(0) == H && qs.size(1) == I, "down weight must be [H,I]");

        std::vector<int8_t>   tmp_qs((size_t)H * (size_t)Ish);
        std::vector<uint16_t> tmp_d ((size_t)H * (size_t)Ish_BLK);

        for (int64_t tp = 0; tp < tp_size_; ++tp) {
            for (int64_t row = 0; row < H; ++row) {
                std::memcpy(tmp_qs.data() + row * Ish,
                            qs_ptr + row * I + tp * Ish,
                            (size_t)Ish * sizeof(int8_t));

                std::memcpy(tmp_d.data() + row * Ish_BLK,
                            d_ptr + row * I_BLK + tp * Ish_BLK,
                            (size_t)Ish_BLK * sizeof(uint16_t));
            }

            int8_t*   dst_qs = down_proj_qs_tp_[(size_t)tp] + expert_idx * (H * Ish);
            uint16_t* dst_d  = down_proj_d_tp_[(size_t)tp]  + expert_idx * (H * Ish_BLK);

            gemm::repack_B_q8_0<uint16_t>(
                H, Ish,
                tmp_qs.data(), tmp_d.data(),
                dst_qs, dst_d
            );
        }
        return;
    }

    TORCH_CHECK(false, "Unknown proj_name: ", proj_name);
}

// Static template wrappers for each configuration
// QT = QuantType, DT = DataType (Half/BFloat16), ST = ScaleType (Half/BFloat16)

template<quant::QuantType QT, typename DT, typename ST>
static void execute_wrapper(
    const void* x_in_ptr, void* y_out_ptr,
    const float* topk_weights_ptr, const int32_t* topk_ids_ptr,
    const void* const* gate_up_qs_tp, const void* const* gate_up_d_tp,
    const void* const* down_proj_qs_tp, const void* const* down_proj_d_tp,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size, int64_t tp_size, int64_t top_k) {
    moe_forward_ptr_impl<QT, DT, ST>(
        static_cast<const DT*>(x_in_ptr),
        static_cast<DT*>(y_out_ptr),
        topk_weights_ptr, topk_ids_ptr,
        gate_up_qs_tp,
        reinterpret_cast<const ST* const*>(gate_up_d_tp),
        down_proj_qs_tp,
        reinterpret_cast<const ST* const*>(down_proj_d_tp),
        num_tokens, hidden_dim, num_experts, intermediate_size, tp_size, top_k
    );
}

void MoEInfer::init_execute_function(at::ScalarType dtype) {
    // All configurations are now fixed, select the appropriate function pointer
    const bool is_bf16_scale = (scale_dtype_ == at::kBFloat16);
    const bool is_q4 = (quant_type_ == quant::QuantType::Q4_0);
    const bool is_bf16_act = (dtype == at::kBFloat16);

    // 2x2x2 = 8 possible configurations
    if (is_q4) {
        if (is_bf16_act) {
            if (is_bf16_scale) {
                execute_fn_ = execute_wrapper<quant::QuantType::Q4_0, at::BFloat16, at::BFloat16>;
            } else {
                execute_fn_ = execute_wrapper<quant::QuantType::Q4_0, at::BFloat16, at::Half>;
            }
        } else {
            if (is_bf16_scale) {
                execute_fn_ = execute_wrapper<quant::QuantType::Q4_0, at::Half, at::BFloat16>;
            } else {
                execute_fn_ = execute_wrapper<quant::QuantType::Q4_0, at::Half, at::Half>;
            }
        }
    } else {
        if (is_bf16_act) {
            if (is_bf16_scale) {
                execute_fn_ = execute_wrapper<quant::QuantType::Q8_0, at::BFloat16, at::BFloat16>;
            } else {
                execute_fn_ = execute_wrapper<quant::QuantType::Q8_0, at::BFloat16, at::Half>;
            }
        } else {
            if (is_bf16_scale) {
                execute_fn_ = execute_wrapper<quant::QuantType::Q8_0, at::Half, at::BFloat16>;
            } else {
                execute_fn_ = execute_wrapper<quant::QuantType::Q8_0, at::Half, at::Half>;
            }
        }
    }
}

MoEInfer::ExecuteFn MoEInfer::get_execute_function(at::ScalarType dtype) {
    if (execute_fn_ == nullptr) {
        init_execute_function(dtype);
    }
    return execute_fn_;
}

void MoEInfer::execute_on_cpu_routed_from_pointers(
    const void* x_in_ptr,
    void* y_out_ptr,
    const int32_t* topk_ids_ptr,
    const float* topk_weights_ptr,
    int64_t num_tokens,
    int64_t top_k,
    at::ScalarType dtype) {

    auto start = std::chrono::high_resolution_clock::now();
    if (execute_fn_ == nullptr) {
        init_execute_function(dtype);
    }
    // Direct call through function pointer - no dispatch overhead after first call
    execute_fn_(
        x_in_ptr, y_out_ptr,
        topk_weights_ptr, topk_ids_ptr,
        reinterpret_cast<const void* const*>(gate_up_qs_tp_.data()),
        reinterpret_cast<const void* const*>(gate_up_d_tp_.data()),
        reinterpret_cast<const void* const*>(down_proj_qs_tp_.data()),
        reinterpret_cast<const void* const*>(down_proj_d_tp_.data()),
        num_tokens, hidden_size_, num_experts_, intermediate_size_, tp_size_, top_k
    );

    auto end = std::chrono::high_resolution_clock::now();
    last_run_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
}
