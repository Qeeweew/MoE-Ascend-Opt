#include "moe_infer.h"
#include "q8_gemm.h"
#include "numa_threadpool.h"

#include <chrono>
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
            (size_t)num_experts_ * (size_t)(2 * Ish) * (size_t)H * sizeof(uint8_t) / 2;
        down_qs_bytes_per_tp_ =
            (size_t)num_experts_ * (size_t)H * (size_t)Ish * sizeof(uint8_t) / 2;
    } else {
        gate_up_qs_bytes_per_tp_ =
            (size_t)num_experts_ * (size_t)(2 * Ish) * (size_t)H * sizeof(int8_t);
        down_qs_bytes_per_tp_ =
            (size_t)num_experts_ * (size_t)H * (size_t)Ish * sizeof(int8_t);
    }

    gate_up_d_bytes_per_tp_  =
        (size_t)num_experts_ * (size_t)(2 * Ish) * (size_t)H_BLK * sizeof(at::Half);
    down_d_bytes_per_tp_  =
        (size_t)num_experts_ * (size_t)H * (size_t)Ish_BLK * sizeof(at::Half);

    gate_up_qs_tp_.resize((size_t)tp_size_, nullptr);
    gate_up_d_tp_.resize((size_t)tp_size_, nullptr);
    down_proj_qs_tp_.resize((size_t)tp_size_, nullptr);
    down_proj_d_tp_.resize((size_t)tp_size_, nullptr);

    for (int tp = 0; tp < (int)tp_size_; ++tp) {
        int node = tp;
        gate_up_qs_tp_[(size_t)tp]   = (int8_t*)numa_alloc_or_throw(gate_up_qs_bytes_per_tp_, node);
        gate_up_d_tp_[(size_t)tp]    = (at::Half*)numa_alloc_or_throw(gate_up_d_bytes_per_tp_, node);
        down_proj_qs_tp_[(size_t)tp] = (int8_t*)numa_alloc_or_throw(down_qs_bytes_per_tp_, node);
        down_proj_d_tp_[(size_t)tp]  = (at::Half*)numa_alloc_or_throw(down_d_bytes_per_tp_, node);
    }
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

    TORCH_CHECK(gate_up_d.scalar_type() == torch::kFloat16, "gate_up_d must be float16");
    TORCH_CHECK(down_proj_d.scalar_type() == torch::kFloat16, "down_proj_d must be float16");

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

    // Dispatch based on quantization type
    if (quant_type_ == quant::QuantType::Q4_0) {
        // Q4_0 path: weights are uint32_t (packed 4-bit)
        TORCH_CHECK(gate_up_qs.scalar_type() == torch::kInt32, "Q4_0 gate_up_qs must be int32 (uint32 storage)");
        TORCH_CHECK(down_proj_qs.scalar_type() == torch::kInt32, "Q4_0 down_proj_qs must be int32 (uint32 storage)");

        const uint32_t* src_gate_up_qs = gate_up_qs.data_ptr<uint32_t>();
        const at::Half*  src_gate_up_d  = (const at::Half*)gate_up_d.data_ptr<at::Half>();
        const uint32_t* src_down_qs     = down_proj_qs.data_ptr<uint32_t>();
        const at::Half*  src_down_d     = (const at::Half*)down_proj_d.data_ptr<at::Half>();

        // gate_up: shard on N (rows): [2I, H] -> tp blocks [2Ish, H]
        for (int64_t exp = 0; exp < num_experts_; ++exp) {
            for (int64_t tp = 0; tp < tp_size_; ++tp) {
                const int64_t Nsh = 2 * Ish;
                const int64_t K   = H;

                const int64_t src_base_qs = exp * (2 * I * H / 2) + tp * (Nsh * K / 2); // Q4_0: half storage
                const int64_t src_base_d  = exp * (2 * I * H_BLK) + tp * (Nsh * H_BLK);

                uint32_t* dst_qs = (uint32_t*)gate_up_qs_tp_[(size_t)tp] + exp * (Nsh * K / 2);
                at::Half* dst_d  = gate_up_d_tp_[(size_t)tp]  + exp * (Nsh * H_BLK);

                gemm::repack_B_q4_0<at::Half>(
                    Nsh, K,
                    src_gate_up_qs + src_base_qs,
                    src_gate_up_d  + src_base_d,
                    dst_qs, dst_d
                );
            }
        }

        // down_proj: shard on K (cols): [H, I] -> tp blocks [H, Ish]
        std::vector<uint32_t> tmp_qs((size_t)H * (size_t)Ish / 2); // Q4_0: half storage
        std::vector<at::Half> tmp_d ((size_t)H * (size_t)Ish_BLK);

        for (int64_t exp = 0; exp < num_experts_; ++exp) {
            for (int64_t tp = 0; tp < tp_size_; ++tp) {
                for (int64_t row = 0; row < H; ++row) {
                    const int64_t src_row_qs_off = exp * (H * I / 2) + row * (I / 2) + tp * (Ish / 2);
                    std::memcpy(tmp_qs.data() + row * (Ish / 2),
                                src_down_qs + src_row_qs_off,
                                (size_t)(Ish / 2) * sizeof(uint32_t));

                    const int64_t src_row_d_off = exp * (H * I_BLK) + row * I_BLK + tp * Ish_BLK;
                    std::memcpy(tmp_d.data() + row * Ish_BLK,
                                src_down_d + src_row_d_off,
                                (size_t)Ish_BLK * sizeof(at::Half));
                }

                uint32_t* dst_qs = (uint32_t*)down_proj_qs_tp_[(size_t)tp] + exp * (H * Ish / 2);
                at::Half*  dst_d  = down_proj_d_tp_[(size_t)tp]  + exp * (H * Ish_BLK);

                gemm::repack_B_q4_0<at::Half>(
                    /*N=*/H, /*K=*/Ish,
                    tmp_qs.data(), tmp_d.data(),
                    dst_qs, dst_d
                );
            }
        }
    } else {
        // Q8_0 path: weights are int8_t
        TORCH_CHECK(gate_up_qs.scalar_type() == torch::kInt8, "Q8_0 gate_up_qs must be int8");
        TORCH_CHECK(down_proj_qs.scalar_type() == torch::kInt8, "Q8_0 down_proj_qs must be int8");

        const int8_t*   src_gate_up_qs = gate_up_qs.data_ptr<int8_t>();
        const at::Half* src_gate_up_d  = (const at::Half*)gate_up_d.data_ptr<at::Half>();
        const int8_t*   src_down_qs    = down_proj_qs.data_ptr<int8_t>();
        const at::Half* src_down_d     = (const at::Half*)down_proj_d.data_ptr<at::Half>();

        // gate_up: shard on N (rows): [2I, H] -> tp blocks [2Ish, H]
        for (int64_t exp = 0; exp < num_experts_; ++exp) {
            for (int64_t tp = 0; tp < tp_size_; ++tp) {
                const int64_t Nsh = 2 * Ish;
                const int64_t K   = H;

                const int64_t src_base_qs = exp * (2 * I * H) + tp * (Nsh * H);
                const int64_t src_base_d  = exp * (2 * I * H_BLK) + tp * (Nsh * H_BLK);

                int8_t*   dst_qs = gate_up_qs_tp_[(size_t)tp] + exp * (Nsh * K);
                at::Half* dst_d  = gate_up_d_tp_[(size_t)tp]  + exp * (Nsh * H_BLK);

                gemm::repack_B_q8_0<at::Half>(
                    Nsh, K,
                    src_gate_up_qs + src_base_qs,
                    src_gate_up_d  + src_base_d,
                    dst_qs, dst_d
                );
            }
        }

        // down_proj: shard on K (cols): [H, I] -> tp blocks [H, Ish]
        std::vector<int8_t>   tmp_qs((size_t)H * (size_t)Ish);
        std::vector<at::Half> tmp_d ((size_t)H * (size_t)Ish_BLK);

        for (int64_t exp = 0; exp < num_experts_; ++exp) {
            for (int64_t tp = 0; tp < tp_size_; ++tp) {
                for (int64_t row = 0; row < H; ++row) {
                    const int64_t src_row_qs_off = exp * (H * I) + row * I + tp * Ish;
                    std::memcpy(tmp_qs.data() + row * Ish,
                                src_down_qs + src_row_qs_off,
                                (size_t)Ish * sizeof(int8_t));

                    const int64_t src_row_d_off = exp * (H * I_BLK) + row * I_BLK + tp * Ish_BLK;
                    std::memcpy(tmp_d.data() + row * Ish_BLK,
                                src_down_d + src_row_d_off,
                                (size_t)Ish_BLK * sizeof(at::Half));
                }

                int8_t*   dst_qs = down_proj_qs_tp_[(size_t)tp] + exp * (H * Ish);
                at::Half* dst_d  = down_proj_d_tp_[(size_t)tp]  + exp * (H * Ish_BLK);

                gemm::repack_B_q8_0<at::Half>(
                    /*N=*/H, /*K=*/Ish,
                    tmp_qs.data(), tmp_d.data(),
                    dst_qs, dst_d
                );
            }
        }
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
    const at::Half* d_ptr  = (const at::Half*)d.data_ptr<at::Half>();

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
            at::Half* dst_d  = gate_up_d_tp_[(size_t)tp]  + expert_idx * (2 * Ish * H_BLK) + row_off * H_BLK;

            repack_B_q8_0_from_ptr<at::Half>(
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
        std::vector<at::Half> tmp_d ((size_t)H * (size_t)Ish_BLK);

        for (int64_t tp = 0; tp < tp_size_; ++tp) {
            for (int64_t row = 0; row < H; ++row) {
                std::memcpy(tmp_qs.data() + row * Ish,
                            qs_ptr + row * I + tp * Ish,
                            (size_t)Ish * sizeof(int8_t));

                std::memcpy(tmp_d.data() + row * Ish_BLK,
                            d_ptr + row * I_BLK + tp * Ish_BLK,
                            (size_t)Ish_BLK * sizeof(at::Half));
            }

            int8_t*   dst_qs = down_proj_qs_tp_[(size_t)tp] + expert_idx * (H * Ish);
            at::Half* dst_d  = down_proj_d_tp_[(size_t)tp]  + expert_idx * (H * Ish_BLK);

            repack_B_q8_0_from_ptr<at::Half>(
                H, Ish,
                tmp_qs.data(), tmp_d.data(),
                dst_qs, dst_d
            );
        }
        return;
    }

    TORCH_CHECK(false, "Unknown proj_name: ", proj_name);
}

void MoEInfer::execute_on_cpu_routed_from_pointers(
    const void* x_in_ptr,
    void* y_out_ptr,
    const int32_t* topk_ids_ptr,
    const float* topk_weights_ptr,
    int64_t num_tokens,
    int64_t top_k,
    at::ScalarType dtype) {

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        dtype, "moe_execute_routed_dispatch",
        [&] {
            // Dispatch based on quantization type to unified template function
            if (quant_type_ == quant::QuantType::Q8_0) {
                moe_forward_ptr_impl<quant::QuantType::Q8_0, scalar_t>(
                    (const scalar_t*)x_in_ptr,
                    (scalar_t*)y_out_ptr,
                    topk_weights_ptr,
                    topk_ids_ptr,
                    reinterpret_cast<const void* const*>(gate_up_qs_tp_.data()),
                    gate_up_d_tp_.data(),
                    reinterpret_cast<const void* const*>(down_proj_qs_tp_.data()),
                    down_proj_d_tp_.data(),
                    num_tokens,
                    hidden_size_,
                    num_experts_,
                    intermediate_size_,
                    tp_size_,
                    top_k
                );
            } else {
                moe_forward_ptr_impl<quant::QuantType::Q4_0, scalar_t>(
                    (const scalar_t*)x_in_ptr,
                    (scalar_t*)y_out_ptr,
                    topk_weights_ptr,
                    topk_ids_ptr,
                    reinterpret_cast<const void* const*>(gate_up_qs_tp_.data()),
                    gate_up_d_tp_.data(),
                    reinterpret_cast<const void* const*>(down_proj_qs_tp_.data()),
                    down_proj_d_tp_.data(),
                    num_tokens,
                    hidden_size_,
                    num_experts_,
                    intermediate_size_,
                    tp_size_,
                    top_k
                );
            }
        }
    );

    // End timing and store result in milliseconds
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    last_run_time_ms_ = diff.count();
}
