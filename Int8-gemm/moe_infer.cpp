#include "moe_infer.h"
#include "q8_gemm.h"

#include <cstring>
#include <stdexcept>

static inline void* aligned_alloc64(size_t bytes) {
    void* p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
    if (!p) throw std::runtime_error("aligned_alloc failed");
    return p;
}

MoEInfer::MoEInfer(int64_t num_experts, int64_t hidden_size, int64_t intermediate_size)
    : num_experts_(num_experts), hidden_size_(hidden_size), intermediate_size_(intermediate_size) {
    // numa_nodes forced to 1
    TORCH_CHECK(hidden_size_ % 32 == 0, "hidden_size must be multiple of 32");
    TORCH_CHECK(intermediate_size_ % 32 == 0, "intermediate_size must be multiple of 32");

    const int64_t gate_up_qs_bytes = num_experts_ * (2 * intermediate_size_) * hidden_size_ * sizeof(int8_t);
    const int64_t gate_up_d_bytes  = num_experts_ * (2 * intermediate_size_) * (hidden_size_ / 32) * sizeof(at::Half);

    const int64_t down_qs_bytes = num_experts_ * hidden_size_ * intermediate_size_ * sizeof(int8_t);
    const int64_t down_d_bytes  = num_experts_ * hidden_size_ * (intermediate_size_ / 32) * sizeof(at::Half);

    gate_up_qs_packed_ = (int8_t*)aligned_alloc64(gate_up_qs_bytes);
    gate_up_d_packed_  = (at::Half*)aligned_alloc64(gate_up_d_bytes);
    down_proj_qs_packed_ = (int8_t*)aligned_alloc64(down_qs_bytes);
    down_proj_d_packed_  = (at::Half*)aligned_alloc64(down_d_bytes);

    std::memset(gate_up_qs_packed_, 0, gate_up_qs_bytes);
    std::memset(gate_up_d_packed_,  0, gate_up_d_bytes);
    std::memset(down_proj_qs_packed_, 0, down_qs_bytes);
    std::memset(down_proj_d_packed_,  0, down_d_bytes);
}

MoEInfer::~MoEInfer() {
    std::free(gate_up_qs_packed_);
    std::free(gate_up_d_packed_);
    std::free(down_proj_qs_packed_);
    std::free(down_proj_d_packed_);
}

void MoEInfer::store_quantized_weights_repack(
    const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d,
    const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d) {

    TORCH_CHECK(gate_up_qs.device().is_cpu() && gate_up_d.device().is_cpu(), "gate_up must be CPU");
    TORCH_CHECK(down_proj_qs.device().is_cpu() && down_proj_d.device().is_cpu(), "down_proj must be CPU");
    TORCH_CHECK(gate_up_qs.is_contiguous() && gate_up_d.is_contiguous(), "gate_up must be contiguous");
    TORCH_CHECK(down_proj_qs.is_contiguous() && down_proj_d.is_contiguous(), "down_proj must be contiguous");

    // Expect row-major:
    // gate_up_qs: [E, 2*I, H] int8
    // gate_up_d:  [E, 2*I, H/32] half
    // down_qs:    [E, H, I] int8
    // down_d:     [E, H, I/32] half
    TORCH_CHECK(gate_up_qs.scalar_type() == torch::kInt8, "gate_up_qs must be int8");
    TORCH_CHECK(gate_up_d.scalar_type() == torch::kFloat16, "gate_up_d must be float16");
    TORCH_CHECK(down_proj_qs.scalar_type() == torch::kInt8, "down_proj_qs must be int8");
    TORCH_CHECK(down_proj_d.scalar_type() == torch::kFloat16, "down_proj_d must be float16");

    const int8_t* src_gate_up_qs = gate_up_qs.data_ptr<int8_t>();
    const at::Half* src_gate_up_d = (const at::Half*)gate_up_d.data_ptr<at::Half>();
    const int8_t* src_down_qs = down_proj_qs.data_ptr<int8_t>();
    const at::Half* src_down_d = (const at::Half*)down_proj_d.data_ptr<at::Half>();

    for (int64_t exp = 0; exp < num_experts_; ++exp) {
        // repack gate_up for this expert: N = 2I, K = H
        {
            const int64_t N = 2 * intermediate_size_;
            const int64_t K = hidden_size_;
            const int64_t K_BLOCKS = K / 32;

            const int8_t* expert_src_qs = src_gate_up_qs + exp * (N * K);
            const at::Half* expert_src_d = src_gate_up_d + exp * (N * K_BLOCKS);

            int8_t* expert_dst_qs = gate_up_qs_packed_ + exp * (N * K);
            at::Half* expert_dst_d = gate_up_d_packed_ + exp * (N * K_BLOCKS);

            repack_B_q8_0_from_ptr<at::Half>(N, K, expert_src_qs, expert_src_d, expert_dst_qs, expert_dst_d);
        }

        // repack down_proj for this expert: N = H, K = I
        {
            const int64_t N = hidden_size_;
            const int64_t K = intermediate_size_;
            const int64_t K_BLOCKS = K / 32;

            const int8_t* expert_src_qs = src_down_qs + exp * (N * K);
            const at::Half* expert_src_d = src_down_d + exp * (N * K_BLOCKS);

            int8_t* expert_dst_qs = down_proj_qs_packed_ + exp * (N * K);
            at::Half* expert_dst_d = down_proj_d_packed_ + exp * (N * K_BLOCKS);

            repack_B_q8_0_from_ptr<at::Half>(N, K, expert_src_qs, expert_src_d, expert_dst_qs, expert_dst_d);
        }
    }
}

void MoEInfer::quantize_and_store_expert(
    int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight_fp32_cpu) {

    TORCH_CHECK(weight_fp32_cpu.device().is_cpu(), "weight must be CPU");
    TORCH_CHECK(weight_fp32_cpu.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight_fp32_cpu.scalar_type() == torch::kFloat32, "weight must be float32");

    std::vector<torch::Tensor> q = quantize_weight_only(weight_fp32_cpu);
    torch::Tensor& qs = q[0]; // [N,K] int8
    torch::Tensor& d  = q[1]; // [N,K/32] half

    const int8_t* qs_ptr = qs.data_ptr<int8_t>();
    const at::Half* d_ptr = (const at::Half*)d.data_ptr<at::Half>();

    if (proj_name == "gate_proj" || proj_name == "up_proj") {
        const int64_t N = intermediate_size_;
        const int64_t K = hidden_size_;
        const int64_t K_BLOCKS = K / 32;

        // gate_up layout in buffer: [E, 2N, K]
        int64_t base_qs_off = expert_idx * (2 * N * K);
        int64_t base_d_off  = expert_idx * (2 * N * K_BLOCKS);

        if (proj_name == "up_proj") {
            base_qs_off += (N * K);
            base_d_off  += (N * K_BLOCKS);
        }

        int8_t* dst_qs = gate_up_qs_packed_ + base_qs_off;
        at::Half* dst_d = gate_up_d_packed_ + base_d_off;

        repack_B_q8_0_from_ptr<at::Half>(N, K, qs_ptr, d_ptr, dst_qs, dst_d);
        return;
    }

    if (proj_name == "down_proj") {
        const int64_t N = hidden_size_;
        const int64_t K = intermediate_size_;
        const int64_t K_BLOCKS = K / 32;

        int64_t base_qs_off = expert_idx * (N * K);
        int64_t base_d_off  = expert_idx * (N * K_BLOCKS);

        int8_t* dst_qs = down_proj_qs_packed_ + base_qs_off;
        at::Half* dst_d = down_proj_d_packed_ + base_d_off;

        repack_B_q8_0_from_ptr<at::Half>(N, K, qs_ptr, d_ptr, dst_qs, dst_d);
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

    TORCH_CHECK(top_k == 1 || top_k == 8, "top_k must be 1 or 8");

    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        dtype, "moe_execute_routed_dispatch",
        [&] {
            moe_q8_forward_ptr_impl<scalar_t>(
                (const scalar_t*)x_in_ptr,
                (scalar_t*)y_out_ptr,
                topk_weights_ptr,
                topk_ids_ptr,
                gate_up_qs_packed_,
                gate_up_d_packed_,
                down_proj_qs_packed_,
                down_proj_d_packed_,
                num_tokens,
                hidden_size_,
                num_experts_,
                intermediate_size_,
                top_k
            );
        }
    );
}
