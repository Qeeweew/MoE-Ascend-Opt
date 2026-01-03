#pragma once
#include <torch/extension.h>
#include <string>
#include <vector>

class MoEInfer {
public:
    MoEInfer(int64_t num_experts, int64_t hidden_size, int64_t intermediate_size);
    ~MoEInfer();

    MoEInfer(const MoEInfer&) = delete;
    MoEInfer& operator=(const MoEInfer&) = delete;

    int64_t num_experts() const { return num_experts_; }
    int64_t hidden_size() const { return hidden_size_; }
    int64_t intermediate_size() const { return intermediate_size_; }

    // Online quantize & store (CPU float32 weight)
    void quantize_and_store_expert(
        int64_t expert_idx,
        const std::string& proj_name, // "gate_proj" / "up_proj" / "down_proj"
        const torch::Tensor& weight_fp32_cpu
    );

    // Optional: load pre-quantized (qs/d) (CPU tensors), then repack & store
    void store_quantized_weights_repack(
        const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d,
        const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d
    );

    // CPU compute: x_in + (topk_ids/topk_weights) -> y_out
    void execute_on_cpu_routed_from_pointers(
        const void* x_in_ptr,
        void* y_out_ptr,
        const int32_t* topk_ids_ptr,
        const float* topk_weights_ptr,
        int64_t num_tokens,
        int64_t top_k,
        at::ScalarType dtype
    );

    // Expose packed weights for debug (raw pointers)
    const int8_t* gate_up_qs_packed() const { return gate_up_qs_packed_; }
    const at::Half* gate_up_d_packed() const { return gate_up_d_packed_; }
    const int8_t* down_proj_qs_packed() const { return down_proj_qs_packed_; }
    const at::Half* down_proj_d_packed() const { return down_proj_d_packed_; }

private:
    const int64_t num_experts_;
    const int64_t hidden_size_;
    const int64_t intermediate_size_;

    // Packed weights (CPU memory)
    int8_t* gate_up_qs_packed_ = nullptr;
    at::Half* gate_up_d_packed_ = nullptr;
    int8_t* down_proj_qs_packed_ = nullptr;
    at::Half* down_proj_d_packed_ = nullptr;
};
