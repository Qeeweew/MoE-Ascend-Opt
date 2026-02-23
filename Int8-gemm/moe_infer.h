#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <string>
#include "numa_threadpool.h"
#include "quant_traits.h"

class MoEInfer {
public:
    MoEInfer(int64_t num_experts, int64_t hidden_size, int64_t intermediate_size,
             quant::QuantType quant_type = quant::QuantType::Q8_0);
    ~MoEInfer();

    MoEInfer(const MoEInfer&) = delete;
    MoEInfer& operator=(const MoEInfer&) = delete;

    int64_t num_experts() const { return num_experts_; }
    int64_t hidden_size() const { return hidden_size_; }
    int64_t intermediate_size() const { return intermediate_size_; }

    // Get last forward execution time in milliseconds
    double get_last_run_time_ms() const { return last_run_time_ms_; }

    // Get the scale dtype (kFloat16 or kBFloat16)
    at::ScalarType get_scale_dtype() const { return scale_dtype_; }

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

    void execute_on_cpu_routed_from_pointers(
        const void* x_in_ptr,
        void* y_out_ptr,
        const int32_t* topk_ids_ptr,
        const float* topk_weights_ptr,
        int64_t num_tokens,
        int64_t top_k,
        at::ScalarType dtype);

private:
    int64_t num_experts_;
    int64_t hidden_size_;
    int64_t intermediate_size_;
    int64_t tp_size_;
    int64_t intermediate_shard_;
    quant::QuantType quant_type_;
    at::ScalarType scale_dtype_ = at::kHalf;  // kHalf or kBFloat16

    // Weights placed per-tp NUMA node
    // Layout per tp:
    // gate_up: [E, 2*Ish, H]
    // down:    [E, H, Ish]
    // For Q4_0, stored as uint32_t (packed 4-bit)
    std::vector<int8_t*>   gate_up_qs_tp_;
    std::vector<uint16_t*> gate_up_d_tp_;
    std::vector<int8_t*>   down_proj_qs_tp_;
    std::vector<uint16_t*> down_proj_d_tp_;

    size_t gate_up_qs_bytes_per_tp_ = 0;
    size_t gate_up_d_bytes_per_tp_  = 0;
    size_t down_qs_bytes_per_tp_    = 0;
    size_t down_d_bytes_per_tp_     = 0;

    // Last forward execution time in milliseconds
    double last_run_time_ms_ = 0.0;

    // Helper to calculate quantized storage bytes
    size_t calculate_qs_bytes(int64_t rows, int64_t cols) const;
};
