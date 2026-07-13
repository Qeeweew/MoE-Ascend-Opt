#pragma once

#include <cstdint>

namespace nanovllm {
class NumaThreadPool;
}

namespace moe::decode {

struct Route {
    int32_t expert_id;
    int32_t output_row;
};

enum class ScaleType {
    Fp16,
    Bf16,
};

struct Q4Args {
    int node;
    int hidden_size;
    int intermediate_size;
    const Route* routes;
    int num_routes;

    const int8_t* x_qs;
    const float* x_d;

    const uint32_t* gate_up_qs;
    const void* gate_up_d;
    const uint32_t* down_qs;
    const void* down_d;
    ScaleType scale_type;

    float* expert_output;
    nanovllm::NumaThreadPool* pool;
};

// Dedicated batch-1 Q4 MoE engine. The caller guarantees one token per route.
// All scheduling and activation scratch is owned by this module.
void run_q4(const Q4Args& args);

} // namespace moe::decode
