#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "moe_infer.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace py = pybind11;

// ------------------------- profiler (optional) -------------------------
struct MoECpuProfiler {
    bool enabled = false;
    std::mutex mtx;
    std::map<int64_t, std::vector<double>> timings;

    MoECpuProfiler() {
        const char* env_p = std::getenv("PROFILE_MOE_CPU");
        if (env_p && (std::string(env_p) == "1" || std::string(env_p) == "true")) {
            enabled = true;
            std::cerr << "[MoECpuProfiler] Profiling enabled.\n";
        }
    }

    ~MoECpuProfiler() {
        if (!enabled || timings.empty()) return;
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "\n--- MoE CPU Task Execution Profile ---\n";
        std::cout << std::left << std::setw(15) << "Num Tokens"
                  << std::setw(15) << "Count"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(15) << "Min (ms)"
                  << std::setw(15) << "Max (ms)" << "\n";
        std::cout << "------------------------------------------------------------\n";
        for (auto& kv : timings) {
            auto& v = kv.second;
            if (v.empty()) continue;
            double sum = std::accumulate(v.begin(), v.end(), 0.0);
            double avg = sum / v.size();
            double mn = *std::min_element(v.begin(), v.end());
            double mx = *std::max_element(v.begin(), v.end());
            std::cout << std::fixed << std::setprecision(4)
                      << std::left << std::setw(15) << kv.first
                      << std::setw(15) << v.size()
                      << std::setw(15) << avg
                      << std::setw(15) << mn
                      << std::setw(15) << mx << "\n";
        }
        std::cout << "------------------------------------------------------------\n";
    }

    void add_record(int64_t tokens, double ms) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mtx);
        timings[tokens].push_back(ms);
    }
};

static MoECpuProfiler profiler_instance;

#ifdef WITH_NPU
#include "acl/acl_rt.h"
#include <atomic>
#include <sstream>
#include <thread>

// ------------------------- NPU callback report thread manager -------------------------
class PYBIND11_EXPORT NpuCallbackManager {
public:
    NpuCallbackManager(uint64_t stream_ptr, int device_id)
        : stream_(reinterpret_cast<aclrtStream>(stream_ptr)),
          device_id_(device_id),
          stop_flag_(false) {

        callback_thread_ = std::thread(&NpuCallbackManager::process_loop, this);

        std::ostringstream oss;
        oss << callback_thread_.get_id();
        uint64_t tid = std::stoull(oss.str());

        aclError err = aclrtSubscribeReport(tid, stream_);
        TORCH_CHECK(err == ACL_SUCCESS, "aclrtSubscribeReport failed: ", err);
    }

    ~NpuCallbackManager() {
        stop_flag_.store(true);

        std::ostringstream oss;
        oss << callback_thread_.get_id();
        uint64_t tid = std::stoull(oss.str());

        aclrtSetDevice(device_id_);
        aclrtUnSubscribeReport(tid, stream_);

        if (callback_thread_.joinable()) callback_thread_.join();
    }

private:
    void process_loop() {
        aclError err = aclrtSetDevice(device_id_);
        if (err != ACL_SUCCESS) {
            fprintf(stderr, "aclrtSetDevice failed in callback thread: %d\n", err);
            return;
        }
        while (!stop_flag_.load()) {
            err = aclrtProcessReport(100);
            if (err != ACL_SUCCESS && err != ACL_ERROR_RT_TASK_TIMEOUT) {
                // ignore
            }
        }
    }

    aclrtStream stream_;
    int device_id_;
    std::thread callback_thread_;
    std::atomic<bool> stop_flag_;
};

// ------------------------- Graph-safe pinned buffer + args -------------------------
struct PinnedBuffer {
    void* ptr = nullptr;
    size_t size = 0;

    ~PinnedBuffer() {
        if (ptr) {
            aclrtFreeHost(ptr);
            ptr = nullptr;
            size = 0;
        }
    }

    void alloc(size_t bytes) {
        if (ptr) {
            TORCH_CHECK(size >= bytes, "PinnedBuffer realloc is not allowed in graph context");
            return;
        }
        size = bytes;
        aclError err = aclrtMallocHost(&ptr, bytes);
        TORCH_CHECK(err == ACL_SUCCESS, "aclrtMallocHost failed: ", err);
    }
};

struct MoECpuTaskArgs {
    const void* x_in_ptr;        // pinned input hidden
    void* y_out_ptr;             // pinned output hidden
    const int32_t* topk_ids_ptr; // pinned ids
    const float* topk_w_ptr;     // pinned weights
    int64_t num_tokens;
    int64_t top_k;
    MoEInfer* moe;
    at::ScalarType dtype;
};

struct MoEGraphContext {
    int64_t tokens_bucket;
    int64_t top_k;
    at::ScalarType dtype;

    PinnedBuffer hidden_in;
    PinnedBuffer hidden_out;
    PinnedBuffer topk_ids;
    PinnedBuffer topk_w;

    MoECpuTaskArgs args;

    MoEGraphContext(MoEInfer* moe, int64_t tokens_bucket_, int64_t top_k_, at::ScalarType dtype_)
        : tokens_bucket(tokens_bucket_), top_k(top_k_), dtype(dtype_) {
        TORCH_CHECK(top_k == 1 || top_k == 8, "top_k must be 1 or 8");
        TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "dtype must be fp16/bf16");

        const int64_t H = moe->hidden_size();

        size_t hidden_bytes = (size_t)tokens_bucket * (size_t)H * (dtype == at::kHalf ? sizeof(at::Half) : sizeof(at::BFloat16));
        size_t ids_bytes    = (size_t)tokens_bucket * (size_t)top_k * sizeof(int32_t);
        size_t w_bytes      = (size_t)tokens_bucket * (size_t)top_k * sizeof(float);

        hidden_in.alloc(hidden_bytes);
        hidden_out.alloc(hidden_bytes);
        topk_ids.alloc(ids_bytes);
        topk_w.alloc(w_bytes);

        args.x_in_ptr = hidden_in.ptr;
        args.y_out_ptr = hidden_out.ptr;
        args.topk_ids_ptr = (const int32_t*)topk_ids.ptr;
        args.topk_w_ptr = (const float*)topk_w.ptr;
        args.num_tokens = tokens_bucket;
        args.top_k = top_k;
        args.moe = moe;
        args.dtype = dtype;
    }
};

// ------------------------- callback fn (graph-safe, no new/delete) -------------------------
static void moe_callback_fn(void* user_data) {
    auto* args = reinterpret_cast<MoECpuTaskArgs*>(user_data);
    if (profiler_instance.enabled) {
        auto st = std::chrono::high_resolution_clock::now();
        args->moe->execute_on_cpu_routed_from_pointers(
            args->x_in_ptr, args->y_out_ptr,
            args->topk_ids_ptr, args->topk_w_ptr,
            args->num_tokens, args->top_k, args->dtype
        );
        auto ed = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = ed - st;
        profiler_instance.add_record(args->num_tokens, ms.count());
    } else {
        args->moe->execute_on_cpu_routed_from_pointers(
            args->x_in_ptr, args->y_out_ptr,
            args->topk_ids_ptr, args->topk_w_ptr,
            args->num_tokens, args->top_k, args->dtype
        );
    }
}

// ------------------------- NPU forward (GRAPH context required) -------------------------
torch::Tensor moe_forward_npu_routed_graph(
    const torch::Tensor& hidden_states_npu,
    const torch::Tensor& topk_ids_npu,
    const torch::Tensor& topk_weights_npu,  // float32
    py::capsule& moe_infer_handle,
    py::capsule& graph_ctx_capsule,
    uint64_t stream_ptr)
{
    TORCH_CHECK(hidden_states_npu.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(topk_ids_npu.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(topk_weights_npu.is_contiguous(), "topk_weights must be contiguous");

    TORCH_CHECK(hidden_states_npu.device().type() == torch::kPrivateUse1, "hidden_states must be NPU");
    TORCH_CHECK(topk_ids_npu.device().type() == torch::kPrivateUse1, "topk_ids must be NPU");
    TORCH_CHECK(topk_weights_npu.device().type() == torch::kPrivateUse1, "topk_weights must be NPU");

    TORCH_CHECK(topk_weights_npu.scalar_type() == torch::kFloat32, "topk_weights must be float32");
    TORCH_CHECK(topk_ids_npu.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(hidden_states_npu.scalar_type() == torch::kFloat16 || hidden_states_npu.scalar_type() == torch::kBFloat16,
                "hidden_states must be fp16/bf16");

    auto* moe = moe_infer_handle.get_pointer<MoEInfer>();
    auto* ctx = graph_ctx_capsule.get_pointer<MoEGraphContext>();

    const int64_t tokens = hidden_states_npu.size(0);
    const int64_t hidden = hidden_states_npu.size(1);
    TORCH_CHECK(hidden == moe->hidden_size(), "hidden_size mismatch");
    TORCH_CHECK(tokens == ctx->tokens_bucket, "Graph mode requires tokens == tokens_bucket");
    TORCH_CHECK(topk_ids_npu.size(0) == tokens && topk_ids_npu.size(1) == ctx->top_k, "topk_ids shape mismatch");
    TORCH_CHECK(topk_weights_npu.size(0) == tokens && topk_weights_npu.size(1) == ctx->top_k, "topk_weights shape mismatch");
    TORCH_CHECK((at::ScalarType)hidden_states_npu.scalar_type() == ctx->dtype, "dtype mismatch with graph ctx");

    aclrtStream stream = reinterpret_cast<aclrtStream>(stream_ptr);

    // output tensor on NPU
    auto output_npu = torch::empty_like(hidden_states_npu);

    const size_t hidden_bytes = hidden_states_npu.nbytes();
    const size_t ids_bytes = topk_ids_npu.nbytes();
    const size_t w_bytes = topk_weights_npu.nbytes();

    // D2H copies into fixed pinned buffers
    aclError err = aclrtMemcpyAsync(
        ctx->hidden_in.ptr, ctx->hidden_in.size,
        hidden_states_npu.data_ptr(), hidden_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    TORCH_CHECK(err == ACL_SUCCESS, "aclrtMemcpyAsync D2H hidden failed: ", err);

    err = aclrtMemcpyAsync(
        ctx->topk_ids.ptr, ctx->topk_ids.size,
        topk_ids_npu.data_ptr(), ids_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    TORCH_CHECK(err == ACL_SUCCESS, "aclrtMemcpyAsync D2H topk_ids failed: ", err);

    err = aclrtMemcpyAsync(
        ctx->topk_w.ptr, ctx->topk_w.size,
        topk_weights_npu.data_ptr(), w_bytes,
        ACL_MEMCPY_DEVICE_TO_HOST, stream);
    TORCH_CHECK(err == ACL_SUCCESS, "aclrtMemcpyAsync D2H topk_weights failed: ", err);

    // callback (user_data is stable address: &ctx->args)
    err = aclrtLaunchCallback(moe_callback_fn, (void*)&ctx->args, ACL_CALLBACK_BLOCK, stream);
    TORCH_CHECK(err == ACL_SUCCESS, "aclrtLaunchCallback failed: ", err);

    // H2D copy result from pinned output -> output_npu
    err = aclrtMemcpyAsync(
        output_npu.data_ptr(), hidden_bytes,
        ctx->hidden_out.ptr, ctx->hidden_out.size,
        ACL_MEMCPY_HOST_TO_DEVICE, stream);
    TORCH_CHECK(err == ACL_SUCCESS, "aclrtMemcpyAsync H2D output failed: ", err);

    return output_npu;
}
#endif // WITH_NPU

// ------------------------- CPU forward (CPU tensors in/out) -------------------------
torch::Tensor execute_moe_cpu_routed(
    py::capsule& moe_infer_handle,
    const torch::Tensor& hidden_states_cpu,
    const torch::Tensor& topk_ids_cpu,
    const torch::Tensor& topk_weights_cpu,
    int64_t top_k)
{
    TORCH_CHECK(hidden_states_cpu.device().is_cpu(), "hidden_states must be CPU");
    TORCH_CHECK(topk_ids_cpu.device().is_cpu(), "topk_ids must be CPU");
    TORCH_CHECK(topk_weights_cpu.device().is_cpu(), "topk_weights must be CPU");

    TORCH_CHECK(hidden_states_cpu.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(topk_ids_cpu.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(topk_weights_cpu.is_contiguous(), "topk_weights must be contiguous");

    TORCH_CHECK(topk_ids_cpu.scalar_type() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topk_weights_cpu.scalar_type() == torch::kFloat32, "topk_weights must be float32");
    TORCH_CHECK(hidden_states_cpu.scalar_type() == torch::kFloat16 || hidden_states_cpu.scalar_type() == torch::kBFloat16,
                "hidden_states must be fp16/bf16");
    TORCH_CHECK(top_k == 1 || top_k == 8, "top_k must be 1 or 8");

    auto* moe = moe_infer_handle.get_pointer<MoEInfer>();
    const int64_t tokens = hidden_states_cpu.size(0);
    const int64_t hidden = hidden_states_cpu.size(1);
    TORCH_CHECK(hidden == moe->hidden_size(), "hidden_size mismatch");
    TORCH_CHECK(topk_ids_cpu.size(0) == tokens && topk_ids_cpu.size(1) == top_k, "topk_ids shape mismatch");
    TORCH_CHECK(topk_weights_cpu.size(0) == tokens && topk_weights_cpu.size(1) == top_k, "topk_weights shape mismatch");

    auto out = torch::empty_like(hidden_states_cpu);

    moe->execute_on_cpu_routed_from_pointers(
        hidden_states_cpu.data_ptr(),
        out.data_ptr(),
        topk_ids_cpu.data_ptr<int32_t>(),
        topk_weights_cpu.data_ptr<float>(),
        tokens,
        top_k,
        (at::ScalarType)hidden_states_cpu.scalar_type()
    );
    return out;
}

// ------------------------- pybind module -------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_moe_infer_handle", [](int64_t num_experts, int64_t hidden_size, int64_t intermediate_size) {
        auto* ptr = new MoEInfer(num_experts, hidden_size, intermediate_size);
        return py::capsule(ptr, [](void* p) { delete reinterpret_cast<MoEInfer*>(p); });
    });

    m.def("moe_infer_quantize_and_store", [](py::capsule& handle, int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight_fp32_cpu) {
        handle.get_pointer<MoEInfer>()->quantize_and_store_expert(expert_idx, proj_name, weight_fp32_cpu);
    });

    m.def("moe_infer_store_quantized_repack", [](py::capsule& handle,
                                                const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d,
                                                const torch::Tensor& down_qs, const torch::Tensor& down_d) {
        handle.get_pointer<MoEInfer>()->store_quantized_weights_repack(gate_up_qs, gate_up_d, down_qs, down_d);
    });

    m.def("execute_moe_cpu_routed", &execute_moe_cpu_routed,
          py::arg("moe_infer_handle"),
          py::arg("hidden_states_cpu"),
          py::arg("topk_ids_cpu"),
          py::arg("topk_weights_cpu"),
          py::arg("top_k"));

#ifdef WITH_NPU
    py::class_<NpuCallbackManager>(m, "NpuCallbackManager")
        .def(py::init<uint64_t, int>(), py::arg("stream_ptr"), py::arg("device_id"));

    m.def("create_moe_graph_context", [](py::capsule& moe_infer_handle,
                                        int64_t tokens_bucket,
                                        int64_t top_k,
                                        int64_t dtype_int) {
        // dtype_int: 0 -> fp16, 1 -> bf16
        auto* moe = moe_infer_handle.get_pointer<MoEInfer>();
        at::ScalarType dtype = (dtype_int == 0) ? at::kHalf : at::kBFloat16;
        auto* ctx = new MoEGraphContext(moe, tokens_bucket, top_k, dtype);
        return py::capsule(ctx, [](void* p) { delete reinterpret_cast<MoEGraphContext*>(p); });
    }, py::arg("moe_infer_handle"), py::arg("tokens_bucket"), py::arg("top_k"), py::arg("dtype_int"));

    m.def("moe_forward_npu_routed_graph", &moe_forward_npu_routed_graph,
          py::arg("hidden_states_npu"),
          py::arg("topk_ids_npu"),
          py::arg("topk_weights_npu"),
          py::arg("moe_infer_handle"),
          py::arg("graph_ctx_capsule"),
          py::arg("stream_ptr"));
#endif
}
