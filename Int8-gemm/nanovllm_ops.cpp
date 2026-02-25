#include <torch/extension.h>
#include <torch/library.h>
#include <torch/custom_class.h>

#include "moe_infer.h"
#include "quant_traits.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <sstream>

#ifdef WITH_NPU
#include "acl/acl_rt.h"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

// ============================================================
// CustomClass: MoEInferHandle
// ============================================================

struct MoEInferHandle : torch::CustomClassHolder {
    quant::QuantType quant_type;
    std::unique_ptr<MoEInfer> impl;

    // Constructor with quantization type (default: Q8_0 for backward compatibility)
    MoEInferHandle(int64_t E, int64_t H, int64_t I, int64_t quant_type_int = 0)
        : quant_type(static_cast<quant::QuantType>(quant_type_int)),
          impl(std::make_unique<MoEInfer>(E, H, I, quant_type)) {}

    // Helper to get quantization type as string
    std::string get_quant_type() const {
        return (quant_type == quant::QuantType::Q4_0) ? "Q4_0" : "Q8_0";
    }
};

#ifdef WITH_NPU
static inline aclrtStream current_acl_stream(c10::DeviceIndex device_index) {
    auto s = c10_npu::getCurrentNPUStream(device_index);
    return s.stream();
}
#endif

// ============================================================
// NPU callback manager: 强关联某条 stream（需要从外部传 stream_ptr）
// 注意：TorchBind integral 只能用 int64_t
// ============================================================

#ifdef WITH_NPU
class NpuCallbackManager : public torch::CustomClassHolder {
public:
    // stream_ptr: Python 侧传入 int(stream) 的数值
    NpuCallbackManager(int64_t stream_ptr, int64_t device_id)
        : device_id_((int)device_id),
          stream_(reinterpret_cast<aclrtStream>((uintptr_t)stream_ptr)),
          stop_flag_(false)
    {
        TORCH_CHECK(stream_ != nullptr, "NpuCallbackManager: stream_ptr is null");

        callback_thread_ = std::thread(&NpuCallbackManager::process_loop, this);

        std::ostringstream oss;
        oss << callback_thread_.get_id();
        uint64_t tid = std::stoull(oss.str());

        aclError err = aclrtSubscribeReport(tid, stream_);
        TORCH_CHECK(err == ACL_SUCCESS, "aclrtSubscribeReport failed: ", err);
    }

    ~NpuCallbackManager() override {
        stop_flag_.store(true);

        std::ostringstream oss;
        oss << callback_thread_.get_id();
        uint64_t tid = std::stoull(oss.str());

        (void)aclrtSetDevice(device_id_);
        (void)aclrtUnSubscribeReport(tid, stream_);

        if (callback_thread_.joinable()) callback_thread_.join();
    }

private:
    void process_loop() {
        aclError err = aclrtSetDevice(device_id_);
        if (err != ACL_SUCCESS) return;

        while (!stop_flag_.load()) {
            (void)aclrtProcessReport(100);
        }
    }

    int device_id_;
    aclrtStream stream_{nullptr};
    std::thread callback_thread_;
    std::atomic<bool> stop_flag_;
};
#endif

// ============================================================
// pinned buffers / callbacks
// ============================================================

#ifdef WITH_NPU
struct PinnedBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    ~PinnedBuffer() {
        if (ptr) { (void)aclrtFreeHost(ptr); ptr = nullptr; size = 0; }
    }
    void alloc(size_t bytes) {
        size = bytes;
        TORCH_CHECK(aclrtMallocHost(&ptr, bytes) == ACL_SUCCESS, "aclrtMallocHost failed");
    }
};

struct MoECpuTaskArgs {
    // Direct function pointer - eliminates virtual/object dispatch
    MoEInfer::ExecuteFn execute_fn;

    // All parameters needed for the call (flattened)
    const void* x_in_ptr;
    void* y_out_ptr;
    const float* topk_weights_ptr;
    const int32_t* topk_ids_ptr;
    const void* const* gate_up_qs_tp;
    const void* const* gate_up_d_tp;
    const void* const* down_proj_qs_tp;
    const void* const* down_proj_d_tp;
    int64_t num_tokens;
    int64_t hidden_size;
    int64_t num_experts;
    int64_t intermediate_size;
    int64_t tp_size;
    int64_t top_k;
};

// Direct callback - calls execute_fn directly, no intermediate function
static void moe_compute_callback(void* user_data) {
    auto* args = reinterpret_cast<MoECpuTaskArgs*>(user_data);
    args->execute_fn(
        args->x_in_ptr,
        args->y_out_ptr,
        args->topk_weights_ptr,
        args->topk_ids_ptr,
        args->gate_up_qs_tp,
        args->gate_up_d_tp,
        args->down_proj_qs_tp,
        args->down_proj_d_tp,
        args->num_tokens,
        args->hidden_size,
        args->num_experts,
        args->intermediate_size,
        args->tp_size,
        args->top_k
    );
}

struct MoEGraphContext : torch::CustomClassHolder {
    PinnedBuffer hidden_in, hidden_out, topk_ids, topk_w;
    MoECpuTaskArgs args;

    MoEGraphContext(const c10::intrusive_ptr<MoEInferHandle>& moe_h,
                    int64_t num_tokens,
                    int64_t top_k,
                    int64_t dtype_int) {

        auto dtype = (dtype_int == 0) ? at::kHalf : at::kBFloat16;
        TORCH_CHECK(top_k == 1 || top_k == 8 || top_k == 10);
        TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16);

        const int64_t H = moe_h->impl->hidden_size();
        const size_t elem = (dtype == at::kHalf) ? sizeof(at::Half) : sizeof(at::BFloat16);

        const size_t hb = (size_t)num_tokens * (size_t)H * elem;
        const size_t ib = (size_t)num_tokens * (size_t)top_k * sizeof(int32_t);
        const size_t wb = (size_t)num_tokens * (size_t)top_k * sizeof(float);

        hidden_in.alloc(hb);
        hidden_out.alloc(hb);
        topk_ids.alloc(ib);
        topk_w.alloc(wb);

        args.x_in_ptr = hidden_in.ptr;
        args.y_out_ptr = hidden_out.ptr;
        args.topk_ids_ptr = (const int32_t*)topk_ids.ptr;
        args.topk_weights_ptr = (const float*)topk_w.ptr;
        args.num_tokens = num_tokens;
        args.top_k = top_k;
        args.hidden_size = moe_h->impl->hidden_size();
        args.num_experts = moe_h->impl->num_experts();
        args.intermediate_size = moe_h->impl->intermediate_size();
        args.tp_size = moe_h->impl->tp_size();
        args.execute_fn = moe_h->impl->get_execute_function(dtype);
        args.gate_up_qs_tp = moe_h->impl->gate_up_qs_tp_data();
        args.gate_up_d_tp = moe_h->impl->gate_up_d_tp_data();
        args.down_proj_qs_tp = moe_h->impl->down_proj_qs_tp_data();
        args.down_proj_d_tp = moe_h->impl->down_proj_d_tp_data();
    }
};

struct StreamCallData {
    int device_id = 0;
    MoECpuTaskArgs args;
    PinnedBuffer hidden_in, hidden_out, topk_ids, topk_w;
};

static void moe_cleanup_callback(void* user_data) {
    auto* p = reinterpret_cast<StreamCallData*>(user_data);
    delete p;
}
#endif

// ============================================================
// CPU op
// ============================================================

static torch::Tensor moe_forward_cpu(
    const torch::Tensor& hidden_cpu,
    const torch::Tensor& topk_ids_cpu,
    const torch::Tensor& topk_w_cpu,
    const c10::intrusive_ptr<MoEInferHandle>& moe_h)
{
    auto out = torch::empty_like(hidden_cpu);
    moe_h->impl->execute_on_cpu_routed_from_pointers(
        hidden_cpu.data_ptr(),
        out.data_ptr(),
        topk_ids_cpu.data_ptr<int32_t>(),
        topk_w_cpu.data_ptr<float>(),
        hidden_cpu.size(0),
        topk_ids_cpu.size(1),
        (at::ScalarType)hidden_cpu.scalar_type()
    );
    return out;
}

#ifdef WITH_NPU
// ============================================================
// NPU: 非 graph stream 版本（内部 empty_like + new/delete）
// stream 在 C++ 侧通过 getCurrentNPUStream 获取
// ============================================================

static torch::Tensor moe_forward_npu_stream(
    const torch::Tensor& hidden_npu,
    const torch::Tensor& topk_ids_npu,
    const torch::Tensor& topk_w_npu,
    const c10::intrusive_ptr<MoEInferHandle>& moe_h)
{
    auto out_npu = torch::empty_like(hidden_npu);

    const int64_t tokens = hidden_npu.size(0);
    const int64_t top_k = topk_ids_npu.size(1);

    auto* cd = new StreamCallData();
    cd->device_id = hidden_npu.get_device();

    const size_t hb = hidden_npu.nbytes();
    const size_t ib = topk_ids_npu.nbytes();
    const size_t wb = topk_w_npu.nbytes();

    cd->hidden_in.alloc(hb);
    cd->hidden_out.alloc(hb);
    cd->topk_ids.alloc(ib);
    cd->topk_w.alloc(wb);

    // Pre-bind execute function and all parameters
    
    cd->args.execute_fn = moe_h->impl->get_execute_function(hidden_npu.scalar_type());
    cd->args.x_in_ptr = cd->hidden_in.ptr;
    cd->args.y_out_ptr = cd->hidden_out.ptr;
    cd->args.topk_ids_ptr = (const int32_t*)cd->topk_ids.ptr;
    cd->args.topk_weights_ptr = (const float*)cd->topk_w.ptr;
    cd->args.num_tokens = tokens;
    cd->args.top_k = top_k;
    cd->args.hidden_size = moe_h->impl->hidden_size();
    cd->args.num_experts = moe_h->impl->num_experts();
    cd->args.intermediate_size = moe_h->impl->intermediate_size();
    cd->args.tp_size = moe_h->impl->tp_size();
    // Weight pointers fetched from moe_h at callback time via get_weight_pointers
    cd->args.gate_up_qs_tp = moe_h->impl->gate_up_qs_tp_data();
    cd->args.gate_up_d_tp = moe_h->impl->gate_up_d_tp_data();
    cd->args.down_proj_qs_tp = moe_h->impl->down_proj_qs_tp_data();
    cd->args.down_proj_d_tp = moe_h->impl->down_proj_d_tp_data();

    aclrtStream stream = current_acl_stream(hidden_npu.get_device());

    TORCH_CHECK(aclrtMemcpyAsync(cd->hidden_in.ptr, hb, hidden_npu.data_ptr(), hb,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(cd->topk_ids.ptr, ib, topk_ids_npu.data_ptr(), ib,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(cd->topk_w.ptr, wb, topk_w_npu.data_ptr(), wb,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);

    TORCH_CHECK(aclrtLaunchCallback(moe_compute_callback, (void*)&cd->args,
                                   ACL_CALLBACK_BLOCK, stream) == ACL_SUCCESS);

    TORCH_CHECK(aclrtMemcpyAsync(out_npu.data_ptr(), hb, cd->hidden_out.ptr, hb,
                                ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS);

    TORCH_CHECK(aclrtLaunchCallback(moe_cleanup_callback, (void*)cd,
                                   ACL_CALLBACK_BLOCK, stream) == ACL_SUCCESS);

    return out_npu;
}

// graph-safe out variant
static void moe_forward_npu_graph_out(
    const torch::Tensor& hidden_npu,
    const torch::Tensor& topk_ids_npu,
    const torch::Tensor& topk_w_npu,
    const c10::intrusive_ptr<MoEInferHandle>& moe_h,
    const c10::intrusive_ptr<MoEGraphContext>& ctx,
    torch::Tensor& out_npu)
{
    (void)moe_h; // ctx->args 已绑定 moe 指针，这里不强用也行

    aclrtStream stream = current_acl_stream(hidden_npu.get_device());

    const size_t hb = hidden_npu.nbytes();
    const size_t ib = topk_ids_npu.nbytes();
    const size_t wb = topk_w_npu.nbytes();

    TORCH_CHECK(aclrtMemcpyAsync(ctx->hidden_in.ptr, ctx->hidden_in.size,
                                hidden_npu.data_ptr(), hb,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);

    TORCH_CHECK(aclrtMemcpyAsync(ctx->topk_ids.ptr, ctx->topk_ids.size,
                                topk_ids_npu.data_ptr(), ib,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);

    TORCH_CHECK(aclrtMemcpyAsync(ctx->topk_w.ptr, ctx->topk_w.size,
                                topk_w_npu.data_ptr(), wb,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);

    TORCH_CHECK(aclrtLaunchCallback(moe_compute_callback, (void*)&ctx->args,
                                   ACL_CALLBACK_BLOCK, stream) == ACL_SUCCESS);

    TORCH_CHECK(aclrtMemcpyAsync(out_npu.data_ptr(), hb,
                                ctx->hidden_out.ptr, ctx->hidden_out.size,
                                ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS);
}
#endif

// ============================================================
// Registration
// ============================================================

TORCH_LIBRARY_FRAGMENT(nanovllm, m) {
    m.class_<MoEInferHandle>("MoEInfer")
        // Main constructor with quantization type support
        // quant_type: 0 = Q8_0 (default), 1 = Q4_0
        .def(torch::init<int64_t,int64_t,int64_t,int64_t>())
        .def("quantize_and_store_expert",
             [](const c10::intrusive_ptr<MoEInferHandle>& self,
                int64_t expert_idx,
                const std::string& proj_name,
                const torch::Tensor& w) {
                 // Q4_0 does not support online quantization
                 if (self->quant_type == quant::QuantType::Q4_0) {
                     TORCH_CHECK(false, "Q4_0 does not support online quantization. "
                                     "Use store_quantized_repack() with pre-quantized weights.");
                 }
                 self->impl->quantize_and_store_expert(expert_idx, proj_name, w);
             })
        .def("store_quantized_repack",
             [](const c10::intrusive_ptr<MoEInferHandle>& self,
                const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d,
                const torch::Tensor& down_qs, const torch::Tensor& down_d) {
                 self->impl->store_quantized_weights_repack(gate_up_qs, gate_up_d, down_qs, down_d);
             })
        .def("get_last_run_time_ms",
             [](const c10::intrusive_ptr<MoEInferHandle>& self) {
                 return self->impl->get_last_run_time_ms();
             })
        .def("get_quant_type",
             [](const c10::intrusive_ptr<MoEInferHandle>& self) {
                 return self->get_quant_type();
             })
        .def("get_num_experts",
             [](const c10::intrusive_ptr<MoEInferHandle>& self) {
                 return self->impl->num_experts();
             })
        .def("get_hidden_size",
             [](const c10::intrusive_ptr<MoEInferHandle>& self) {
                 return self->impl->hidden_size();
             })
        .def("get_intermediate_size",
             [](const c10::intrusive_ptr<MoEInferHandle>& self) {
                 return self->impl->intermediate_size();
             });

#ifdef WITH_NPU
    // 关键：用 int64_t, 不要 torch::arg(...)
    m.class_<NpuCallbackManager>("NpuCallbackManager")
        .def(torch::init<int64_t, int64_t>());

    m.class_<MoEGraphContext>("MoEGraphContext")
        .def(torch::init<const c10::intrusive_ptr<MoEInferHandle>&, int64_t, int64_t, int64_t>());
#endif

    m.def("moe_forward(Tensor hidden, Tensor topk_ids, Tensor topk_w, __torch__.torch.classes.nanovllm.MoEInfer moe) -> Tensor");

#ifdef WITH_NPU
    m.def("moe_forward_npu_stream(Tensor hidden, Tensor topk_ids, Tensor topk_w, __torch__.torch.classes.nanovllm.MoEInfer moe) -> Tensor");
    m.def("moe_forward_npu_graph_out(Tensor hidden, Tensor topk_ids, Tensor topk_w, __torch__.torch.classes.nanovllm.MoEInfer moe, __torch__.torch.classes.nanovllm.MoEGraphContext ctx, Tensor(a!) out) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(nanovllm, CPU, m) {
    m.impl("moe_forward", &moe_forward_cpu);
}

#ifdef WITH_NPU
TORCH_LIBRARY_IMPL(nanovllm, PrivateUse1, m) {
    m.impl("moe_forward_npu_stream", &moe_forward_npu_stream);
    m.impl("moe_forward_npu_graph_out",
           [](const torch::Tensor& hidden,
              const torch::Tensor& ids,
              const torch::Tensor& w,
              const c10::intrusive_ptr<MoEInferHandle>& moe,
              const c10::intrusive_ptr<MoEGraphContext>& ctx,
              torch::Tensor out) {
               moe_forward_npu_graph_out(hidden, ids, w, moe, ctx, out);
           });
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "nanovllm_ext (dispatcher registered ops)";
}
