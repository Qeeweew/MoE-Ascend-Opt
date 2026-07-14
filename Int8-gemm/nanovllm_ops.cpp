#include <torch/extension.h>
#include <torch/library.h>
#include <torch/custom_class.h>

#include "moe_infer.h"
#include "quant_traits.h"

#include <atomic>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <sstream>
#include <vector>

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

// ============================================================
// CustomClass: ExpertCacheScheduler
// ============================================================

class ExpertCacheScheduler : public torch::CustomClassHolder {
    using PlanRow = std::array<int64_t, 5>;

public:
    ExpertCacheScheduler(int64_t cache_size, int64_t swap_per_update,
                         int64_t update_interval, int64_t warmup_steps,
                         double decay)
        : cache_size_(cache_size),
          swap_per_update_(swap_per_update),
          update_interval_(update_interval),
          warmup_steps_(warmup_steps),
          decay_(decay) {
        TORCH_CHECK(cache_size_ > 0, "cache size must be positive");
        TORCH_CHECK(swap_per_update_ > 0, "swap_per_update must be positive");
        TORCH_CHECK(update_interval_ > 0, "update_interval must be positive");
        TORCH_CHECK(warmup_steps_ >= 0, "warmup_steps must be non-negative");
        TORCH_CHECK(decay_ >= 0.0 && decay_ < 1.0, "decay must be in [0,1)");
        const int64_t physical = cache_size_ + swap_per_update_;
        slot_owner_.assign((size_t)physical, -1);
        for (int64_t slot = 0; slot < physical; ++slot) {
            free_slots_.push_back(slot);
        }
    }

    void register_layer(int64_t layer_idx,
                        const c10::intrusive_ptr<MoEInferHandle>& handle) {
        TORCH_CHECK(layer_idx >= 0, "layer index must be non-negative");
        TORCH_CHECK(handle && handle->impl, "invalid MoEInfer handle");
        if (num_experts_ < 0) {
            num_experts_ = handle->impl->num_experts();
            hidden_size_ = handle->impl->hidden_size();
            intermediate_size_ = handle->impl->intermediate_size();
            scale_dtype_ = handle->impl->get_scale_dtype();
            export_groups_ = handle->impl->npu_layout_export_groups();
        } else {
            TORCH_CHECK(handle->impl->num_experts() == num_experts_
                            && handle->impl->hidden_size() == hidden_size_
                            && handle->impl->intermediate_size() == intermediate_size_
                            && handle->impl->get_scale_dtype() == scale_dtype_
                            && handle->impl->npu_layout_export_groups() == export_groups_,
                        "all cache layers must have identical expert shapes");
        }
        if ((int64_t)sources_.size() <= layer_idx) {
            sources_.resize((size_t)layer_idx + 1);
        }
        TORCH_CHECK(!sources_[(size_t)layer_idx], "layer already registered");
        sources_[(size_t)layer_idx] = handle;
        const size_t required = sources_.size() * (size_t)num_experts_;
        freq_.resize(required, 0.0);
        owner_slot_.resize(required, -1);
        cooldown_.resize(required, 0);
    }

    int64_t plan_out(int64_t valid_tokens, const torch::Tensor& output) {
        TORCH_CHECK(output.device().is_cpu()
                        && output.scalar_type() == at::kLong
                        && output.is_contiguous() && output.dim() == 2
                        && output.size(1) == 5
                        && output.size(0) == swap_per_update_,
                    "plan output must be contiguous int64 CPU [swap_per_update,5]");
        std::vector<PlanRow> rows;
        const int64_t count = schedule(valid_tokens, rows);
        if (count > 0) {
            auto* output_ptr = output.data_ptr<int64_t>();
            for (int64_t row = 0; row < count; ++row) {
                std::copy(rows[(size_t)row].begin(), rows[(size_t)row].end(),
                          output_ptr + row * 5);
            }
        }
        return count;
    }

    std::vector<torch::Tensor> last_stats() const {
        auto counters = torch::tensor(
            {replay_steps_, active_slots_, total_swaps_, last_window_total_,
             last_window_miss_, last_interval_, last_update_swaps_},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto rates = torch::tensor(
            {last_hit_rate_},
            torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
        return {counters, rates};
    }

private:
    int64_t schedule(int64_t valid_tokens, std::vector<PlanRow>& rows) {
        TORCH_CHECK(num_experts_ > 0 && !sources_.empty(),
                    "expert cache scheduler has no registered layers");
        if (!started_) {
            for (const auto& source : sources_) {
                if (source) source->impl->reset_routing_stats();
            }
            started_ = true;
        }
        for (const auto& source : sources_) {
            if (source) source->impl->set_valid_tokens(valid_tokens);
        }

        ++replay_steps_;
        int64_t interval = update_interval_;
        if (active_slots_ >= cache_size_) {
            interval *= steady_interval_multiplier_;
        }
        last_interval_ = interval;
        if (replay_steps_ <= warmup_steps_
                || (replay_steps_ - warmup_steps_) % interval != 0) {
            return -1;
        }

        int64_t window_total = 0;
        int64_t window_miss = 0;
        for (size_t layer = 0; layer < sources_.size(); ++layer) {
            const auto& source = sources_[layer];
            if (!source) continue;
            auto stats = source->impl->take_routing_stats_snapshot();
            TORCH_CHECK((int64_t)stats.counts.size() == num_experts_,
                        "routing stats size mismatch");
            const size_t base = layer * (size_t)num_experts_;
            for (int64_t expert = 0; expert < num_experts_; ++expert) {
                freq_[base + (size_t)expert] =
                    freq_[base + (size_t)expert] * decay_
                    + (double)stats.counts[(size_t)expert];
            }
            window_total += stats.total;
            window_miss += stats.misses;
        }
        const double hit_rate = window_total == 0
            ? 0.0 : 1.0 - (double)window_miss / (double)window_total;

        for (int& value : cooldown_) {
            if (value > 0) --value;
        }

        int64_t limit = std::min<int64_t>(swap_per_update_, 8);
        if (active_slots_ < cache_size_) {
            limit = std::min<int64_t>(
                swap_per_update_, cache_size_ - active_slots_);
        }

        struct ScoredOwner {
            double score;
            int64_t flat;
        };
        std::vector<ScoredOwner> candidates;
        candidates.reserve(freq_.size());
        for (size_t layer = 0; layer < sources_.size(); ++layer) {
            if (!sources_[layer]) continue;
            const size_t base = layer * (size_t)num_experts_;
            for (int64_t expert = 0; expert < num_experts_; ++expert) {
                const int64_t flat = (int64_t)(base + (size_t)expert);
                if (owner_slot_[(size_t)flat] >= 0 || cooldown_[(size_t)flat] > 0) {
                    continue;
                }
                const double score = freq_[(size_t)flat];
                if (score > 0.0) candidates.push_back({score, flat});
            }
        }
        const size_t candidate_count = std::min<size_t>(
            (size_t)limit, candidates.size());
        std::partial_sort(
            candidates.begin(), candidates.begin() + candidate_count,
            candidates.end(), [](const ScoredOwner& a, const ScoredOwner& b) {
                return a.score > b.score
                    || (a.score == b.score && a.flat > b.flat);
            });

        std::vector<ScoredOwner> victims;
        if (active_slots_ >= cache_size_) {
            victims.reserve((size_t)cache_size_);
            for (int64_t flat : slot_owner_) {
                if (flat < 0 || cooldown_[(size_t)flat] > 0) continue;
                victims.push_back({freq_[(size_t)flat], flat});
            }
            const size_t victim_count = std::min<size_t>(
                candidate_count, victims.size());
            std::partial_sort(
                victims.begin(), victims.begin() + victim_count,
                victims.end(), [](const ScoredOwner& a, const ScoredOwner& b) {
                    return a.score < b.score
                        || (a.score == b.score && a.flat < b.flat);
                });
        }

        rows.reserve(candidate_count);
        for (size_t i = 0; i < candidate_count; ++i) {
            if (free_slots_.empty()) break;
            int64_t victim_flat = -1;
            if (active_slots_ >= cache_size_) {
                if (i >= victims.size()) break;
                if (candidates[i].score <= victims[i].score * 1.10) break;
                victim_flat = victims[i].flat;
            }
            const int64_t slot = free_slots_.front();
            free_slots_.pop_front();
            const int64_t new_flat = candidates[i].flat;
            rows.push_back({
                new_flat / num_experts_, new_flat % num_experts_, slot,
                victim_flat < 0 ? -1 : victim_flat / num_experts_,
                victim_flat < 0 ? -1 : victim_flat % num_experts_,
            });

            if (victim_flat >= 0) {
                const int64_t old_slot = owner_slot_[(size_t)victim_flat];
                owner_slot_[(size_t)victim_flat] = -1;
                slot_owner_[(size_t)old_slot] = -1;
                cooldown_[(size_t)victim_flat] = 2;
                free_slots_.push_back(old_slot);
            } else {
                ++active_slots_;
            }
            owner_slot_[(size_t)new_flat] = slot;
            slot_owner_[(size_t)slot] = new_flat;
        }

        total_swaps_ += (int64_t)rows.size();
        update_backoff(hit_rate, (int64_t)rows.size());
        last_window_total_ = window_total;
        last_window_miss_ = window_miss;
        last_hit_rate_ = hit_rate;
        last_update_swaps_ = (int64_t)rows.size();
        return last_update_swaps_;
    }

public:

    void export_plan_out(
        const torch::Tensor& plan,
        const torch::Tensor& w13, const torch::Tensor& s13,
        const torch::Tensor& w2, const torch::Tensor& s2) const {
        TORCH_CHECK(plan.device().is_cpu() && plan.scalar_type() == at::kLong
                        && plan.is_contiguous() && plan.dim() == 2
                        && plan.size(1) == 5,
                    "plan must be contiguous int64 CPU [B,5]");
        const int64_t batch = plan.size(0);
        TORCH_CHECK(batch > 0, "cannot export an empty replacement plan");
        for (const auto& tensor : {w13, s13, w2, s2}) {
            TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous()
                            && tensor.is_pinned(),
                        "replacement staging tensors must be contiguous pinned CPU tensors");
        }
        TORCH_CHECK(w13.scalar_type() == at::kInt
                        && w13.sizes() == at::IntArrayRef(
                            {batch, hidden_size_, 2 * intermediate_size_ / 8}),
                    "w13 staging shape mismatch");
        TORCH_CHECK(s13.scalar_type() == scale_dtype_
                        && s13.sizes() == at::IntArrayRef(
                            {batch, hidden_size_ / 32, 2 * intermediate_size_}),
                    "s13 staging shape mismatch");
        TORCH_CHECK(w2.scalar_type() == at::kInt
                        && w2.sizes() == at::IntArrayRef(
                            {batch, intermediate_size_, hidden_size_ / 8}),
                    "w2 staging shape mismatch");
        TORCH_CHECK(s2.scalar_type() == scale_dtype_
                        && s2.sizes() == at::IntArrayRef(
                            {batch, intermediate_size_ / 32, hidden_size_}),
                    "s2 staging shape mismatch");

        const auto* plan_ptr = plan.data_ptr<int64_t>();
        auto* w13_base = reinterpret_cast<uint32_t*>(w13.data_ptr<int32_t>());
        auto* s13_base = reinterpret_cast<uint16_t*>(s13.data_ptr());
        auto* w2_base = reinterpret_cast<uint32_t*>(w2.data_ptr<int32_t>());
        auto* s2_base = reinterpret_cast<uint16_t*>(s2.data_ptr());
        const int64_t w13_stride = hidden_size_ * (2 * intermediate_size_ / 8);
        const int64_t s13_stride = (hidden_size_ / 32) * (2 * intermediate_size_);
        const int64_t w2_stride = intermediate_size_ * (hidden_size_ / 8);
        const int64_t s2_stride = (intermediate_size_ / 32) * hidden_size_;

        for (int64_t row = 0; row < batch; ++row) {
            const int64_t layer = plan_ptr[row * 5];
            const int64_t expert = plan_ptr[row * 5 + 1];
            TORCH_CHECK(layer >= 0 && layer < (int64_t)sources_.size()
                            && sources_[(size_t)layer],
                        "replacement plan references an unregistered layer");
            TORCH_CHECK(expert >= 0 && expert < num_experts_,
                        "replacement plan references an invalid expert");
        }

        nanovllm::NumaExecutorManager::get(0)->pool->parallel_for_static(
            0, batch * export_groups_, [&](int64_t task) {
                const int64_t row = task / export_groups_;
                const int64_t group = task % export_groups_;
                const int64_t layer = plan_ptr[row * 5];
                const int64_t expert = plan_ptr[row * 5 + 1];
                sources_[(size_t)layer]->impl->export_expert_npu_layout_group(
                    expert, group,
                    w13_base + row * w13_stride,
                    s13_base + row * s13_stride,
                    w2_base + row * w2_stride,
                    s2_base + row * s2_stride);
            });
    }

    void reset_routing_stats() {
        for (const auto& source : sources_) {
            if (source) source->impl->reset_routing_stats();
        }
    }

private:
    void update_backoff(double hit_rate, int64_t swaps) {
        if (active_slots_ < cache_size_) {
            steady_interval_multiplier_ = 8;
            stable_no_swap_windows_ = 0;
            last_window_hit_rate_ = hit_rate;
            return;
        }
        const bool hit_rate_dropped = hit_rate + 0.05 < last_window_hit_rate_;
        if (swaps > 0 || hit_rate_dropped) {
            steady_interval_multiplier_ = 8;
            stable_no_swap_windows_ = 0;
            last_window_hit_rate_ = hit_rate;
            return;
        }
        ++stable_no_swap_windows_;
        if (stable_no_swap_windows_ >= 2 && steady_interval_multiplier_ < 32) {
            steady_interval_multiplier_ = std::min<int64_t>(
                steady_interval_multiplier_ * 2, 32);
            stable_no_swap_windows_ = 0;
        }
        last_window_hit_rate_ = hit_rate;
    }

    int64_t cache_size_;
    int64_t swap_per_update_;
    int64_t update_interval_;
    int64_t warmup_steps_;
    double decay_;
    int64_t num_experts_ = -1;
    int64_t hidden_size_ = -1;
    int64_t intermediate_size_ = -1;
    int64_t export_groups_ = -1;
    at::ScalarType scale_dtype_ = at::kHalf;
    std::vector<c10::intrusive_ptr<MoEInferHandle>> sources_;
    std::vector<double> freq_;
    std::vector<int64_t> owner_slot_;
    std::vector<int64_t> slot_owner_;
    std::vector<int> cooldown_;
    std::deque<int64_t> free_slots_;
    int64_t replay_steps_ = 0;
    int64_t active_slots_ = 0;
    int64_t total_swaps_ = 0;
    int64_t steady_interval_multiplier_ = 8;
    int64_t stable_no_swap_windows_ = 0;
    double last_window_hit_rate_ = 0.0;
    int64_t last_window_total_ = 0;
    int64_t last_window_miss_ = 0;
    int64_t last_interval_ = 0;
    int64_t last_update_swaps_ = 0;
    double last_hit_rate_ = 0.0;
    bool started_ = false;
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
    const int32_t* routing_ids_ptr = nullptr;
    MoEInfer* moe_impl = nullptr;
    bool record_routing = false;
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
static void run_moe_compute(MoECpuTaskArgs* args) {
    if (args->record_routing && args->moe_impl != nullptr && args->routing_ids_ptr != nullptr) {
        args->moe_impl->record_routing(
            args->routing_ids_ptr, args->topk_ids_ptr,
            args->num_tokens, args->top_k);
    }
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

static void moe_compute_callback(void* user_data) {
    run_moe_compute(reinterpret_cast<MoECpuTaskArgs*>(user_data));
}

struct MoEAsyncState {
    std::mutex mutex;
    std::condition_variable cv;
    bool in_flight = false;
    bool done = false;
    bool failed = false;
};

struct MoEAsyncJob {
    MoECpuTaskArgs* args = nullptr;
    MoEAsyncState* state = nullptr;
};

// A coordinator thread is deliberately separate from the NUMA GEMV worker
// pools.  execute_fn submits work to those pools and waits for completion, so
// running the coordinator inside one of them can deadlock when all workers are
// occupied by the MoE kernels themselves.
class MoEAsyncExecutor {
public:
    static MoEAsyncExecutor& instance() {
        static MoEAsyncExecutor executor;
        return executor;
    }

    void submit(MoEAsyncJob job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(job);
        }
        cv_.notify_one();
    }

private:
    MoEAsyncExecutor() : worker_(&MoEAsyncExecutor::worker_loop, this) {}

    ~MoEAsyncExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    void worker_loop() {
        for (;;) {
            MoEAsyncJob job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty()) return;
                job = jobs_.front();
                jobs_.pop_front();
            }

            bool failed = false;
            try {
                run_moe_compute(job.args);
            } catch (...) {
                failed = true;
            }
            {
                std::lock_guard<std::mutex> lock(job.state->mutex);
                job.state->failed = failed;
                job.state->done = true;
            }
            job.state->cv.notify_one();
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<MoEAsyncJob> jobs_;
    bool stop_ = false;
    std::thread worker_;
};

struct MoEGraphContext : torch::CustomClassHolder {
    PinnedBuffer hidden_in, hidden_out, topk_ids, routing_ids, topk_w;
    MoECpuTaskArgs args;
    MoEAsyncState async_state;

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
        routing_ids.alloc(ib);
        topk_w.alloc(wb);

        args.x_in_ptr = hidden_in.ptr;
        args.y_out_ptr = hidden_out.ptr;
        args.topk_ids_ptr = (const int32_t*)topk_ids.ptr;
        args.routing_ids_ptr = (const int32_t*)routing_ids.ptr;
        args.moe_impl = moe_h->impl.get();
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

static void moe_async_start_callback(void* user_data) {
    auto* ctx = reinterpret_cast<MoEGraphContext*>(user_data);
    {
        std::lock_guard<std::mutex> lock(ctx->async_state.mutex);
        if (ctx->async_state.in_flight && !ctx->async_state.done) {
            std::terminate();
        }
        ctx->async_state.in_flight = true;
        ctx->async_state.done = false;
        ctx->async_state.failed = false;
    }
    MoEAsyncExecutor::instance().submit({&ctx->args, &ctx->async_state});
}

static void moe_async_join_callback(void* user_data) {
    auto* ctx = reinterpret_cast<MoEGraphContext*>(user_data);
    std::unique_lock<std::mutex> lock(ctx->async_state.mutex);
    ctx->async_state.cv.wait(lock, [&] { return ctx->async_state.done; });
    if (ctx->async_state.failed) {
        std::terminate();
    }
    ctx->async_state.in_flight = false;
}

struct StreamCallData {
    int device_id = 0;
    MoECpuTaskArgs args;
    PinnedBuffer hidden_in, hidden_out, topk_ids, routing_ids, topk_w;
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

// Cache partial variant.  ``compute_ids`` contains -1 for cache hits while
// ``routing_ids`` preserves the original expert ids for LFU telemetry.
static torch::Tensor moe_forward_npu_stream_partial(
    const torch::Tensor& hidden_npu,
    const torch::Tensor& compute_ids_npu,
    const torch::Tensor& routing_ids_npu,
    const torch::Tensor& topk_w_npu,
    const c10::intrusive_ptr<MoEInferHandle>& moe_h)
{
    auto out_npu = torch::empty_like(hidden_npu);
    const int64_t tokens = hidden_npu.size(0);
    const int64_t top_k = compute_ids_npu.size(1);
    TORCH_CHECK(compute_ids_npu.sizes() == routing_ids_npu.sizes(),
                "compute_ids and routing_ids shape mismatch");

    auto* cd = new StreamCallData();
    cd->device_id = hidden_npu.get_device();
    const size_t hb = hidden_npu.nbytes();
    const size_t ib = compute_ids_npu.nbytes();
    const size_t wb = topk_w_npu.nbytes();
    cd->hidden_in.alloc(hb); cd->hidden_out.alloc(hb);
    cd->topk_ids.alloc(ib); cd->routing_ids.alloc(ib); cd->topk_w.alloc(wb);

    cd->args.execute_fn = moe_h->impl->get_execute_function(hidden_npu.scalar_type());
    cd->args.x_in_ptr = cd->hidden_in.ptr;
    cd->args.y_out_ptr = cd->hidden_out.ptr;
    cd->args.topk_ids_ptr = (const int32_t*)cd->topk_ids.ptr;
    cd->args.routing_ids_ptr = (const int32_t*)cd->routing_ids.ptr;
    cd->args.moe_impl = moe_h->impl.get();
    cd->args.record_routing = true;
    cd->args.topk_weights_ptr = (const float*)cd->topk_w.ptr;
    cd->args.num_tokens = tokens; cd->args.top_k = top_k;
    cd->args.hidden_size = moe_h->impl->hidden_size();
    cd->args.num_experts = moe_h->impl->num_experts();
    cd->args.intermediate_size = moe_h->impl->intermediate_size();
    cd->args.tp_size = moe_h->impl->tp_size();
    cd->args.gate_up_qs_tp = moe_h->impl->gate_up_qs_tp_data();
    cd->args.gate_up_d_tp = moe_h->impl->gate_up_d_tp_data();
    cd->args.down_proj_qs_tp = moe_h->impl->down_proj_qs_tp_data();
    cd->args.down_proj_d_tp = moe_h->impl->down_proj_d_tp_data();

    aclrtStream stream = current_acl_stream(hidden_npu.get_device());
    TORCH_CHECK(aclrtMemcpyAsync(cd->hidden_in.ptr, hb, hidden_npu.data_ptr(), hb,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(cd->topk_ids.ptr, ib, compute_ids_npu.data_ptr(), ib,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(cd->routing_ids.ptr, ib, routing_ids_npu.data_ptr(), ib,
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

static void moe_forward_npu_graph_partial_out(
    const torch::Tensor& hidden_npu,
    const torch::Tensor& compute_ids_npu,
    const torch::Tensor& routing_ids_npu,
    const torch::Tensor& topk_w_npu,
    const c10::intrusive_ptr<MoEInferHandle>& moe_h,
    const c10::intrusive_ptr<MoEGraphContext>& ctx,
    torch::Tensor& out_npu)
{
    (void)moe_h;
    TORCH_CHECK(compute_ids_npu.sizes() == routing_ids_npu.sizes(),
                "compute_ids and routing_ids shape mismatch");
    aclrtStream stream = current_acl_stream(hidden_npu.get_device());
    const size_t hb = hidden_npu.nbytes();
    const size_t ib = compute_ids_npu.nbytes();
    const size_t wb = topk_w_npu.nbytes();
    ctx->args.record_routing = true;
    TORCH_CHECK(aclrtMemcpyAsync(ctx->hidden_in.ptr, ctx->hidden_in.size,
                                hidden_npu.data_ptr(), hb, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(ctx->topk_ids.ptr, ctx->topk_ids.size,
                                compute_ids_npu.data_ptr(), ib, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(ctx->routing_ids.ptr, ctx->routing_ids.size,
                                routing_ids_npu.data_ptr(), ib, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(ctx->topk_w.ptr, ctx->topk_w.size,
                                topk_w_npu.data_ptr(), wb, ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtLaunchCallback(moe_compute_callback, (void*)&ctx->args,
                                   ACL_CALLBACK_BLOCK, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(out_npu.data_ptr(), hb, ctx->hidden_out.ptr, ctx->hidden_out.size,
                                ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS);
}

// Single-stream split callback path used by expert caching:
//   D2H -> async-start callback -> NPU cached MoE -> join callback -> H2D.
// The start callback only queues CPU work onto MoEAsyncExecutor and returns,
// allowing the same stream to proceed into the NPU kernel without a side
// stream or cross-stream events.
static void moe_forward_npu_graph_partial_start(
    const torch::Tensor& hidden_npu,
    const torch::Tensor& compute_ids_npu,
    const torch::Tensor& routing_ids_npu,
    const torch::Tensor& topk_w_npu,
    const c10::intrusive_ptr<MoEInferHandle>& moe_h,
    const c10::intrusive_ptr<MoEGraphContext>& ctx)
{
    (void)moe_h;
    TORCH_CHECK(compute_ids_npu.sizes() == routing_ids_npu.sizes(),
                "compute_ids and routing_ids shape mismatch");
    aclrtStream stream = current_acl_stream(hidden_npu.get_device());
    const size_t hb = hidden_npu.nbytes();
    const size_t ib = compute_ids_npu.nbytes();
    const size_t wb = topk_w_npu.nbytes();
    ctx->args.record_routing = true;

    TORCH_CHECK(aclrtMemcpyAsync(ctx->hidden_in.ptr, ctx->hidden_in.size,
                                hidden_npu.data_ptr(), hb,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(ctx->topk_ids.ptr, ctx->topk_ids.size,
                                compute_ids_npu.data_ptr(), ib,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(ctx->routing_ids.ptr, ctx->routing_ids.size,
                                routing_ids_npu.data_ptr(), ib,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(ctx->topk_w.ptr, ctx->topk_w.size,
                                topk_w_npu.data_ptr(), wb,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtLaunchCallback(moe_async_start_callback, ctx.get(),
                                   ACL_CALLBACK_BLOCK, stream) == ACL_SUCCESS);
}

static void moe_forward_npu_graph_partial_wait_out(
    torch::Tensor& out_npu,
    const c10::intrusive_ptr<MoEGraphContext>& ctx)
{
    aclrtStream stream = current_acl_stream(out_npu.get_device());
    const size_t hb = out_npu.nbytes();
    TORCH_CHECK(hb == ctx->hidden_out.size,
                "output size does not match MoEGraphContext");
    TORCH_CHECK(aclrtLaunchCallback(moe_async_join_callback, ctx.get(),
                                   ACL_CALLBACK_BLOCK, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(out_npu.data_ptr(), hb,
                                ctx->hidden_out.ptr, ctx->hidden_out.size,
                                ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS);
}

struct MoEContextKeepAlive {
    c10::intrusive_ptr<MoEGraphContext> ctx;
};

static void moe_context_release_callback(void* user_data) {
    delete reinterpret_cast<MoEContextKeepAlive*>(user_data);
}

// Eager counterpart of the graph wait operation.  Prefill shapes are dynamic,
// so Python creates a call-scoped MoEGraphContext instead of retaining one in
// the graph-context cache.  Keep one intrusive reference alive until the
// stream has completed the join and H2D copy, then release it from a final
// callback.  The graph path deliberately keeps its original wait operation:
// a one-shot heap pointer must never be captured into a replayable graph.
static void moe_forward_npu_eager_partial_wait_out(
    torch::Tensor& out_npu,
    const c10::intrusive_ptr<MoEGraphContext>& ctx)
{
    aclrtStream stream = current_acl_stream(out_npu.get_device());
    const size_t hb = out_npu.nbytes();
    TORCH_CHECK(hb == ctx->hidden_out.size,
                "output size does not match MoEGraphContext");

    TORCH_CHECK(aclrtLaunchCallback(moe_async_join_callback, ctx.get(),
                                   ACL_CALLBACK_BLOCK, stream) == ACL_SUCCESS);
    TORCH_CHECK(aclrtMemcpyAsync(out_npu.data_ptr(), hb,
                                ctx->hidden_out.ptr, ctx->hidden_out.size,
                                ACL_MEMCPY_HOST_TO_DEVICE, stream) == ACL_SUCCESS);
    auto* keep_alive = new MoEContextKeepAlive{ctx};
    const aclError release_status = aclrtLaunchCallback(
        moe_context_release_callback, keep_alive, ACL_CALLBACK_BLOCK, stream);
    if (release_status != ACL_SUCCESS) {
        // The stream already owns callbacks that reference ctx.  Retaining the
        // holder is safer than releasing the buffers while those callbacks are
        // in flight; this exceptional path intentionally leaks one call context.
        TORCH_CHECK(false, "failed to enqueue MoE context release callback: ",
                    release_status);
    }
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
        .def("export_expert_npu_layout",
             [](const c10::intrusive_ptr<MoEInferHandle>& self, int64_t expert_idx) {
                 return self->impl->export_expert_npu_layout(expert_idx);
             })
        .def("export_expert_npu_layout_out",
             [](const c10::intrusive_ptr<MoEInferHandle>& self, int64_t expert_idx,
                const torch::Tensor& w13, const torch::Tensor& s13,
                const torch::Tensor& w2, const torch::Tensor& s2) {
                 self->impl->export_expert_npu_layout_out(expert_idx, w13, s13, w2, s2);
             })
        .def("export_experts_npu_layout_out",
             [](const c10::intrusive_ptr<MoEInferHandle>& self,
                const torch::Tensor& expert_indices,
                const torch::Tensor& w13, const torch::Tensor& s13,
                const torch::Tensor& w2, const torch::Tensor& s2) {
                 self->impl->export_experts_npu_layout_out(
                     expert_indices, w13, s13, w2, s2);
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
             })
        .def("set_valid_tokens",
             [](const c10::intrusive_ptr<MoEInferHandle>& self, int64_t n) {
                 self->impl->set_valid_tokens(n);
             })
        .def("take_routing_stats",
             [](const c10::intrusive_ptr<MoEInferHandle>& self) {
                 return self->impl->take_routing_stats();
             })
        .def("reset_routing_stats",
             [](const c10::intrusive_ptr<MoEInferHandle>& self) {
                 self->impl->reset_routing_stats();
             });

    // ExpertCacheScheduler::register_layer accepts MoEInferHandle, so the
    // referenced TorchBind class must be registered first.
    m.class_<ExpertCacheScheduler>("ExpertCacheScheduler")
        .def(torch::init<int64_t,int64_t,int64_t,int64_t,double>())
        .def("register_layer", &ExpertCacheScheduler::register_layer)
        .def("plan_out", &ExpertCacheScheduler::plan_out)
        .def("last_stats", &ExpertCacheScheduler::last_stats)
        .def("export_plan_out", &ExpertCacheScheduler::export_plan_out)
        .def("reset_routing_stats", &ExpertCacheScheduler::reset_routing_stats);

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
    m.def("moe_forward_npu_stream_partial(Tensor hidden, Tensor compute_ids, Tensor routing_ids, Tensor topk_w, __torch__.torch.classes.nanovllm.MoEInfer moe) -> Tensor");
    m.def("moe_forward_npu_graph_out(Tensor hidden, Tensor topk_ids, Tensor topk_w, __torch__.torch.classes.nanovllm.MoEInfer moe, __torch__.torch.classes.nanovllm.MoEGraphContext ctx, Tensor(a!) out) -> ()");
    m.def("moe_forward_npu_graph_partial_out(Tensor hidden, Tensor compute_ids, Tensor routing_ids, Tensor topk_w, __torch__.torch.classes.nanovllm.MoEInfer moe, __torch__.torch.classes.nanovllm.MoEGraphContext ctx, Tensor(a!) out) -> ()");
    m.def("moe_forward_npu_graph_partial_start(Tensor hidden, Tensor compute_ids, Tensor routing_ids, Tensor topk_w, __torch__.torch.classes.nanovllm.MoEInfer moe, __torch__.torch.classes.nanovllm.MoEGraphContext ctx) -> ()");
    m.def("moe_forward_npu_graph_partial_wait_out(Tensor(a!) out, __torch__.torch.classes.nanovllm.MoEGraphContext ctx) -> ()");
    m.def("moe_forward_npu_eager_partial_wait_out(Tensor(a!) out, __torch__.torch.classes.nanovllm.MoEGraphContext ctx) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(nanovllm, CPU, m) {
    m.impl("moe_forward", &moe_forward_cpu);
}

#ifdef WITH_NPU
TORCH_LIBRARY_IMPL(nanovllm, PrivateUse1, m) {
    m.impl("moe_forward_npu_stream", &moe_forward_npu_stream);
    m.impl("moe_forward_npu_stream_partial", &moe_forward_npu_stream_partial);
    m.impl("moe_forward_npu_graph_out",
           [](const torch::Tensor& hidden,
              const torch::Tensor& ids,
              const torch::Tensor& w,
              const c10::intrusive_ptr<MoEInferHandle>& moe,
              const c10::intrusive_ptr<MoEGraphContext>& ctx,
              torch::Tensor out) {
               moe_forward_npu_graph_out(hidden, ids, w, moe, ctx, out);
           });
    m.impl("moe_forward_npu_graph_partial_out",
           [](const torch::Tensor& hidden,
              const torch::Tensor& compute_ids,
              const torch::Tensor& routing_ids,
              const torch::Tensor& w,
              const c10::intrusive_ptr<MoEInferHandle>& moe,
              const c10::intrusive_ptr<MoEGraphContext>& ctx,
              torch::Tensor out) {
               moe_forward_npu_graph_partial_out(hidden, compute_ids, routing_ids, w, moe, ctx, out);
           });
    m.impl("moe_forward_npu_graph_partial_start",
           [](const torch::Tensor& hidden,
              const torch::Tensor& compute_ids,
              const torch::Tensor& routing_ids,
              const torch::Tensor& w,
              const c10::intrusive_ptr<MoEInferHandle>& moe,
              const c10::intrusive_ptr<MoEGraphContext>& ctx) {
               moe_forward_npu_graph_partial_start(
                   hidden, compute_ids, routing_ids, w, moe, ctx);
           });
    m.impl("moe_forward_npu_graph_partial_wait_out",
           [](torch::Tensor out,
              const c10::intrusive_ptr<MoEGraphContext>& ctx) {
               moe_forward_npu_graph_partial_wait_out(out, ctx);
           });
    m.impl("moe_forward_npu_eager_partial_wait_out",
           [](torch::Tensor out,
              const c10::intrusive_ptr<MoEGraphContext>& ctx) {
               moe_forward_npu_eager_partial_wait_out(out, ctx);
           });
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "nanovllm_ext (dispatcher registered ops)";
}
