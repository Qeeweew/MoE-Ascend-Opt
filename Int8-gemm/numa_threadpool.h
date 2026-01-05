#pragma once

#include <numa.h>
#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstring> // for memset

namespace nanovllm {

namespace detail {

inline void bind_this_thread_to_node(int node) {
    // Prefer local memory allocations for this thread
    (void)numa_run_on_node(node);

    // Strong affinity to CPUs belonging to this NUMA node
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    struct bitmask* bm = numa_allocate_cpumask();
    numa_node_to_cpus(node, bm);
    for (int cpu = 0; cpu < (int)bm->size; ++cpu) {
        if (numa_bitmask_isbitset(bm, cpu)) CPU_SET(cpu, &cpuset);
    }
    numa_free_cpumask(bm);

    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

inline int64_t read_env_int64(const char* name, int64_t def) {
    const char* s = std::getenv(name);
    if (!s || !*s) return def;

    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        // Fallback or throw, here we just return def on error to be safe
        return def;
    }
    return (int64_t)v;
}

} // namespace detail

// -------------------------
// NumaThreadPool (worker pool)
// -------------------------
class NumaThreadPool {
public:
    enum class Schedule { StaticBlock, Dynamic };

    NumaThreadPool(int node, int num_threads)
        : node_(node), stop_(false), num_threads_(std::max(1, num_threads)) {
        workers_.reserve((size_t)num_threads_);
        for (int tid = 0; tid < num_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~NumaThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            // bump generation to wake workers
            gen_++;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    int node() const { return node_; }

    // 默认：静态划分（最快）
    template <class F>
    void parallel_for(int64_t begin, int64_t end, F&& fn) {
        parallel_for_impl(begin, end, Schedule::Dynamic, /*grain*/1, std::forward<F>(fn));
    }

    template <class F>
    void parallel_for_static(int64_t begin, int64_t end, F&& fn) {
        parallel_for_impl(begin, end, Schedule::StaticBlock, /*grain*/0, std::forward<F>(fn));
    }

    template <class F>
    void parallel_for_dynamic(int64_t begin, int64_t end, int64_t grain, F&& fn) {
        parallel_for_impl(begin, end, Schedule::Dynamic, grain, std::forward<F>(fn));
    }

    static int cpu_count_in_node(int node) {
        struct bitmask* bm = numa_allocate_cpumask();
        numa_node_to_cpus(node, bm);
        int cnt = 0;
        for (int cpu = 0; cpu < (int)bm->size; ++cpu) {
            if (numa_bitmask_isbitset(bm, cpu)) cnt++;
        }
        numa_free_cpumask(bm);
        return cnt > 0 ? cnt : 1;
    }

private:
    // 判断当前线程是否本 pool 的 worker（避免嵌套死锁）
    bool is_worker_thread() const {
        return tls_pool_ == this;
    }

    template <class F>
    void parallel_for_impl(int64_t begin, int64_t end, Schedule sched, int64_t grain, F&& fn) {
        if (end <= begin) return;

        // 避免 worker 内嵌套调用导致死锁：直接串行跑
        if (is_worker_thread()) {
            for (int64_t i = begin; i < end; ++i) fn(i);
            return;
        }

        // 把任意 lambda 变成 std::function（一次赋值，job 期间保持有效）
        std::function<void(int64_t)> fwrap = std::forward<F>(fn);

        {
            std::lock_guard<std::mutex> lk(mu_);
            job_fn_ = std::move(fwrap);
            begin_ = begin;
            end_ = end;
            sched_ = sched;
            grain_ = (sched == Schedule::Dynamic ? std::max<int64_t>(1, grain) : 0);
            next_.store(begin_, std::memory_order_relaxed);
            remaining_.store(num_threads_, std::memory_order_release);
            gen_++;
        }

        cv_work_.notify_all();

        // barrier wait
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&]() { return remaining_.load(std::memory_order_acquire) == 0; });
    }

    void worker_loop(int tid) {
        detail::bind_this_thread_to_node(node_);
        tls_pool_ = this;
        tls_tid_ = tid;

        uint64_t my_gen = 0;
        while (true) {
            std::function<void(int64_t)> fn_local;
            int64_t begin, end, grain;
            Schedule sched;

            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_work_.wait(lk, [&]() { return stop_ || gen_ != my_gen; });
                if (stop_) return;

                my_gen = gen_;
                // 拷贝/引用 job 描述（fn 用拷贝，保证 job 中安全）
                fn_local = job_fn_;
                begin = begin_;
                end = end_;
                sched = sched_;
                grain = grain_;
            }

            if (sched == Schedule::StaticBlock) {
                // static block: tid 固定区间
                const int T = num_threads_;
                const int64_t n = end - begin;
                const int64_t chunk = (n + T - 1) / T;
                const int64_t s = begin + (int64_t)tid * chunk;
                const int64_t e = std::min(end, s + chunk);
                for (int64_t i = s; i < e; ++i) fn_local(i);
            } else {
                // dynamic: 分块抢任务，但不 enqueue
                while (true) {
                    int64_t s = next_.fetch_add(grain, std::memory_order_relaxed);
                    if (s >= end) break;
                    int64_t e = std::min(end, s + grain);
                    for (int64_t i = s; i < e; ++i) fn_local(i);
                }
            }

            // worker done
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(mu_);
                cv_done_.notify_one();
            }
        }
    }

private:
    int node_;
    std::atomic<bool> stop_;
    int num_threads_;

    std::vector<std::thread> workers_;

    // job shared state
    mutable std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;

    uint64_t gen_ = 0;

    std::function<void(int64_t)> job_fn_;
    int64_t begin_ = 0;
    int64_t end_ = 0;
    Schedule sched_ = Schedule::StaticBlock;
    int64_t grain_ = 0;

    std::atomic<int64_t> next_{0};
    std::atomic<int> remaining_{0};

    // TLS for nested detection
    static inline thread_local NumaThreadPool* tls_pool_ = nullptr;
    static inline thread_local int tls_tid_ = -1;
};

// -------------------------
// NumaLauncher
// -------------------------
class NumaLauncher {
public:
    using Job = std::function<void()>;

    explicit NumaLauncher(int node) : node_(node), stop_(false) {
        th_ = std::thread([this]() { loop(); });
    }

    ~NumaLauncher() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

    int node() const { return node_; }

    template <class F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        
        // FIX: Use shared_ptr to wrap packaged_task.
        // std::packaged_task is MoveOnly, but std::function requires CopyConstructible.
        // std::shared_ptr is CopyConstructible.
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        
        auto fut = task_ptr->get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            // Lambda captures shared_ptr by value (copy), which is valid for std::function
            q_.emplace_back([task_ptr]() { (*task_ptr)(); });
        }
        cv_.notify_one();
        return fut;
    }

private:
    void loop() {
        detail::bind_this_thread_to_node(node_);

        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                job = std::move(q_.front());
                q_.pop_front();
            }
            job();
        }
    }

private:
    int node_;
    bool stop_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> q_;
    std::thread th_;
};

// -------------------------
// Combined executor per node
// -------------------------
struct NumaNodeExecutor {
    std::shared_ptr<NumaThreadPool> pool;
    std::shared_ptr<NumaLauncher> launcher;
};

// -------------------------
// Global manager
// -------------------------
class NumaExecutorManager {
public:
    static std::shared_ptr<NumaNodeExecutor> get(int node) {
        if (numa_available() == -1) throw std::runtime_error("libnuma not available");

        std::lock_guard<std::mutex> lk(mu());
        auto& mp = execs();

        auto it = mp.find(node);
        if (it != mp.end()) return it->second;   // 强引用直接返回

        int threads = threads_per_node_from_env(node);
        auto exec = std::make_shared<NumaNodeExecutor>();
        exec->pool = std::make_shared<NumaThreadPool>(node, threads);
        exec->launcher = std::make_shared<NumaLauncher>(node);

        mp[node] = exec; // 强引用保活
        return exec;
    }

private:
    static int threads_per_node_from_env(int node) {
        const int cpus = NumaThreadPool::cpu_count_in_node(node);
        int64_t v = detail::read_env_int64("NANOVLLM_TP_THREADS_PER_NODE", 0);
        if (v <= 0) return cpus;
        if (v > cpus) v = cpus;
        return (int)v;
    }

    static std::mutex& mu() {
        static std::mutex m;
        return m;
    }

    static std::unordered_map<int, std::shared_ptr<NumaNodeExecutor>>& execs() {
        static std::unordered_map<int, std::shared_ptr<NumaNodeExecutor>> mp;
        return mp;
    }
};

} // namespace nanovllm
