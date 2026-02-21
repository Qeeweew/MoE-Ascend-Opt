#pragma once

#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sys/time.h>

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
#include <cstring>
#include <climits>
#include <new>

namespace nanovllm {

namespace detail {

// -------------------------
// Futex Helper for Linux
// -------------------------
namespace FutexImpl {
    inline void wait(std::atomic<uint32_t>& atom, uint32_t expected) {
        while (atom.load(std::memory_order_acquire) == expected) {
            syscall(SYS_futex, &atom, FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
        }
    }

    inline void wait(std::atomic<int>& atom, int expected) {
        while (atom.load(std::memory_order_acquire) == expected) {
            syscall(SYS_futex, &atom, FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
        }
    }

    inline void notify_one(std::atomic<uint32_t>& atom) {
        syscall(SYS_futex, &atom, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
    }
    
    inline void notify_one(std::atomic<int>& atom) {
        syscall(SYS_futex, &atom, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
    }

    inline void notify_all(std::atomic<uint32_t>& atom) {
        syscall(SYS_futex, &atom, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
    }
}

inline void bind_this_thread_to_node(int node) {
    (void)numa_run_on_node(node);
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
    if (errno != 0 || end == s || *end != '\0') return def;
    return (int64_t)v;
}

} // namespace detail

// -------------------------
// NumaThreadPool
// -------------------------
class NumaThreadPool {
public:
    // Unified execution mode: all modes are mutually exclusive
    enum class ExecutionMode {
        StaticBlock,  // Static block scheduling (parallel_for)
        Dynamic,      // Dynamic scheduling (parallel_for with grain)
        PerThread     // Direct thread-id based execution (execute_per_thread)
    };

    // 定义缓存行大小，通常为64字节
    static constexpr size_t CACHE_LINE_SIZE = 64;

    // 核心控制块：分配在目标 NUMA 节点上
    struct alignas(CACHE_LINE_SIZE) ControlBlock {
        // 使用 alignas 强制每个热点变量独占 Cache Line，避免 False Sharing

        // 1. Worker 等待的信号
        alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> gen_{0};

        // 2. 主线程等待的计数器
        alignas(CACHE_LINE_SIZE) std::atomic<int> remaining_{0};

        // 3. 动态调度时的任务游标 (Worker 间高频竞争)
        alignas(CACHE_LINE_SIZE) std::atomic<int64_t> next_{0};

        // 4. 停止信号
        alignas(CACHE_LINE_SIZE) std::atomic<bool> stop_{false};

        // 5. 任务参数 (主要是读，竞争少，但也隔离一下)
        alignas(CACHE_LINE_SIZE) std::function<void(int64_t)> job_fn_;
        int64_t begin_ = 0;
        int64_t end_ = 0;
        int64_t grain_ = 0;
        ExecutionMode mode_ = ExecutionMode::StaticBlock;

        // 6. Per-thread execution function
        alignas(CACHE_LINE_SIZE) std::function<void(int)> job_fn_per_thread_;
        int active_threads_ = 0;  // Number of threads participating in execute_per_thread

        // 7. Barrier support
        alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> barrier_phase_{0};
        alignas(CACHE_LINE_SIZE) std::atomic<int> barrier_count_{0};
    };

    NumaThreadPool(int node, int num_threads)
        : node_(node), num_threads_(std::max(1, num_threads)) {
        
        // 关键修改：在目标 NUMA 节点上分配控制块内存
        // 这样 Worker 线程访问这些原子变量时，是访问的本地内存 (Local Access)
        void* raw_mem = numa_alloc_onnode(sizeof(ControlBlock), node_);
        if (!raw_mem) {
            throw std::runtime_error("Failed to allocate memory on NUMA node " + std::to_string(node_));
        }
        // 使用 Placement New 初始化对象
        cb_ = new (raw_mem) ControlBlock();

        workers_.reserve((size_t)num_threads_);
        for (int tid = 0; tid < num_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~NumaThreadPool() {
        cb_->stop_.store(true, std::memory_order_release);
        cb_->gen_.fetch_add(1, std::memory_order_release);
        detail::FutexImpl::notify_all(cb_->gen_);
        
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }

        // 手动析构并释放 NUMA 内存
        cb_->~ControlBlock();
        numa_free(cb_, sizeof(ControlBlock));
    }

    int node() const { return node_; }
    int num_threads() const { return num_threads_; }

    template <class F>
    void parallel_for(int64_t begin, int64_t end, F&& fn) {
        parallel_for_impl(begin, end, ExecutionMode::Dynamic, 1, std::forward<F>(fn));
    }

    template <class F>
    void parallel_for_static(int64_t begin, int64_t end, F&& fn) {
        parallel_for_impl(begin, end, ExecutionMode::StaticBlock, 0, std::forward<F>(fn));
    }

    template <class F>
    void parallel_for_dynamic(int64_t begin, int64_t end, int64_t grain, F&& fn) {
        parallel_for_impl(begin, end, ExecutionMode::Dynamic, grain, std::forward<F>(fn));
    }

    template <class F>
    void execute_per_thread(int num_threads, F&& fn) {
        if (num_threads <= 0 || num_threads > num_threads_) {
            throw std::runtime_error("Invalid num_threads: " + std::to_string(num_threads));
        }

        std::function<void(int)> fwrap = std::forward<F>(fn);

        {
            std::lock_guard<std::mutex> lk(mu_dispatch_);

            cb_->job_fn_per_thread_ = std::move(fwrap);
            cb_->active_threads_ = num_threads;
            cb_->mode_ = ExecutionMode::PerThread;
            cb_->barrier_phase_.store(0, std::memory_order_release);
            cb_->barrier_count_.store(0, std::memory_order_release);

            // All threads participate in the completion notification
            cb_->remaining_.store(num_threads_, std::memory_order_release);

            cb_->gen_.fetch_add(1, std::memory_order_release);
            detail::FutexImpl::notify_all(cb_->gen_);

            int rem;
            while ((rem = cb_->remaining_.load(std::memory_order_acquire)) > 0) {
                detail::FutexImpl::wait(cb_->remaining_, rem);
            }

            // Reset mode back to StaticBlock for subsequent calls
            cb_->mode_ = ExecutionMode::StaticBlock;
        }
    }

    // Simple barrier for synchronization within execute_per_thread
    void barrier() {
        int tid = tls_tid_;
        if (tid < 0) return;  // Not a worker thread

        // Check if this thread should participate in barrier
        if (tid >= cb_->active_threads_) return;

        uint32_t expected_phase = cb_->barrier_phase_.load(std::memory_order_acquire);
        int count = cb_->barrier_count_.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (count == cb_->active_threads_) {
            // Last thread resets the counter and advances phase
            cb_->barrier_count_.store(0, std::memory_order_release);
            cb_->barrier_phase_.fetch_add(1, std::memory_order_release);
            detail::FutexImpl::notify_all(cb_->barrier_phase_);
        } else {
            // Wait for phase to change
            while (cb_->barrier_phase_.load(std::memory_order_acquire) == expected_phase) {
                detail::FutexImpl::wait(cb_->barrier_phase_, expected_phase);
            }
        }
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
    bool is_worker_thread() const {
        return tls_pool_ == this;
    }

    template <class F>
    void parallel_for_impl(int64_t begin, int64_t end, ExecutionMode mode, int64_t grain, F&& fn) {
        if (end <= begin) return;

        if (end - begin == 1) {
            fn(begin);
            return;
        }

        if (is_worker_thread()) {
            for (int64_t i = begin; i < end; ++i) fn(i);
            return;
        }

        std::function<void(int64_t)> fwrap = std::forward<F>(fn);

        {
            // 此锁保护的是任务分发逻辑，只需简单的互斥
            std::lock_guard<std::mutex> lk(mu_dispatch_);

            // 写入位于目标 NUMA 节点的控制块
            cb_->job_fn_ = std::move(fwrap);
            cb_->begin_ = begin;
            cb_->end_ = end;
            cb_->grain_ = (mode == ExecutionMode::Dynamic ? std::max<int64_t>(1, grain) : 0);
            cb_->next_.store(begin, std::memory_order_relaxed);
            cb_->mode_ = mode;

            cb_->remaining_.store(num_threads_, std::memory_order_release);

            cb_->gen_.fetch_add(1, std::memory_order_release);
            detail::FutexImpl::notify_all(cb_->gen_);

            int rem;
            while ((rem = cb_->remaining_.load(std::memory_order_acquire)) > 0) {
                detail::FutexImpl::wait(cb_->remaining_, rem);
            }
        }
    }

    void worker_loop(int tid) {
        detail::bind_this_thread_to_node(node_);
        tls_pool_ = this;
        tls_tid_ = tid;

        uint32_t my_gen = cb_->gen_.load(std::memory_order_acquire);

        while (true) {
            detail::FutexImpl::wait(cb_->gen_, my_gen);
            
            if (cb_->stop_.load(std::memory_order_acquire)) return;

            uint32_t current_gen = cb_->gen_.load(std::memory_order_acquire);
            if (current_gen == my_gen) continue;
            my_gen = current_gen;

            // 拷贝参数到本地栈
            auto fn_local = cb_->job_fn_;
            int64_t begin = cb_->begin_;
            int64_t end = cb_->end_;
            int64_t grain = cb_->grain_;
            ExecutionMode mode = cb_->mode_;

            // Special handling for PerThread mode: only active threads participate
            if (mode == ExecutionMode::PerThread) {
                if (tid < cb_->active_threads_) {
                    // This thread is active, execute the function
                    auto fn_per_thread_local = cb_->job_fn_per_thread_;
                    fn_per_thread_local(tid);
                    if (cb_->remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        detail::FutexImpl::notify_one(cb_->remaining_);
                    }
                } else {
                    // This thread is not active, skip execution but decrement remaining
                    if (cb_->remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        detail::FutexImpl::notify_one(cb_->remaining_);
                    }
                }
                continue;
            }

            if (mode == ExecutionMode::StaticBlock) {
                const int T = num_threads_;
                const int64_t n = end - begin;
                const int64_t chunk = (n + T - 1) / T;
                const int64_t s = begin + (int64_t)tid * chunk;
                const int64_t e = std::min(end, s + chunk);
                for (int64_t i = s; i < e; ++i) fn_local(i);
            } else {  // ExecutionMode::Dynamic
                while (true) {
                    // 访问位于本地 NUMA 内存的原子变量，即使高频争抢也比跨节点快得多
                    int64_t s = cb_->next_.fetch_add(grain, std::memory_order_relaxed);
                    if (s >= end) break;
                    int64_t e = std::min(end, s + grain);
                    for (int64_t i = s; i < e; ++i) fn_local(i);
                }
            }

            if (cb_->remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                detail::FutexImpl::notify_one(cb_->remaining_);
            }
        }
    }

private:
    int node_;
    int num_threads_;
    std::vector<std::thread> workers_;
    std::mutex mu_dispatch_; // 本地锁，无需放到远端

    // 指向远端（或本地）NUMA 内存的指针
    ControlBlock* cb_ = nullptr;

    static inline thread_local NumaThreadPool* tls_pool_ = nullptr;
    static inline thread_local int tls_tid_ = -1;
};

// -------------------------
// NumaLauncher
// -------------------------
class NumaLauncher {
public:
    using Job = std::function<void()>;
    static constexpr size_t CACHE_LINE_SIZE = 64;

    // 同样将 Launcher 的热点同步变量放在目标节点
    struct alignas(CACHE_LINE_SIZE) LaunchState {
        alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> signal_{0};
        alignas(CACHE_LINE_SIZE) std::atomic<bool> stop_{false};
    };

    explicit NumaLauncher(int node) : node_(node) {
        // 在目标节点分配同步状态
        void* raw = numa_alloc_onnode(sizeof(LaunchState), node_);
        if (!raw) throw std::runtime_error("NumaLauncher alloc failed");
        state_ = new (raw) LaunchState();

        th_ = std::thread([this]() { loop(); });
    }

    ~NumaLauncher() {
        state_->stop_.store(true, std::memory_order_release);
        state_->signal_.fetch_add(1, std::memory_order_release);
        detail::FutexImpl::notify_one(state_->signal_);
        
        if (th_.joinable()) th_.join();

        state_->~LaunchState();
        numa_free(state_, sizeof(LaunchState));
    }

    int node() const { return node_; }

    template <class F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut = task_ptr->get_future();
        {
            // 锁保护队列，队列在内存中可能不在本地，但这不可避免
            // 因为这是 Producer-Consumer 模型
            std::lock_guard<std::mutex> lk(mu_);
            q_.emplace_back([task_ptr]() { (*task_ptr)(); });
        }
        // 通知信号位于消费者所在的 NUMA 节点
        state_->signal_.fetch_add(1, std::memory_order_release);
        detail::FutexImpl::notify_one(state_->signal_);
        return fut;
    }

private:
    void loop() {
        detail::bind_this_thread_to_node(node_);

        uint32_t last_signal = state_->signal_.load(std::memory_order_acquire);

        while (true) {
            Job job;
            bool has_job = false;

            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!q_.empty()) {
                    job = std::move(q_.front());
                    q_.pop_front();
                    has_job = true;
                } else if (state_->stop_.load(std::memory_order_acquire)) {
                    return;
                }
            }

            if (has_job) {
                job();
            } else {
                detail::FutexImpl::wait(state_->signal_, last_signal);
                last_signal = state_->signal_.load(std::memory_order_acquire);
            }
        }
    }

private:
    int node_;
    std::mutex mu_;
    std::deque<Job> q_;
    std::thread th_;

    LaunchState* state_ = nullptr;
};

// -------------------------
// Combined executor per node
// -------------------------
struct NumaNodeExecutor {
    std::shared_ptr<NumaThreadPool> pool;
    std::shared_ptr<NumaLauncher> launcher;
};

class NumaExecutorManager {
public:
    static std::shared_ptr<NumaNodeExecutor> get(int node) {
        if (numa_available() == -1) throw std::runtime_error("libnuma not available");

        std::lock_guard<std::mutex> lk(mu());
        auto& mp = execs();

        auto it = mp.find(node);
        if (it != mp.end()) return it->second;

        int threads = threads_per_node_from_env(node);
        auto exec = std::make_shared<NumaNodeExecutor>();
        exec->pool = std::make_shared<NumaThreadPool>(node, threads);
        exec->launcher = std::make_shared<NumaLauncher>(node);

        mp[node] = exec;
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
