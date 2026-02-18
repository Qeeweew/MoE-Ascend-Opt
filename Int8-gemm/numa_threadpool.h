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

namespace nanovllm {

namespace detail {

// -------------------------
// Futex Helper for Linux
// -------------------------
namespace FutexImpl {
    inline void wait(std::atomic<uint32_t>& atom, uint32_t expected) {
        // FUTEX_WAIT_PRIVATE: Assuming threads share the same memory space (process)
        // Check if value is still 'expected', if so, sleep.
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
    if (errno != 0 || end == s || *end != '\0') {
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
        stop_.store(true, std::memory_order_release);
        // Bump generation to wake workers
        gen_.fetch_add(1, std::memory_order_release);
        // System call wake
        detail::FutexImpl::notify_all(gen_);
        
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    int node() const { return node_; }

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
    bool is_worker_thread() const {
        return tls_pool_ == this;
    }

    template <class F>
    void parallel_for_impl(int64_t begin, int64_t end, Schedule sched, int64_t grain, F&& fn) {
        if (end <= begin) return;

        if (is_worker_thread()) {
            for (int64_t i = begin; i < end; ++i) fn(i);
            return;
        }

        std::function<void(int64_t)> fwrap = std::forward<F>(fn);

        {
            std::lock_guard<std::mutex> lk(mu_dispatch_);
            job_fn_ = std::move(fwrap);
            begin_ = begin;
            end_ = end;
            sched_ = sched;
            grain_ = (sched == Schedule::Dynamic ? std::max<int64_t>(1, grain) : 0);
            next_.store(begin_, std::memory_order_relaxed);
            
            // Set remaining count
            remaining_.store(num_threads_, std::memory_order_release);
            
            // Update generation to wake workers
            gen_.fetch_add(1, std::memory_order_release);
            detail::FutexImpl::notify_all(gen_);

            // Wait until remaining_ becomes 0
            // Futex wait handles the loop
            int rem;
            while ((rem = remaining_.load(std::memory_order_acquire)) > 0) {
                detail::FutexImpl::wait(remaining_, rem);
            }
        }
    }

    void worker_loop(int tid) {
        detail::bind_this_thread_to_node(node_);
        tls_pool_ = this;
        tls_tid_ = tid;

        // Change gen_ to uint32_t for futex compatibility
        uint32_t my_gen = gen_.load(std::memory_order_acquire);

        while (true) {
            // Wait for generation change
            detail::FutexImpl::wait(gen_, my_gen);
            
            if (stop_.load(std::memory_order_acquire)) return;

            uint32_t current_gen = gen_.load(std::memory_order_acquire);
            if (current_gen == my_gen) {
                // Spurious wakeup
                continue;
            }
            my_gen = current_gen;

            // Load job params
            std::function<void(int64_t)> fn_local = job_fn_;
            int64_t begin = begin_;
            int64_t end = end_;
            Schedule sched = sched_;
            int64_t grain = grain_;

            if (sched == Schedule::StaticBlock) {
                const int T = num_threads_;
                const int64_t n = end - begin;
                const int64_t chunk = (n + T - 1) / T;
                const int64_t s = begin + (int64_t)tid * chunk;
                const int64_t e = std::min(end, s + chunk);
                for (int64_t i = s; i < e; ++i) fn_local(i);
            } else {
                while (true) {
                    int64_t s = next_.fetch_add(grain, std::memory_order_relaxed);
                    if (s >= end) break;
                    int64_t e = std::min(end, s + grain);
                    for (int64_t i = s; i < e; ++i) fn_local(i);
                }
            }

            // Decrement remaining count
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                // Last thread wakes up the submitter
                detail::FutexImpl::notify_one(remaining_);
            }
        }
    }

private:
    int node_;
    std::atomic<bool> stop_;
    int num_threads_;

    std::vector<std::thread> workers_;
    std::mutex mu_dispatch_;

    // Use uint32_t for futex compatibility (linux futex is 32-bit)
    std::atomic<uint32_t> gen_{0};
    std::atomic<int> remaining_{0};

    std::function<void(int64_t)> job_fn_;
    int64_t begin_ = 0;
    int64_t end_ = 0;
    Schedule sched_ = Schedule::StaticBlock;
    int64_t grain_ = 0;

    std::atomic<int64_t> next_{0};

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
        stop_.store(true, std::memory_order_release);
        signal_.fetch_add(1, std::memory_order_release);
        detail::FutexImpl::notify_one(signal_);
        
        if (th_.joinable()) th_.join();
    }

    int node() const { return node_; }

    template <class F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto task_ptr = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut = task_ptr->get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.emplace_back([task_ptr]() { (*task_ptr)(); });
        }
        signal_.fetch_add(1, std::memory_order_release);
        detail::FutexImpl::notify_one(signal_);
        return fut;
    }

private:
    void loop() {
        detail::bind_this_thread_to_node(node_);

        uint32_t last_signal = signal_.load(std::memory_order_acquire);

        while (true) {
            Job job;
            bool has_job = false;

            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!q_.empty()) {
                    job = std::move(q_.front());
                    q_.pop_front();
                    has_job = true;
                } else if (stop_.load(std::memory_order_acquire)) {
                    return;
                }
            }

            if (has_job) {
                job();
            } else {
                // Wait for signal change using futex
                detail::FutexImpl::wait(signal_, last_signal);
                last_signal = signal_.load(std::memory_order_acquire);
            }
        }
    }

private:
    int node_;
    std::atomic<bool> stop_;
    std::mutex mu_;
    std::deque<Job> q_;
    
    // uint32_t for futex
    std::atomic<uint32_t> signal_{0};
    
    std::thread th_;
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
