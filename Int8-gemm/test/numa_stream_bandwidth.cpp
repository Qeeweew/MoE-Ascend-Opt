#include <numa.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class KernelKind {
    Init,
    Read,
    Read3,
    Copy,
    Triad,
};

struct NodeArrays {
    int node = -1;
    size_t elems = 0;
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
};

struct SharedState {
    KernelKind kernel = KernelKind::Init;
    float triad_scalar = 1.1f;
    std::atomic<bool> stop{false};
    std::vector<double> read_sums;
    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;
};

struct WorkerCtx {
    SharedState* shared = nullptr;
    NodeArrays* arrays = nullptr;
    int global_tid = -1;
    int local_tid = -1;
    int cpu = -1;
    int threads_per_node = 1;
};

std::vector<int> cpus_for_node(int node) {
    std::vector<int> cpus;
    struct bitmask* bm = numa_allocate_cpumask();
    if (!bm) {
        throw std::runtime_error("numa_allocate_cpumask failed");
    }
    if (numa_node_to_cpus(node, bm) != 0) {
        numa_free_cpumask(bm);
        throw std::runtime_error("numa_node_to_cpus failed for node " + std::to_string(node));
    }
    for (int cpu = 0; cpu < static_cast<int>(bm->size); ++cpu) {
        if (numa_bitmask_isbitset(bm, cpu)) {
            cpus.push_back(cpu);
        }
    }
    numa_free_cpumask(bm);
    return cpus;
}

void bind_to_cpu(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        throw std::runtime_error("pthread_setaffinity_np failed for cpu " + std::to_string(cpu));
    }
}

void* worker_main(void* arg) {
    auto* ctx = static_cast<WorkerCtx*>(arg);
    bind_to_cpu(ctx->cpu);

    NodeArrays& arr = *ctx->arrays;
    SharedState& shared = *ctx->shared;

    const size_t chunk = (arr.elems + static_cast<size_t>(ctx->threads_per_node) - 1) /
                         static_cast<size_t>(ctx->threads_per_node);
    const size_t begin = std::min(arr.elems, static_cast<size_t>(ctx->local_tid) * chunk);
    const size_t end = std::min(arr.elems, begin + chunk);

    while (true) {
        pthread_barrier_wait(&shared.start_barrier);
        if (shared.stop.load(std::memory_order_acquire)) {
            break;
        }

        switch (shared.kernel) {
            case KernelKind::Init:
                for (size_t i = begin; i < end; ++i) {
                    arr.a[i] = 1.0f + static_cast<float>((i + ctx->local_tid) & 0xF);
                    arr.b[i] = 2.0f + static_cast<float>((i + ctx->local_tid) & 0x7);
                    arr.c[i] = 0.0f;
                }
                break;
            case KernelKind::Read: {
                double s0 = 0.0;
                double s1 = 0.0;
                double s2 = 0.0;
                double s3 = 0.0;
                size_t i = begin;
                for (; i + 3 < end; i += 4) {
                    s0 += static_cast<double>(arr.a[i + 0]);
                    s1 += static_cast<double>(arr.a[i + 1]);
                    s2 += static_cast<double>(arr.a[i + 2]);
                    s3 += static_cast<double>(arr.a[i + 3]);
                }
                for (; i < end; ++i) {
                    s0 += static_cast<double>(arr.a[i]);
                }
                shared.read_sums[static_cast<size_t>(ctx->global_tid)] = s0 + s1 + s2 + s3;
                break;
            }
            case KernelKind::Read3: {
                double s0 = 0.0;
                double s1 = 0.0;
                double s2 = 0.0;
                double s3 = 0.0;
                size_t i = begin;
                for (; i + 3 < end; i += 4) {
                    s0 += static_cast<double>(arr.a[i + 0]) + static_cast<double>(arr.b[i + 0]) + static_cast<double>(arr.c[i + 0]);
                    s1 += static_cast<double>(arr.a[i + 1]) + static_cast<double>(arr.b[i + 1]) + static_cast<double>(arr.c[i + 1]);
                    s2 += static_cast<double>(arr.a[i + 2]) + static_cast<double>(arr.b[i + 2]) + static_cast<double>(arr.c[i + 2]);
                    s3 += static_cast<double>(arr.a[i + 3]) + static_cast<double>(arr.b[i + 3]) + static_cast<double>(arr.c[i + 3]);
                }
                for (; i < end; ++i) {
                    s0 += static_cast<double>(arr.a[i]) + static_cast<double>(arr.b[i]) + static_cast<double>(arr.c[i]);
                }
                shared.read_sums[static_cast<size_t>(ctx->global_tid)] = s0 + s1 + s2 + s3;
                break;
            }
            case KernelKind::Copy:
                for (size_t i = begin; i < end; ++i) {
                    arr.c[i] = arr.a[i];
                }
                break;
            case KernelKind::Triad:
                for (size_t i = begin; i < end; ++i) {
                    arr.c[i] = arr.a[i] + shared.triad_scalar * arr.b[i];
                }
                break;
        }

        pthread_barrier_wait(&shared.end_barrier);
    }

    return nullptr;
}

double run_kernel(SharedState& shared, KernelKind kernel, int warmup, int repeat) {
    shared.kernel = kernel;
    for (int i = 0; i < warmup; ++i) {
        pthread_barrier_wait(&shared.start_barrier);
        pthread_barrier_wait(&shared.end_barrier);
    }

    double best_sec = 1e30;
    for (int i = 0; i < repeat; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        pthread_barrier_wait(&shared.start_barrier);
        pthread_barrier_wait(&shared.end_barrier);
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        best_sec = std::min(best_sec, sec);
    }
    return best_sec;
}

std::string kernel_name(KernelKind kernel) {
    switch (kernel) {
        case KernelKind::Init: return "init";
        case KernelKind::Read: return "read";
        case KernelKind::Read3: return "read3";
        case KernelKind::Copy: return "copy";
        case KernelKind::Triad: return "triad";
    }
    return "unknown";
}

double bytes_per_kernel(KernelKind kernel, size_t elems_total) {
    const double bytes = static_cast<double>(elems_total) * sizeof(float);
    switch (kernel) {
        case KernelKind::Read:
            return bytes;
        case KernelKind::Read3:
            return 3.0 * bytes;
        case KernelKind::Copy:
            return 2.0 * bytes;
        case KernelKind::Triad:
            return 3.0 * bytes;
        case KernelKind::Init:
            return 3.0 * bytes;
    }
    return 0.0;
}

void free_node_arrays(NodeArrays& arr) {
    const size_t bytes = arr.elems * sizeof(float);
    if (arr.a) numa_free(arr.a, bytes);
    if (arr.b) numa_free(arr.b, bytes);
    if (arr.c) numa_free(arr.c, bytes);
    arr.a = nullptr;
    arr.b = nullptr;
    arr.c = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    if (numa_available() < 0) {
        std::cerr << "libnuma is not available\n";
        return 1;
    }

    const int threads_per_node = (argc > 1) ? std::stoi(argv[1]) : 20;
    const size_t mib_per_array = (argc > 2) ? static_cast<size_t>(std::stoull(argv[2])) : 256;
    const int warmup = (argc > 3) ? std::stoi(argv[3]) : 2;
    const int repeat = (argc > 4) ? std::stoi(argv[4]) : 5;

    const int max_nodes = numa_max_node() + 1;
    const std::vector<int> test_node_counts = {1, 2, 4, 8};
    const size_t bytes_per_array = mib_per_array * 1024ull * 1024ull;
    const size_t elems_per_array = bytes_per_array / sizeof(float);

    std::cout << "NUMA STREAM-like bandwidth benchmark\n";
    std::cout << "threads_per_node=" << threads_per_node
              << ", array_size_per_node=" << mib_per_array << " MiB"
              << ", warmup=" << warmup
              << ", repeat=" << repeat << "\n\n";

    std::cout << std::left << std::setw(8) << "Nodes"
              << std::setw(8) << "Threads"
              << std::setw(14) << "Kernel"
              << std::setw(14) << "Best(ms)"
              << std::setw(16) << "BW(GB/s)"
              << std::setw(18) << "Bytes(GB)"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    volatile double sink = 0.0;

    for (int node_count : test_node_counts) {
        if (node_count > max_nodes) {
            continue;
        }

        std::vector<NodeArrays> nodes(static_cast<size_t>(node_count));
        for (int i = 0; i < node_count; ++i) {
            nodes[static_cast<size_t>(i)].node = i;
            nodes[static_cast<size_t>(i)].elems = elems_per_array;
            nodes[static_cast<size_t>(i)].a =
                static_cast<float*>(numa_alloc_onnode(bytes_per_array, i));
            nodes[static_cast<size_t>(i)].b =
                static_cast<float*>(numa_alloc_onnode(bytes_per_array, i));
            nodes[static_cast<size_t>(i)].c =
                static_cast<float*>(numa_alloc_onnode(bytes_per_array, i));
            if (!nodes[static_cast<size_t>(i)].a ||
                !nodes[static_cast<size_t>(i)].b ||
                !nodes[static_cast<size_t>(i)].c) {
                throw std::runtime_error("numa_alloc_onnode failed for node " + std::to_string(i));
            }
        }

        const int total_threads = node_count * threads_per_node;
        SharedState shared;
        shared.read_sums.resize(static_cast<size_t>(total_threads), 0.0);
        pthread_barrier_init(&shared.start_barrier, nullptr, static_cast<unsigned>(total_threads + 1));
        pthread_barrier_init(&shared.end_barrier, nullptr, static_cast<unsigned>(total_threads + 1));

        std::vector<WorkerCtx> worker_ctxs(static_cast<size_t>(total_threads));
        std::vector<pthread_t> workers(static_cast<size_t>(total_threads));

        int global_tid = 0;
        for (int node_idx = 0; node_idx < node_count; ++node_idx) {
            const std::vector<int> cpus = cpus_for_node(node_idx);
            if (static_cast<int>(cpus.size()) < threads_per_node) {
                throw std::runtime_error(
                    "node " + std::to_string(node_idx) + " has only " +
                    std::to_string(cpus.size()) + " CPUs, less than requested threads_per_node");
            }
            for (int local_tid = 0; local_tid < threads_per_node; ++local_tid, ++global_tid) {
                WorkerCtx& ctx = worker_ctxs[static_cast<size_t>(global_tid)];
                ctx.shared = &shared;
                ctx.arrays = &nodes[static_cast<size_t>(node_idx)];
                ctx.global_tid = global_tid;
                ctx.local_tid = local_tid;
                ctx.cpu = cpus[static_cast<size_t>(local_tid)];
                ctx.threads_per_node = threads_per_node;
                if (pthread_create(&workers[static_cast<size_t>(global_tid)], nullptr, worker_main, &ctx) != 0) {
                    throw std::runtime_error("pthread_create failed");
                }
            }
        }

        run_kernel(shared, KernelKind::Init, 0, 1);

        const size_t elems_total = static_cast<size_t>(node_count) * elems_per_array;
        for (KernelKind kernel : {KernelKind::Read, KernelKind::Read3, KernelKind::Copy, KernelKind::Triad}) {
            const double best_sec = run_kernel(shared, kernel, warmup, repeat);
            const double bytes = bytes_per_kernel(kernel, elems_total);
            const double bw_gbs = bytes / best_sec / 1e9;
            std::cout << std::left << std::setw(8) << node_count
                      << std::setw(8) << total_threads
                      << std::setw(14) << kernel_name(kernel)
                      << std::setw(14) << std::fixed << std::setprecision(3) << best_sec * 1e3
                      << std::setw(16) << std::fixed << std::setprecision(2) << bw_gbs
                      << std::setw(18) << std::fixed << std::setprecision(2) << (bytes / 1e9)
                      << "\n";
            if (kernel == KernelKind::Read || kernel == KernelKind::Read3) {
                sink += std::accumulate(shared.read_sums.begin(), shared.read_sums.end(), 0.0);
            }
        }

        shared.stop.store(true, std::memory_order_release);
        pthread_barrier_wait(&shared.start_barrier);
        for (pthread_t& th : workers) {
            pthread_join(th, nullptr);
        }
        pthread_barrier_destroy(&shared.start_barrier);
        pthread_barrier_destroy(&shared.end_barrier);

        for (NodeArrays& arr : nodes) {
            free_node_arrays(arr);
        }
    }

    if (sink == 0.123456789) {
        std::cerr << "ignore " << sink << "\n";
    }
    return 0;
}
