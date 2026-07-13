#include "moe_decode_gemv.h"

#include "gemm_kernels.h"
#include "numa_threadpool.h"

#include <ATen/ATen.h>
#include <numa.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace moe::decode {
namespace {

constexpr int kQk = 32;
constexpr int kMr = 4;
constexpr int kColumnTile = 64;

size_t align64(size_t value) {
    return (value + 63) & ~size_t(63);
}

class Scratch {
public:
    ~Scratch() {
        if (base_) numa_free(base_, bytes_);
    }

    void ensure(int node, int hidden, int intermediate, int routes) {
        const size_t x_qs_bytes = (size_t)kMr * hidden;
        const size_t x_d_bytes =
            (size_t)kMr * (hidden / kQk) * sizeof(float);
        const size_t gate_bytes =
            (size_t)routes * (2 * intermediate) * sizeof(float);
        const size_t down_qs_bytes =
            (size_t)routes * kMr * intermediate;
        const size_t down_d_bytes =
            (size_t)routes * kMr * (intermediate / kQk) * sizeof(float);

        const size_t off_x_qs = 0;
        const size_t off_x_d = align64(off_x_qs + x_qs_bytes);
        const size_t off_gate = align64(off_x_d + x_d_bytes);
        const size_t off_down_qs = align64(off_gate + gate_bytes);
        const size_t off_down_d = align64(off_down_qs + down_qs_bytes);
        const size_t required = align64(off_down_d + down_d_bytes);

        if (required > bytes_ || node != node_) {
            if (base_) numa_free(base_, bytes_);
            base_ = static_cast<char*>(numa_alloc_onnode(required, node));
            if (!base_) throw std::bad_alloc();
            bytes_ = required;
            node_ = node;
        }

        x_qs = reinterpret_cast<int8_t*>(base_ + off_x_qs);
        x_d = reinterpret_cast<float*>(base_ + off_x_d);
        gate = reinterpret_cast<float*>(base_ + off_gate);
        down_qs = reinterpret_cast<int8_t*>(base_ + off_down_qs);
        down_d = reinterpret_cast<float*>(base_ + off_down_d);
    }

    int8_t* x_qs = nullptr;
    float* x_d = nullptr;
    float* gate = nullptr;
    int8_t* down_qs = nullptr;
    float* down_d = nullptr;

private:
    char* base_ = nullptr;
    size_t bytes_ = 0;
    int node_ = -1;
};

thread_local Scratch g_scratch;

struct RouteState {
    std::atomic<int> gate_left{0};
};

struct Scheduler {
    std::atomic<int> next_gate{0};
    std::atomic<int> ready_routes{0};
    std::atomic<int> next_down{0};
    std::unique_ptr<RouteState[]> route_state;

    std::mutex phase_mutex;
    std::condition_variable phase_cv;
    bool down_phase_ready = false;

    int route_capacity = 0;

    void reset(int routes, int gate_tiles) {
        if (routes > route_capacity) {
            route_state = std::make_unique<RouteState[]>((size_t)routes);
            route_capacity = routes;
        }
        next_gate.store(0, std::memory_order_relaxed);
        ready_routes.store(0, std::memory_order_relaxed);
        next_down.store(0, std::memory_order_relaxed);
        down_phase_ready = false;
        for (int route = 0; route < routes; ++route) {
            route_state[(size_t)route].gate_left.store(
                gate_tiles, std::memory_order_relaxed);
        }
    }
};

thread_local Scheduler g_scheduler;

template <typename Scale>
void run_q4_typed(const Q4Args& args) {
    const int routes = args.num_routes;
    const int hidden = args.hidden_size;
    const int intermediate = args.intermediate_size;
    const int hidden_blocks = hidden / kQk;
    const int intermediate_blocks = intermediate / kQk;
    const int gate_tiles = (2 * intermediate + kColumnTile - 1) / kColumnTile;
    const int down_tiles = (hidden + kColumnTile - 1) / kColumnTile;
    const int total_gate = routes * gate_tiles;
    const int total_down = routes * down_tiles;

    Scratch& scratch = g_scratch;
    scratch.ensure(args.node, hidden, intermediate, routes);

    gemm::MoETokenInfo token{0, 0};
    gemm::pack_A_q8_0_from_quantized_indirect(
        1, hidden, args.x_qs, args.x_d, &token, 0,
        scratch.x_qs, scratch.x_d);

    Scheduler& scheduler = g_scheduler;
    scheduler.reset(routes, gate_tiles);

    const auto* gate_scale = static_cast<const Scale*>(args.gate_up_d);
    const auto* down_scale = static_cast<const Scale*>(args.down_d);

    args.pool->execute_per_thread(args.pool->num_threads(), [&](int) {
        auto activate_and_publish = [&](int route) {
            float* gate = scratch.gate +
                (size_t)route * (size_t)(2 * intermediate);
            int8_t* down_a_qs = scratch.down_qs +
                (size_t)route * kMr * (size_t)intermediate;
            float* down_a_d = scratch.down_d +
                (size_t)route * kMr * (size_t)intermediate_blocks;

            gemm::silu_and_mul(gate, 1, 2 * intermediate);
            gemm::pack_A_q8_0(
                1, intermediate, gate, 2 * intermediate,
                down_a_qs, down_a_d);

            if (scheduler.ready_routes.fetch_add(
                    1, std::memory_order_acq_rel) + 1 == routes) {
                {
                    std::lock_guard<std::mutex> lock(scheduler.phase_mutex);
                    scheduler.down_phase_ready = true;
                }
                scheduler.phase_cv.notify_all();
            }
        };

        auto run_gate = [&](int task) {
            // Tile-major order advances every route uniformly.
            const int route = task % routes;
            const int tile = task / routes;
            const int column = tile * kColumnTile;
            const int columns = std::min(kColumnTile, 2 * intermediate - column);
            const int expert = args.routes[route].expert_id;
            const uint32_t* weight = args.gate_up_qs
                + (size_t)expert * (2 * intermediate) * hidden / 8
                + (size_t)column * hidden / 8;
            const Scale* scale = gate_scale
                + (size_t)expert * (2 * intermediate) * hidden_blocks
                + (size_t)column * hidden_blocks;
            float* output = scratch.gate
                + (size_t)route * (2 * intermediate) + column;

            gemm::gemm_q4_0_compute_packed(
                1, columns, hidden,
                scratch.x_qs, scratch.x_d,
                weight, scale, output, 2 * intermediate);

            if (scheduler.route_state[(size_t)route].gate_left.fetch_sub(
                    1, std::memory_order_acq_rel) == 1) {
                activate_and_publish(route);
            }
        };

        auto run_down = [&](int task) {
            const int route = task / down_tiles;
            const int tile = task % down_tiles;
            const int column = tile * kColumnTile;
            const int columns = std::min(kColumnTile, hidden - column);
            const int expert = args.routes[route].expert_id;
            const uint32_t* weight = args.down_qs
                + (size_t)expert * hidden * intermediate / 8
                + (size_t)column * intermediate / 8;
            const Scale* scale = down_scale
                + (size_t)expert * hidden * intermediate_blocks
                + (size_t)column * intermediate_blocks;
            const int8_t* down_a_qs = scratch.down_qs
                + (size_t)route * kMr * (size_t)intermediate;
            const float* down_a_d = scratch.down_d
                + (size_t)route * kMr * (size_t)intermediate_blocks;
            float* output = args.expert_output
                + (size_t)args.routes[route].output_row * hidden + column;

            gemm::gemm_q4_0_compute_packed(
                1, columns, intermediate,
                down_a_qs, down_a_d,
                weight, scale, output, hidden);

        };

        while (true) {
            const int gate_task = scheduler.next_gate.fetch_add(
                1, std::memory_order_relaxed);
            if (gate_task >= total_gate) break;
            run_gate(gate_task);
        }

        {
            std::unique_lock<std::mutex> lock(scheduler.phase_mutex);
            scheduler.phase_cv.wait(lock, [&] {
                return scheduler.down_phase_ready;
            });
        }

        while (true) {
            const int down_task = scheduler.next_down.fetch_add(
                1, std::memory_order_relaxed);
            if (down_task >= total_down) break;
            // Tile-major order balances all experts across the fixed pool.
            const int route = down_task % routes;
            const int tile = down_task / routes;
            run_down(route * down_tiles + tile);
        }
    });
}

} // namespace

void run_q4(const Q4Args& args) {
    if (!args.pool || !args.routes || args.num_routes <= 0) {
        throw std::invalid_argument("invalid batch-1 MoE GEMV arguments");
    }
    if (args.hidden_size % kQk != 0 ||
        args.intermediate_size % kQk != 0) {
        throw std::invalid_argument("MoE GEMV dimensions must be multiples of 32");
    }

    if (args.scale_type == ScaleType::Fp16) {
        run_q4_typed<at::Half>(args);
    } else {
        run_q4_typed<at::BFloat16>(args);
    }
}

} // namespace moe::decode
