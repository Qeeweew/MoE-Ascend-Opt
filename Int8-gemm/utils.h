#pragma once
#include <ATen/Parallel.h>

enum class ExecutionPolicy { Parallel, Sequential };

template <ExecutionPolicy Policy, typename F>
inline void dispatch_for(int64_t begin, int64_t end, F&& func) {
    if constexpr (Policy == ExecutionPolicy::Parallel) {
        at::parallel_for(begin, end, 0, [&](int64_t s, int64_t e) {
            for (int64_t i = s; i < e; ++i) func(i);
        });
    } else {
        for (int64_t i = begin; i < end; ++i) func(i);
    }
}
