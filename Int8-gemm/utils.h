#pragma once
#include <omp.h>

enum class ExecutionPolicy { Parallel, Sequential };

template <ExecutionPolicy Policy, typename F>
inline void dispatch_for(int64_t begin, int64_t end, F&& func) {
    if constexpr (Policy == ExecutionPolicy::Parallel) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = begin; i < end; ++i) {
            func(i);
        }
    } else {
        for (int64_t i = begin; i < end; ++i) {
            func(i);
        }
    }
}
