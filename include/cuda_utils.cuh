#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <source_location>

inline void cuda_check(cudaError_t err, std::source_location loc = std::source_location::current()) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s:%u cuda_check: %s\n", loc.file_name(), loc.line(), cudaGetErrorString(err));
        std::exit(1);
    }
}

inline int get_sm_count() {
    static int count = 0;
    if (count == 0) {
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0);
    }
    return count;
}