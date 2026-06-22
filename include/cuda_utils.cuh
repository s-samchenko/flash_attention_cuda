#pragma once
#include <cuda_runtime.h>

inline int get_sm_count() {
    static int count = 0;
    if (count == 0) {
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0);
    }
    return count;
}