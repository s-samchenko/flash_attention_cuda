#pragma once
#include <cuda_runtime.h>

__device__ inline float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }

    if (lane == 0) {
        warp_sums[warp] = val;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
    }

    __shared__ float total;
    if (threadIdx.x == 0) {
        total = val;
    }
    __syncthreads();
    return total;
}
