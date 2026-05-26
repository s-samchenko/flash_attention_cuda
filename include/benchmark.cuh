#pragma once
#include <cuda_runtime.h>
#include <functional>

// Returns average ms per run over 'runs' iterations after 'warmup' warmup runs.
inline float benchmark_kernel(std::function<void()> kernel_fn,
                               int warmup = 3,
                               int runs = 10) {
    for (int i = 0; i < warmup; i++) kernel_fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) kernel_fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / runs;
}
