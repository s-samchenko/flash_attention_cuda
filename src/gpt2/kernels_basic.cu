#include "gpt2_kernels.h"
#include "cuda_utils.cuh"

namespace {

    constexpr int kBlockSize = 256;   // all these ops are memory-bound; tuning buys nothing
    constexpr float c = 0.7978845608028654f;

    inline int grid_for(int n) {
        return (n + kBlockSize - 1) / kBlockSize;
    }

    __global__ void residual_add_kernel(float* x, const float* y, int n) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            x[i] += y[i];
        }
    }

    __global__ void bias_add_kernel(float* x, const float* bias, int rows, int cols) {
        const int n = rows * cols;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            x[i] += bias[i % cols];
        }
    }

    __global__ void gelu_new_kernel(float* x, int n) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            const float v = x[i];
            x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        }
    }

}

namespace gpt2 {

    void residual_add(float* x, const float* y, int n) {
        residual_add_kernel<<<grid_for(n), kBlockSize>>>(x, y, n);
        cuda_check(cudaGetLastError());
    }

    void bias_add(float* x, const float* bias, int rows, int cols) {
        bias_add_kernel<<<grid_for(rows * cols), kBlockSize>>>(x, bias, rows, cols);
        cuda_check(cudaGetLastError());
    }

    void gelu_new(float* x, int n) {
        gelu_new_kernel<<<grid_for(n), kBlockSize>>>(x, n);
        cuda_check(cudaGetLastError());
    }

}
