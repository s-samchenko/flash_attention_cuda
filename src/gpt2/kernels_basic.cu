#include "gpt2_kernels.h"
#include "cuda_utils.cuh"
#include "block_reduce.cuh"

namespace {

    constexpr int kBlockSize = 256;   // all these ops are memory-bound; tuning buys nothing
    constexpr float c = 0.7978845608028654f;

    inline int grid_for(int n) {
        return (n + kBlockSize - 1) / kBlockSize;
    }

    __global__ void layernorm_kernel(const float* x, const float* w, const float* b, float* out, int C) {
        const int row = blockIdx.x;
        const float* xr = x + row * C;
        float* orow = out + row * C;

        // pass 1: mean = (1/C) * sum(x)
        float s = 0.0f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            s += xr[i];
        }
        const float mean = block_reduce_sum(s) / C;

        // pass 2: var = (1/C) * sum((x - mean)^2)
        float vs = 0.0f;
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            const float diff = xr[i] - mean;
            vs += diff * diff;
        }
        const float inv_std = rsqrtf(block_reduce_sum(vs) / C + 1e-5f);

        // normalize
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            orow[i] = (xr[i] - mean) * inv_std * w[i] + b[i];
        }
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
            x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
        }
    }

    __global__ void embedding_gather_kernel(const int* ids, const float* wte, const float* wpe,
                                            float* out, int n, int d) {
        const int i = blockIdx.x;
        const int tok = ids[i];
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            out[i * d + k] = wte[tok * d + k] + wpe[i * d + k];
        }
    }

    __global__ void qkv_split_kernel(const float* qkv, float* Q, float* K, float* V,
                                     int n_real, int n_pad, int h, int d) {
        const int hd = h * d;
        const int row_stride = 3 * hd;
        // rows [n_real, n_pad) are never written and remain zero-init
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_real * hd; idx += gridDim.x * blockDim.x) {
            const int nn = idx / hd;
            const int r  = idx % hd;
            const int hh = r / d;
            const int dd = r % d;
            const int dst = hh * n_pad * d + nn * d + dd;
            const int src = nn * row_stride + r;
            Q[dst] = qkv[src];
            K[dst] = qkv[src + hd];
            V[dst] = qkv[src + 2 * hd];
        }
    }

    __global__ void merge_heads_kernel(const float* O, float* out, int n_real, int n_pad, int h, int d) {
        const int hd = h * d;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_real * hd; idx += gridDim.x * blockDim.x) {
            const int nn = idx / hd;
            const int r = idx % hd;
            const int hh = r / d;
            const int dd = r % d;
            out[nn * hd + r] = O[hh * n_pad * d + nn * d + dd];
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

    void layernorm(const float* x, const float* w, const float* b, float* out, int N, int C) {
        layernorm_kernel<<<N, kBlockSize>>>(x, w, b, out, C);
        cuda_check(cudaGetLastError());
    }

    void embedding_gather(const int* ids, const float* wte, const float* wpe,
                          float* out, int n, int d) {
        embedding_gather_kernel<<<n, kBlockSize>>>(ids, wte, wpe, out, n, d);
        cuda_check(cudaGetLastError());
    }

    void qkv_split(const float* qkv, float* Q, float* K, float* V,
                   int n_real, int n_pad, int h, int d) {
        qkv_split_kernel<<<grid_for(n_real * h * d), kBlockSize>>>(qkv, Q, K, V, n_real, n_pad, h, d);
        cuda_check(cudaGetLastError());
    }

    void merge_heads(const float* O, float* out, int n_real, int n_pad, int h, int d) {
        merge_heads_kernel<<<grid_for(n_real * h * d), kBlockSize>>>(O, out, n_real, n_pad, h, d);
        cuda_check(cudaGetLastError());
    }

}
