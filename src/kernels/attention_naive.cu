#include "attention.h"
#include <float.h>
#include <math.h>

//naive attention using 3 separate kernels
// QK^T -> softmax -> AV

#define TILE 32

__global__ void qk_matmul_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int N,
    int D,
    float scale)
{
    int b   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    const float* Qb = Q + (long)b * N * D;
    const float* Kb = K + (long)b * N * D;
    float* Sb = S + (long)b * N * N;

    float acc = 0.0f;
    for (int k = 0; k < D; k++)
        acc += Qb[row * D + k] * Kb[col * D + k];
    Sb[row * N + col] = acc * scale;
}

__global__ void softmax_kernel(float* __restrict__ S, int N)
{
    int b = blockIdx.y;
    int row = blockIdx.x;
    float* row_ptr = S + (long)b * N * N + row * N;

    __shared__ float smem[1024];

    // pass 1: find row max
    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        mx = fmaxf(mx, row_ptr[j]);
    }
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    mx = smem[0];
    __syncthreads();

    // pass 2: exp(x - max) and accumulate sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float e = expf(row_ptr[j] - mx);
        row_ptr[j] = e;
        sum += e;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // pass 3: normalize
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        row_ptr[j] /= sum;
}

__global__ void av_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int D)
{
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= D) return;

    const float* Ab = A + (long)b * N * N;
    const float* Vb = V + (long)b * N * D;
    float* Ob = O + (long)b * N * D;

    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += Ab[row * N + k] * Vb[k * D + col];
    Ob[row * D + col] = acc;
}

static int softmax_threads(int N)
{
    int t = 1;
    while (t * 2 <= N && t * 2 <= 1024) t *= 2;
    return t;
}

void attention_naive(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* S_buf,
    const AttentionParams& p)
{
    int B = p.batch * p.n_heads;
    int N = p.seq_len;
    int D = p.head_dim;

    // QK^T
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE-1)/TILE, (N + TILE-1)/TILE, B);
        qk_matmul_kernel<<<grid, block>>>(Q, K, S_buf, N, D, p.scale);
    }

    // softmax
    {
        int threads = softmax_threads(N);
        softmax_kernel<<<dim3(N, B), dim3(threads)>>>(S_buf, N);
    }

    // AV
    {
        dim3 block(TILE, TILE);
        dim3 grid((D + TILE-1)/TILE, (N + TILE-1)/TILE, B);
        av_matmul_kernel<<<grid, block>>>(S_buf, V, O, N, D);
    }
}
