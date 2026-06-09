#include "attention.h"
#include <float.h>
#include <math.h>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int nthreads = (BM / TM) * (BN / TN); // 256

// QK^T
// As[k][m] = Q[b, row+m, j+k]   (Q loaded row-major)
// Bs[k][n] = K[b, col+n, j+k]   (K loaded row-major)
static __global__ void qk_matmul_kernel(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        float* __restrict__ S,
        int N,
        int D,
        float scale)
{
    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;

    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;

    const float* Q_b = Q + (long)b * N * D;
    const float* K_b = K + (long)b * N * D;
    float* S_b = S + (long)b * N * N;

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    float acc[TM][TN] = {};

    for (int j = 0; j < D; j += BK) {

        // Load Q
        for (int i = tid; i < BM * BK / 4; i += nthreads) {
            int m = i / (BK / 4);
            int k4 = i % (BK / 4);
            int gr = row + m;
            float4 tmp = {0.f, 0.f, 0.f, 0.f};
            if (gr < N) {
                tmp = *reinterpret_cast<const float4 *>(&Q_b[gr * D + j + k4 * 4]);
            }
            As[k4 * 4 + 0][m] = tmp.x;
            As[k4 * 4 + 1][m] = tmp.y;
            As[k4 * 4 + 2][m] = tmp.z;
            As[k4 * 4 + 3][m] = tmp.w;
        }

        // Load K^T
        for (int i = tid; i < BN * BK / 4; i += nthreads) {
            int n = i / (BK / 4);
            int k4 = i % (BK / 4);
            int gc = col + n;
            float4 tmp = {0.f, 0.f, 0.f, 0.f};
            if (gc < N){
                tmp = *reinterpret_cast<const float4*>(&K_b[gc * D + j + k4 * 4]);
            }
            Bs[k4 * 4 + 0][n] = tmp.x;
            Bs[k4 * 4 + 1][n] = tmp.y;
            Bs[k4 * 4 + 2][n] = tmp.z;
            Bs[k4 * 4 + 3][n] = tmp.w;
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float a[TM], bv[TN];
            for (int m = 0; m < TM; ++m) a[m]  = As[k][ty * TM + m];
            for (int n = 0; n < TN; ++n) bv[n] = Bs[k][tx * TN + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * bv[n];
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++) {
            int r = row + ty * TM + m;
            int c = col + tx * TN + n;
            if (r < N && c < N)
                S_b[r * N + c] = acc[m][n] * scale;
        }
}

// Fused softmax + AV
static __global__ void softmax_av_fused_kernel(
    const float* __restrict__ S, // (B, N, N)
    const float* __restrict__ V, // (B, N, D)
    float* __restrict__ O, // (B, N, D)
    int N,
    int D)
{
    int b = blockIdx.y;
    int row = blockIdx.x;

    const float* S_row = S + (long)b * N * N + row * N;
    const float* Vb = V + (long)b * N * D;
    float* O_row = O + (long)b * N * D + row * D;

    extern __shared__ float smem[];
    float* attn = smem;
    float* scratch = smem + N;

    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        mx = fmaxf(mx, S_row[j]);
    scratch[threadIdx.x] = mx;

    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        }
        __syncthreads();
    }
    mx = scratch[0];

    __syncthreads();

    float sum = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float e = expf(S_row[j] - mx);
        attn[j] = e;
        sum += e;
    }
    scratch[threadIdx.x] = sum;

    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            scratch[threadIdx.x] += scratch[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = scratch[0];

    for (int j = threadIdx.x; j < N; j += blockDim.x)
        attn[j] /= sum;

    __syncthreads(); // all threads see the full normalized attn[] before AV

    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < N; k++)
            acc += attn[k] * Vb[k * D + col];
        O_row[col] = acc;
    }
}

void attention_fused_softmax(
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
        dim3 block(BN / TN, BM / TM);
        dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM, B);
        qk_matmul_kernel<<<grid, block>>>(Q, K, S_buf, N, D, p.scale);
    }

    // Fused softmax + AV
    {
        size_t smem_bytes = (size_t)(N + D) * sizeof(float);
        softmax_av_fused_kernel<<<dim3(N, B), D, smem_bytes>>>(S_buf, V, O, N, D);
    }
}
