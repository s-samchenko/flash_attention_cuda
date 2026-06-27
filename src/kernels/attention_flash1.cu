#include "attention.h"
#include <float.h>
#include <math.h>
#include <cstdio>

template<int D, int Br, int Bc, int BK, int TM, int TN>
__global__ void attention_flash_1_kernel(
        const float* __restrict Q,
        const float* __restrict K,
        const float* __restrict V,
        float* O,
        int N,
        float scale){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;
    int Tc = (N + Bc - 1) / Bc;


    extern __shared__ float smem[];
    float* Q_i = smem;                 // [Br][D+1]
    float* K_j = Q_i + Br * (D+1);     // [D][Bc+1]
    float* V_j = K_j + D * (Bc+1);     // [Bc+1][D]
    float* S_ij = V_j + Bc * D;        // [Br][Bc+1]

    constexpr int TD = D * TN / Bc;
    constexpr int tpr = Bc / TN;
    static_assert(tpr <= 32 && (tpr & (tpr - 1)) == 0,
                  "tpr must be a power of two <= 32 for warp shuffle");

    float o_reg[TM][TD] = {};
    float m_acc[TM], l_acc[TM];
    for (int m = 0; m < TM; ++m) {
        m_acc[m] = -INFINITY;
        l_acc[m] = 0.f;
    }

    const int q_base = (blockIdx.y * N + blockIdx.x * Br) * D;
    for (int i = tid; i < Br * D; i += total_threads){
        int r = i / D;
        int d = i - r * D;
        Q_i[r * (D+1) + d] = Q[q_base + i];
    }

    for (int j = 0; j < Tc; ++j) {
        for (int c = ty; c < Bc; c += blockDim.y) {
            for (int d = tx; d < D; d += blockDim.x) {
                int kv_idx = (blockIdx.y * N + j * Bc + c) * D + d;
                K_j[d * (Bc+1) + c] = K[kv_idx];
                V_j[c * D + d] = V[kv_idx];
            }
        }
        __syncthreads();

        float acc[TM][TN] = {};
        for (int d = 0; d < D; ++d) {

            float qf[TM], kf[TN];
            for (int m = 0; m < TM; ++m) {
                qf[m] = Q_i[(ty*TM + m) * (D+1) + d];
            }
            for (int n = 0; n < TN; ++n) {
                kf[n] = K_j[d * (Bc+1) + (tx*TN + n)];
            }
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] += qf[m] * kf[n];
                }
            }
        }

        float alpha[TM];
        for (int m = 0; m < TM; ++m) {

            float pmax = -INFINITY;
            for (int n = 0; n < TN; ++n) {
                acc[m][n] *= scale;
                pmax = fmaxf(pmax, acc[m][n]);
            }
            for (int off = tpr / 2; off > 0; off >>= 1)
                pmax = fmaxf(pmax, __shfl_xor_sync(0xffffffffu, pmax, off));

            float m_old = m_acc[m];
            float m_new = fmaxf(m_old, pmax);
            alpha[m] = __expf(m_old - m_new);

            float psum = 0.f;
            for (int n = 0; n < TN; ++n) {
                float e = __expf(acc[m][n] - m_new);
                acc[m][n] = e;
                psum += e;
            }
            for (int off = tpr / 2; off > 0; off >>= 1)
                psum += __shfl_xor_sync(0xffffffffu, psum, off);

            l_acc[m] = alpha[m] * l_acc[m] + psum;
            m_acc[m] = m_new;
        }

        for (int m = 0; m < TM; ++m) {
            for (int n = 0; n < TN; ++n) {
                S_ij[(ty*TM + m) * (Bc+1) + tx*TN + n] = acc[m][n];
            }
        }

        for (int m = 0; m < TM; ++m) {
            for (int nv = 0; nv < TD; ++nv)
                o_reg[m][nv] *= alpha[m];
        }
        __syncthreads();

        for (int c = 0; c < Bc; ++c) {
            float pf[TM], vf[TD];
            for (int m = 0; m < TM; ++m)
                pf[m] = S_ij[(ty * TM + m) * (Bc+1) + c];

            for (int nv = 0; nv < TD; ++nv)
                vf[nv] = V_j[c * D + tx * TD + nv];

            for (int m = 0; m < TM; ++m)
                for (int nv = 0; nv < TD; ++nv)
                    o_reg[m][nv] += pf[m] * vf[nv];
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        int row = ty * TM + m;
        float linv = 1.f / l_acc[m];
        for (int nv = 0; nv < TD; ++nv)
            O[(blockIdx.y * N + blockIdx.x * Br + row) * D + tx * TD + nv] = o_reg[m][nv] * linv;
    }
}

template<int D, int Br, int Bc, int BK, int TM, int TN>
static void launch(const float* Q, const float* K, const float* V,
                   float* O, int N, float scale, int B)
{
    static_assert(Bc % TN == 0, "Bc must be divisible by TN");
    static_assert(Br % TM == 0, "Br must be divisible by TM");
    static_assert((D * TN) % Bc == 0, "D*TN must be divisible by Bc");
    static_assert(D % (Bc / TN) == 0, "D must be divisible by Bc/TN");
    static_assert((Bc / TN) * (Br / TM) <= 1024, "block exceeds 1024 threads");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t smem_bytes = (Br * (D + 1) + D * (Bc + 1) + Bc * D + Br * (Bc + 1)) * sizeof(float);

    if (smem_bytes > prop.sharedMemPerBlockOptin) {
        fprintf(stderr,
                "fa1: requested %zu B shared > device max %zu B "
                "(D=%d Br=%d Bc=%d TN=%d TM=%d)\n",
                smem_bytes, prop.sharedMemPerBlockOptin, D, Br, Bc, TN, TM);
        exit(1);
    }

    cudaError_t e = cudaFuncSetAttribute(
            attention_flash_1_kernel<D, Br, Bc, BK, TM, TN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "fa1: cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(e));
        exit(1);
    }

    int Tr = (N + Br - 1) / Br;
    attention_flash_1_kernel<D, Br, Bc, BK, TM, TN>
            <<<dim3(Tr, B), dim3(Bc / TN, Br / TM), smem_bytes>>>
            (Q, K, V, O, N, scale
    );
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "fa1 launch failed: %s\n", cudaGetErrorString(launchErr));
        exit(1);
    }
}

void attention_flash1(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        float*,
        const AttentionParams& p)
{
    int B = p.batch * p.n_heads;
    if (p.head_dim ==  64) {
        launch<64, 64, 32, 16, 2, 4>(Q, K, V, O, p.seq_len, p.scale, B);
    }
    else if (p.head_dim == 128) {
        launch<128, 64, 32, 16, 4, 2>(Q, K, V, O, p.seq_len, p.scale, B);
    }
    else exit(-1);
}