#include "attention.h"
#include <math.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

namespace {
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
constexpr int WARP_SIZE = 32;

using q_frag_t = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>;
using k_frag_t = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major>;
using p_frag_t = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>;
using v_frag_t = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>;
using acc_t = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

template<class Frag>
    __device__ __forceinline__ void truncate_tf32(Frag& f) {
    #pragma unroll
        for (int i = 0; i < f.num_elements; ++i)
            f.x[i] = wmma::__float_to_tf32(f.x[i]);
    }
}

template<int D, int Br, int Bc>
__global__ void attention_flash_2_tf32_kernel(
        const float* __restrict Q,
        const float* __restrict K,
        const float* __restrict V,
        float* O,
        int N,
        float scale)
{
    static_assert(Br % WMMA_M == 0, "Br must be a multiple of 16");
    static_assert(Bc % WMMA_N == 0, "Bc must be a multiple of 16");
    static_assert(D % WMMA_K == 0, "D must be a multiple of the MMA K-step");

    constexpr int FRAG_N = Bc / WMMA_N;
    constexpr int FRAG_D = D  / WMMA_N;
    constexpr int QK_STEPS = D  / WMMA_K;
    constexpr int PV_STEPS = Bc / WMMA_K;

    constexpr int lpr = WARP_SIZE / WMMA_M;
    constexpr int TS = Bc / lpr;
    constexpr int TD = D / lpr;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int total_threads = blockDim.x;
    const int cs = (lane_id % lpr) * TS;
    const int cd = (lane_id % lpr) * TD;
    const int row = warp_id * WMMA_M + lane_id / lpr;

    const int bh_base = blockIdx.y * N * D;
    const int q_base = bh_base + blockIdx.x * Br * D;

    extern __shared__ float smem[];
    float* Q_i = smem;              // [Br][D]
    float* K_j = Q_i + Br * D;      // [Bc][D]
    float* V_j = K_j + Bc * D;      // [Bc][D]
    float* S_ij = V_j + Bc * D;     // [Br][Bc]
    float* O_i = Q_i;               // reuse Q_i to reduce shared memory since Q_i isn't needed when we start computing O_i

    float m_acc = -INFINITY;
    float l_acc = 0.f;

    for (int i = tid; i < Br * D; i += total_threads) {
        Q_i[i] = Q[q_base + i];
    }
    __syncthreads();

    q_frag_t q_frag[QK_STEPS];
    #pragma unroll
    for (int k = 0; k < QK_STEPS; ++k) {
        wmma::load_matrix_sync(q_frag[k], Q_i + warp_id * WMMA_M * D + k * WMMA_K, D);
        truncate_tf32(q_frag[k]);
    }

    __syncthreads();
    for (int i = tid; i < Br * D; i += total_threads) {
        O_i[i] = 0.f;
    }

    const int Tc = (N + Bc - 1) / Bc;
    for (int j = 0; j < Tc; ++j) {
        for (int i = tid; i < Bc * D; i += total_threads) {
            int kv_idx = bh_base + j * Bc * D + i;
            K_j[i] = K[kv_idx];
            V_j[i] = V[kv_idx];
        }
        __syncthreads();

        acc_t s_frag[FRAG_N];
        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
            wmma::fill_fragment(s_frag[n], 0.f);
        }

        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
            #pragma unroll
            for (int k = 0; k < QK_STEPS; ++k) {
                k_frag_t k_frag;
                wmma::load_matrix_sync(k_frag, K_j + n * WMMA_N * D + k * WMMA_K, D);
                truncate_tf32(k_frag);
                wmma::mma_sync(s_frag[n], q_frag[k], k_frag, s_frag[n]);
            }
        }

        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
            #pragma unroll
            for (int i = 0; i < s_frag[n].num_elements; ++i) {
                s_frag[n].x[i] *= scale;
            }
            wmma::store_matrix_sync(S_ij + warp_id * WMMA_M * Bc + n * WMMA_N, s_frag[n], Bc, wmma::mem_row_major);
        }

        float* row_ptr = S_ij + row * Bc + cs;

        float pmax = -INFINITY;
        #pragma unroll
        for (int c = 0; c < TS; ++c) {
            pmax = fmaxf(pmax, row_ptr[c]);
        }
        pmax = fmaxf(pmax, __shfl_xor_sync(0xffffffffu, pmax, 1));

        float m_new = fmaxf(m_acc, pmax);
        // sm_86: __expf(-inf) = 0, so alpha=0 on the first tile is safe.
        float alpha = __expf(m_acc - m_new);

        float psum = 0.f;
        #pragma unroll
        for (int c = 0; c < TS; ++c) {
            float e = __expf(row_ptr[c] - m_new);
            row_ptr[c] = e;
            psum += e;
        }
        psum += __shfl_xor_sync(0xffffffffu, psum, 1);

        l_acc = alpha * l_acc + psum;
        m_acc = m_new;

        #pragma unroll
        for (int c = 0; c < TD; ++c) {
            O_i[row * D + cd + c] *= alpha;
        }
        __syncwarp();

        #pragma unroll
        for (int d = 0; d < FRAG_D; ++d) {
            acc_t o_frag;
            wmma::load_matrix_sync(o_frag, O_i + warp_id * WMMA_M * D + d * WMMA_N, D, wmma::mem_row_major);

            #pragma unroll
            for (int k = 0; k < PV_STEPS; ++k) {
                p_frag_t p_frag;
                v_frag_t v_frag;
                wmma::load_matrix_sync(p_frag, S_ij + warp_id * WMMA_M * Bc + k * WMMA_K, Bc);
                truncate_tf32(p_frag);
                wmma::load_matrix_sync(v_frag, V_j + k * WMMA_K * D + d * WMMA_N, D);
                truncate_tf32(v_frag);
                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            wmma::store_matrix_sync(O_i + warp_id * WMMA_M * D + d * WMMA_N, o_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    float linv = (l_acc > 0.f) ? (1.f / l_acc) : 0.f;
    const int row_g = blockIdx.x * Br + row;
    if (row_g < N) {
        #pragma unroll
        for (int c = 0; c < TD; ++c) {
            O[bh_base + row_g * D + cd + c] = O_i[row * D + cd + c] * linv;
        }
    }
}

template<int D, int Br, int Bc>
static void launch(const float* Q, const float* K, const float* V,
                   float* O, int N, float scale, int B)
{
    static_assert(Br % WMMA_M == 0 && Bc % WMMA_N == 0 && D % 16 == 0, "WMMA tile dims must be multiples of 16");

    constexpr int WARPS   = Br / WMMA_M;
    constexpr int THREADS = WARPS * WARP_SIZE;

    size_t smem_bytes = (Br * D + Bc * D + Bc * D +  Br * Bc) * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_bytes > prop.sharedMemPerBlockOptin) {
        fprintf(stderr, "fa2_tf32: requested %zu B smem > device max %zu B (D=%d Br=%d Bc=%d)\n",
                smem_bytes, prop.sharedMemPerBlockOptin, D, Br, Bc);
        exit(1);
    }

    cudaError_t e = cudaFuncSetAttribute(
            attention_flash_2_tf32_kernel<D, Br, Bc>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "fa2_tf32: cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(e));
        exit(1);
    }

    int Tr = (N + Br - 1) / Br;
    attention_flash_2_tf32_kernel<D, Br, Bc> <<<dim3(Tr, B), THREADS, smem_bytes>>>(Q, K, V, O, N, scale);

    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "fa2_tf32 launch failed: %s\n", cudaGetErrorString(le));
        exit(1);
    }
}

void attention_flash2_tf32(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        float*,
        const AttentionParams& p)
{
    int B = p.batch * p.n_heads;
    if (p.head_dim == 64)
        launch<64, 64, 32>(Q, K, V, O, p.seq_len, p.scale, B);
    else if (p.head_dim == 128)
        launch<128, 64, 32>(Q, K, V, O, p.seq_len, p.scale, B);
    else exit(-1);
}
