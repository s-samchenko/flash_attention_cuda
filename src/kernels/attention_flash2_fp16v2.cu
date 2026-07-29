#include "attention.h"
#include <math.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>

using namespace nvcuda;

namespace {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;

    constexpr int PAD_S = 4;
    constexpr int PAD_KV = 8;

    using q_frag_t = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using k_frag_t = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;
    using p_frag_t = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using v_frag_t = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using acc_t = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

    __device__ __forceinline__ int frag_row(int lane, int i) {
        return (lane / 4) + ((i & 2) ? 8 : 0);
    }
    __device__ __forceinline__ int frag_col(int lane, int i) {
        return (lane & 3) * 2 + (i & 1) + ((i & 4) ? 8 : 0);
    }
}

template<int D, int Br, int Bc, int MIN_BLOCKS, bool CAUSAL>
__global__ __launch_bounds__((Br / WMMA_M) * WARP_SIZE, MIN_BLOCKS)
void attention_flash_2_fp16v2_kernel(
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
    constexpr int FRAG_D = D / WMMA_N;
    constexpr int QK_STEPS = D / WMMA_K;
    constexpr int PV_STEPS = Bc / WMMA_K;
    constexpr int ACC_NUM = 8;

    constexpr int S_STRIDE = Bc + PAD_S;         // floats per S row
    constexpr int P_STRIDE = S_STRIDE * 2;       // halves per aliased P row
    constexpr int KV_STRIDE = D + PAD_KV;        // halves per K/V row

    static_assert(S_STRIDE % 4 == 0, "S ldm must be a multiple of 4 for WMMA float store");
    static_assert(P_STRIDE % 8 == 0, "P ldm must be a multiple of 8 for WMMA half load");
    static_assert(KV_STRIDE % 8 == 0, "KV ldm must be a multiple of 8 for WMMA half load");

    constexpr int lpr = WARP_SIZE / WMMA_M;
    constexpr int TS = Bc / lpr;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int total_threads = blockDim.x;
    const int cs = (lane_id % lpr) * TS;
    const int row = warp_id * WMMA_M + lane_id / lpr;

    const int bh_base = blockIdx.y * N * D;
    const int q_base = bh_base + blockIdx.x * Br * D;

    extern __shared__ float smem[];
    constexpr size_t Q_BYTES = size_t(2) * Br * D;
    constexpr size_t SP_BYTES = size_t(4) * Br * S_STRIDE;
    constexpr size_t KV_OFF = (Q_BYTES > SP_BYTES) ? Q_BYTES : SP_BYTES;

    half* Q_i = reinterpret_cast<half*>(smem);                                      // [Br][D]
    float* S_ij = reinterpret_cast<float*>(smem);                                   // [Br][S_STRIDE]
    half* P_ij = reinterpret_cast<half*>(smem);                                     // aliased with S_ij
    half* K_j = reinterpret_cast<half*>(reinterpret_cast<char*>(smem) + KV_OFF);    // [Bc][KV_STRIDE]
    half* V_j = K_j + Bc * KV_STRIDE;                                               // [Bc][KV_STRIDE]

    float m_acc = -INFINITY;
    float l_acc = 0.f;

    acc_t o_frag[FRAG_D];
#pragma unroll
    for (int d = 0; d < FRAG_D; ++d) {
        wmma::fill_fragment(o_frag[d], 0.f);
    }

    for (int i = tid; i < Br * D; i += total_threads) {
        Q_i[i] = __float2half(Q[q_base + i]);
    }
    __syncthreads();

    q_frag_t q_frag[QK_STEPS];
#pragma unroll
    for (int k = 0; k < QK_STEPS; ++k) {
        wmma::load_matrix_sync(q_frag[k], Q_i + warp_id * WMMA_M * D + k * WMMA_K, D);
    }
    __syncthreads();

    const int row0 = blockIdx.x * Br;
    const int Tc = (N + Bc - 1) / Bc;

    // Causal loop cap: any tile whose min column > this block's max row is fully masked.
    const int Tc_eff = CAUSAL ? min(Tc, (row0 + Br + Bc - 1) / Bc) : Tc;
    for (int j = 0; j < Tc_eff; ++j) {
        for (int i = tid; i < Bc * D; i += total_threads) {
            int r = i / D;
            int c = i - r * D;
            int kv_idx = bh_base + j * Bc * D + i;
            K_j[r * KV_STRIDE + c] = __float2half(K[kv_idx]);
            V_j[r * KV_STRIDE + c] = __float2half(V[kv_idx]);
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
                wmma::load_matrix_sync(k_frag, K_j + n * WMMA_N * KV_STRIDE + k * WMMA_K, KV_STRIDE);
                wmma::mma_sync(s_frag[n], q_frag[k], k_frag, s_frag[n]);
            }
        }

        // Global row range this warp owns: [q_row_warp, q_row_warp + WMMA_M).
        const int q_row_warp = row0 + warp_id * WMMA_M;
        const int tile_col0 = j * Bc;

#pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
#pragma unroll
            for (int i = 0; i < s_frag[n].num_elements; ++i) {
                s_frag[n].x[i] *= scale;
            }
            if (CAUSAL) {
                const int frag_col0 = tile_col0 + n * WMMA_N;
                if (frag_col0 + WMMA_N - 1 > q_row_warp) {
#pragma unroll
                    for (int i = 0; i < s_frag[n].num_elements; ++i) {
                        int r = q_row_warp + frag_row(lane_id, i);
                        int c = frag_col0 + frag_col(lane_id, i);
                        if (c > r) s_frag[n].x[i] = -INFINITY;
                    }
                }
            }
            wmma::store_matrix_sync(S_ij + warp_id * WMMA_M * S_STRIDE + n * WMMA_N,
                                    s_frag[n], S_STRIDE, wmma::mem_row_major);
        }
        __syncwarp();

        float* row_ptr = S_ij + row * S_STRIDE + cs;
        half* p_ptr = reinterpret_cast<half*>(S_ij + row * S_STRIDE) + cs;

        float pmax = -INFINITY;
#pragma unroll
        for (int c = 0; c < TS; ++c) {
            pmax = fmaxf(pmax, row_ptr[c]);
        }
        pmax = fmaxf(pmax, __shfl_xor_sync(0xffffffffu, pmax, 1));

        float m_new = fmaxf(m_acc, pmax);
        float alpha = __expf(m_acc - m_new);

        float e_reg[TS];
        float psum = 0.f;
#pragma unroll
        for (int c = 0; c < TS; ++c) {
            float e = __expf(row_ptr[c] - m_new);
            e_reg[c] = e;
            psum += e;
        }
        __syncwarp();
#pragma unroll
        for (int c = 0; c < TS; ++c) {
            p_ptr[c] = __float2half(e_reg[c]);
        }
        psum += __shfl_xor_sync(0xffffffffu, psum, 1);

        l_acc = alpha * l_acc + psum;
        m_acc = m_new;

        float alpha_table[ACC_NUM];
#pragma unroll
        for (int i = 0; i < ACC_NUM; ++i) {
            alpha_table[i] = __shfl_sync(0xffffffffu, alpha, frag_row(lane_id, i) * 2);
        }

#pragma unroll
        for (int d = 0; d < FRAG_D; ++d) {
#pragma unroll
            for (int i = 0; i < ACC_NUM; ++i) {
                o_frag[d].x[i] *= alpha_table[i];
            }
        }
        __syncwarp();

        p_frag_t p_frag[PV_STEPS];
#pragma unroll
        for (int k = 0; k < PV_STEPS; ++k) {
            wmma::load_matrix_sync(p_frag[k], P_ij + warp_id * WMMA_M * P_STRIDE + k * WMMA_K, P_STRIDE);
        }

#pragma unroll
        for (int d = 0; d < FRAG_D; ++d) {
#pragma unroll
            for (int k = 0; k < PV_STEPS; ++k) {
                v_frag_t v_frag;
                wmma::load_matrix_sync(v_frag, V_j + k * WMMA_K * KV_STRIDE + d * WMMA_N, KV_STRIDE);
                wmma::mma_sync(o_frag[d], p_frag[k], v_frag, o_frag[d]);
            }
        }
        __syncthreads();
    }

    float linv = (l_acc > 0.f) ? (1.f / l_acc) : 0.f;
    float linv_table[ACC_NUM];
#pragma unroll
    for (int i = 0; i < ACC_NUM; ++i) {
        linv_table[i] = __shfl_sync(0xffffffffu, linv, frag_row(lane_id, i) * 2);
    }

#pragma unroll
    for (int d = 0; d < FRAG_D; ++d) {
#pragma unroll
        for (int i = 0; i < ACC_NUM; ++i) {
            o_frag[d].x[i] *= linv_table[i];
        }
        wmma::store_matrix_sync(
                O + bh_base + (blockIdx.x * Br + warp_id * WMMA_M) * D + d * WMMA_N,
                o_frag[d], D, wmma::mem_row_major);
    }
}

template<int D, int Br, int Bc, int MIN_BLOCKS, bool CAUSAL>
static void launch(const float* Q, const float* K, const float* V,
                   float* O, int N, float scale, int B)
{
    static_assert(Br % WMMA_M == 0 && Bc % WMMA_N == 0 && D % 16 == 0, "WMMA tile dims must be multiples of 16");

    constexpr int WARPS = Br / WMMA_M;
    constexpr int THREADS = WARPS * WARP_SIZE;

    constexpr int S_STRIDE = Bc + PAD_S;
    constexpr int KV_STRIDE = D + PAD_KV;

    constexpr size_t Q_BYTES  = size_t(2) * Br * D;
    constexpr size_t SP_BYTES = size_t(4) * Br * S_STRIDE;
    constexpr size_t KV_OFF   = (Q_BYTES > SP_BYTES) ? Q_BYTES : SP_BYTES;
    constexpr size_t smem_bytes = KV_OFF + size_t(2) * (size_t(2) * Bc * KV_STRIDE);

    static const bool configured = [] {
        int max_smem = 0;
        cudaDeviceGetAttribute(&max_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if ((int)smem_bytes > max_smem) {
            fprintf(stderr, "fa2v2: requested %zu B smem > device max %d B (D=%d Br=%d Bc=%d)\n",
                    smem_bytes, max_smem, D, Br, Bc);
            exit(1);
        }
        cudaError_t e = cudaFuncSetAttribute(
                attention_flash_2_fp16v2_kernel<D, Br, Bc, MIN_BLOCKS, CAUSAL>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "fa2v2: cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(e));
            exit(1);
        }
        return true;
    }();
    (void)configured;

    int Tr = (N + Br - 1) / Br;
    attention_flash_2_fp16v2_kernel<D, Br, Bc, MIN_BLOCKS, CAUSAL> <<<dim3(Tr, B), THREADS, smem_bytes>>>(Q, K, V, O, N, scale);

    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "fa2v2 launch failed: %s\n", cudaGetErrorString(le));
        exit(1);
    }
}

void attention_flash2_fp16v2(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        float*,
        const AttentionParams& p)
{
    int B = p.batch * p.n_heads;
    if (p.head_dim == 64) {
        if (p.causal) launch<64, 64, 32, /*MIN_BLOCKS=*/5, true >(Q, K, V, O, p.seq_len, p.scale, B);
        else launch<64, 64, 32, /*MIN_BLOCKS=*/5, false>(Q, K, V, O, p.seq_len, p.scale, B);
    } else if (p.head_dim == 128) {
        if (p.causal) launch<128, 64, 32, /*MIN_BLOCKS=*/3, true >(Q, K, V, O, p.seq_len, p.scale, B);
        else launch<128, 64, 32, /*MIN_BLOCKS=*/3, false>(Q, K, V, O, p.seq_len, p.scale, B);
    } else exit(-1);
}