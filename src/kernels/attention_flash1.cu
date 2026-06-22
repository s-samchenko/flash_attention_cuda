#include "attention.h"
#include <float.h>
#include <math.h>

template<int D, int Br, int Bc, int BK, int TM, int TN>
__global__ void attention_flash_1_kernel(
        const float* __restrict Q,
        const float* __restrict K,
        const float* __restrict V,
        float* O,
        int N,
        float scale){
    unsigned int tile_i = blockIdx.x;
    unsigned int tile_j = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    int Tc = (N + Bc - 1) / Bc;

    extern __shared__ float smem[];
    float* Q_i = smem;                // [Br][D]
    float* K_j = Q_i + Br * D;        // [D][Bc]
    float* V_j = K_j + Bc * D;        // [Bc][D]
    float* S_ij = V_j + Br * D;       // [Br][Bc]
    float* m_i = S_ij + Br * Bc;      // [Br]
    float* l_i = m_i + Br;            // [Br]
    float o_reg[D / Bc] = {};

    for (int i = 0; i < D / Bc; ++i) {
        Q_i[ty * D + tx + i * Bc] = Q[(tile_j * N + tile_i * Br + ty) * D + tx + i * Bc];
    }
    if (tx == 0) {
        m_i[ty] = -INFINITY;
        l_i[ty] = 0.f;
    }

    for (int j = 0; j < Tc; ++j) {
        for (int i = 0; i < D / Br; ++i) {
            int d = ty + Br * i;
            K_j[d * Bc + tx] = K[(tile_j * N + j * Bc + tx) * D + d];
            V_j[tx * D + d] = V[(tile_j * N + j * Bc + tx) * D + d];
        }
        __syncthreads();

        float s_acc = 0.f;
        for (int d = 0; d < D; ++d) {
            s_acc += Q_i[ty * D + d] * K_j[d * Bc + tx];
        }
        S_ij[ty * Bc + tx] = s_acc * scale;

        const unsigned sub = (Bc < 32) ? ty % (32 / Bc) : 0;
        const unsigned mask = (Bc < 32) ? (((1u << Bc) - 1u) << (sub * Bc)) : 0xffffffffu;
        const unsigned base_lane = sub * Bc;

        float m_ij = S_ij[ty * Bc + tx];
        for (int offset = Bc / 2; offset > 0; offset >>= 1)
            m_ij = fmaxf(m_ij, __shfl_down_sync(0xffffffff, m_ij, offset));
        m_ij = __shfl_sync(mask, m_ij, base_lane);

        S_ij[ty * Bc + tx] = __expf(S_ij[ty * Bc + tx] - m_ij);

        float l_ij = S_ij[ty * Bc + tx];
        for (int offset = Bc / 2; offset > 0; offset >>= 1)
            l_ij += __shfl_down_sync(0xffffffff, l_ij, offset);
        l_ij = __shfl_sync(mask, l_ij, base_lane);

        float m_i_new = fmaxf(m_i[ty],m_ij);
        float alpha = __expf(m_i[ty] - m_i_new);
        float beta = __expf(m_ij - m_i_new);
        float l_i_new = alpha * l_i[ty] + beta * l_ij;

        __syncthreads();

        float o_acc[D / Bc] = {};
        for (int k = 0; k < Bc; ++k) {
            float p = S_ij[ty * Bc + k];
            for (int i = 0; i < D / Bc; ++i) {
                o_acc[i] += p * V_j[k * D + tx + i * Bc];
            }
        }
        for (int i = 0; i < D / Bc; ++i) {
            o_reg[i] = alpha * o_reg[i] + beta * o_acc[i];
        }

        m_i[ty] = m_i_new;
        l_i[ty] = l_i_new;

        __syncthreads();
    }

    for (int i = 0; i < D / Bc; ++i) {
        O[(tile_j * N + tile_i * Br + ty) * D + tx + i * Bc] = o_reg[i] / l_i[ty];
    }
}

template<int D, int Br, int Bc, int BK, int TM, int TN>
static void launch(const float* Q, const float* K, const float* V,
                   float* O, int N, float scale, int B)
{
    size_t smem_bytes = (Br * D + 2 * Bc * D + 2 * Br + Br * Bc) * sizeof(float);
    cudaFuncSetAttribute(
            attention_flash_1_kernel<D, Br, Bc, BK, TM, TN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
    );

    int Tr = (N + Br - 1) / Br;
    attention_flash_1_kernel<D, Br, Bc, BK, TM, TN><<<dim3(Tr, B), dim3(Bc, Br), smem_bytes>>>(
            Q, K, V, O, N, scale
    );
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
        launch< 64, 32, 32, 16, 4, 4>(Q, K, V, O, p.seq_len, p.scale, B);
    }
    else if (p.head_dim == 128) {
        launch<128, 16, 16, 16, 4, 4>(Q, K, V, O, p.seq_len, p.scale, B);
    }
    else exit(-1);
}