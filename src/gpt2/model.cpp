#include "gpt2_model.hpp"
#include "gpt2_kernels.hpp"
#include "gpt2_gemm.hpp"
#include "attention.h"
#include "cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <cassert>

namespace {

    constexpr int N_MAX = 1024;
    constexpr int D = 768;
    constexpr int H = 12;
    constexpr int DH = 64;
    constexpr int QKV = 3 * D; // 2304
    constexpr int FF = 4 * D;  // 3072

    int pad_to_64(int n) {
        return ((n + 63) / 64) * 64;
    }

    void run_attention(gpt2::Activations& a, int n_real, int n_pad) {
        AttentionParams p;
        p.batch = 1;
        p.n_heads = H;
        p.seq_len = n_pad;
        p.head_dim = DH;
        p.causal = true;
        p.scale = 0.125f;
        assert(p.causal || n_pad == n_real);
        attention_flash2_fp16v2(a.Q, a.K, a.V, a.O_heads, nullptr, p);
    }

}

namespace gpt2 {

    void activations_init(Activations& a) {
        const size_t ND = size_t(N_MAX) * D;
        const size_t sizes[] = {
            ND,                   // x
            ND,                   // ln_out
            size_t(N_MAX) * QKV,
            ND, ND, ND,           // Q, K, V
            ND,                   // O_heads
            ND,                   // attn_merged
            ND,                   // proj_out
            size_t(N_MAX) * FF,
        };
        size_t total = 0;
        for (size_t s : sizes) total += s;

        a.arena_floats = total;
        cuda_check(cudaMalloc(&a.arena, total * sizeof(float)));
        cuda_check(cudaMemset(a.arena, 0, total * sizeof(float)));

        float* p = a.arena;
        auto carve = [&](size_t n) { float* q = p; p += n; return q; };
        a.x = carve(sizes[0]);
        a.ln_out = carve(sizes[1]);
        a.qkv = carve(sizes[2]);
        a.Q = carve(sizes[3]);
        a.K = carve(sizes[4]);
        a.V = carve(sizes[5]);
        a.O_heads = carve(sizes[6]);
        a.attn_merged = carve(sizes[7]);
        a.proj_out = carve(sizes[8]);
        a.mlp_h = carve(sizes[9]);

        cuda_check(cudaMalloc(&a.d_ids, size_t(N_MAX) * sizeof(int)));
        cuda_check(cudaMemset(a.d_ids, 0, size_t(N_MAX) * sizeof(int)));

        if (cublasCreate(&a.cublas) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "cublasCreate failed\n");
            std::exit(1);
        }
    }

    void activations_free(Activations& a) {
        cudaFree(a.arena);
        cudaFree(a.d_ids);
        cublasDestroy(a.cublas);
        a.arena = nullptr;
        a.d_ids = nullptr;
    }

    void block_forward(const GPT2Weights& w, Activations& a, int layer, int n_real, int n_pad) {
        const BlockWeights& b = w.h[layer];

        layernorm(a.x, b.ln1_w, b.ln1_b, a.ln_out, n_real, D);
        gemm_rm(a.cublas, a.ln_out, b.attn_qkv_w, a.qkv, n_real, D, QKV);
        bias_add(a.qkv, b.attn_qkv_b, n_real, QKV);
        qkv_split(a.qkv, a.Q, a.K, a.V, n_real, n_pad, H, DH);

        run_attention(a, n_real, n_pad);

        merge_heads(a.O_heads, a.attn_merged, n_real, n_pad, H, DH);
        gemm_rm(a.cublas, a.attn_merged, b.attn_proj_w, a.proj_out, n_real, D, D);
        bias_add(a.proj_out, b.attn_proj_b, n_real, D);
        residual_add(a.x, a.proj_out, n_real * D);

        layernorm(a.x, b.ln2_w, b.ln2_b, a.ln_out, n_real, D);
        gemm_rm(a.cublas, a.ln_out, b.mlp_fc_w, a.mlp_h, n_real, D, FF);
        bias_add(a.mlp_h, b.mlp_fc_b, n_real, FF);
        gelu_new(a.mlp_h, n_real * FF);
        gemm_rm(a.cublas, a.mlp_h, b.mlp_proj_w, a.proj_out, n_real, FF, D);
        bias_add(a.proj_out, b.mlp_proj_b, n_real, D);
        residual_add(a.x, a.proj_out, n_real * D);
    }

    void gpt2_forward(const GPT2Weights& w, Activations& a, const int* ids_host, int n_real) {
        cuda_check(cudaMemcpy(a.d_ids, ids_host, size_t(n_real) * sizeof(int), cudaMemcpyHostToDevice));
        embedding_gather(a.d_ids, w.wte, w.wpe, a.x, n_real, D);

        const int n_pad = pad_to_64(n_real);
        for (int i = 0; i < w.config.n_layer; ++i) {
            block_forward(w, a, i, n_real, n_pad);
        }
    }

}
