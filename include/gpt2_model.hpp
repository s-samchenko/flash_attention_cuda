#pragma once
#include "gpt2.hpp"
#include <cublas_v2.h>
#include <cstddef>

namespace gpt2 {

    struct Activations {
        float* arena;
        size_t arena_floats;
        float *x, *ln_out, *qkv, *Q, *K, *V, *O_heads, *attn_merged, *proj_out, *mlp_h;
        int* d_ids;
        cublasHandle_t cublas;
    };

    void activations_init(Activations& a);
    void activations_free(Activations& a);

    // residual stream
    void block_forward(const GPT2Weights& w, Activations& a, int layer, int n_real, int n_pad);

    // Embedding + all blocks
    void gpt2_forward(const GPT2Weights& w, Activations& a, const int* ids_host, int n_real);

}
