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

    void block_forward(const GPT2Weights& w, Activations& a, int layer, int n_real, int n_pad, const char* dump_dir = nullptr);

    void gpt2_forward(const GPT2Weights& w, Activations& a, const int* ids_host, int n_real, const char* dump_dir = nullptr);

}
