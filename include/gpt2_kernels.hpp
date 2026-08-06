#pragma once

namespace gpt2 {

    // in-place elementwise gelu_new using tanh approximation
    void gelu_new(float* x, int n);

    // per-row (C-wide) layernorm: out = (x - mean) * rsqrt(var + eps) * w + b
    void layernorm(const float* x, const float* w, const float* b, float* out, int N, int C = 768);

    // x[i] += bias[i % cols], bias broadcast down the rows
    void bias_add(float* x, const float* bias, int rows, int cols);

    // x[i] += y[i].
    void residual_add(float* x, const float* y, int n);

    // out[i*d + k] = wte[ids[i]*d + k] + wpe[i*d + k]
    void embedding_gather(const int* ids, const float* wte, const float* wpe,
                          float* out, int n, int d);

    // fused [n_real, 3*h*d] -> head-contiguous Q,K,V each [h, n_pad, d]
    void qkv_split(const float* qkv, float* Q, float* K, float* V,
                   int n_real, int n_pad, int h, int d);

    // head-contiguous O [h, n_pad, d] -> [n_real, h*d]
    void merge_heads(const float* O, float* out, int n_real, int n_pad, int h, int d);

}

int gpt2_test_kernels(const char* fixtures_dir);
