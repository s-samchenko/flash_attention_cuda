#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Tensor shape convention: (batch*n_heads, seq_len, head_dim), row-major.

struct AttentionParams {
    int   batch;
    int   n_heads;
    int   seq_len;
    int   head_dim;
    bool  causal;
    float scale;   // 1 / sqrt(head_dim)
};

// src/kernels/attention_naive.cu
void attention_naive(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* S_buf,
    const AttentionParams& p);
