#pragma once
#include <cstddef>

struct GPT2Config {
    int n_layer;
    int n_head;
    int d_model;
    int n_ctx;
    int vocab;
};

struct BlockWeights {
    float *ln1_w, *ln1_b;
    float *attn_qkv_w, *attn_qkv_b;
    float *attn_proj_w, *attn_proj_b;
    float *ln2_w, *ln2_b;
    float *mlp_fc_w, *mlp_fc_b;
    float *mlp_proj_w, *mlp_proj_b;
};

struct GPT2Weights {
    GPT2Config config;
    float* arena;
    size_t arena_bytes;
    float *wte, *wpe;
    BlockWeights h[12];
    float *lnf_w, *lnf_b;
};

// Allocates one device arena and carves per-tensor pointers into it.
// If `manifest_out_path` is non-null, also writes a JSON manifest
// (per-tensor shape/sum/mean/first/last) computed on the host copy
void gpt2_load(const char* bin_path, GPT2Weights& out, const char* manifest_out_path = nullptr);

void gpt2_free(GPT2Weights& w);
