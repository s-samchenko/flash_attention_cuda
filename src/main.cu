#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "benchmark.cuh"
#include "attention.h"

// ── helpers ──────────────────────────────────────────────────────────────────

static void fill_rand(float* d_ptr, long n)
{
    std::vector<float> h(n);
    for (long i = 0; i < n; i++) h[i] = (float)rand() / RAND_MAX - 0.5f;
    cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

// Bytes touched by the naive 3-kernel attention (N×N matrix materialised)
static long naive_bytes(int batch, int n_heads, long N, long D)
{
    long BH = batch * n_heads;
    return BH * (4L * N * D + N * N) * sizeof(float);
}

// ── benchmark ───────────────────────────────────────────────────────────

typedef void (*AttnFn)(const float*, const float*, const float*,
                       float*, float*, const AttentionParams&);

static void bench_kernel(const char* name, AttnFn fn,
                         long (*bytes_fn)(int, int, long, long))
{
    const int seq_lens[]  = {512, 1024, 2048, 4096};
    const int head_dims[] = {64, 128};
    const int batch = 1, n_heads = 8;

    for (int hd : head_dims) {
        for (int sl : seq_lens) {
            int  BH = batch * n_heads;
            long N  = sl, D = hd;

            float *d_Q, *d_K, *d_V, *d_O, *d_S;
            cudaMalloc(&d_Q, BH * N * D * sizeof(float));
            cudaMalloc(&d_K, BH * N * D * sizeof(float));
            cudaMalloc(&d_V, BH * N * D * sizeof(float));
            cudaMalloc(&d_O, BH * N * D * sizeof(float));
            cudaMalloc(&d_S, BH * N * N * sizeof(float));

            fill_rand(d_Q, BH * N * D);
            fill_rand(d_K, BH * N * D);
            fill_rand(d_V, BH * N * D);

            AttentionParams p;
            p.batch    = batch;
            p.n_heads  = n_heads;
            p.seq_len  = sl;
            p.head_dim = hd;
            p.causal   = false;
            p.scale    = 1.0f / sqrtf((float)hd);

            float ms = benchmark_kernel([&]() { fn(d_Q, d_K, d_V, d_O, d_S, p); });

            long flops = 4L * batch * n_heads * N * N * D;
            long bytes = bytes_fn(batch, n_heads, N, D);

            printf("ms: %.4f flops: %ld bytes: %ld seq_len: %d head_dim: %d batch: %d n_heads: %d\n",
                   ms, flops, bytes, sl, hd, batch, n_heads);

            cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
            cudaFree(d_O); cudaFree(d_S);
        }
    }
}

static void run_bench(const char* kernel)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("GPU: %s\n", props.name);

    if (strcmp(kernel, "naive") == 0)
        bench_kernel("naive", attention_naive, naive_bytes);
    else {
        fprintf(stderr, "unknown kernel: %s\n", kernel);
        exit(1);
    }
}

// ── validate ─────────────────────────────────────────────────────────────

static void read_bin(const char* path, std::vector<float>& buf)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t got = fread(buf.data(), sizeof(float), buf.size(), f);
    fclose(f);
    if (got != buf.size()) {
        fprintf(stderr, "short read from %s: expected %zu got %zu\n", path, buf.size(), got);
        exit(1);
    }
}

static void run_validate(const char* kernel,
                         int seq_len, int head_dim, int batch, int n_heads,
                         const char* io_dir)
{
    int  BH           = batch * n_heads;
    long tensor_elems = (long)BH * seq_len * head_dim;

    std::vector<float> h_Q(tensor_elems), h_K(tensor_elems),
                       h_V(tensor_elems), h_O(tensor_elems);

    char path[512];
    snprintf(path, sizeof(path), "%s/Q.bin", io_dir); read_bin(path, h_Q);
    snprintf(path, sizeof(path), "%s/K.bin", io_dir); read_bin(path, h_K);
    snprintf(path, sizeof(path), "%s/V.bin", io_dir); read_bin(path, h_V);

    float *d_Q, *d_K, *d_V, *d_O, *d_S;
    cudaMalloc(&d_Q, tensor_elems * sizeof(float));
    cudaMalloc(&d_K, tensor_elems * sizeof(float));
    cudaMalloc(&d_V, tensor_elems * sizeof(float));
    cudaMalloc(&d_O, tensor_elems * sizeof(float));
    cudaMalloc(&d_S, (long)BH * seq_len * seq_len * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), tensor_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), tensor_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), tensor_elems * sizeof(float), cudaMemcpyHostToDevice);

    AttentionParams p;
    p.batch    = batch;
    p.n_heads  = n_heads;
    p.seq_len  = seq_len;
    p.head_dim = head_dim;
    p.causal   = false;
    p.scale    = 1.0f / sqrtf((float)head_dim);

    if (strcmp(kernel, "naive") == 0) {
        attention_naive(d_Q, d_K, d_V, d_O, d_S, p);
    } else {
        fprintf(stderr, "unknown kernel: %s\n", kernel);
        exit(1);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_O.data(), d_O, tensor_elems * sizeof(float), cudaMemcpyDeviceToHost);

    snprintf(path, sizeof(path), "%s/O.bin", io_dir);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite(h_O.data(), sizeof(float), h_O.size(), f);
    fclose(f);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_S);
}

// ── entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        if (argc < 3) {
            fprintf(stderr, "usage: flash_attn bench <kernel>\n");
            return 1;
        }
        run_bench(argv[2]);
    } else if (argc > 1 && strcmp(argv[1], "validate") == 0) {
        if (argc < 8) {
            fprintf(stderr,
                "usage: flash_attn validate <kernel> <seq_len> <head_dim> "
                "<batch> <n_heads> <io_dir>\n");
            return 1;
        }
        run_validate(argv[2], atoi(argv[3]), atoi(argv[4]),
                     atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else {
        fprintf(stderr, "usage: flash_attn bench <kernel>\n"
                        "       flash_attn validate <kernel> <seq_len> <head_dim> <batch> <n_heads> <io_dir>\n");
        return 1;
    }
    return 0;
}
