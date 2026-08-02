#include "gpt2_kernels.h"
#include "cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

    std::vector<float> load_floats(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "cannot open fixture %s\n", path.c_str());
            std::exit(1);
        }

        std::fseek(f, 0, SEEK_END);
        const long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<float> v(size_t(bytes) / sizeof(float));

        if (std::fread(v.data(), sizeof(float), v.size(), f) != v.size()) {
            std::fprintf(stderr, "short read on %s\n", path.c_str());
            std::exit(1);
        }

        std::fclose(f);
        return v;
    }

    float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
        float worst = 0.0f;
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i) {
            const float d = std::fabs(a[i] - b[i]);
            if (std::isnan(d)) {
                return INFINITY;
            }
            if (d > worst) {
                worst = d;
            }
        }
        return worst;
    }

    std::vector<int> load_ints(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "cannot open fixture %s\n", path.c_str());
            std::exit(1);
        }

        std::fseek(f, 0, SEEK_END);
        const long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<int> v(size_t(bytes) / sizeof(int));

        if (std::fread(v.data(), sizeof(int), v.size(), f) != v.size()) {
            std::fprintf(stderr, "short read on %s\n", path.c_str());
            std::exit(1);
        }

        std::fclose(f);
        return v;
    }

    float* to_device(const std::vector<float>& h) {
        float* d = nullptr;
        cuda_check(cudaMalloc(&d, h.size() * sizeof(float)));
        cuda_check(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    }

    int* to_device(const std::vector<int>& h) {
        int* d = nullptr;
        cuda_check(cudaMalloc(&d, h.size() * sizeof(int)));
        cuda_check(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
        return d;
    }

    std::vector<float> to_host(const float* d, size_t n) {
        std::vector<float> h(n);
        cuda_check(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }

    bool report(const char* name, float diff, float tol) {
        const bool pass = diff <= tol;
        std::printf("%-16s max_abs_diff %.3e   %s\n", name, diff, pass ? "PASS" : "FAIL");
        return pass;
    }

    bool test_residual_add(const std::string& dir) {
        auto x = load_floats(dir + "/residual_add_x.bin");
        auto y = load_floats(dir + "/residual_add_y.bin");
        auto ref = load_floats(dir + "/residual_add_out.bin");
        float* dx = to_device(x);
        float* dy = to_device(y);

        gpt2::residual_add(dx, dy, int(x.size()));
        cuda_check(cudaDeviceSynchronize());
        auto out = to_host(dx, x.size());

        cudaFree(dx);
        cudaFree(dy);

        return report("residual_add", max_abs_diff(out, ref), 1e-5f);
    }

    bool test_bias_add(const std::string& dir) {
        auto x = load_floats(dir + "/bias_add_x.bin");
        auto bias = load_floats(dir + "/bias_add_bias.bin");
        auto ref = load_floats(dir + "/bias_add_out.bin");
        const int cols = int(bias.size());
        const int rows = int(x.size() / bias.size());
        float* dx = to_device(x);
        float* db = to_device(bias);

        gpt2::bias_add(dx, db, rows, cols);
        cuda_check(cudaDeviceSynchronize());
        auto out = to_host(dx, x.size());

        cudaFree(dx);
        cudaFree(db);

        return report("bias_add", max_abs_diff(out, ref), 1e-5f);
    }

    bool test_gelu_new(const std::string& dir) {
        auto x = load_floats(dir + "/gelu_new_in.bin");
        auto ref = load_floats(dir + "/gelu_new_out.bin");
        float* dx = to_device(x);

        gpt2::gelu_new(dx, int(x.size()));
        cuda_check(cudaDeviceSynchronize());
        auto out = to_host(dx, x.size());

        cudaFree(dx);

        return report("gelu_new", max_abs_diff(out, ref), 1e-5f);
    }

    bool test_embedding_gather(const std::string& dir) {
        auto ids = load_ints(dir + "/embedding_gather_ids.bin");
        auto wte = load_floats(dir + "/embedding_gather_wte.bin");
        auto wpe = load_floats(dir + "/embedding_gather_wpe.bin");
        auto ref = load_floats(dir + "/embedding_gather_out.bin");
        const int n = int(ids.size());
        const int d = int(ref.size() / ids.size());   // out is [n, d]

        int* d_ids = to_device(ids);
        float* d_wte = to_device(wte);
        float* d_wpe = to_device(wpe);
        float* d_out = nullptr;
        cuda_check(cudaMalloc(&d_out, ref.size() * sizeof(float)));

        gpt2::embedding_gather(d_ids, d_wte, d_wpe, d_out, n, d);
        cuda_check(cudaDeviceSynchronize());
        auto out = to_host(d_out, ref.size());

        cudaFree(d_ids);
        cudaFree(d_wte);
        cudaFree(d_wpe);
        cudaFree(d_out);

        return report("embedding_gather", max_abs_diff(out, ref), 1e-5f);
    }

    bool test_qkv_split(const std::string& dir) {
        auto qkv  = load_floats(dir + "/qkv_split_in.bin");
        auto refQ = load_floats(dir + "/qkv_split_Q.bin");
        auto refK = load_floats(dir + "/qkv_split_K.bin");
        auto refV = load_floats(dir + "/qkv_split_V.bin");

        const int h = 12, d = 64, n_pad = 64;
        const int n_real = int(qkv.size() / (3 * h * d));
        const size_t buf = size_t(h) * n_pad * d; // floats per Q/K/V buffer

        float* d_qkv = to_device(qkv);

        float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
        cuda_check(cudaMalloc(&d_Q, buf * sizeof(float)));
        cuda_check(cudaMalloc(&d_K, buf * sizeof(float)));
        cuda_check(cudaMalloc(&d_V, buf * sizeof(float)));
        cuda_check(cudaMemset(d_Q, 0, buf * sizeof(float)));
        cuda_check(cudaMemset(d_K, 0, buf * sizeof(float)));
        cuda_check(cudaMemset(d_V, 0, buf * sizeof(float)));

        gpt2::qkv_split(d_qkv, d_Q, d_K, d_V, n_real, n_pad, h, d);
        cuda_check(cudaDeviceSynchronize());
        auto Q = to_host(d_Q, buf);
        auto K = to_host(d_K, buf);
        auto V = to_host(d_V, buf);

        cudaFree(d_qkv);
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);

        float diff = max_abs_diff(Q, refQ);
        diff = std::max(diff, max_abs_diff(K, refK));
        diff = std::max(diff, max_abs_diff(V, refV));
        return report("qkv_split", diff, 1e-5f);
    }

    bool test_merge_heads(const std::string& dir) {
        auto in  = load_floats(dir + "/merge_heads_in.bin");
        auto ref = load_floats(dir + "/merge_heads_out.bin");

        const int h = 12, d = 64, n_pad = 64;
        const int n_real = int(ref.size() / (h * d));

        float* d_in = to_device(in);
        float* d_out = nullptr;
        cuda_check(cudaMalloc(&d_out, ref.size() * sizeof(float)));

        gpt2::merge_heads(d_in, d_out, n_real, n_pad, h, d);
        cuda_check(cudaDeviceSynchronize());
        auto out = to_host(d_out, ref.size());

        cudaFree(d_in);
        cudaFree(d_out);

        return report("merge_heads", max_abs_diff(out, ref), 1e-5f);
    }

}

int gpt2_test_kernels(const char* fixtures_dir) {
    const std::string dir = fixtures_dir;
    int failed = 0;
    failed += !test_residual_add(dir);
    failed += !test_bias_add(dir);
    failed += !test_gelu_new(dir);
    failed += !test_embedding_gather(dir);
    failed += !test_qkv_split(dir);
    failed += !test_merge_heads(dir);
    std::printf("\n%d ops failed\n", failed);
    return failed;
}
