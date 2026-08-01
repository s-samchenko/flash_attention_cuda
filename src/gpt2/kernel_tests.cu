#include "gpt2_kernels.h"
#include "cuda_utils.cuh"

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

    float* to_device(const std::vector<float>& h) {
        float* d = nullptr;
        cuda_check(cudaMalloc(&d, h.size() * sizeof(float)));
        cuda_check(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
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

}

int gpt2_test_kernels(const char* fixtures_dir) {
    const std::string dir = fixtures_dir;
    int failed = 0;
    failed += !test_residual_add(dir);
    failed += !test_bias_add(dir);
    failed += !test_gelu_new(dir);
    std::printf("\n%d ops failed\n", failed);
    return failed;
}
