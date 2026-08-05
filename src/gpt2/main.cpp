#include "gpt2.hpp"
#include "gpt2_kernels.hpp"
#include "gpt2_model.hpp"
#include "cuda_utils.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

// Token ids for "The quick brown fox" under the gpt2 BPE encoding (assets/ref/meta.json)
constexpr std::array<int, 4> kRefPromptIds = {464, 2068, 7586, 21831};

static void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  gpt2 --verify-weights [bin] [manifest_out]\n"
        "       defaults: bin=assets/gpt2_weights.bin manifest_out=assets/gpt2_weights.loader.json\n"
        "  gpt2 --test-kernels [fixtures_dir]\n"
        "       default: fixtures_dir=assets/fixtures\n"
        "  gpt2 --forward [bin]\n"
        "       runs embedding + 12 blocks on the ref prompt and prints residual-stream stats\n"
        "  gpt2 --dump-activations [bin] [out_dir]\n"
        "       runs the ref prompt and dumps x_emb, x_block{NN}, and block-0 internals\n"
        "       default: out_dir=assets/dump\n");
}

static int cmd_forward(const char* bin) {
    const int n_real = int(kRefPromptIds.size());

    GPT2Weights w{};
    gpt2_load(bin, w);

    gpt2::Activations a{};
    gpt2::activations_init(a);
    gpt2::gpt2_forward(w, a, kRefPromptIds.data(), n_real);
    cuda_check(cudaDeviceSynchronize());

    std::vector<float> x(size_t(n_real) * 768);
    cuda_check(cudaMemcpy(x.data(), a.x, x.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float lo = x[0], hi = x[0];
    double sum = 0.0;
    int nans = 0;
    for (float v : x) {
        if (std::isnan(v) || std::isinf(v)) {
            ++nans;
            continue;
        }
        lo = v < lo ? v : lo;
        hi = v > hi ? v : hi;
        sum += v;
    }
    std::printf("forward ok: n_real=%d  min=%.4f  max=%.4f  mean=%.6f  non-finite=%d\n",
                n_real, lo, hi, sum / x.size(), nans);

    gpt2::activations_free(a);
    gpt2_free(w);
    return nans == 0 ? 0 : 1;
}

static int cmd_dump(const char* bin, const char* out_dir) {
    const int n_real = int(kRefPromptIds.size());
    std::filesystem::create_directories(out_dir);

    GPT2Weights w{};
    gpt2_load(bin, w);

    gpt2::Activations a{};
    gpt2::activations_init(a);
    gpt2::gpt2_forward(w, a, kRefPromptIds.data(), n_real, out_dir);
    cuda_check(cudaDeviceSynchronize());

    std::printf("dumped activations for n_real=%d to %s/\n", n_real, out_dir);

    gpt2::activations_free(a);
    gpt2_free(w);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    if (std::strcmp(argv[1], "--verify-weights") == 0) {
        const char* bin = (argc > 2) ? argv[2] : "assets/gpt2_weights.bin";
        const char* out = (argc > 3) ? argv[3] : "assets/gpt2_weights.loader.json";
        GPT2Weights w{};
        gpt2_load(bin, w, out);
        std::fprintf(stderr, "loaded %zu bytes into device arena; wrote manifest to %s\n", w.arena_bytes, out);
        gpt2_free(w);
        return 0;
    }

    if (std::strcmp(argv[1], "--test-kernels") == 0) {
        const char* dir = (argc > 2) ? argv[2] : "assets/fixtures";
        return gpt2_test_kernels(dir);
    }

    if (std::strcmp(argv[1], "--forward") == 0) {
        const char* bin = (argc > 2) ? argv[2] : "assets/gpt2_weights.bin";
        return cmd_forward(bin);
    }

    if (std::strcmp(argv[1], "--dump-activations") == 0) {
        const char* bin = (argc > 2) ? argv[2] : "assets/gpt2_weights.bin";
        const char* dir = (argc > 3) ? argv[3] : "assets/dump";
        return cmd_dump(bin, dir);
    }

    usage();
    return 1;
}
