#include "gpt2.h"
#include "gpt2_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  gpt2 --verify-weights [bin] [manifest_out]\n"
        "       defaults: bin=assets/gpt2_weights.bin manifest_out=assets/gpt2_weights.loader.json\n"
        "  gpt2 --test-kernels [fixtures_dir]\n"
        "       default: fixtures_dir=assets/fixtures\n");
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

    usage();
    return 1;
}
