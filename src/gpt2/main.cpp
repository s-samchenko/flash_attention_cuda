#include "gpt2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void usage() {
    std::fprintf(stderr,
        "usage: gpt2 --verify-weights [bin] [manifest_out]\n"
        "defaults: bin=assets/gpt2_weights.bin\n"
        "manifest_out=assets/gpt2_weights.loader.json\n");
}

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--verify-weights") != 0) {
        usage();
        return 1;
    }
    const char* bin = (argc > 2) ? argv[2] : "assets/gpt2_weights.bin";
    const char* out = (argc > 3) ? argv[3] : "assets/gpt2_weights.loader.json";

    GPT2Weights w{};
    gpt2_load(bin, w, out);
    std::fprintf(stderr, "loaded %zu bytes into device arena; wrote manifest to %s\n", w.arena_bytes, out);
    gpt2_free(w);
    return 0;
}
