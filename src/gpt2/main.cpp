#include "gpt2.hpp"
#include "gpt2_kernels.hpp"
#include "gpt2_model.hpp"
#include "gpt2_generate.hpp"
#include "cuda_utils.cuh"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <random>
#include <string_view>
#include <vector>

// Token ids for "The quick brown fox" under the gpt2 BPE encoding (assets/ref/fox/meta.json)
constexpr std::array<int, 4> kRefPromptIds = {464, 2068, 7586, 21831};

static std::vector<int> parse_ids(std::string_view csv) {
    std::vector<int> ids;
    while (!csv.empty()) {
        const auto comma = csv.find(',');
        std::string_view tok = csv.substr(0, comma);
        while (!tok.empty() && tok.front() == ' ') {
            tok.remove_prefix(1);
        }
        while (!tok.empty() && tok.back() == ' ') {
            tok.remove_suffix(1);
        }

        int value{};
        const auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), value);
        if (ec != std::errc{} || ptr != tok.data() + tok.size()) {
            std::fprintf(stderr, "invalid token id: '%.*s'\n", int(tok.size()), tok.data());
            std::exit(1);
        }
        ids.push_back(value);

        if (comma == std::string_view::npos) {
            break;
        }
        csv.remove_prefix(comma + 1);
    }
    return ids;
}

static void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  gpt2 --verify-weights [bin] [manifest_out]\n"
        "       defaults: bin=assets/gpt2_weights.bin manifest_out=assets/gpt2_weights.loader.json\n"
        "  gpt2 --test-kernels [fixtures_dir]\n"
        "       default: fixtures_dir=assets/fixtures\n"
        "  gpt2 --forward [bin]\n"
        "       runs embedding + 12 blocks on the ref prompt and prints residual-stream stats\n"
        "  gpt2 --dump-activations [bin] [--tokens id,id,...] [--out dir]\n"
        "       runs a prompt and dumps x_emb, x_block{NN}, block-0 internals,\n"
        "       x_lnf, and logits. defaults: the fox prompt, out=assets/dump\n"
        "  gpt2 --tokens id,id,... --generate N [--temp T] [--top-k K] [--seed S] [--delay MS | --maxspeed]\n"
        "       re-prefill generation, streams token bytes to stdout.\n"
        "       temp<=0 is greedy. defaults: temp=0.8 top-k=40 delay=30ms\n"
        "       no --seed means a fresh random seed each run; pass --seed to pin it\n"
        "       --maxspeed disables the per-token pause.\n"
        "       --time prints tokens/s of the generation loop to stderr (implies --maxspeed)\n"
        "  gpt2 --selftest\n"
        "       greedy generation on the fox prompt; checks reproducibility\n");
}

static int cmd_forward(const char* bin) {
    const int n_real = int(kRefPromptIds.size());

    GPT2Weights w{};
    gpt2_load(bin, w);

    gpt2::Activations a{};
    gpt2::activations_init(a, w.config, n_real);
    gpt2::gpt2_forward(w, a, kRefPromptIds.data(), n_real);
    cuda_check(cudaDeviceSynchronize());

    std::vector<float> x(size_t(n_real) * w.config.d_model);
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

static int cmd_dump(const char* bin, const char* out_dir, const std::vector<int>& ids) {
    const int n_real = int(ids.size());
    if (n_real < 1 || n_real > 1024) {
        std::fprintf(stderr, "prompt length %d out of range [1, 1024]\n", n_real);
        return 1;
    }
    std::filesystem::create_directories(out_dir);

    GPT2Weights w{};
    gpt2_load(bin, w);

    gpt2::Activations a{};
    gpt2::activations_init(a, w.config, n_real);
    gpt2::gpt2_forward(w, a, ids.data(), n_real, out_dir);
    gpt2::lm_head(w, a, n_real, /*all_positions=*/true, out_dir);
    cuda_check(cudaDeviceSynchronize());

    std::printf("dumped activations for n_real=%d to %s/\n", n_real, out_dir);

    gpt2::activations_free(a);
    gpt2_free(w);
    return 0;
}

static int cmd_generate(int argc, char** argv) {
    const char* bin = "assets/gpt2_weights.bin";
    const char* tokens_bin = "assets/gpt2_tokens.bin";
    std::vector<int> ids;
    gpt2::GenConfig cfg;
    cfg.delay_ms = 30;
    bool maxspeed = false;
    bool seed_set = false;
    bool time_it = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            ids = parse_ids(argv[++i]);
        } else if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            cfg.max_new = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            cfg.temp = float(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            cfg.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = unsigned(std::strtoul(argv[++i], nullptr, 10));
            seed_set = true;
        } else if (std::strcmp(argv[i], "--delay") == 0 && i + 1 < argc) {
            cfg.delay_ms = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--maxspeed") == 0) {
            maxspeed = true;
        } else if (std::strcmp(argv[i], "--time") == 0) {
            time_it = true;
        } else if (std::strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            bin = argv[++i];
        }
    }
    if (maxspeed || time_it) {
        cfg.delay_ms = 0; // never benchmark a paced run
    }
    if (!seed_set) {
        cfg.seed = std::random_device{}();
    }

    if (ids.empty()) {
        std::fprintf(stderr, "generation needs --tokens id,id,...  (see python/encode.py)\n");
        return 1;
    }
    if (int(ids.size()) >= 1024) {
        std::fprintf(stderr, "prompt already at context limit (%zu tokens)\n", ids.size());
        return 1;
    }

    GPT2Weights w{};
    gpt2_load(bin, w);
    gpt2::TokenTable tok = gpt2::load_token_table(tokens_bin);

    gpt2::Activations a{};
    gpt2::activations_init(a, w.config, /*logit_rows=*/1);

    std::vector<int> tokens = ids;
    const bool greedy = cfg.temp <= 0.0f;

    cuda_check(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    gpt2::generate(w, a, tok, tokens, cfg, greedy);
    cuda_check(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    std::putchar('\n');

    if (time_it) {
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int generated = int(tokens.size()) - int(ids.size());
        std::fprintf(stderr,
            "prompt=%d generated=%d ctx=%d  total=%.1fms  %.2f ms/token  %.1f tok/s\n",
            int(ids.size()), generated, int(tokens.size()), ms,
            generated ? ms / generated : 0.0,
            generated ? generated * 1000.0 / ms : 0.0);
    }

    gpt2::activations_free(a);
    gpt2_free(w);
    return 0;
}

static int cmd_selftest() {
    GPT2Weights w{};
    gpt2_load("assets/gpt2_weights.bin", w);
    gpt2::TokenTable tok = gpt2::load_token_table("assets/gpt2_tokens.bin");

    gpt2::Activations a{};
    gpt2::activations_init(a, w.config, /*logit_rows=*/1);

    gpt2::GenConfig cfg;
    cfg.max_new = 20;
    std::vector<int> tokens(kRefPromptIds.begin(), kRefPromptIds.end());
    gpt2::generate(w, a, tok, tokens, cfg, /*greedy=*/true);
    cuda_check(cudaDeviceSynchronize());
    std::putchar('\n');

    std::printf("greedy ids:");
    for (int id : tokens) {
        std::printf(" %d", id);
    }
    std::putchar('\n');

    static const std::vector<int> expected = {
        464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 198, 198,
        464, 3105, 7586, 21831, 279, 18058, 510, 465, 7721, 290, 29959, 465,
    };

    int rc = 0;
    if (expected.empty()) {
        std::printf("selftest: no pinned sequence yet (paste greedy ids into main.cpp)\n");
    } else if (tokens != expected) {
        std::fprintf(stderr, "selftest FAILED: greedy output drifted from pinned sequence\n");
        rc = 1;
    } else {
        std::printf("selftest OK: greedy reproduces pinned sequence\n");
    }

    gpt2::activations_free(a);
    gpt2_free(w);
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) {
            return cmd_selftest();
        }
        if (std::strcmp(argv[i], "--generate") == 0) {
            return cmd_generate(argc, argv);
        }
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
        const char* bin = "assets/gpt2_weights.bin";
        const char* dir = "assets/dump";
        std::vector<int> ids;
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
                ids = parse_ids(argv[++i]);
            } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                dir = argv[++i];
            } else if (argv[i][0] != '-') {
                bin = argv[i];
            }
        }
        if (ids.empty()) {
            ids.assign(kRefPromptIds.begin(), kRefPromptIds.end());
        }
        return cmd_dump(bin, dir, ids);
    }

    usage();
    return 1;
}
