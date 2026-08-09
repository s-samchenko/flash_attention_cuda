#include "gpt2_generate.hpp"
#include "cuda_utils.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>

namespace gpt2 {

    TokenTable load_token_table(const char* path) {
        FILE* f = std::fopen(path, "rb");
        if (!f) {
            std::fprintf(stderr, "cannot open token table %s\n", path);
            std::exit(1);
        }

        uint32_t count = 0;
        if (std::fread(&count, sizeof(count), 1, f) != 1) {
            std::fprintf(stderr, "%s: truncated header\n", path);
            std::exit(1);
        }

        TokenTable t;
        t.offsets.resize(size_t(count) + 1);
        if (std::fread(t.offsets.data(), sizeof(uint32_t), t.offsets.size(), f) != t.offsets.size()) {
            std::fprintf(stderr, "%s: truncated offsets\n", path);
            std::exit(1);
        }

        t.blob.resize(t.offsets.back());
        if (std::fread(t.blob.data(), 1, t.blob.size(), f) != t.blob.size()) {
            std::fprintf(stderr, "%s: truncated blob\n", path);
            std::exit(1);
        }

        std::fclose(f);
        return t;
    }

    int sample_greedy(const float* logits, int vocab) {
        return int(std::max_element(logits, logits + vocab) - logits);
    }

    int sample_topk(const float* logits, int vocab, float temp, int k, std::mt19937& rng) {
        const int kk = (k <= 0 || k > vocab) ? vocab : k;

        std::vector<int> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });

        const float maxv = logits[idx[0]] / temp;
        std::vector<float> probs(kk);
        float sum = 0.f;
        for (int i = 0; i < kk; ++i) {
            probs[i] = expf(logits[idx[i]] / temp - maxv);
            sum += probs[i];
        }

        std::uniform_real_distribution<float> u(0.f, 1.f);
        const float r = u(rng) * sum;
        float cdf = 0.f;
        for (int i = 0; i < kk; ++i) {
            cdf += probs[i];
            if (cdf >= r) {
                return idx[i];
            }
        }

        return idx[kk - 1];
    }

    void generate(const GPT2Weights& w, Activations& a, const TokenTable& tok,
                  std::vector<int>& tokens, const GenConfig& cfg, bool greedy) {
        const int vocab = w.config.vocab;
        std::mt19937 rng(cfg.seed);
        std::vector<float> logits(vocab);

        for (int s = 0; s < cfg.max_new && int(tokens.size()) < 1024; ++s) {
            const int n_real = int(tokens.size());
            gpt2_forward(w, a, tokens.data(), n_real);
            lm_head(w, a, n_real, /*all_positions=*/false);
            cuda_check(cudaMemcpy(logits.data(), a.logits, size_t(vocab) * sizeof(float), cudaMemcpyDeviceToHost));

            const int next = greedy ? sample_greedy(logits.data(), vocab)
                                    : sample_topk(logits.data(), vocab, cfg.temp, cfg.top_k, rng);
            if (next == kEotToken) {
                break;
            }

            tokens.push_back(next);
            const std::string_view b = tok.bytes(next);
            std::fwrite(b.data(), 1, b.size(), stdout);
            std::fflush(stdout);

            if (cfg.delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.delay_ms));
            }
        }
    }

}
