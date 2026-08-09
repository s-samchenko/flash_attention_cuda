#pragma once
#include "gpt2.hpp"
#include "gpt2_model.hpp"

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

namespace gpt2 {

    constexpr int kEotToken = 50256;

    struct TokenTable {
        std::vector<uint32_t> offsets;
        std::vector<unsigned char> blob;

        std::string_view bytes(int id) const {
            const uint32_t s = offsets[id], e = offsets[id + 1];
            return {reinterpret_cast<const char*>(blob.data()) + s, e - s};
        }
    };

    TokenTable load_token_table(const char* path);

    struct GenConfig {
        int max_new = 100;
        float temp = 0.8f;
        int top_k = 40;   // <= 0 disables top-k truncation
        unsigned seed = 42;
        int delay_ms = 0; // per-token pause; 0 = full speed.
    };

    int sample_greedy(const float* logits, int vocab);
    int sample_topk(const float* logits, int vocab, float temp, int k, std::mt19937& rng);

    // Re-prefill generation. Appends sampled ids to `tokens` and streams their bytes to stdout.
    // greedy == true ignores temp/top_k (deterministic).
    void generate(const GPT2Weights& w, Activations& a, const TokenTable& tok,
                  std::vector<int>& tokens, const GenConfig& cfg, bool greedy);

}
