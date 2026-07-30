#include "gpt2.h"
#include "cuda_utils.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

    constexpr uint32_t EXPECTED_MAGIC = 0x47505432; // "GPT2"
    constexpr uint32_t EXPECTED_VERSION = 1;

    struct Header {
        uint32_t magic;
        uint32_t version;
        int32_t n_layer, n_head, d_model, n_ctx, vocab, pad;
    };
    static_assert(sizeof(Header) == 32, "GPT-2 weights file header must be 32 bytes");

    struct TensorDesc {
        std::string key;
        std::vector<int> shape;
        size_t count() const {
            size_t n = 1;
            for (int d : shape) n *= size_t(d);
            return n;
        }
    };

    std::vector<TensorDesc> tensor_descs(const GPT2Config& c) {
        std::vector<TensorDesc> t;
        t.push_back({"transformer.wte.weight", {c.vocab, c.d_model}});
        t.push_back({"transformer.wpe.weight", {c.n_ctx, c.d_model}});
        for (int i = 0; i < c.n_layer; ++i) {
            std::string p = "transformer.h." + std::to_string(i) + ".";
            t.push_back({p + "ln_1.weight", {c.d_model}});
            t.push_back({p + "ln_1.bias", {c.d_model}});
            t.push_back({p + "attn.c_attn.weight", {c.d_model, 3 * c.d_model}});
            t.push_back({p + "attn.c_attn.bias", {3 * c.d_model}});
            t.push_back({p + "attn.c_proj.weight", {c.d_model, c.d_model}});
            t.push_back({p + "attn.c_proj.bias", {c.d_model}});
            t.push_back({p + "ln_2.weight", {c.d_model}});
            t.push_back({p + "ln_2.bias", {c.d_model}});
            t.push_back({p + "mlp.c_fc.weight", {c.d_model, 4 * c.d_model}});
            t.push_back({p + "mlp.c_fc.bias", {4 * c.d_model}});
            t.push_back({p + "mlp.c_proj.weight", {4 * c.d_model, c.d_model}});
            t.push_back({p + "mlp.c_proj.bias", {c.d_model}});
        }
        t.push_back({"transformer.ln_f.weight", {c.d_model}});
        t.push_back({"transformer.ln_f.bias", {c.d_model}});
        return t;
    }

    void write_manifest(const char* path, const Header& h, const float* buf, const std::vector<TensorDesc>& descs) {
        FILE* f = std::fopen(path, "w");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", path);
            std::exit(1);
        }

        std::fprintf(f, "{\n");
        std::fprintf(f, "  \"magic\": %u,\n", h.magic);
        std::fprintf(f, "  \"version\": %u,\n", h.version);
        std::fprintf(f,
            "  \"config\": {\"n_layer\": %d, \"n_head\": %d, \"d_model\": %d, "
            "\"n_ctx\": %d, \"vocab\": %d},\n",
            h.n_layer, h.n_head, h.d_model, h.n_ctx, h.vocab);
        std::fprintf(f, "  \"header_bytes\": %zu,\n", sizeof(Header));
        std::fprintf(f, "  \"tensors\": [\n");

        size_t off = 0;
        for (size_t i = 0; i < descs.size(); ++i) {
            const auto& d = descs[i];
            const size_t n = d.count();
            const float* p = buf + off;

            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += double(p[k]);
            }

            const double mean = sum / double(n);
            const double first = double(p[0]);
            const double last = double(p[n - 1]);

            std::fprintf(f, "    {\"shape\": [");
            for (size_t s = 0; s < d.shape.size(); ++s) {
                std::fprintf(f, "%s%d", s ? ", " : "", d.shape[s]);
            }
            std::fprintf(f, "], \"sum\": %.17g, \"mean\": %.17g, "
                            "\"first\": %.17g, \"last\": %.17g, \"key\": \"%s\"}%s\n",
                            sum, mean, first, last, d.key.c_str(),
                            i + 1 == descs.size() ? "" : ",");

            off += n;
        }

        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
    }

}

void gpt2_load(const char* bin_path, GPT2Weights& out, const char* manifest_out_path) {
    FILE* f = std::fopen(bin_path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", bin_path);
        std::exit(1);
    }

    Header h;
    if (std::fread(&h, sizeof(h), 1, f) != 1) {
        std::fprintf(stderr, "short read on header\n");
        std::exit(1);
    }
    if (h.magic != EXPECTED_MAGIC) {
        std::fprintf(stderr, "bad magic 0x%08X (expected 0x%08X)\n", h.magic, EXPECTED_MAGIC);
        std::exit(1);
    }
    if (h.version != EXPECTED_VERSION) {
        std::fprintf(stderr, "unsupported version %u (expected %u)\n", h.version, EXPECTED_VERSION);
        std::exit(1);
    }
    out.config = {h.n_layer, h.n_head, h.d_model, h.n_ctx, h.vocab};

    const auto descs = tensor_descs(out.config);
    size_t total_floats = 0;
    for (const auto& d : descs) total_floats += d.count();
    const size_t total_bytes = total_floats * sizeof(float);
.
    std::fseek(f, 0, SEEK_END);
    const long file_size = std::ftell(f);
    std::fseek(f, sizeof(Header), SEEK_SET);
    const size_t expected_size = sizeof(Header) + total_bytes;
    if (size_t(file_size) != expected_size) {
        std::fprintf(stderr, "size mismatch: file=%ld expected=%zu\n", file_size, expected_size);
        std::exit(1);
    }

    std::vector<float> host(total_floats);
    if (std::fread(host.data(), sizeof(float), total_floats, f) != total_floats) {
        std::fprintf(stderr, "short read on tensor data\n");
        std::exit(1);
    }
    std::fclose(f);

    if (manifest_out_path) {
        write_manifest(manifest_out_path, h, host.data(), descs);
    }

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&out.arena), total_bytes));
    out.arena_bytes = total_bytes;
    cuda_check(cudaMemcpy(out.arena, host.data(), total_bytes, cudaMemcpyHostToDevice));

    size_t off = 0;
    auto slice = [&](size_t n) -> float* {
        float* p = out.arena + off;
        off += n;
        return p;
    };

    const int d = h.d_model;
    out.wte = slice(size_t(h.vocab) * d);
    out.wpe = slice(size_t(h.n_ctx) * d);
    for (int i = 0; i < h.n_layer; ++i) {
        auto& b = out.h[i];
        b.ln1_w = slice(d);
        b.ln1_b = slice(d);
        b.attn_qkv_w = slice(size_t(d) * 3 * d);
        b.attn_qkv_b = slice(3 * d);
        b.attn_proj_w = slice(size_t(d) * d);
        b.attn_proj_b = slice(d);
        b.ln2_w = slice(d);
        b.ln2_b = slice(d);
        b.mlp_fc_w = slice(size_t(d) * 4 * d);
        b.mlp_fc_b = slice(4 * d);
        b.mlp_proj_w = slice(size_t(4 * d) * d);
        b.mlp_proj_b = slice(d);
    }
    out.lnf_w = slice(d);
    out.lnf_b = slice(d);

    if (off != total_floats) {
        std::fprintf(stderr, "internal carve mismatch: %zu vs %zu\n", off, total_floats);
        std::exit(1);
    }
}

void gpt2_free(GPT2Weights& w) {
    if (w.arena) {
        cudaFree(w.arena);
        w.arena = nullptr;
    }
    w.arena_bytes = 0;
}
