# GPT-2 on a Hand-Written Flash Attention 2 Kernel

A Flash Attention 2 forward pass written from scratch in CUDA (FP16, WMMA tensor cores),
built through five progressively optimized kernel versions and then wired into a full GPT-2
inference path validated against HuggingFace at logit level.

**Headline results (RTX 3090, sm_86):**

- Final kernel: **20.5 TFLOPS** at seq=4096, head_dim=64. That is 42x over the naive baseline
  and roughly 40-43% of PyTorch SDPA's flash backend at hd=64.
- End-to-end text generation with GPT-2 XL (1.5B), streamed token by token, on a network
  whose attention is entirely this repo's kernel. Everything around the attention (QKV split,
  head merge, LayerNorm, GELU, embeddings) is also hand-written CUDA; the dense projections
  use cuBLAS.
- Output validated against HuggingFace at logit level, with argmax agreement at 1000/1000
  positions.

![Throughput vs sequence length](benchmarks%2Fplots%2Fperfomance-comparison.png)

| Kernel        | seq=4096, hd=64 |  TFLOPS | vs naive |
|---------------|-----------------|--------:|---------:|
| naive         | 67.0 ms         |   0.513 |     1.0× |
| fused_softmax | 14.2 ms         |   2.412 |     4.7× |
| fa1 (FP32)    |  4.66 ms        |   7.368 |    14.4× |
| fa2_tf32      |  5.69 ms        |   6.044 |    11.8× |
| fa2_fp16 v1   |  2.76 ms        |  12.459 |    24.3× |
| **fa2_fp16 v2** | **1.68 ms**   | **20.494** | **40.0×** |

Best single result: 42.1x over naive at seq=2048, hd=64. Full tables, all shapes, and dated
Nsight profiles are in [`benchmarks/results.md`](benchmarks/results.md).

Almost every from-scratch GPT implementation imports its attention, either `flash_attn` or
`F.scaled_dot_product_attention`. It gets outsourced because it is the hard part. This project
is that part, built up from a three-kernel naive implementation, plus the inference machinery
to prove it works inside a real transformer.

---

## The algorithm in brief

Attention is `O = softmax(QKᵀ / √d) · V`. Implemented directly, the N×N score matrix has to
cross HBM several times (64 MB per head at seq=4096), and the kernel becomes memory-bound
long before the ALUs matter.

Flash Attention's key insight is that softmax can be computed incrementally over K/V tiles,
so the N×N matrix never needs to exist in HBM. Each query row keeps a small amount of extra
state, O(N) in total: a running max `m`, a normalizer `l`, and an unnormalized output
accumulator. Whenever a new tile raises the max, the previous contributions are rescaled by
`exp(m_old − m_new)`. This cuts memory traffic from O(N²) to O(N·d), about 22x less at
seq=1024, d=64, and the result is exact attention, not an approximation. FA2 then adds the
work partitioning this implementation uses: one CUDA block per Q-tile per batch·head, split-Q
within the block so each warp owns its query rows with no cross-warp softmax reductions, and
normalization deferred to a single pass at the end.

Full derivations: [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) and
[FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691).
 
---

## Implementation overview

```
├── src/
│   ├── kernels/
│   │   ├── attention_naive.cu           v1  three kernels, S in HBM
│   │   ├── attention_fused_softmax.cu   v2  softmax+AV fused, S still in HBM
│   │   ├── attention_flash1.cu          v3  tiled online softmax, scalar FP32
│   │   ├── attention_flash2_tf32.cu     v4  FA2 work partitioning, TF32 tensor cores
│   │   ├── attention_flash2_fp16.cu     v5a FP16 WMMA (FP32 accumulators)
│   │   └── attention_flash2_fp16v2.cu   v5b same, with padded smem strides and causal masking
│   ├── main.cu                          benchmark / validation driver
│   └── gpt2/                            loader, cuBLAS GEMM wrappers, elementwise kernels, forward pass, sampling, CLI
├── include/                             headers + benchmark.cuh (cudaEvent timing)
├── python/
│   ├── validate.py                      kernel correctness vs PyTorch SDPA
│   ├── benchmark_compare.py             collection, results table, plots
│   ├── export_gpt2.py                   converts HF weights to the binary format the loader reads
│   ├── gpt2_logit_check.py              full-ladder logit validation vs HuggingFace
│   ├── gpt2_block_check.py              block-0 stage-by-stage divergence isolation
│   └── verify_manifest.py               loader round-trip check (export vs C++ reader)
├── benchmarks/results.md                append-only log: every run, dated, with roofline analysis and Nsight profiles
└── demo.sh                              takes a prompt and streams the generated text
```

**Final kernel structure (fp16v2, Br=64, Bc=32).** Each block handles one Q-tile for one
(batch, head), and its 4 warps each own a 16-row WMMA fragment strip.

For every K/V tile the kernel does three things. It computes QKᵀ as a chain of m16n16k16
`mma_sync` calls, applying the scale and the causal mask directly to the accumulator
fragments before storing S to shared memory. It then runs the online-softmax update in
scalar code, keeping m and l in registers and reducing across each row-pair with a single
`__shfl_xor`. Finally it feeds the resulting probabilities back through WMMA for PV,
accumulating O in FP32 fragments that stay in registers across all tiles.

The softmax writes those FP16 probabilities straight back into the buffer S just occupied.
By aliasing float to half, we cut the shared memory footprint in half and buy the occupancy
that makes v5b fast, at the cost of one `__syncwarp` to separate the read and write phases.
Everything numerically sensitive stays in FP32: the running max, the normalizer, and the
output accumulator. Only storage and matmul inputs are FP16. Causal masking skips
fully-masked K-tiles outright and applies −inf in-fragment on the diagonal tile.
 
---

## Optimization journey

Every step follows the same loop: profile with Nsight, create a hypothesis, make the
change, run `validate.py`, benchmark, then profile again to confirm the bottleneck actually
moved. Numbers below are seq=4096, hd=64 unless noted. The full sweep across
{512, 1024, 2048, 4096} × {64, 128} is in [`benchmarks/results.md`](benchmarks/results.md).

### v1: naive (0.51 TFLOPS, 1×)

Three kernels: QKᵀ, row-wise softmax, AV. The N×N matrix crosses HBM four times. The kernel is
memory-bound at 17–33 GB/s (2–4% of peak bandwidth) with an arithmetic intensity of 14–31
FLOP/B, below the 3090's 38 FLOP/B roofline crossover. Naive QKᵀ alone takes 86% of GPU time.
This version exists to make the problem measurable.

### v2: fused softmax (2.4 TFLOPS, 4.7×)

Softmax and AV fused into one kernel, with a register-blocked, float4-vectorized QKᵀ. S is now
written to HBM once and read once instead of four passes. Arithmetic intensity doubles and
bandwidth jumps to 42–115 GB/s. The new bottleneck is the AV loop, which still reads all of V
from HBM once per query row: O(N²·d) traffic that only tiling can remove. This is the reason
Flash Attention exists, measured on my own kernels.

### v3: FA1, FP32 scalar (7.4 TFLOPS, 14.4×)

Online softmax, Q/K/V tiled through shared memory, sequence-parallel grid. DRAM throughput
collapses to 0.64%: the N×N traffic is gone. The kernel flips from memory-bound to something
the DRAM/FLOP roofline does not model, the shared-memory MIO issue pipe (L1/TEX at 96%, 0.43
eligible warps per scheduler). Around 700 scalar 4-byte smem loads per K/V tile saturate the
pipe on instruction count alone. Padding away the 5.9-way bank conflicts changed nothing,
which showed that issue *rate* was the ceiling and motivated tensor cores: WMMA fragments
load the same tiles in far fewer, wider instructions.

One crossover worth knowing: FA1 only beats v2 past seq of roughly 1024–2048. At seq=512 the
fused kernel is faster, because the N×N matrix is small enough that per-tile rescaling
overhead costs more than it saves.

### v4: FA2 restructure + TF32 tensor cores (6.0 TFLOPS, 0.85× of FA1)

A negative result, kept in the ladder on purpose. Two things landed here: the FA2 split-Q warp
partitioning, and a first tensor-core port in TF32 (FP32-range numerics make for a forgiving
bring-up). It lost to scalar FA1. Keeping the FP32 O-accumulator in shared memory (chosen to
avoid depending on undocumented fragment lane layouts) plus FP32 Q/K/V staging pushed the
block to 40–72 KB of smem, capping occupancy at 1–2 blocks per SM: 8% achieved at hd=128
against FA1's 32%. TF32 tensor cores on GA102 also run at the same 35.6 TFLOPS as the plain
FP32 cores, so there was no arithmetic headroom to buy the occupancy loss back. The analysis
predicted FP16 would fix both problems at once, and it did.

### v5a: FP16 WMMA (12.5 TFLOPS, 24×)

Halving the element size fixed both v4 problems at once. Smem per block drops from 40 to
20.5 KB at hd=64, occupancy rises to 4 blocks per SM, and the mma K-step widens from 8 to 16,
halving the mma instruction count per tile. The arithmetic ceiling finally moves too: FP16
with FP32 accumulation runs at 71 TFLOPS on GA102 (the 142 TFLOPS headline halves when
accumulating in FP32, which softmax numerics require). Accumulators stay FP32, and validation
holds against SDPA-FP16 within its own noise floor.

Nsight on v5a showed 22.9-way average bank conflicts on shared loads, with 87.8% of load
wavefronts conflicted. Every smem row stride was a multiple of 128 B, so WMMA fragment loads
and the softmax kept landing on the same banks.

### v5b: de-aligned smem strides + causal (20.5 TFLOPS, 40–42×)

Pad the S and K/V row strides by a few elements so consecutive rows start in different banks,
and reorder the softmax read/write phases so the newly-aliased S/P buffer stays correct. At
hd=64 the L1-pipe busy share falls from 76% to 32%, IPC rises 2.5x, and the kernel gets 1.64x
faster end to end. At hd=128 it changes nothing: the profiler shows 16.05% achieved occupancy
against a 16.67% theoretical ceiling: the 33.8 KB smem footprint fits only 2 blocks per SM,
8 warps in total, which is too little parallel work to hide memory latency. The padding
fixed a problem hd=128 never had; its wall is the footprint itself.
 
---

## Roofline analysis

RTX 3090: 35.6 TFLOPS FP32, 71 TFLOPS FP16 tensor cores with FP32 accumulate, 936 GB/s. The
DRAM crossover sits at ~38 FLOP/B.

| Kernel | AI (FLOP/B) | Roofline bound | Achieved | Actual bound (Nsight) |
|---|---|---|---|---|
| naive | 14–31 | memory | 17–33 GB/s (2–4%) | N×N matrix in HBM, 4 passes |
| fused_softmax | 26–60 | memory | 42–115 GB/s (4–12%) | O(N²d) V re-reads |
| fa1 | 128–1024 | compute | 7.4 TFLOPS (20% FP32) | MIO smem issue pipe (not on the roofline) |
| fa2_tf32 | 128–1024 | compute | 6.0 TFLOPS (16% FP32) | occupancy (smem footprint) |
| fp16 v1 | 128–1024 | compute | 12.5 TFLOPS (17.5% of 71) | smem bank conflicts + occupancy |
| fp16 v2 | 128–1024 | compute | 20.5 TFLOPS (29% of 71) | hd=64: latency at 33% occupancy; hd=128: hard occupancy wall |

Past FA1, the classic DRAM/FLOP roofline stops being the binding
constraint. Every remaining bottleneck (MIO issue rate, bank-conflict replays, occupancy
ceilings) is an on-chip resource the roofline does not model and only the profiler exposes.
 
---

## GPT-2 integration

The kernel runs inside a complete GPT-2 XL (1.5B) forward pass: embeddings → 48× (LayerNorm →
QKV → **this repo's attention** → projection → residual → LayerNorm → MLP+GELU → residual) →
final LayerNorm → LM head. XL's attention shape, 25 heads at head_dim=64, lands on the
kernel's best-performing configuration. LayerNorm, GELU, bias/residual adds, embedding gather,
and the QKV head split/merge are hand-written CUDA, kept simple on purpose since they are not
the point. The dense projections use cuBLAS. Weights are exported from HuggingFace into a flat
binary with a validated header, and the loader cross-checks per-tensor statistics against the
export manifest (`verify_manifest.py`).

**Correctness.** The numerics were validated at GPT-2 (124M) before scaling up.
`gpt2_logit_check.py` walks the per-block hidden-state ladder, so any divergence localizes to
the block that introduced it, then compares the final logits against HuggingFace (FP32
network + fp16v2 attention):

| tokens | max abs diff | argmax agreement |
|-------:|-------------:|-----------------:|
|      4 |      1.52e-2 |              4/4 |
|     49 |      3.08e-2 |            49/49 |
|     64 |      1.56e-1 |            64/64 |
|   1000 |      1.13e+0 |        1000/1000 |

The absolute drift grows with depth and length, since FP16 rounding in attention compounds
through the residual blocks, but it never flips a single argmax. An argmax mismatch is only
tolerated when HF's own top-1/top-2 margin is smaller than the accumulated numeric error,
i.e. when the position is a coin flip. The same code path then runs XL by swapping the
exported weights; a pinned greedy sequence (`gpt2 --selftest`) guards against regressions.

**Generation** uses naive re-prefill: there is no KV cache, and each new token recomputes the
full prefill over the context so far. That is O(N²) overall, slow but exactly correct. An
efficient decode path needs a separate seq=1 kernel and is out of scope by design.

| Model | context 16→80 | 256→320 | 768→832 |
|-------|--------------:|--------:|--------:|
| 124M  | 289 tok/s | 165 tok/s | 79 tok/s |
| 1.5B  |  57 tok/s |  18 tok/s | 6.9 tok/s |

The 48x prompt-length range compresses to only a 3.6x throughput range at 124M but 8.3x at
1.5B. Short contexts are kernel-launch-bound (~195 launches per forward at 124M, ~780 at
1.5B); long contexts are compute-bound.
 
---

## What I'd do next

- **A KV-cache decode kernel.** Decoding with seq=1 queries is a different, memory-bound
  problem with paged layouts, which is why generation here stayed at re-prefill.
- **Backward pass.** Roughly the forward's complexity again, with recomputation strategy as
  the new core problem.
- FP8 / Hopper `wgmma` paths, variable-length sequence packing.
---

## Building and running

Requires CUDA 12.x, CMake >= 3.24, a GPU with sm_80+ (developed on an RTX 3090), and Python
with PyTorch for validation.

Build from the repo root, passing the target architecture (86 for an RTX 3090):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

If `CMAKE_CUDA_ARCHITECTURES` is not set, CMake falls back to detecting the local GPU with
`nvidia-smi`, which only works when building on the machine you intend to run on.

**Attention kernels** (no weights or downloads needed):

```bash
# correctness vs PyTorch SDPA, all shapes x 3 input regimes
python3 python/validate.py --kernel fa2_fp16v2
 
# benchmark one kernel, or `bench all` for the full sweep
./build/flash_attn bench fa2_fp16v2
```

**GPT-2** (exports GPT-2 XL weights from HuggingFace, several GB):

```bash
python3 python/export_gpt2.py weights
python3 python/export_gpt2.py tokens
./build/gpt2 --verify-weights
 
./demo.sh "The quick brown fox" 100 0.8
```

`validate.py` tests more than random inputs. Alongside standard Gaussian tensors it runs
adversarial cases (`wide_logits`, and `late_spike`, which places a score spike in the final
K-tile) built to expose incorrect online-softmax rescaling, and FP16 kernels are checked
against SDPA-FP16's own numerical noise floor rather than a fixed tolerance.

## Known limitations

- `seq_len` must be a multiple of 64 (the tile size); the GPT-2 path pads internally and masks
  the padding via causality. head_dim ∈ {64, 128}.
- Causal masking is implemented only in the final kernel (fp16v2), the version the transformer
  uses. Earlier ladder versions reject the flag.
- Q/K/V are FP32 in HBM and converted to FP16 in-kernel.
- Forward pass only; generation is naive re-prefill (no KV cache) by design.
- The causal mask relies on the WMMA accumulator fragment's lane layout, which is
  implementation-defined: stable on sm_80/86, not guaranteed elsewhere.
 