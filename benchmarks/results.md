# Benchmark Results
> Hardware: RTX 3090 (Vast.ai) | Peak: 35.5 TFLOPS FP32, 142 TFLOPS FP16 TC, 936 GB/s
> All times: average of 10 runs after 3 warmup runs
> Bytes formula: `naive = BH*(4*N*D + 4*N*N)*4`, `fused_softmax = BH*(4*N*D + 2*N*N)*4`, `fa1 = BH*(4*N*D)*4` 

---

## Attention Kernels

| Date       | GPU                     | Kernel        | seq_len | head_dim | ms/run   |  GB/s | TFLOPS | AI (F/B) | vs Naive | Notes                       |
|------------|-------------------------|---------------|---------|----------|----------|-------|--------|----------|----------|-----------------------------|
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |     512 |       64 |   1.264  |  29.9 |  0.425 |     14.2 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |    1024 |       64 |   5.084  |  28.0 |  0.422 |     15.1 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |    2048 |       64 |  17.956  |  30.8 |  0.478 |     15.5 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |    4096 |       64 |  67.024  |  32.5 |  0.513 |     15.8 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |     512 |      128 |   2.091  |  20.1 |  0.514 |     25.6 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |    1024 |      128 |   8.396  |  18.0 |  0.512 |     28.4 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |    2048 |      128 |  32.928  |  17.3 |  0.522 |     30.1 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | naive         |    4096 |      128 | 134.408  |  16.5 |  0.511 |     31.0 | 1x       | baseline                    |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |     512 |       64 |   0.182  | 115.0 |  2.943 |     25.6 | 6.9x     | vec4 QK^T, fused softmax+AV |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |    1024 |       64 |   0.789  |  95.7 |  2.722 |     28.4 | 6.4x     | vec4 QK^T, fused softmax+AV |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |    2048 |       64 |   3.166  |  90.1 |  2.713 |     30.1 | 5.7x     | vec4 QK^T, fused softmax+AV |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |    4096 |       64 |  19.326  |  57.3 |  1.778 |     31.0 | 3.5x     | vec4 QK^T, fused softmax+AV |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |     512 |      128 |   0.289  |  87.1 |  3.715 |     42.7 | 7.2x     | vec4 QK^T, fused softmax+AV |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |    1024 |      128 |   1.720  |  48.8 |  2.498 |     51.2 | 4.9x     | vec4 QK^T, fused softmax+AV |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |    2048 |      128 |   7.284  |  41.5 |  2.358 |     56.9 | 4.5x     | vec4 QK^T, fused softmax+AV |
| 2026-06-07 | NVIDIA GeForce RTX 3090 | fused_softmax |    4096 |      128 |  20.543  |  55.5 |  3.345 |     60.2 | 6.5x     | vec4 QK^T, fused softmax+AV |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |     512 |       64 |   1.145  |   3.7 |  0.469 |    128.0 | 1.1x     | online softmax, warp-shuffle reduce, 64x32 tile; **fused still 6.3x faster (pre-crossover)** |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |    1024 |       64 |   1.322  |   6.3 |  1.625 |    256.0 | 3.8x     | fused still 1.7x faster (pre-crossover) |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |    2048 |       64 |   1.575  |  10.7 |  5.455 |    512.0 | 11.4x    | crossover: 2.0x over fused |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |    4096 |       64 |   5.063  |   6.6 |  6.786 |   1024.0 | 13.2x    | 3.8x over fused; MIO-bound (L1 95.8%) |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |     512 |      128 |   1.297  |   6.5 |  0.828 |    128.0 | 1.6x     | 64x32 tile, TN=2; **fused 4.5x faster (pre-crossover)** |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |    1024 |      128 |   1.363  |  12.3 |  3.152 |    256.0 | 6.2x     | crossover: 1.3x over fused |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |    2048 |      128 |   3.269  |  10.3 |  5.255 |    512.0 | 10.1x    | 2.2x over fused |
| 2026-06-27 | NVIDIA GeForce RTX 3090 | fa1           |    4096 |      128 |  11.329  |   5.9 |  6.066 |   1024.0 | 11.9x    | 1.8x over fused |

> Note: hd=64 rows use the 64x32 tile (16 warps, 33% occ). The 64x64 tile (8 warps, 16.7% occ) measured **faster** — 7.30 TFLOPS @ seq=4096 — see "Config is head-dim-dependent" below. Logged the 64x32 config for consistency with hd=128.

---

## Nsight Compute — FA1 (single fused kernel, hd=64 seq=4096, 64x32 tile)

```
 GPU Speed of Light
 -------------------
   L1/TEX cache throughput   95.76%   <- the wall: shared-memory pipe saturated
   Compute (SM) throughput   33.77%
   Memory (DRAM) throughput   0.64%   <- not DRAM-bound; flash attention removed the NxN traffic
   Achieved occupancy        31.66%   (shared-mem limited, Block Limit SMem = 2)

 Scheduler / warp state
 ----------------------
   Eligible warps/scheduler   0.43    <- 69.9% of cycles issue nothing
   Stall: MIO throttle        35.2%   
   Stall: short scoreboard    30.7%   both = shared-memory / MIO instruction queue
```

Bottleneck: **MIO (shared-memory) issue pipe**, not DRAM and not the FP32 ALUs.
The inner loops issue ~700 scalar 4-byte shared loads per tile; the pipe saturates on
instruction count. Padding removed the 5.9-way bank conflicts but did NOT 
move throughput, confirming the limit is issue rate, not conflicts.

---

## Roofline Analysis

### fa1
```
Arithmetic intensity: 128–1024 FLOP/B  (= N/4; scales with seq_len because flash drops N×N)
Roofline bound:       compute, per the simple model (AI >> 38 FLOP/B DRAM crossover)
Achieved bandwidth:   3.7–12.3 GB/s / 936 GB/s = <2% peak   (low by design, not memory-bound)
Achieved compute:     6.8 TFLOPS / 35.5 TFLOPS = 19% peak
Actual bound:         shared-memory / MIO issue pipe, a resource the basic DRAM-vs-FLOP
                      roofline doesn't plot. L1/TEX at 95.8%, 0.43 eligible warps/sched.
Bottleneck:           ~700 scalar smem loads/tile saturate the MIO pipe; padding fixed
                      bank conflicts but issue rate is the ceiling

```
### naive
```
Arithmetic intensity: 14–31 FLOP/B (varies with seq_len/head_dim)
Roofline bound:       memory (all shapes below 38 FLOP/B crossover)
Achieved bandwidth:   17–33 GB/s / 936 GB/s = 2–4% peak
Achieved compute:     0.4–0.5 TFLOPS / 35.5 TFLOPS = 1–2% peak
Bottleneck:           N×N attention matrix materialised to HBM (3 kernels, 4 passes of N×N buffer)
                      + naive QK^T dominates at 86% of total GPU time
Next step:            fuse softmax into AV epilogue (eliminate P write+read), faster QK^T
```

### fused_softmax
```
Arithmetic intensity: 26–60 FLOP/B (higher than naive — 2×N×N bytes instead of 4×N×N)
Roofline bound:       memory (hd=64 below 38 FLOP/B; hd=128 at large seq_len crosses over)
Achieved bandwidth:   42–115 GB/s / 936 GB/s = 4–12% peak
Achieved compute:     1.8–3.7 TFLOPS / 35.5 TFLOPS = 5–10% peak
Speedup vs naive:     3.5–7.2x (vec4 QK^T + eliminated P round-trip)
Bottleneck:           AV loop reads all of V from HBM once per query row = O(N²D) V traffic
                      FA1 eliminates this by tiling the outer loop so V stays in smem
Next step:            Flash Attention 1 — online softmax, tile over sequence blocks
```

---

## Observations after implementing FA1

### Crossover with fused_softmax
FA1 only beats `fused_softmax` past seq≈1024–2048. At seq=512 fused is 4–6x faster, the N×N matrix is small enough that flash's online-softmax                                                                                                                               
overhead (per-tile rescaling, more passes over Q) costs more than it saves.

By seq=4096 flash wins 3.8x at hd=64 and 1.8x at hd=128. hd=128 crosses earlier but with a smaller long-seq margin: its per-row D work amortizes the overhead by seq=1024 already.

### Optimal tile depends on head_dim
Going 64x64 → 64x32 (warps 8→16) gave 4x on hd=128 but regressed hd=64 from 7.30 to 6.79 TFLOPS at seq=4096. 

hd=128 was latency-bound at 2 warps/SM so more warps helped hide stalls; hd=64 was already MIO-bound, 
and the extra warps just piled more pressure on the saturated L1 pipe.

### Stopping FP32 work at 19% peak
A tuned FP32 FA1 tops out around 9–14 TFLOPS (as per google), and float4 smem vectorization may get closer to it, but it also fights the padding and likely won't yield a sufficient win.
Since my ideas to scale the performance up *will* conflict with this improvement, I will not do so.