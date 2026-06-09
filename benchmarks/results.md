# Benchmark Results
> Hardware: RTX 3090 (Vast.ai) | Peak: 35.5 TFLOPS FP32, 142 TFLOPS FP16 TC, 936 GB/s
> All times: average of 10 runs after 3 warmup runs
> Bytes formula: `naive = BH*(4*N*D + 4*N*N)*4`, `fused_softmax = BH*(4*N*D + 2*N*N)*4`

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

---

## Nsight Systems — Kernel Time Breakdown

### naive (2026-06-07)

```
 Time (%)   Kernel
 --------   ------
    85.9%   qk_matmul_kernel      bottleneck: naive QK^T dominates, D=64 inner dim
    11.5%   av_matmul_kernel
     2.6%   softmax_kernel
```

### fused_softmax (2026-06-07)

```
 Time (%)   Kernel
 --------   ------
    84.4%   softmax_av_fused_kernel   bottleneck shifted: AV loop reads V N times from HBM
    15.6%   qk_matmul_kernel          vec4 collapsed this from 85.9% to 15.6%
```

---

## Roofline Analysis

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
