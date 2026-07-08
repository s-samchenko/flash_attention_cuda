# Benchmark Results

**Hardware:** NVIDIA RTX 3090 (GA102, 82 SM)
**Compute:** 35.6 TFLOPS FP32 | 142 TFLOPS FP16 tensor cores | 936 GB/s HBM
**Method:** average of 10 timed runs after 3 warmup runs

**Notation:** `N` — sequence length * `D` — head dim * `BH` — batch x heads * `Br`, `Bc` — per-block Q and K/V tile sizes.

**Bytes per launch (arithmetic-intensity denominator):**

- `naive`: `BH*(4*N*D + 4*N*N)*4` – attention matrix materialized in HBM; 4 passes over it
- `fused_softmax`: `BH*(4*N*D + 2*N*N)*4` — attention matrix materialized; 2 passes
- `fa1`, `fa2_tf32`: `BH*4*N*D*4` — attention matrix stays in SRAM

---

## Attention Kernels

| Date       | Kernel        |  seq | head_dim | ms/run  |  GB/s | TFLOPS | AI (F/B) | vs. naive |
|------------|---------------|-----:|---------:|--------:|------:|-------:|---------:|----------:|
| 2026-06-07 | naive         |  512 |       64 |   1.264 |  29.9 |  0.425 |     14.2 |      1.0x |
| 2026-06-07 | naive         | 1024 |       64 |   5.084 |  28.0 |  0.422 |     15.1 |      1.0x |
| 2026-06-07 | naive         | 2048 |       64 |  17.956 |  30.8 |  0.478 |     15.5 |      1.0x |
| 2026-06-07 | naive         | 4096 |       64 |  67.024 |  32.5 |  0.513 |     15.8 |      1.0x |
| 2026-06-07 | naive         |  512 |      128 |   2.091 |  20.1 |  0.514 |     25.6 |      1.0x |
| 2026-06-07 | naive         | 1024 |      128 |   8.396 |  18.0 |  0.512 |     28.4 |      1.0x |
| 2026-06-07 | naive         | 2048 |      128 |  32.928 |  17.3 |  0.522 |     30.1 |      1.0x |
| 2026-06-07 | naive         | 4096 |      128 | 134.408 |  16.5 |  0.511 |     31.0 |      1.0x |
| 2026-06-07 | fused_softmax |  512 |       64 |   0.182 | 115.0 |  2.943 |     25.6 |      6.9x |
| 2026-06-07 | fused_softmax | 1024 |       64 |   0.789 |  95.7 |  2.722 |     28.4 |      6.4x |
| 2026-06-07 | fused_softmax | 2048 |       64 |   3.166 |  90.1 |  2.713 |     30.1 |      5.7x |
| 2026-06-07 | fused_softmax | 4096 |       64 |  19.326 |  57.3 |  1.778 |     31.0 |      3.5x |
| 2026-06-07 | fused_softmax |  512 |      128 |   0.289 |  87.1 |  3.715 |     42.7 |      7.2x |
| 2026-06-07 | fused_softmax | 1024 |      128 |   1.720 |  48.8 |  2.498 |     51.2 |      4.9x |
| 2026-06-07 | fused_softmax | 2048 |      128 |   7.284 |  41.5 |  2.358 |     56.9 |      4.5x |
| 2026-06-07 | fused_softmax | 4096 |      128 |  20.543 |  55.5 |  3.345 |     60.2 |      6.5x |
| 2026-06-27 | fa1           |  512 |       64 |   1.145 |   3.7 |  0.469 |    128.0 |      1.1x |
| 2026-06-27 | fa1           | 1024 |       64 |   1.322 |   6.3 |  1.625 |    256.0 |      3.8x |
| 2026-06-27 | fa1           | 2048 |       64 |   1.575 |  10.7 |  5.455 |    512.0 |     11.4x |
| 2026-06-27 | fa1           | 4096 |       64 |   5.063 |   6.6 |  6.786 |   1024.0 |     13.2x |
| 2026-06-27 | fa1           |  512 |      128 |   1.297 |   6.5 |  0.828 |    128.0 |      1.6x |
| 2026-06-27 | fa1           | 1024 |      128 |   1.363 |  12.3 |  3.152 |    256.0 |      6.2x |
| 2026-06-27 | fa1           | 2048 |      128 |   3.269 |  10.3 |  5.255 |    512.0 |     10.1x |
| 2026-06-27 | fa1           | 4096 |      128 |  11.329 |   5.9 |  6.066 |   1024.0 |     11.9x |
| 2026-07-08 | fa2_tf32      |  512 |       64 |   1.226 |   3.4 |  0.438 |    128.0 |      1.0x |
| 2026-07-08 | fa2_tf32      | 1024 |       64 |   1.555 |   5.4 |  1.381 |    256.0 |      3.3x |
| 2026-07-08 | fa2_tf32      | 2048 |       64 |   1.752 |   9.6 |  4.904 |    512.0 |     10.3x |
| 2026-07-08 | fa2_tf32      | 4096 |       64 |   5.922 |   5.7 |  5.802 |   1024.0 |     11.3x |
| 2026-07-08 | fa2_tf32      |  512 |      128 |   1.293 |   6.5 |  0.830 |    128.0 |      1.6x |
| 2026-07-08 | fa2_tf32      | 1024 |      128 |   1.412 |  11.9 |  3.041 |    256.0 |      5.9x |
| 2026-07-08 | fa2_tf32      | 2048 |      128 |   3.563 |   9.4 |  4.821 |    512.0 |      9.2x |
| 2026-07-08 | fa2_tf32      | 4096 |      128 |  12.217 |   5.5 |  5.625 |   1024.0 |     11.0x |

---

## Roofline Analysis

### fa2_tf32
```
Arithmetic intensity: 128–1024 FLOP/B (identical to fa1)
Roofline bound:       compute
Achieved compute:     5.8 TFLOPS of 35.6 TFLOPS (16% peak; 0.85x fa1 at seq=4096 hd=64)
Achieved bandwidth:   3.4–11.9 GB/s of 936 GB/s (<2% peak)
Actual bound:         warp-scheduler latency. Achieved occupancy is 8% at
                      hd=128 and 16% at hd=64, vs. fa1's 32%. Warps stall on
                      WMMA fragment loads; neither pipe reaches saturation.
Root cause:           low occupancy from SRAM pressure. Per-block SRAM at 40 KB
                      (hd=64) or 72 KB (hd=128) caps Blocks/SM at 2 and 1,
                      combined with 4 warps/block. TF32 tensor cores on the
                      3090 do not provide enough arithmetic headroom over
                      FP32 CUDA cores to offset the loss.
```

### fa1
```
Arithmetic intensity: 128–1024 FLOP/B (= N/4; scales with seq_len because the
                      NxN attention matrix never leaves SRAM)
Roofline bound:       compute (AI far above the 38 FLOP/B DRAM crossover)
Achieved bandwidth:   3.7–12.3 GB/s of 936 GB/s (<2% peak; low by design)
Achieved compute:     6.8 TFLOPS of 35.6 TFLOPS (19% peak)
Actual bound:         shared-memory MIO issue pipe — a resource not modeled by
                      the DRAM/FLOP roofline. L1/TEX at 96%, 0.43 eligible
                      warps/scheduler.
Bottleneck:           ~700 scalar SRAM loads per K/V tile saturate the MIO pipe
                      on instruction count. Padding removed bank conflicts
                      without moving throughput, confirming issue rate is
                      the ceiling.
```

### fused_softmax
```
Arithmetic intensity: 26–60 FLOP/B (higher than naive — 2*N*N bytes instead of 4*N*N)
Roofline bound:       memory (hd=64 below crossover; hd=128 crosses at large seq)
Achieved bandwidth:   42–115 GB/s of 936 GB/s (4–12% peak)
Achieved compute:     1.8–3.7 TFLOPS of 35.6 TFLOPS (5–10% peak)
Speedup vs. naive:    3.5–7.2x (vec4 QK^T + eliminated P round-trip through HBM)
Bottleneck:           AV loop reads all of V from HBM once per query row ->
                      O(N²D) V traffic. FA1 removes this by tiling the outer
                      loop so V stays in SRAM.
```

### naive
```
Arithmetic intensity: 14–31 FLOP/B (varies with seq_len and head_dim)
Roofline bound:       memory (all shapes below the 38 FLOP/B DRAM crossover)
Achieved bandwidth:   17–33 GB/s of 936 GB/s (2–4% peak)
Achieved compute:     0.4–0.5 TFLOPS of 35.6 TFLOPS (1–2% peak)
Bottleneck:           full NxN attention matrix materialized in HBM (3 kernels,
                      4 passes over the NxN buffer). Naive QK^T dominates 86%
                      of total GPU time.
```

---

## Nsight Compute Profiles

### fa2_tf32 — hd=128, seq=4096
```
 Occupancy
 ─────────
   Achieved occupancy         8.33%     (4 warps/SM; fa1 = 31.66%)
   Theoretical occupancy      8.33%     (kernel already at block-imposed ceiling)
   Block Limit Shared Mem        1      (72 KB block, 100 KB per-SM opt-in ceiling)
   Block Limit Registers         3      (not binding)
   Block Limit Warps            12      (not binding — block has only 4 warps)
```

**Bottleneck:** low occupancy driven by per-block SRAM pressure. The FA2 layout keeps an FP32 O accumulator in SRAM alongside FP32 Q/K/V staging, exceeding the 48 KB default SRAM budget and requiring opt-in mode, where only one 72 KB block fits per SM. With four warps to schedule, WMMA fragment load stalls expose latency the scheduler cannot hide. The SRAM round-trip on the O accumulator is a small direct cost; the occupancy loss it caused is the dominant effect.


### fa1 — hd=64, seq=4096, 64x32 tile
```
 GPU Speed of Light
 ─────────────────
   L1/TEX cache throughput    95.76%   (the wall: shared-memory pipe saturated)
   Compute (SM) throughput    33.77%
   Memory (DRAM) throughput    0.64%   (flash attention removed NxN HBM traffic)
   Achieved occupancy         31.66%   (Block Limit SMem = 2)

 Scheduler / warp state
 ──────────────────────
   Eligible warps/scheduler    0.43    (69.9% of cycles issue nothing)
   Stall: MIO throttle        35.2%
   Stall: short scoreboard    30.7%    (shared-memory instruction queue)
```

**Bottleneck:** MIO (shared-memory) issue pipe, not DRAM and not the FP32 ALUs. The inner loops issue ~700 scalar 4-byte shared loads per K/V tile; the pipe saturates on instruction count. Padding removed 5.9-way bank conflicts without moving throughput, confirming issue rate is the ceiling.

---

## Observations

### fused_softmax vs. naive
Fusing softmax into the AV epilogue eliminates one of the two HBM round-trips through the NxN attention matrix, doubling arithmetic intensity and delivering 3.5–7.2x speedup. The remaining O(N²D) V traffic — V loaded once per query row — is what FA1 attacks next.

### fa1 crossover with fused_softmax
Flash Attention only outperforms fused softmax past seq ≈ 1024–2048. At seq=512 the fused kernel is 4–6x faster because the NxN matrix is small enough that flash's per-tile softmax rescaling overhead costs more than it saves. By seq=4096, fa1 wins 3.8x at hd=64 and 1.8x at hd=128. Head dim 128 crosses earlier (at seq=1024) because its larger per-row work amortizes the overhead sooner.

### fa1 tile choice depends on head dim
Moving from a 64x64 tile to 64x32 (8 -> 16 warps per block) gave 4x throughput at hd=128 but regressed hd=64 from 7.30 to 6.79 TFLOPS at seq=4096. At hd=128 the kernel was latency-bound at 2 warps/SM, so extra warps helped hide stalls; at hd=64 the kernel was already MIO-bound, and additional warps added pressure to the already-saturated L1 pipe. The 64x32 config was selected for consistency across both head dims.

### Ceiling on FP32 flash-attention
fa1 achieves 6.8 TFLOPS at seq=4096 hd=64, or 19% of the 3090's FP32 peak. Reported tuned FP32 flash-attention implementations peak in the 9–14 TFLOPS range. Further FP32 gains would require float4 SRAM vectorization, which conflicts with the padding used to eliminate bank conflicts. Optimization effort was redirected to the tensor-core port (FA2), which has a higher throughput ceiling.

### fa2_tf32 does not beat fa1 on the 3090
FA2 with TF32 tensor cores achieves 0.85–0.93x of fa1 at seq=4096. Two factors account for this:

1. **Limited arithmetic headroom.** On the 3090, TF32 tensor-core throughput is only modestly above the FP32 CUDA-core rate — unlike the A100, where TF32 TC delivers 8x the FP32 rate. fa1 was already at 19% of the 35.6 TFLOPS FP32 ceiling; swapping to TF32 tensor cores on the same silicon raises the ceiling only slightly.
2. **Occupancy regression.** The FP32 O accumulator held in SRAM (chosen for portability across GPU generations — the alternative depends on documented but implementation-defined accumulator lane layouts) plus FP32 Q/K/V staging exceeds the 48 KB default and forces opt-in SRAM mode. Achieved occupancy drops from fa1's 32% to 16% at hd=64 and 8% at hd=128, and the tensor-core arithmetic does not compensate.

Both factors resolve at FP16: FP16 tensor cores deliver 71 TFLOPS on the 3090 (2x FP32/TF32), and Q/K/V storage halves, freeing SRAM budget for more blocks per SM. FP16 numbers will be added when landed.
