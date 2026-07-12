# Benchmark Results

**Hardware:** NVIDIA RTX 3090 (GA102, 82 SM)
**Compute:** 35.6 TFLOPS FP32 | 142 TFLOPS FP16 tensor cores | 936 GB/s HBM
**Method:** average of 10 timed runs after 3 warmup runs

**Notation:** `N` — sequence length * `D` — head dim * `BH` — batch x heads * `Br`, `Bc` — per-block Q and K/V tile sizes.

**Bytes per launch (arithmetic-intensity denominator):**

- `naive`: `BH*(4*N*D + 4*N*N)*4` – attention matrix materialized in HBM; 4 passes over it
- `fused_softmax`: `BH*(4*N*D + 2*N*N)*4` — attention matrix materialized; 2 passes
- `fa1`, `fa2_tf32`, `fa2_fp16v1`: `BH*4*N*D*4` — attention matrix stays in SRAM (fa2_fp16v1 reads fp32 globals, converts to half in smem)

---

## Attention Kernels

| Date       | Kernel        |  seq | head_dim |  ms/run |  GB/s | TFLOPS | AI (F/B) | vs. naive |
|------------|---------------|-----:|---------:|--------:|------:|-------:|---------:|----------:|
| 2026-06-07 | naive         |  512 |       64 |   1.264 |  29.9 |  0.425 |     14.2 |      1.0x |
| 2026-06-07 | naive         | 1024 |       64 |   5.084 |  28.0 |  0.422 |     15.1 |      1.0x |
| 2026-06-07 | naive         | 2048 |       64 |  17.956 |  30.8 |  0.478 |     15.5 |      1.0x |
| 2026-06-07 | naive         | 4096 |       64 |  67.024 |  32.5 |  0.513 |     15.8 |      1.0x |
| 2026-06-07 | naive         |  512 |      128 |   2.091 |  20.1 |  0.514 |     25.6 |      1.0x |
| 2026-06-07 | naive         | 1024 |      128 |   8.396 |  18.0 |  0.512 |     28.4 |      1.0x |
| 2026-06-07 | naive         | 2048 |      128 |  32.928 |  17.3 |  0.522 |     30.1 |      1.0x |
| 2026-06-07 | naive         | 4096 |      128 | 134.408 |  16.5 |  0.511 |     31.0 |      1.0x |
| 2026-06-07 | fused_softmax |  512 |       64 |   0.172 | 121.9 |  3.121 |     25.6 |      7.3x |
| 2026-06-07 | fused_softmax | 1024 |       64 |   0.936 |  80.6 |  2.293 |     28.4 |      5.4x |
| 2026-06-07 | fused_softmax | 2048 |       64 |   3.681 |  77.5 |  2.334 |     30.1 |      4.9x |
| 2026-06-07 | fused_softmax | 4096 |       64 |  14.245 |  77.7 |  2.412 |     31.0 |      4.7x |
| 2026-06-07 | fused_softmax |  512 |      128 |   0.428 |  58.8 |  2.510 |     42.7 |      4.9x |
| 2026-06-07 | fused_softmax | 1024 |      128 |   1.855 |  45.2 |  2.315 |     51.2 |      4.5x |
| 2026-06-07 | fused_softmax | 2048 |      128 |   7.560 |  39.9 |  2.272 |     56.9 |      4.4x |
| 2026-06-07 | fused_softmax | 4096 |      128 |  29.659 |  38.5 |  2.317 |     60.2 |      4.5x |
| 2026-06-27 | fa1           |  512 |       64 |   1.740 |   2.4 |  0.309 |    128.0 |      0.7x |
| 2026-06-27 | fa1           | 1024 |       64 |   1.735 |   4.8 |  1.238 |    256.0 |      2.9x |
| 2026-06-27 | fa1           | 2048 |       64 |   2.060 |   8.1 |  4.171 |    512.0 |      8.7x |
| 2026-06-27 | fa1           | 4096 |       64 |   4.663 |   7.2 |  7.368 |   1024.0 |     14.4x |
| 2026-06-27 | fa1           |  512 |      128 |   1.874 |   4.5 |  0.573 |    128.0 |      1.1x |
| 2026-06-27 | fa1           | 1024 |      128 |   2.067 |   8.1 |  2.078 |    256.0 |      4.1x |
| 2026-06-27 | fa1           | 2048 |      128 |   3.244 |  10.3 |  5.296 |    512.0 |     10.2x |
| 2026-06-27 | fa1           | 4096 |      128 |  11.032 |   6.1 |  6.229 |   1024.0 |     12.2x |
| 2026-07-08 | fa2_tf32      |  512 |       64 |   2.050 |   2.0 |  0.262 |    128.0 |      0.6x |
| 2026-07-08 | fa2_tf32      | 1024 |       64 |   1.770 |   4.7 |  1.213 |    256.0 |      2.9x |
| 2026-07-08 | fa2_tf32      | 2048 |       64 |   1.943 |   8.6 |  4.421 |    512.0 |      9.2x |
| 2026-07-08 | fa2_tf32      | 4096 |       64 |   5.685 |   5.9 |  6.044 |   1024.0 |     11.8x |
| 2026-07-08 | fa2_tf32      |  512 |      128 |   1.839 |   4.6 |  0.584 |    128.0 |      1.1x |
| 2026-07-08 | fa2_tf32      | 1024 |      128 |   1.857 |   9.0 |  2.313 |    256.0 |      4.5x |
| 2026-07-08 | fa2_tf32      | 2048 |      128 |   3.763 |   8.9 |  4.565 |    512.0 |      8.7x |
| 2026-07-08 | fa2_tf32      | 4096 |      128 |  12.726 |   5.3 |  5.400 |   1024.0 |     10.6x |
| 2026-07-11 | fa2_fp16v1    |  512 |       64 |   0.088 |  47.9 |  6.125 |    128.0 |     14.4x |
| 2026-07-11 | fa2_fp16v1    | 1024 |       64 |   0.270 |  31.0 |  7.947 |    256.0 |     18.8x |
| 2026-07-11 | fa2_fp16v1    | 2048 |       64 |   0.681 |  24.6 | 12.618 |    512.0 |     26.4x |
| 2026-07-11 | fa2_fp16v1    | 4096 |       64 |   2.758 |  12.2 | 12.459 |   1024.0 |     24.3x |
| 2026-07-11 | fa2_fp16v1    |  512 |      128 |   0.239 |  35.1 |  4.487 |    128.0 |      8.7x |
| 2026-07-11 | fa2_fp16v1    | 1024 |      128 |   0.463 |  36.3 |  9.283 |    256.0 |     18.1x |
| 2026-07-11 | fa2_fp16v1    | 2048 |      128 |   1.822 |  18.4 |  9.428 |    512.0 |     18.1x |
| 2026-07-11 | fa2_fp16v1    | 4096 |      128 |   5.967 |  11.2 | 11.516 |   1024.0 |     22.5x |

---

## Roofline Analysis

### fa2_fp16v1
```
Arithmetic intensity: 128–1024 FLOP/B (identical to fa1/fa2_tf32; NxN stays in SRAM)
Roofline bound:       compute
Achieved compute:     12.5 TFLOPS at seq=4096 hd=64 — 17.5% of effective peak.
                      Effective peak here is 71 TFLOPS, not 142: GA102 halves
                      FP16 tensor-core rate when accumulating in FP32, and the
                      softmax needs FP32 accumulation.
Achieved bandwidth:   11–48 GB/s of 936 GB/s (<5% peak; low by design)
Actual bound:         shared-memory latency. DRAM sits at 1%, tensor cores at
                      16% — the busy unit is the L1/smem pipe, and mostly on
                      replays rather than useful traffic: shared loads average
                      a 23-way bank conflict, and with 3–4 blocks/SM there are
                      only ~3 warps per scheduler to hide the stalls. An
                      instruction issues about once every 11 cycles.
Bottleneck:           every smem row stride (K/V tiles, S and P tiles) is a
                      multiple of 128 B, so WMMA fragment loads and the
                      softmax keep landing on the same banks. Occupancy is
                      capped by registers and smem simultaneously. On top of
                      that, fp32 inputs are converted to half in smem on
                      every K/V tile, keeping global loads on the critical
                      path. Next: de-align the smem strides, then raise
                      blocks/SM, then store fp16 in global so K/V tiles can
                      be prefetched with cp.async.
```

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

### fa2_fp16v1 — seq=4096, Br=64, Bc=32
```
 Speed of Light                        hd=64      hd=128
 ─────────────────────────────────────────────────────────
   L1/TEX cache throughput             76.38%     64.64%
   Compute (SM) throughput             15.65%     12.62%
   DRAM throughput                      1.01%      1.08%

 Shared-memory conflicts
 ─────────────────────────────────────────────────────────
   Avg. bank conflict per shared load  22.9-way   25.7-way
   Conflicted share of load wavefronts 87.8%      87.7%
   Excessive shared wavefronts (all)   84%        84%

 Scheduler / warp state
 ─────────────────────────────────────────────────────────
   Eligible warps/scheduler            0.10       0.09
   Warp cycles per issued instruction  36.2       35.8
   Top stall                           MIO short  L1TEX long
                                       scoreboard scoreboard
                                       43.7%      45.9%

 Occupancy / launch
 ─────────────────────────────────────────────────────────
   Registers/thread                    98         158
   Dynamic SRAM/block                  20.5 KB    32.8 KB
   Blocks/SM (regs AND smem limited)   4          3
   Waves per SM                        1.56       2.08
```

**Bottleneck:** SRAM latency, twice over. Every SRAM row stride is a multiple
of 128 B, so WMMA fragment loads and the scalar softmax hit near-worst-case
bank conflicts — the L1 pipe is ~76% busy doing ~8x redundant wavefronts. With
only 3–4 blocks/SM (dual-limited by registers and SRAM), ~3 warps per scheduler
cannot hide the serialized accesses: an instruction issues once every ~11
cycles. At hd=128 the synchronous FP32 global staging adds an equal share of
long-scoreboard stalls. The 1.56-wave launch at hd=64 seq=4096 adds a
partial-wave tail on top.

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
fa1 achieves 7.3 TFLOPS at seq=4096 hd=64, or 20% of the 3090's FP32 peak. Reported tuned FP32 flash-attention implementations peak in the 9–14 TFLOPS range. Further FP32 gains would require float4 SRAM vectorization, which conflicts with the padding used to eliminate bank conflicts. Optimization effort was redirected to the tensor-core port (FA2), which has a higher throughput ceiling.

### fa2_tf32 does not beat fa1 on the 3090
FA2 with TF32 tensor cores achieves 0.85–0.93x of fa1 at seq=4096. The biggest factor:

**Occupancy regression.** The FP32 O accumulator held in SRAM (chosen for portability across GPU generations — the alternative depends on documented but implementation-defined accumulator lane layouts) plus FP32 Q/K/V staging exceeds the 48 KB default and forces opt-in SRAM mode. Achieved occupancy drops from fa1's 32% to 16% at hd=64 and 8% at hd=128, and the tensor-core arithmetic does not compensate.

It will resolve at FP16: FP16 tensor cores deliver 71 TFLOPS on the 3090 (2x FP32/TF32), and Q/K/V storage halves, freeing SRAM budget for more blocks per SM. FP16 numbers will be added when landed.

### fa2_fp16v1: where the 2x over fa2_tf32 comes from
Halving the element size fixed both problems from the fa2_tf32 analysis at
once. Per-block SRAM for Q/K/V staging drops from 40 KB to 20.5 KB at hd=64
and from 72 KB to 32.8 KB at hd=128, lifting occupancy from 2/1 blocks per SM
to 4/3 — from 8/4 active warps per SM to ~13. For a latency-bound kernel this
is the main lever: three times as many warps for the scheduler to hide the
same stalls with. And the arithmetic ceiling actually moves this time: TF32
tensor cores on GA102 run at 35.6 TFLOPS, the same rate as the plain FP32
cores, so fa2_tf32 had nothing to gain even at full utilization. FP16 with
FP32 accumulation runs at 71 TFLOPS.

Two smaller effects point the same way: the mma K-step widens from 8 to 16,
halving the mma instruction count per tile, and fragment loads move half the
bytes through the smem pipe that is already the bottleneck. The softmax
accumulators (running max, normalizer, O) stayed in FP32, so the numerics of
the online softmax are unchanged — only storage and matmul inputs narrowed —
and the kernel still validates against the PyTorch SDPA reference.

Net: 5.4–6.0 -> 11.5–12.5 TFLOPS at seq=4096, 2.1x over fa2_tf32 and 1.7–1.8x
over fa1. Best result so far is 26x over naive, at seq=2048 hd=64.

### fa2_fp16v1: seq=2048 slightly beats seq=4096 at hd=64
12.62 vs 12.46 TFLOPS. This is wave quantization, not a kernel property: at
4 blocks/SM the GPU holds 328 blocks in flight, so seq=2048 (256 blocks) runs
as a single wave while seq=4096 (512 blocks) runs one full wave plus a
184-block tail on a mostly idle machine. Not worth fixing directly — raising
blocks/SM shrinks it as a side effect.

### fa2_fp16v1: where the remaining headroom is
Nsight attributes 84% of shared-memory wavefronts to bank conflicts: every
smem row stride is 128 B-aligned, so fragment loads and the softmax hit the
same banks over and over (estimated 55–67% speedup from fixing this alone).
Behind that, occupancy is capped at 3–4 blocks/SM by registers and smem
together, and K/V tiles are loaded from global and converted fp32 -> half
synchronously inside the inner loop. The plan, in order: de-align the smem
strides, then free enough registers/smem for a fifth block per SM, then store
fp16 in global and double-buffer the K/V loads with cp.async.