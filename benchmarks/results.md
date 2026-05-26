# Benchmark Results
> Hardware: RTX 3090 (Vast.ai) | Peak: 35.5 TFLOPS FP32, 142 TFLOPS FP16 TC, 936 GB/s
> All times: average of 10 runs after 3 warmup runs

---

## Attention Kernels

| Date       | GPU                      | Kernel | seq_len | head_dim | ms/run   | GB/s | TFLOPS | vs Naive | Notes    |
|------------|--------------------------|--------|---------|----------|----------|------|--------|----------|----------|
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |     512 |       64 |   1.2642 | 10.0 | 0.4247 | 1x       | baseline |
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |    1024 |       64 |   5.0977 |  8.2 | 0.4213 | 1x       | baseline |
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |    2048 |       64 |  17.8396 |  8.5 | 0.4815 | 1x       | baseline |
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |    4096 |       64 |  66.0179 |  8.6 | 0.5205 | 1x       | baseline |
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |     512 |      128 |   2.0607 |  8.1 | 0.5211 | 1x       | baseline |
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |    1024 |      128 |   8.2714 |  6.1 | 0.5193 | 1x       | baseline |
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |    2048 |      128 |  32.4367 |  5.2 | 0.5296 | 1x       | baseline |
| 2026-05-25 | NVIDIA GeForce RTX 3090  | naive  |    4096 |      128 | 129.4242 |  4.7 | 0.5310 | 1x       | baseline |
