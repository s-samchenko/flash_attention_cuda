#!/usr/bin/env python3
"""Correctness tests against PyTorch SDPA.

Usage:
  python3 python/validate.py --kernel <kernel_name>
  python3 python/validate.py --kernel naive        # test naive kernel
  python3 python/validate.py --kernel naive --verbose
  python3 python/validate.py --kernel naive --save-failures
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

BINARY = os.path.join(os.path.dirname(__file__), '..', 'build', 'flash_attn')

TEST_SHAPES = [
    # (seq_len, head_dim, batch, n_heads)
    (512,  64,  1, 8),
    (1024, 64,  1, 8),
    (2048, 64,  1, 8),
    (4096, 64,  1, 8),
    (512,  128, 1, 8),
    (1024, 128, 1, 8),
    (2048, 128, 1, 8),
    (4096, 128, 1, 8),
]

# Kernels that run in FP16 internally
FP16_KERNELS = {'fa2_tf32', 'fa2_fp16', 'fa2_fp16v2'}

# Multiplier on top of the FP16 noise floor.  ~2.5× keeps a small safety
# margin over the ~2× the plan targets while still catching bugs that widen
# the gap substantially.
NOISE_FLOOR_MULT = 2.5


def _sdpa(Q, K, V, dtype):
    """Q, K, V: np.float32 (BH, N, D). Returns np.float32 in `dtype` precision."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Qt = torch.from_numpy(Q).to(device=device, dtype=dtype).unsqueeze(1)
    Kt = torch.from_numpy(K).to(device=device, dtype=dtype).unsqueeze(1)
    Vt = torch.from_numpy(V).to(device=device, dtype=dtype).unsqueeze(1)
    with torch.no_grad():
        out = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
    return out.squeeze(1).float().cpu().numpy()

def sdpa_reference(Q, K, V):
    return _sdpa(Q, K, V, torch.float32)


def run_kernel(kernel, seq_len, head_dim, batch, n_heads, Q, K, V, io_dir):
    Q.tofile(os.path.join(io_dir, 'Q.bin'))
    K.tofile(os.path.join(io_dir, 'K.bin'))
    V.tofile(os.path.join(io_dir, 'V.bin'))

    result = subprocess.run(
        [BINARY, 'validate', kernel,
         str(seq_len), str(head_dim), str(batch), str(n_heads), io_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"binary error:\n{result.stderr.strip()}")

    BH = batch * n_heads
    raw = np.fromfile(os.path.join(io_dir, 'O.bin'), dtype=np.float32)
    return raw.reshape(BH, seq_len, head_dim)


# ── input generators ─────────────────────────────────────────────────────────
def _gen_standard(rng, BH, N, D):
    Q = rng.standard_normal((BH, N, D)).astype(np.float32)
    K = rng.standard_normal((BH, N, D)).astype(np.float32)
    V = rng.standard_normal((BH, N, D)).astype(np.float32)
    return Q, K, V


def _gen_wide_logits(rng, BH, N, D):
    Q, K, V = _gen_standard(rng, BH, N, D)
    Q *= 10.0
    return Q, K, V


def _gen_late_spike(rng, BH, N, D, Bc=32):
    Q, K, V = _gen_standard(rng, BH, N, D)
    last_tile = slice(max(0, N - Bc), N)
    K[:, last_tile, :] += 8.0
    return Q, K, V


INPUT_CASES = [
    ('standard', _gen_standard),
    ('wide_logits', _gen_wide_logits),
    ('late_spike', _gen_late_spike),
]


# ── testing ──────────────────────────────────────────────────────────────────

def test_shape(kernel, seq_len, head_dim, batch, n_heads, verbose, save_failures):
    BH = batch * n_heads
    is_fp16 = kernel in FP16_KERNELS
    all_passed = True

    for case_name, gen in INPUT_CASES:
        rng = np.random.default_rng(hash((seq_len, head_dim, case_name)) & 0xffffffff)
        Q, K, V = gen(rng, BH, seq_len, head_dim)

        expected_fp32 = sdpa_reference(Q, K, V)

        if is_fp16:
            sdpa_fp16 = _sdpa(Q, K, V, torch.float16)
            noise = float(np.max(np.abs(sdpa_fp16 - expected_fp32)))
            tol = max(noise * NOISE_FLOOR_MULT, 5e-3)
            expected = sdpa_fp16
        else:
            noise = 0.0
            tol = 1e-4
            expected = expected_fp32

        with tempfile.TemporaryDirectory() as tmpdir:
            got = run_kernel(kernel, seq_len, head_dim, batch, n_heads,
                             Q, K, V, tmpdir)

        max_err = float(np.max(np.abs(got - expected)))
        passed = max_err < tol
        label = "PASS" if passed else "FAIL"
        all_passed &= passed

        if verbose or not passed:
            noise_str = f" noise={noise:.2e}" if is_fp16 else ""
            print(f"  {label}  seq={seq_len:5d}  head_dim={head_dim:3d}  "
                  f"batch={batch}  n_heads={n_heads}  case={case_name:11s}  "
                  f"max_err={max_err:.2e} tol={tol:.2e}{noise_str}")

        if not passed and save_failures:
            fail_dir = os.path.join('benchmarks', 'failures')
            os.makedirs(fail_dir, exist_ok=True)
            tag = f"{kernel}_seq{seq_len}_hd{head_dim}_{case_name}"
            np.save(os.path.join(fail_dir, f'{tag}_Q.npy'), Q)
            np.save(os.path.join(fail_dir, f'{tag}_K.npy'), K)
            np.save(os.path.join(fail_dir, f'{tag}_V.npy'), V)
            np.save(os.path.join(fail_dir, f'{tag}_expected.npy'), expected)
            np.save(os.path.join(fail_dir, f'{tag}_got.npy'), got)
            print(f"    failure tensors saved to benchmarks/failures/{tag}_*.npy")

    return all_passed


def reference_self_check():
    N, D = 64, 32
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((1, N, D)).astype(np.float32)
    K = rng.standard_normal((1, N, D)).astype(np.float32)
    V = rng.standard_normal((1, N, D)).astype(np.float32)
    out = sdpa_reference(Q, K, V)
    assert out.shape == (1, N, D), f"shape mismatch: {out.shape}"
    print("reference self-check: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-failures', action='store_true')
    args = parser.parse_args()

    if args.kernel is None:
        reference_self_check()
        return

    print(f"validating kernel: {args.kernel}")
    passed = failed = 0
    for seq_len, head_dim, batch, n_heads in TEST_SHAPES:
        ok = test_shape(args.kernel, seq_len, head_dim, batch, n_heads,
                        args.verbose, args.save_failures)
        if ok: passed += 1
        else: failed += 1

    print(f"\n{passed}/{passed+failed} shapes passed")
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
