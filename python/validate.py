#!/usr/bin/env python3
"""Correctness tests against PyTorch SDPA.

Usage:
  python3 python/validate.py                       # reference self-check only
  python3 python/validate.py --kernel naive        # test naive kernel
  python3 python/validate.py --kernel naive --verbose
  python3 python/validate.py --kernel naive --save-failures
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch

BINARY = os.path.join(os.path.dirname(__file__), '..', 'build', 'flash_attn')

# All shapes run on every kernel
TEST_SHAPES = [
    # (seq_len, head_dim, batch, n_heads)
    (512,  64,  1, 8),
    (1024, 64,  1, 8),
    (2048, 64,  1, 8),
    (4096, 64,  1, 8),
    (512,  128, 1, 8),
    (1024, 128, 1, 8),
]


def sdpa_reference(Q, K, V):
    """
    Q, K, V: np.float32 arrays of shape (BH, N, D).
    Returns np.float32 of the same shape.
    """
    Qt = torch.from_numpy(Q).unsqueeze(1)  # (BH, 1, N, D)
    Kt = torch.from_numpy(K).unsqueeze(1)
    Vt = torch.from_numpy(V).unsqueeze(1)
    with torch.no_grad():
        out = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt)
    return out.squeeze(1).numpy()


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


def test_shape(kernel, seq_len, head_dim, batch, n_heads, verbose, save_failures):
    BH = batch * n_heads
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((BH, seq_len, head_dim)).astype(np.float32)
    K = rng.standard_normal((BH, seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((BH, seq_len, head_dim)).astype(np.float32)

    expected = sdpa_reference(Q, K, V)

    with tempfile.TemporaryDirectory() as tmpdir:
        got = run_kernel(kernel, seq_len, head_dim, batch, n_heads,
                         Q, K, V, tmpdir)

    max_err = float(np.max(np.abs(got - expected)))
    passed  = max_err < 1e-4
    label   = "PASS" if passed else "FAIL"

    if verbose or not passed:
        print(f"  {label}  seq={seq_len:5d}  head_dim={head_dim:3d}  "
              f"batch={batch}  n_heads={n_heads}  max_err={max_err:.2e}")

    if not passed and save_failures:
        fail_dir = os.path.join('benchmarks', 'failures')
        os.makedirs(fail_dir, exist_ok=True)
        tag = f"{kernel}_seq{seq_len}_hd{head_dim}"
        np.save(os.path.join(fail_dir, f'{tag}_Q.npy'), Q)
        np.save(os.path.join(fail_dir, f'{tag}_K.npy'), K)
        np.save(os.path.join(fail_dir, f'{tag}_V.npy'), V)
        np.save(os.path.join(fail_dir, f'{tag}_expected.npy'), expected)
        np.save(os.path.join(fail_dir, f'{tag}_got.npy'), got)
        print(f"    failure tensors saved to benchmarks/failures/{tag}_*.npy")

    return passed


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
        else:  failed += 1

    print(f"\n{passed}/{passed+failed} shapes passed")
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
