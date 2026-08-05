#!/usr/bin/env python3
"""Compare the C++ block-0 activations against the HuggingFace reference ladder.

  python3 python/gpt2_block_check.py [--ref assets/ref] [--dump assets/dump]

Each stage isolates a slice of the pipeline. Stages up to and including 'qkv' are
pure fp32, so they hold at ~1e-4 and pin down GEMM/layout/bias. Stages downstream
of attention inherit fp16 rounding from the fp16v2 kernel.
"""

import argparse
import json
import os

import numpy as np

D = 768
N_HEAD = 12
HEAD_DIM = 64


def load(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def causal_mha(c_attn, n_head=N_HEAD, head_dim=HEAD_DIM):
    """fp32 causal multi-head attention from a [N, 3*768] c_attn output.

    Returns the merged attention output [N, 768], before the c_proj projection.
    """
    n = c_attn.shape[0]
    q, k, v = np.split(c_attn, 3, axis=-1)

    def to_heads(x):
        return x.reshape(n, n_head, head_dim).transpose(1, 0, 2)  # [H, n, d]

    q, k, v = to_heads(q), to_heads(k), to_heads(v)

    scores = (q @ k.transpose(0, 2, 1)) * (1.0 / np.sqrt(head_dim))  # [H, n, n]
    causal = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores[:, causal] = -np.inf

    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)

    out = w @ v  # [H, n, d]
    return out.transpose(1, 0, 2).reshape(n, D).astype(np.float32)


def max_abs(a, b):
    return float(np.max(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='assets/ref')
    ap.add_argument('--dump', default='assets/dump')
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.ref, 'meta.json')))
    n = meta['n_tokens']

    def R(name, cols):
        return load(os.path.join(args.ref, name + '.bin'), (n, cols))

    def G(name, cols):
        return load(os.path.join(args.dump, name + '.bin'), (n, cols))

    c_attn_ref = R('block0_c_attn', 3 * D)
    oracle = causal_mha(c_attn_ref)

    ladder = [
        ('ln1',       G('block00_ln1', D),          R('block0_ln1', D),  1e-4),
        ('qkv',       G('block00_qkv', 3 * D),       c_attn_ref,          1e-3),
        ('attn_core', G('block00_attn_merged', D),   oracle,              3e-3),
        ('attn_proj', G('block00_attn_proj', D),     R('block0_attn', D), 5e-3),
        ('mlp_proj',  G('block00_mlp_proj', D),      R('block0_mlp', D),  5e-3),
        ('block00',   G('x_block00', D),             R('h01', D),         1.5e-2),
    ]

    fails = 0
    for label, ours, ref, tol in ladder:
        d = max_abs(ours, ref)
        ok = d <= tol
        print(f"{label:12s} max_abs {d:.3e}  tol {tol:.1e}  {'OK' if ok else 'FAIL'}")
        fails += not ok

    print(f"\n{fails} stage(s) failed")
    raise SystemExit(1 if fails else 0)


if __name__ == '__main__':
    main()
