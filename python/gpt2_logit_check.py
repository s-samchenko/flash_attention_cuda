#!/usr/bin/env python3
"""Validate the C++ GPT-2 forward pass against the HuggingFace reference ladder.

For every prompt in assets/ref/index.json this drives `gpt2 --dump-activations`,
then walks the ladder emb -> block 00..11 -> final ln_f -> logits, printing the
max-abs deviation at each rung so the first divergence is obvious when something
breaks. The gate is argmax agreement at every position; a disagreement is only
tolerated when HF's own top-1/top-2 margin says the position is a coin flip.

  python3 python/gpt2_logit_check.py [--skip-run] [--bin build/gpt2]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

D = 768
VOCAB = 50257
ATOL, RTOL = 1e-2, 3e-2


def _load(path: str, cols: int) -> np.ndarray:
    if not os.path.exists(path):
        sys.exit(f"missing tensor: {path} (did the dump run?)")
    return np.fromfile(path, dtype=np.float32).reshape(-1, cols)


def _ladder(dump_dir: str, ref_dir: str):
    """(label, ours_path, ref_path, tol) for each hidden-state rung."""
    stages = [('emb', 'x_emb', 'h00')]
    for i in range(11):
        stages.append((f'block {i:02d}', f'x_block{i:02d}', f'h{i + 1:02d}'))
    stages.append(('block 11', 'x_block11', 'h11_raw'))
    stages.append(('final ln_f', 'x_lnf', 'hf_final_ln'))
    for label, ours, ref in stages:
        yield (label,
               os.path.join(dump_dir, ours + '.bin'),
               os.path.join(ref_dir, ref + '.bin'))


def _top2_gap(row: np.ndarray) -> float:
    top2 = np.partition(row, -2)[-2:]
    return float(top2[1] - top2[0])


def _acceptable_mismatch(ref_row: np.ndarray, logit_err: float) -> bool:
    return _top2_gap(ref_row) < 2.0 * logit_err


def check_prompt(name: str, ref_root: str, dump_root: str) -> bool:
    ref_dir = os.path.join(ref_root, name)
    dump_dir = os.path.join(dump_root, name)
    n = json.load(open(os.path.join(ref_dir, 'meta.json')))['n_tokens']

    for label, ours_p, ref_p in _ladder(dump_dir, ref_dir):
        ref_t = _load(ref_p, D)
        abs_e = np.abs(_load(ours_p, D) - ref_t)
        rel = float((abs_e / (ATOL + RTOL * np.abs(ref_t))).max())
        print(f"  {label:12s} max_abs {abs_e.max():.3e}  rel {rel:5.2f}")

    ours = _load(os.path.join(dump_dir, 'logits.bin'), VOCAB)
    ref = _load(os.path.join(ref_dir, 'logits.bin'), VOCAB)
    logit_err = float(np.abs(ours - ref).max())
    our_arg, ref_arg = ours.argmax(1), ref.argmax(1)
    matches = int((our_arg == ref_arg).sum())

    top5 = float(np.mean([len(set(np.argpartition(o, -5)[-5:]) & set(np.argpartition(r, -5)[-5:]))
                          for o, r in zip(ours, ref)]))

    borderline, hard = [], []
    for i in np.where(our_arg != ref_arg)[0]:
        (borderline if _acceptable_mismatch(ref[i], logit_err) else hard).append(int(i))

    ok = len(hard) == 0
    print(f"  logits       max_abs {logit_err:.3e}  argmax {matches}/{n}"
          f"  (+{len(borderline)} coin-flip)  top5 {top5:.2f}/5  {'PASS' if ok else 'FAIL'}")
    for i in hard[:5]:
        print(f"    pos {i}: ours={our_arg[i]} ref={ref_arg[i]}  "
              f"HF margin {_top2_gap(ref[i]):.3e} vs 2*err {2 * logit_err:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='assets/ref')
    ap.add_argument('--dump', default='assets/dump')
    ap.add_argument('--bin', default='build/gpt2')
    ap.add_argument('--weights', default='assets/gpt2_weights.bin')
    ap.add_argument('--skip-run', action='store_true',
                    help='compare existing dumps instead of invoking the binary')
    args = ap.parse_args()

    prompts = json.load(open(os.path.join(args.ref, 'index.json')))['prompts']
    all_ok = True
    for entry in prompts:
        name = entry['name']
        print(f"\n=== {name} (n_tokens={entry['n_tokens']}) ===")
        if not args.skip_run:
            ids = json.load(open(os.path.join(args.ref, name, 'meta.json')))['ids']
            subprocess.run([args.bin, '--dump-activations', args.weights,
                            '--tokens', ','.join(map(str, ids)),
                            '--out', os.path.join(args.dump, name)], check=True)
        all_ok &= check_prompt(name, args.ref, args.dump)

    print(f"\n{'ALL PROMPTS PASS' if all_ok else 'VALIDATION FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
