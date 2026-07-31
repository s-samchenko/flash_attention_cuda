#!/usr/bin/env python3
"""Diff the C++ loader's manifest against the Python export manifest.

Usage:
  python3 python/verify_manifest.py
  python3 python/verify_manifest.py --sum-rtol 1e-3

`first` and `last` must match exactly (same float32 bits round-tripped through
both writers). `sum` is compared with a relative tolerance because summation
order differs: numpy sums in float64 across a flat array, the C++ loader
accumulates float-by-float into a double register.
"""
import argparse
import json
import os
import sys

DEFAULT_EXPORT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'gpt2_weights.manifest.json')
DEFAULT_LOADER = os.path.join(os.path.dirname(__file__), '..', 'assets', 'gpt2_weights.loader.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default=DEFAULT_EXPORT)
    ap.add_argument('--loader', default=DEFAULT_LOADER)
    ap.add_argument('--sum-rtol', type=float, default=1e-3)
    args = ap.parse_args()

    with open(args.export) as f: exp = json.load(f)
    with open(args.loader) as f: got = json.load(f)

    def die(msg):
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)

    if exp['magic'] != got['magic']: die(f"magic: {exp['magic']} vs {got['magic']}")
    if exp['version'] != got['version']: die(f"version: {exp['version']} vs {got['version']}")
    if exp['config'] != got['config']: die(f"config: {exp['config']} vs {got['config']}")
    if len(exp['tensors']) != len(got['tensors']):
        die(f"tensor count: {len(exp['tensors'])} vs {len(got['tensors'])}")

    fails = 0
    for e, g in zip(exp['tensors'], got['tensors']):
        if e['key'] != g['key']:
            die(f"order mismatch at some position: export={e['key']} loader={g['key']}")
        errs = []
        if e['shape'] != g['shape']:
            errs.append(f"shape {e['shape']} vs {g['shape']}")
        if e['first'] != g['first']:
            errs.append(f"first {e['first']!r} vs {g['first']!r}")
        if e['last'] != g['last']:
            errs.append(f"last {e['last']!r} vs {g['last']!r}")
        rel = abs(e['sum'] - g['sum']) / (abs(e['sum']) + 1e-30)
        if rel > args.sum_rtol:
            errs.append(f"sum rel_err={rel:.2e} > {args.sum_rtol:.0e}")
        if errs:
            fails += 1
            print(f"FAIL {e['key']}: {'; '.join(errs)}")

    n = len(exp['tensors'])
    if fails == 0:
        print(f"OK: {n}/{n} tensors match (first/last exact, sums within {args.sum_rtol:.0e})")
    else:
        print(f"FAIL: {fails}/{n} tensors had check failures", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
