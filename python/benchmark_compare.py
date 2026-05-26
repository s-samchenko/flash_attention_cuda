#!/usr/bin/env python3
"""Run benchmarks and generate comparison plots.

Usage:
  python3 python/benchmark_compare.py --run                       # collect pytorch_sdpa + all kernels
  python3 python/benchmark_compare.py --run --kernel naive        # collect one kernel only
  python3 python/benchmark_compare.py --table                     # print results.md rows for all kernels
  python3 python/benchmark_compare.py --table --kernel naive      # print rows for one kernel
  python3 python/benchmark_compare.py --plot                      # generate plots (week 10)
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

BINARY      = os.path.join(os.path.dirname(__file__), '..', 'build', 'flash_attn')
RESULTS_DIR = 'benchmarks/benchmark_results'
PLOTS_DIR   = 'benchmarks/plots'

# Locked shapes (seq_len, head_dim, batch, n_heads)
SHAPES = [
    (512,  64, 1, 8),
    (1024, 64, 1, 8),
    (2048, 64, 1, 8),
    (4096, 64, 1, 8),
]


# ── data collection ───────────────────────────────────────────────────────────

def bench_sdpa(seq_len, head_dim, batch, n_heads, warmup=3, runs=10):
    device = 'cuda'
    Q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    fn = lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


def run_sdpa(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for seq_len, head_dim, batch, n_heads in SHAPES:
        ms    = bench_sdpa(seq_len, head_dim, batch, n_heads)
        flops = 4 * batch * n_heads * seq_len * seq_len * head_dim
        # Flash-style byte count (no N×N materialized)
        bytes_accessed = batch * n_heads * 4 * seq_len * head_dim * 4
        line = (f"ms: {ms:.4f} flops: {flops} bytes: {bytes_accessed} "
                f"seq_len: {seq_len} head_dim: {head_dim} batch: {batch} n_heads: {n_heads}")
        print(line)
        lines.append(line)
    path = os.path.join(out_dir, 'results.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  → {path}")


def run_cuda_kernel(kernel_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run([BINARY, 'bench', kernel_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"binary failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    ms_lines = [l for l in result.stdout.splitlines() if l.startswith('ms:')]
    for line in ms_lines:
        print(line)
    path = os.path.join(out_dir, 'results.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(ms_lines) + '\n')
    print(f"saved → {path}")


# All CUDA kernels in implementation order — add each as it's implemented.
CUDA_KERNELS = [
    'naive',
    # 'fused_softmax',
    # 'fa1_fp32',
    # 'fa2_fp32',
    # 'fa2_fp16',
]


# ── data parsing ──────────────────────────────────────────────────────────────

def parse_results(path):
    records = []
    with open(path) as f:
        for line in f:
            pairs = re.findall(r'(\w+):\s+([\d.]+)', line)
            if not pairs:
                continue
            d = {k: float(v) for k, v in pairs}
            records.append(d)
    return records


def load_all_results():
    data = {}
    if not os.path.isdir(RESULTS_DIR):
        return data
    for name in sorted(os.listdir(RESULTS_DIR)):
        p = os.path.join(RESULTS_DIR, name, 'results.txt')
        if os.path.exists(p):
            data[name] = parse_results(p)
    return data


# ── results.md table ─────────────────────────────────────────────────────────

def get_gpu_name():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        return r.stdout.strip().split('\n')[0]
    except Exception:
        return 'RTX 3090'


def print_table(kernel_filter=None):
    import datetime
    data = load_all_results()
    if not data:
        print("no results found — run with --run first")
        return

    date = datetime.date.today().isoformat()
    gpu  = get_gpu_name()

    naive_ms = {}
    if 'naive' in data:
        for r in data['naive']:
            naive_ms[(int(r['seq_len']), int(r['head_dim']))] = r['ms']

    kernels = [kernel_filter] if kernel_filter else sorted(data.keys())
    for name in kernels:
        if name not in data:
            print(f"no data for kernel '{name}' — run with --run --kernel {name} first")
            continue
        rows = sorted(data[name], key=lambda r: (r['head_dim'], r['seq_len']))
        for r in rows:
            ms     = r['ms']
            sl     = int(r['seq_len'])
            hd     = int(r['head_dim'])
            gbps   = r['bytes'] / (ms / 1000) / 1e9
            tflops = r['flops'] / (ms / 1000) / 1e12
            key    = (sl, hd)
            if name == 'naive':
                vs = '1x'
            elif key in naive_ms and naive_ms[key]:
                vs = f'{naive_ms[key] / ms:.1f}x'
            else:
                vs = '-'
            notes = 'baseline' if name == 'naive' else ''
            print(f"| {date} | {gpu} | {name} | {sl} | {hd} | {ms:.4f} | {gbps:.1f} | {tflops:.4f} | {vs} | {notes} |")


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    'naive':        'tab:blue',
    'pytorch_sdpa': 'tab:green',
    'flash_attn':   'tab:red',
}

def kernel_color(name):
    return COLORS.get(name, None)


def plot_tflops(data, hd=64):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, records in data.items():
        rows = sorted([r for r in records if r.get('head_dim') == hd],
                      key=lambda r: r['seq_len'])
        if not rows:
            continue
        xs = [int(r['seq_len']) for r in rows]
        ys = [r['flops'] / (r['ms'] / 1000) / 1e12 for r in rows]
        ax.plot(xs, ys, marker='o', label=name, color=kernel_color(name))
    ax.set_xlabel('sequence length')
    ax.set_ylabel('TFLOPS')
    ax.set_title(f'Throughput vs sequence length (head_dim={hd}, batch=1, n_heads=8)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bandwidth(data, hd=64):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, records in data.items():
        rows = sorted([r for r in records if r.get('head_dim') == hd],
                      key=lambda r: r['seq_len'])
        if not rows:
            continue
        xs = [int(r['seq_len']) for r in rows]
        ys = [r['bytes'] / (r['ms'] / 1000) / 1e9 for r in rows]
        ax.plot(xs, ys, marker='o', label=name, color=kernel_color(name))
    ax.axhline(936, color='black', linestyle='--', linewidth=0.8, label='RTX 3090 peak (936 GB/s)')
    ax.set_xlabel('sequence length')
    ax.set_ylabel('GB/s')
    ax.set_title(f'Memory bandwidth vs sequence length (head_dim={hd})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pct_ceiling(data, ceiling_key='pytorch_sdpa', hd=64):
    if ceiling_key not in data:
        return None
    ceiling_map = {int(r['seq_len']): r['flops'] / (r['ms'] / 1000) / 1e12
                   for r in data[ceiling_key] if r.get('head_dim') == hd}
    if not ceiling_map:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, records in data.items():
        if name == ceiling_key:
            continue
        rows = sorted([r for r in records if r.get('head_dim') == hd],
                      key=lambda r: r['seq_len'])
        if not rows:
            continue
        xs, ys = [], []
        for r in rows:
            sl = int(r['seq_len'])
            if sl not in ceiling_map or ceiling_map[sl] == 0:
                continue
            xs.append(sl)
            tflops = r['flops'] / (r['ms'] / 1000) / 1e12
            ys.append(100.0 * tflops / ceiling_map[sl])
        if xs:
            ax.plot(xs, ys, marker='o', label=name, color=kernel_color(name))
    ax.axhline(100, color='tab:green', linestyle='--', linewidth=0.8,
               label=f'{ceiling_key} (100%)')
    ax.set_xlabel('sequence length')
    ax.set_ylabel(f'% of {ceiling_key} throughput')
    ax.set_title(f'Relative throughput vs {ceiling_key} (head_dim={hd})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_plots():
    data = load_all_results()
    if not data:
        print("no results found — run with --run first")
        return
    if not HAS_MPL:
        print("matplotlib not available; install it to generate plots")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    for fig, fname in [
        (plot_tflops(data),           'tflops_vs_seqlen_hd64.png'),
        (plot_bandwidth(data),        'bandwidth_vs_seqlen_hd64.png'),
        (plot_pct_ceiling(data),      'pct_ceiling_hd64.png'),
    ]:
        if fig is None:
            continue
        path = os.path.join(PLOTS_DIR, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  → {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',    action='store_true', help='collect benchmark data')
    parser.add_argument('--table',  action='store_true', help='print results.md rows')
    parser.add_argument('--plot',   action='store_true', help='generate plots (week 10)')
    parser.add_argument('--kernel', default=None,        help='target a specific kernel')
    args = parser.parse_args()

    if args.run:
        if args.kernel in (None, 'pytorch_sdpa'):
            print("collecting pytorch_sdpa baseline...")
            run_sdpa(os.path.join(RESULTS_DIR, 'pytorch_sdpa'))
        kernels = [args.kernel] if args.kernel and args.kernel != 'pytorch_sdpa' else CUDA_KERNELS
        for name in kernels:
            print(f"\ncollecting {name}...")
            run_cuda_kernel(name, os.path.join(RESULTS_DIR, name))

    if args.table:
        print_table(args.kernel)

    if args.plot:
        print("\ngenerating plots...")
        generate_plots()


if __name__ == '__main__':
    main()
