#!/usr/bin/env python3
"""Export GPT-2 weights and tokenizer table for the C++ loader.

Usage:
  python3 python/export_gpt2.py weights [--out-dir assets]
  python3 python/export_gpt2.py tokens  [--out-dir assets]
"""

import argparse
import json
import os
import struct

import numpy as np
import torch
from transformers import GPT2LMHeadModel

# Header layout (little-endian, 32 bytes total):
# uint32 magic = 0x47505432 ("GPT2")
# uint32 version = 1
# int32 n_layer, n_head, d_model, n_ctx, vocab, pad
MAGIC = 0x47505432
VERSION = 1
CONFIG = dict(n_layer=12, n_head=12, d_model=768, n_ctx=1024, vocab=50257)
HEADER_BYTES = 32

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')


def _tensor_order():
    order = ['transformer.wte.weight', 'transformer.wpe.weight']
    per_block = [
        'ln_1.weight', 'ln_1.bias',
        'attn.c_attn.weight', 'attn.c_attn.bias',
        'attn.c_proj.weight', 'attn.c_proj.bias',
        'ln_2.weight', 'ln_2.bias',
        'mlp.c_fc.weight', 'mlp.c_fc.bias',
        'mlp.c_proj.weight', 'mlp.c_proj.bias',
    ]
    for i in range(CONFIG['n_layer']):
        for name in per_block:
            order.append(f'transformer.h.{i}.{name}')
    order += ['transformer.ln_f.weight', 'transformer.ln_f.bias']
    return order


def _tensor_summary(t: np.ndarray) -> dict:
    flat = t.reshape(-1).astype(np.float64)
    return {
        'shape': list(t.shape),
        'sum': float(flat.sum()),
        'mean': float(flat.mean()),
        'first': float(flat[0]),
        'last': float(flat[-1]),
    }


def cmd_weights(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, 'gpt2_weights.bin')
    manifest_path = os.path.join(out_dir, 'gpt2_weights.manifest.json')

    print("loading gpt2 from huggingface...")
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    sd = model.state_dict()

    order = _tensor_order()
    manifest = {
        'magic': MAGIC, 'version': VERSION, 'config': CONFIG,
        'header_bytes': HEADER_BYTES, 'tensors': [],
    }
    total_params = 0

    with open(bin_path, 'wb') as f:
        f.write(struct.pack(
            '<IIiiiiii',
            MAGIC, VERSION,
            CONFIG['n_layer'], CONFIG['n_head'], CONFIG['d_model'],
            CONFIG['n_ctx'], CONFIG['vocab'], 0,
        ))
        for key in order:
            if key not in sd:
                raise KeyError(f"missing weight key: {key}")
            t = sd[key].detach().to(torch.float32).contiguous().numpy()
            f.write(t.tobytes())
            summary = _tensor_summary(t)
            summary['key'] = key
            manifest['tensors'].append(summary)
            total_params += t.size

    actual = os.path.getsize(bin_path)
    expected = HEADER_BYTES + 4 * total_params
    assert actual == expected, f"size mismatch: got {actual}, expected {expected}"
    print(f"wrote {bin_path}: {actual:,} bytes ({actual/1e6:.1f} MB), "
          f"{len(order)} tensors, {total_params:,} params")

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


def cmd_tokens(out_dir: str):
    """Byte-level decode table for GPT-2"""
    import tiktoken
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, 'gpt2_tokens.bin')

    enc = tiktoken.get_encoding('gpt2')
    count = CONFIG['vocab']
    blobs = []
    for tid in range(count):
        try:
            blobs.append(enc.decode_single_token_bytes(tid))
        except KeyError:
            blobs.append(b'')

    offsets = [0]
    for b in blobs:
        offsets.append(offsets[-1] + len(b))
    blob = b''.join(blobs)

    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<I', count))
        f.write(np.asarray(offsets, dtype=np.uint32).tobytes())
        f.write(blob)

    def _decode(tid: int) -> str:
        s, e = offsets[tid], offsets[tid + 1]
        try:
            return blob[s:e].decode('utf-8')
        except UnicodeDecodeError:
            return f"<bytes {blob[s:e]!r}>"

    print(f"wrote {bin_path}: {os.path.getsize(bin_path):,} bytes, "
          f"{count} tokens, blob {len(blob):,} bytes")
    print(f"spot check: 464='{_decode(464)}' 995='{_decode(995)}' "
          f"7586='{_decode(7586)}' 50256={_decode(50256)!r}")


COMMANDS = {
    'weights': cmd_weights,
    'tokens': cmd_tokens,
}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        sp.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    COMMANDS[args.cmd](args.out_dir)


if __name__ == '__main__':
    main()
