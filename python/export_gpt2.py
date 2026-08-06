#!/usr/bin/env python3
"""Export GPT-2 weights, tokenizer, reference activations, and kernel fixtures.

Usage:
  python3 python/export_gpt2.py weights  [--out-dir assets]
  python3 python/export_gpt2.py tokens   [--out-dir assets]
  python3 python/export_gpt2.py ref      [--out-dir assets] [--dump-block0]
  python3 python/export_gpt2.py fixtures [--out-dir assets]
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


REF_PROMPT = "The quick brown fox"

PARA_PROMPT = (
    "Machine translation systems have improved dramatically over the past decade, "
    "driven by larger models, richer training data, and architectural advances such "
    "as attention. They still stumble on rare idioms and the long-range context that "
    "human translators resolve almost without thinking."
)
FILLER_TEXT = (
    "The history of computing is a history of abstraction, each layer resting on the "
    "one beneath it so a programmer need not reason about electrons to write a loop. "
)


def _ids_of_length(enc, n: int) -> list:
    """Deterministic list of exactly n token ids, by repeating FILLER_TEXT."""
    base = enc.encode(FILLER_TEXT)
    ids = []
    while len(ids) < n:
        ids += base
    return ids[:n]


def _dump_tensor(path: str, t: torch.Tensor, meta: dict, key: str):
    """Save a torch tensor to raw fp32 and record its shape in `meta`."""
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
    arr.tofile(path)
    meta[key] = {'shape': list(arr.shape), 'path': os.path.basename(path)}


def _hook_output(store: dict, key: str):
    """Forward hook that captures o (or o[0] if o is a tuple) into store[key]."""
    def _hook(_m, _inp, out):
        store[key] = out[0] if isinstance(out, tuple) else out
    return _hook


def _dump_one(model, ids, ref_dir: str, text=None, dump_block0: bool = False) -> int:
    """Run the model on `ids` and dump the reference activation ladder into ref_dir.

    Returns the argmax over the last position's logits.
    """
    os.makedirs(ref_dir, exist_ok=True)
    x = torch.tensor([ids])

    # Hook block 11 directly to capture its unnormalized output.
    raw11 = {}
    h11_handle = model.transformer.h[11].register_forward_hook(_hook_output(raw11, 'x'))

    block0 = {}
    block0_handles = []
    if dump_block0:
        h0 = model.transformer.h[0]
        block0_handles = [
            h0.ln_1.register_forward_hook(_hook_output(block0, 'ln1')),
            h0.attn.c_attn.register_forward_hook(_hook_output(block0, 'c_attn')),
            h0.attn.register_forward_hook(_hook_output(block0, 'attn')),
            h0.mlp.register_forward_hook(_hook_output(block0, 'mlp')),
        ]

    with torch.no_grad():
        out = model(x, output_hidden_states=True)

    h11_handle.remove()
    for h in block0_handles:
        h.remove()

    hs = out.hidden_states
    assert len(hs) == 13, f"unexpected hidden_states length: {len(hs)}"

    meta = {
        'prompt': text,
        'ids': ids,
        'n_tokens': len(ids),
        'note': ('h{i}.bin holds hidden_states[i] under HF indexing: '
                 'h00 = embeddings, h01..h11 = outputs of blocks 0..10, '
                 'h11_raw = raw block-11 output (via hook), '
                 'hf_final_ln = hidden_states[12] = post-ln_f'),
        'tensors': {},
    }

    for i in range(12):
        _dump_tensor(os.path.join(ref_dir, f'h{i:02d}.bin'),
                     hs[i].squeeze(0), meta['tensors'], f'h{i:02d}')
    _dump_tensor(os.path.join(ref_dir, 'h11_raw.bin'),
                 raw11['x'].squeeze(0), meta['tensors'], 'h11_raw')
    _dump_tensor(os.path.join(ref_dir, 'hf_final_ln.bin'),
                 hs[12].squeeze(0), meta['tensors'], 'hf_final_ln')
    _dump_tensor(os.path.join(ref_dir, 'logits.bin'),
                 out.logits.squeeze(0), meta['tensors'], 'logits')

    if dump_block0:
        for k in ('ln1', 'c_attn', 'attn', 'mlp'):
            _dump_tensor(os.path.join(ref_dir, f'block0_{k}.bin'),
                         block0[k].squeeze(0), meta['tensors'], f'block0_{k}')

    with open(os.path.join(ref_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return int(out.logits[0, -1].argmax())


def cmd_ref(out_dir: str, dump_block0: bool = False):
    import tiktoken
    ref_root = os.path.join(out_dir, 'ref')
    os.makedirs(ref_root, exist_ok=True)

    print("loading gpt2 for reference dump...")
    model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
    enc = tiktoken.get_encoding('gpt2')

    prompts = [
        ('fox', enc.encode(REF_PROMPT), REF_PROMPT),
        ('para', enc.encode(PARA_PROMPT), PARA_PROMPT),
        ('ctx64', _ids_of_length(enc, 64), None),
        ('ctx1000', _ids_of_length(enc, 1000), None),
    ]

    index = []
    for name, ids, text in prompts:
        argmax_last = _dump_one(model, ids, os.path.join(ref_root, name),
                                text=text, dump_block0=(dump_block0 and name == 'fox'))
        index.append({'name': name, 'n_tokens': len(ids), 'argmax_last': argmax_last})
        print(f"  {name:8s} n_tokens={len(ids):4d}  argmax@last={argmax_last}")

    with open(os.path.join(ref_root, 'index.json'), 'w') as f:
        json.dump({'prompts': index}, f, indent=2)

    print(f"wrote {ref_root}/ for {len(prompts)} prompts (index.json + per-prompt meta.json)")


def _write_fixture(fx_dir: str, name: str, arr: np.ndarray, meta: dict):
    """Write an ndarray to fixtures/<name>.bin and record shape+dtype."""
    arr = np.ascontiguousarray(arr)
    arr.tofile(os.path.join(fx_dir, f'{name}.bin'))
    meta[name] = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}


def cmd_fixtures(out_dir: str):
    from transformers.activations import NewGELUActivation
    fx_dir = os.path.join(out_dir, 'fixtures')
    os.makedirs(fx_dir, exist_ok=True)
    rng = np.random.default_rng(0xF1)
    meta = {}

    D_MODEL = CONFIG['d_model']          # 768
    N_HEADS = CONFIG['n_head']           # 12
    D_HEAD = D_MODEL // N_HEADS          # 64
    D_FF = 4 * D_MODEL                   # 3072 (GPT-2 MLP intermediate)
    ROWS = 64                            # fixture row count;

    # layernorm: rows=ROWS, C=D_MODEL, eps=1e-5
    x = rng.standard_normal((ROWS, D_MODEL)).astype(np.float32)
    w = rng.standard_normal(D_MODEL).astype(np.float32)
    b = rng.standard_normal(D_MODEL).astype(np.float32)
    y = torch.nn.functional.layer_norm(
        torch.from_numpy(x), [D_MODEL],
        torch.from_numpy(w), torch.from_numpy(b), eps=1e-5,
    ).numpy()
    _write_fixture(fx_dir, 'layernorm_x', x, meta)
    _write_fixture(fx_dir, 'layernorm_w', w, meta)
    _write_fixture(fx_dir, 'layernorm_b', b, meta)
    _write_fixture(fx_dir, 'layernorm_out', y, meta)

    # gelu_new: elementwise, x over [-6, 6]
    x = np.linspace(-6.0, 6.0, 4096).astype(np.float32)
    y = NewGELUActivation()(torch.from_numpy(x)).numpy()
    _write_fixture(fx_dir, 'gelu_new_in', x, meta)
    _write_fixture(fx_dir, 'gelu_new_out', y, meta)

    # qkv_split: N_real=7 with N_pad=64.
    # Kernel writes rows [0, N_real) into a zero-init [H, N_pad, D] buffer.
    N_real, N_pad, H, D = 7, 64, N_HEADS, D_HEAD
    qkv = rng.standard_normal((N_real, 3 * H * D)).astype(np.float32)
    Qp, Kp, Vp = (np.zeros((H, N_pad, D), np.float32) for _ in range(3))
    q, k, v = torch.from_numpy(qkv).split(H * D, dim=-1)
    Qp[:, :N_real] = q.view(N_real, H, D).permute(1, 0, 2).contiguous().numpy()
    Kp[:, :N_real] = k.view(N_real, H, D).permute(1, 0, 2).contiguous().numpy()
    Vp[:, :N_real] = v.view(N_real, H, D).permute(1, 0, 2).contiguous().numpy()
    _write_fixture(fx_dir, 'qkv_split_in', qkv, meta)
    _write_fixture(fx_dir, 'qkv_split_Q',  Qp,  meta)
    _write_fixture(fx_dir, 'qkv_split_K',  Kp,  meta)
    _write_fixture(fx_dir, 'qkv_split_V',  Vp,  meta)
    meta['qkv_split_in']['n_real'] = N_real
    meta['qkv_split_in']['n_pad'] = N_pad

    # merge_heads: reads only rows [0, N_real). Padded input rows are garbage-by-design
    O_heads = rng.standard_normal((H, N_pad, D)).astype(np.float32)
    O_out = O_heads[:, :N_real, :].transpose(1, 0, 2).reshape(N_real, H * D).copy()
    _write_fixture(fx_dir, 'merge_heads_in', O_heads, meta)
    _write_fixture(fx_dir, 'merge_heads_out', O_out, meta)
    meta['merge_heads_in']['n_real'] = N_real
    meta['merge_heads_in']['n_pad'] = N_pad

    # bias_add: exercises the widest bias (MLP hidden)
    x = rng.standard_normal((ROWS, D_FF)).astype(np.float32)
    bias = rng.standard_normal(D_FF).astype(np.float32)
    _write_fixture(fx_dir, 'bias_add_x', x, meta)
    _write_fixture(fx_dir, 'bias_add_bias', bias, meta)
    _write_fixture(fx_dir, 'bias_add_out', x + bias, meta)

    # residual_add: elementwise
    x = rng.standard_normal((ROWS, D_MODEL)).astype(np.float32)
    y = rng.standard_normal((ROWS, D_MODEL)).astype(np.float32)
    _write_fixture(fx_dir, 'residual_add_x', x, meta)
    _write_fixture(fx_dir, 'residual_add_y', y, meta)
    _write_fixture(fx_dir, 'residual_add_out', x + y, meta)

    # gemm_rm: C[m,n] = A[m,k] * B[k,n]
    M, K, N = 5, 7, 3
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    _write_fixture(fx_dir, 'gemm_rm_A', A, meta)
    _write_fixture(fx_dir, 'gemm_rm_B', B, meta)
    _write_fixture(fx_dir, 'gemm_rm_C', A @ B, meta)

    # gemm_rm_bt: C[m,n] = A[m,k] * B[n,k]^T
    Abt = rng.standard_normal((M, K)).astype(np.float32)
    Bbt = rng.standard_normal((N, K)).astype(np.float32)
    _write_fixture(fx_dir, 'gemm_bt_A', Abt, meta)
    _write_fixture(fx_dir, 'gemm_bt_B', Bbt, meta)
    _write_fixture(fx_dir, 'gemm_bt_C', Abt @ Bbt.T, meta)

    # embedding_gather: small toy vocab so the fixture is tiny.
    V_small, N_ctx, D_emb = 100, 16, 32
    ids = rng.integers(0, V_small, size=N_ctx).astype(np.int32)
    wte = rng.standard_normal((V_small, D_emb)).astype(np.float32)
    wpe = rng.standard_normal((N_ctx, D_emb)).astype(np.float32)
    out_arr = wte[ids] + wpe
    _write_fixture(fx_dir, 'embedding_gather_ids', ids, meta)
    _write_fixture(fx_dir, 'embedding_gather_wte', wte, meta)
    _write_fixture(fx_dir, 'embedding_gather_wpe', wpe, meta)
    _write_fixture(fx_dir, 'embedding_gather_out', out_arr, meta)

    with open(os.path.join(fx_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {fx_dir}/ ({len(meta)} tensors across 7 kernels + 2 gemm wrappers)")


COMMANDS = {
    'weights':  cmd_weights,
    'tokens':   cmd_tokens,
    'ref':      cmd_ref,
    'fixtures': cmd_fixtures,
}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        sp.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
        if name == 'ref':
            sp.add_argument('--dump-block0', action='store_true',
                            help='also dump intra-block-0 hooks for block-forward debugging')
    args = parser.parse_args()
    if args.cmd == 'ref':
        cmd_ref(args.out_dir, dump_block0=args.dump_block0)
    else:
        COMMANDS[args.cmd](args.out_dir)


if __name__ == '__main__':
    main()
