#!/usr/bin/env python3
"""Correctness check: dumped logits vs HF gpt2-xl for one prompt.

  ./build/gpt2 --dump-activations --tokens "464,2068,7586,21831" --out assets/dump/fox
  python3 python/gpt2xl_check.py assets/dump/fox/logits.bin "464,2068,7586,21831"
"""
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel

VOCAB = 50257
dump = sys.argv[1] if len(sys.argv) > 1 else 'assets/dump/fox/logits.bin'
arg_ids = sys.argv[2].strip() if len(sys.argv) > 2 else ''
ids = [int(x) for x in arg_ids.split(',')] if arg_ids else [464, 2068, 7586, 21831]

ours = np.fromfile(dump, dtype=np.float32).reshape(-1, VOCAB)
assert ours.shape[0] == len(ids), f"dump has {ours.shape[0]} rows, prompt has {len(ids)}"

model = GPT2LMHeadModel.from_pretrained('gpt2-xl').eval()
with torch.no_grad():
    ref = model(torch.tensor([ids])).logits[0].numpy()

max_abs = float(np.abs(ours - ref).max())
matches = int((ours.argmax(1) == ref.argmax(1)).sum())
n = len(ids)
print(f"logits: max_abs {max_abs:.3e}  argmax {matches}/{n}  "
      f"{'PASS' if matches == n else 'FAIL'}")
