#!/usr/bin/env python3
"""Encode a prompt to comma-separated GPT-2 BPE ids for `gpt2 --tokens`."""
import sys
import tiktoken

enc = tiktoken.get_encoding("gpt2")
text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
print(",".join(str(i) for i in enc.encode(text)))
