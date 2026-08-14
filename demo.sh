#!/usr/bin/env bash
# demo.sh "prompt" [max_new] [temp]
# Streams a continuation for the prompt, typed token-by-token. Run from repo root.
set -euo pipefail

prompt="${1-}"
max_new="${2:-100}"
temp="${3:-0.8}"

if [ -z "$prompt" ]; then
    prompt="$(cat)"
fi

ids="$(printf '%s' "$prompt" | python3 python/encode.py)"
printf '%s' "$prompt"
./build/gpt2 --tokens "$ids" --generate "$max_new" --temp "$temp"
