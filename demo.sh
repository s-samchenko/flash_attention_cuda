#!/usr/bin/env bash
# demo.sh "Once upon a time" [max_new]
# Streams generated text for the prompt, typed token-by-token. Run from the repo root.
set -euo pipefail

prompt="${1:-Once upon a time}"
max_new="${2:-100}"

ids="$(python3 python/encode.py "$prompt")"
printf '%s' "$prompt"
./build/gpt2 --tokens "$ids" --generate "$max_new" --temp 0.8
