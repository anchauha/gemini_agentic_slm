#!/usr/bin/env bash
# Serve gemma-4-E2B-it-qat-w4a16-ct on an 8 GB laptop GPU (RTX 4060, WSL) with the competition flags.
# Laptop-only additions, none of which changes model outputs:
#   --language-model-only      skip vision/audio encoders (text-only agent)
#   --enforce-eager            torch.compile copies the 4.7 GB per-layer embedding table and runs out of memory
#   --gpu-memory-utilization   0.86 is the most WSL leaves free (6.93 of 8 GiB)
#   --max-num-seqs 1 / --max-num-batched-tokens 1024   smaller activation buffers; the harness sends one request at a time
# Usage (repo root): bash local/serve_laptop.sh [--submission-dir local/lora_agent]
# On the H100: python local/serve_vllm.py --model ~/models/gemma-4-31B-it-qat-w4a16-ct [--tensor-parallel-size N]
cd "$(dirname "$0")/.."
exec ~/venvs/harness/bin/python local/serve_vllm.py --model ~/models/gemma-4-E2B-it-qat-w4a16-ct \
  --gpu-memory-utilization 0.86 "$@" -- \
  --language-model-only --enforce-eager --max-num-seqs 1 --max-num-batched-tokens 1024
