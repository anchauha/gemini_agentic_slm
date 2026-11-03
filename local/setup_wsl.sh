#!/usr/bin/env bash
# One-time setup inside WSL Ubuntu 24.04 (or any Linux host with an NVIDIA GPU and Docker).
# Needs: build-essential (sudo apt-get install -y build-essential), Docker reachable as `docker`.
# Run from the repository root: bash local/setup_wsl.sh [hf-model-id]
set -euo pipefail
MODEL_ID=${1:-google/gemma-4-E2B-it-qat-w4a16-ct}
VLLM_WHEEL=vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Harness host: organizer wheels plus the versions known to work together (see local/README.md).
[ -d ~/venvs/harness ] || uv venv -q --python 3.12 ~/venvs/harness
uv pip install -q --python ~/venvs/harness/bin/python --no-deps host-wheelhouse/*.whl
uv pip install -q --python ~/venvs/harness/bin/python google-adk==1.36.1 litellm==1.99.4 mcp==1.30.0 \
  google-genai networkx numpy docker cachetools pandas rich python-dotenv pyyaml pydantic

# vLLM: uv-managed Python (ships Python.h, which Triton compiles against), the organizer's patched
# vLLM wheel (Gemma 4 LoRA support), and the wheelhouse's transformers stack.
[ -d ~/venvs/vllm ] || uv venv -q --managed-python --python 3.12 ~/venvs/vllm
uv pip install -q --python ~/venvs/vllm/bin/python vllm==0.19.1 transformers==5.13.1 tokenizers==0.22.2 \
  compressed-tensors==0.15.0.1 safetensors==0.8.0 "huggingface_hub[cli]"
mkdir -p ~/wheels
if [ ! -f ~/wheels/$VLLM_WHEEL ]; then
  echo "Download $VLLM_WHEEL from kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse into ~/wheels, then rerun." >&2
  exit 1
fi
uv pip install -q --python ~/venvs/vllm/bin/python --no-deps --reinstall ~/wheels/$VLLM_WHEEL
grep -q SupportsLoRA ~/venvs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/gemma4_mm.py

# Weights and the sandbox images (organizer image, then the local one with missing test wheels).
~/venvs/vllm/bin/hf download "$MODEL_ID" --local-dir ~/models/"${MODEL_ID#*/}" >/dev/null
docker build -q -t swebench-sandbox:latest -f docker/Dockerfile.sandbox docker
docker build -q -t swebench-sandbox:local -f local/Dockerfile.sandbox-extras local
echo "setup done: ~/venvs/harness, ~/venvs/vllm, ~/models/${MODEL_ID#*/}, swebench-sandbox:{latest,local}"
