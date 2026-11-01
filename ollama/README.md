# Ollama setup for local harness runs

Local additions on top of the Kaggle files at the repository root. None of this is part of the competition download.

| File | Purpose |
| --- | --- |
| `models.yaml` | `swegemma eval --models-yaml` registry. Routes the competition alias `gemma-4-31b-it-qat-w4a16-ct` to `ollama_chat/gemma4:31b-cloud` at `http://localhost:11434`. |
| `smoke_ollama_agent.py` | Runs an ADK agent with in-memory fake tools through the same model registry that the `swegemma` CLI builds. Checks that Gemma emits parseable tool calls. No repository code runs. |
| `requirements.lock.txt` | `uv pip freeze` of `.venv` (Windows, Python 3.12.6), checked 2026-11-01. |
| `../host-wheelhouse/` | Harness wheels from the organizer dataset [`metric/gemma-4-developer-agent-wheelhouse`](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse): `swegemma` 0.2.11, `adk-submission` 0.2.13, `adk-eval-core` 0.1.0. |

## Setup

```text
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r ollama/requirements.lock.txt
python ollama/smoke_ollama_agent.py --models-yaml ollama/models.yaml
```

Run commands from the repository root. The smoke test prints `PASS` when the model reads the file, makes the correct edit, and calls `submit_patch`.

## Known constraints

- `gemma4:31b-cloud` is not the competition's QAT W4A16 build. Record the served model in every run's metadata. The closest local tag is `gemma4:31b-it-qat` (18.9 GB).
- `litellm` is pinned to 1.99.4. Version 1.106.6 imports `mcp.types.CacheableResult` (mcp 2.x) on every completion call, while `google-adk` 1.36.1 requires `mcp<2`.
- `swegemma` is installed with `--no-deps`. Its `torchvision`, `transformers` and `accelerate` requirements serve local weights only and are not needed with Ollama.
- Full task runs need the Docker sandbox (`docker/Dockerfile.sandbox`), so they need WSL2. The subprocess sandbox uses POSIX-only calls (`os.killpg`, `/bin/bash`) and does not run on native Windows.
- `snapshots/` is not stored in Git. Verify it with `sha256sum -c snapshots.sha256`.
