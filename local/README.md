# Local harness runs

Local additions on top of the Kaggle files at the repository root. None of this is part of the competition download. Everything runs in WSL Ubuntu 24.04 (or any Linux host): vLLM serves the model on `localhost:8000`, and the official `swegemma` harness drives Docker sandboxes.

| File | Purpose |
| --- | --- |
| `setup_wsl.sh` | One-time setup: `~/venvs/harness` (organizer harness wheels), `~/venvs/vllm` (organizer vLLM wheel), model weights, `swebench-sandbox:latest` and `:local` images. |
| `Dockerfile.sandbox-extras` | `swebench-sandbox:local`: the organizer image plus 7 test packages missing from the public `wheels/` cache. Default for `run_task.sh` and `verify_gold.py`. |
| `serve_vllm.py` | Builds the vLLM command with `adk_submission`'s own `VllmServer` and the Kaggle settings, served under the competition alias. Adapters in `<submission>/adapters/` are mounted as on Kaggle. |
| `serve_laptop.sh` | `serve_vllm.py` for `gemma-4-E2B-it-qat-w4a16-ct` on the 8 GB RTX 4060, with the laptop-only flags explained inline. |
| `run_task.sh` | Runs one task through `swegemma eval` against the local server (quick checks; budgets 50 calls / 30 min, below the scorer's). |
| `raw_agent/agent.yaml` | Minimal single agent for raw-model studies: all 9 harness tools, no adapters, no sub-agents, one-line instruction. |
| `lora_agent/` | `raw_agent` plus `adapter: main_lora`, to test adapter serving end to end. |
| `make_test_adapter.py` | Writes a zero-effect PEFT adapter shaped like the sample submission's for any Gemma 4 base (the sample adapters fit only 31B). |
| `verify_gold.py` | Feeds each task's reference patch through the harness's `verify_task` (Container B). No model is called. |
| `gold_resolvable.txt` | The 56 tasks whose reference patch resolves locally on `swebench-sandbox:local` (2026-11-02). |
| `repair/` | Repaired environment (`swebench-sandbox:repaired`): `build_image.sh` builds it, `repair_env.py` runs in every container after the harness's setup. See "Environment modes". |
| `swegemma_repaired.py` | The `swegemma` CLI with `repair_env.py` hooked into the harness's `install_test_dependencies`. Used by `REPAIRED=1 run_task.sh` and `verify_gold.py --repaired`. |
| `gold_resolvable_repaired.txt` | The 118 tasks whose reference patch resolves in the repaired environment (2026-11-03). `gold_resolvable_faithful.txt` lists the 43 for the organizer image. |
| `runlog.py` | Runs the harness task by task with the scorer's budgets and records every run in `../logs/`. See "Run log". |
| `../host-wheelhouse/` | Harness wheels from the organizer dataset [`metric/gemma-4-developer-agent-wheelhouse`](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse): `swegemma` 0.2.11, `adk-submission` 0.2.13, `adk-eval-core` 0.1.0. |

## Usage

```text
bash local/setup_wsl.sh                                   # once
bash local/serve_laptop.sh                                # terminal 1; add --submission-dir local/lora_agent for adapters
bash local/run_task.sh e2b_rich_3061 rich_3061            # terminal 2; results in runs/e2b_rich_3061
```

On the H100, change only the model and GPU count; the agent YAML, served name and flags stay the same:

```text
~/venvs/harness/bin/python local/serve_vllm.py --model ~/models/gemma-4-31B-it-qat-w4a16-ct --tensor-parallel-size <N>
```

The server always answers as `gemma-4-31b-it-qat-w4a16-ct`, whatever checkpoint it loads. Record the real `--model` in every run's metadata.

## Environment modes

Label every run with its mode; scores from different modes are not comparable.

| Mode | How | Gold patches resolve |
| --- | --- | --- |
| Faithful | `--image swebench-sandbox:latest` (organizer image as shipped) | 43/129 |
| Local | default `run_task.sh` (`swebench-sandbox:local`: + 7 test packages) | 56/129 |
| Repaired | `REPAIRED=1 bash local/run_task.sh ...`, `verify_gold.py --repaired` | 118/129 |

Repaired mode follows the scorer-like sandbox of Kaggle discussion 751487 (Zbigniew Kirchner, 127/129 in a notebook), adapted to the Docker backend. After the harness's own setup, `repair/repair_env.py`:

1. installs 64 test packages missing from `wheels/` (versions pinned in `repair/wheels.txt`, downloaded from PyPI by `build_image.sh`), with pytest 8.4.2 and pytest-timeout 2.1.0 as reported for the scorer image instead of the image's newest pytest; brotli, httpbin and similar go only to the repository that needs them;
2. removes the repository's own released wheel, which the harness injects into site-packages and which shadows `/workspace` (`requests_6644`, `httpx_3672`);
3. installs, for each dependency in the snapshot's `pyproject.toml`, the newest wheel the snapshot allows (the harness always injects the newest, so old fastapi commits got Starlette 1.6);
4. applies a per-task calibration taken from the same source: period Pygments for 7 rich tasks, werkzeug 2.3.8/flask 2.3.3 for 3 requests tasks, an asyncio test runner for 3 fastapi tasks.

Calibrations were found by trying versions with the gold patch, so they make tasks gradable; they are not information an agent could have. Ceiling in this mode is 118: `rich_3486` and `fastapi_14186` require tests with opposite skip conditions; 8 requests tasks fail `TestTimeout` connect tests because the harness runs containers with `network_mode='none'` (no route, so the connection fails instead of timing out); `rich_3472` needs Python 3.12 (`requests_6757` also needs 3.12 and certificate-time checks off). Whether the hosted scorer uses Docker or the subprocess sandbox is unconfirmed; the organizers report that hidden tasks' gold patches all pass on the scorer.

## Run log

`runlog.py run --name <run> --mode <faithful|local|repaired> [--submission-dir ...] [--tasks ...] [--notes ...]` runs each task through the unmodified harness CLI (budgets default to the scorer's: 100 tool calls, 60 min, 500 turns, 300 s per command) and writes:

| Path | Tracked in Git | Content |
| --- | --- | --- |
| `logs/<run>/manifest.json` | yes | Checkpoint actually loaded (read from vLLM `/v1/models`, since the alias is always the 31B name), vLLM command line, agent.yaml text and hash, mode and image id, budgets, package versions, repo commit |
| `logs/<run>/results.jsonl` | yes | Per task: resolved, outcome category, patch submitted, tool calls and tool errors, tokens, duration, first failing test |
| `logs/<run>/summary.md` | yes | Totals by repo and outcome, over all tasks and over tasks whose gold patch resolves in that mode |
| `logs/ledger.csv`, `logs/runs.csv` | yes | Every task of every run, and one line per run, for comparisons over time |
| `runs/<run>/<task>/` | no | Harness output: patch, trace, test output, harness log |

Reusing a name resumes the run; it refuses to continue if the checkpoint, agent config, mode or budgets changed. `runlog.py compare <runA> <runB>` lists tasks that flipped and outcome changes. Outcome categories: `resolved`, `tests_failed`, `tests_not_collected` (import or syntax error, usually from the patch), `no_patch`, `budget_exhausted_no_patch`, `patch_not_applied`, `harness_error`.

## Verified on the laptop (2026-11-02)

- E2B QAT W4A16 serves with a full 32k context: 6.37 GiB weights, 0.53 GiB KV cache (18,592 tokens), about 19 tokens/s in eager mode.
- `gemma4` tool-call and reasoning parsers return clean tool calls with reasoning separated.
- `raw_agent` on `rich_3061` ran end to end (5 tool calls, patch extracted, verified in Container B; not resolved).
- `lora_agent` loads `main_lora` and the harness routes requests through it.

## Known constraints

- **Use the organizer's vLLM wheel.** The public PyPI `vllm==0.19.1` rejects Gemma 4 adapters (`Gemma4ForConditionalGeneration does not support LoRA yet`). The wheelhouse build (433,132,704 bytes, sha256 `73c69baf…051310`) patches `gemma4.py`, `gemma4_mm.py` (adds `SupportsLoRA`) and `lora/model_manager.py`. 31B uses the same architecture, so this applies on the H100 too.
- Pin `transformers==5.13.1` and `tokenizers==0.22.2` (wheelhouse versions). `transformers` 5.22 breaks vLLM 0.19.1 on the Gemma 4 config (`'head_dim' is a per-layer attribute`).
- Triton compiles C at startup: it needs `build-essential` and Python headers. The system Python lacks `Python.h`, so the vLLM venv uses uv's managed Python.
- vLLM CPU weight offload does not work for this model (`Cannot re-initialize the input batch when CPU weight offloading is enabled`).
- Run the harness on Linux only. On native Windows it resolves `Path('/workspace')` as `C:\workspace`, so tool paths are silently rewritten and created files land in the patch under that name (seen 2026-11-01 on `requests_6644`). The subprocess sandbox also uses POSIX-only calls (`os.killpg`).
- `litellm` must stay at 1.99.4 (or another release that works with `mcp<2`). Version 1.106.6 imports `mcp.types.CacheableResult` (mcp 2.x) on every completion call, while `google-adk` 1.36.1 requires `mcp<2`.
- Reference patches resolve on 43 of 129 tasks with the organizer image and on 56 with `swebench-sandbox:local`, with no task lost. The public `wheels/` cache lacks `typing_inspection`, `dirty_equals`, `attrs`, `inline_snapshot` (+ `asttokens`, `executing`) and `python-multipart`; the local image adds them. `inline_snapshot`'s pytest plugin entry point is removed there: auto-loaded in every run, it imported `rich` from site-packages before the workspace copy and broke 7 rich reference patches.
- Remaining 73 failures on `swebench-sandbox:local` (repaired mode fixes 62 of them): 48 fastapi tasks get Starlette 1.6.0 (`Router.__init__() got an unexpected keyword argument 'on_startup'`) because both `sandbox/setup.py` and `swegemma` keep only the highest version of each wheel and strip version pins, even when the snapshot requires `starlette<0.51.0`; 8 requests tests need network; 2 import a site-packages copy instead of `/workspace` (`requests_6644`, `httpx_3672`); 2 need `ujson`/`orjson`; 13 other assertion or exit failures. Whether Kaggle scoring shares these gaps is unverified.
- `snapshots/` is not stored in Git. Verify it with `sha256sum -c snapshots.sha256`.
