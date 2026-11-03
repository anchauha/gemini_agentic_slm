"""Launch vLLM with the competition's serving flags, built by adk_submission's own VllmServer.

Kaggle settings (HARNESS_README section 3.1): gemma4 tool and reasoning parsers, thinking on,
max_model_len 32768, gpu_memory_utilization 0.80, LoRA flags only when the submission has adapters.
The server is exposed under the competition alias, which is where `swegemma eval` sends requests
(http://localhost:8000/v1, model gemma-4-31b-it-qat-w4a16-ct), so moving hardware only changes
--model and --tensor-parallel-size. Record the real --model in every run's metadata.

Run from the repository root with the harness venv; vLLM runs in its own venv:
    python local/serve_vllm.py --model ~/models/gemma-4-E2B-it-qat-w4a16-ct -- --limit-mm-per-prompt '{"image":0,"audio":0}'
    python local/serve_vllm.py --model ~/models/gemma-4-31B-it-qat-w4a16-ct --tensor-parallel-size 4
Arguments after `--` are laptop-only additions appended to the competition command.
"""

import argparse
import os
import shlex
from pathlib import Path

from adk_submission.discovery import discover_adapters
from adk_submission.server import VllmConfig, VllmServer

COMPETITION_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Local checkpoint directory or HF id")
    p.add_argument("--submission-dir", default="local/raw_agent", help="Adapters are read from <dir>/adapters")
    p.add_argument("--vllm-python", default=os.path.expanduser("~/venvs/vllm/bin/python"))
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Kaggle uses 4")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--dry-run", action="store_true", help="Print the command and exit")
    args, extra = p.parse_known_args()
    extra = [a for a in extra if a != "--"]

    config = VllmConfig(
        model=os.path.expanduser(args.model),
        tool_call_parser="gemma4",
        reasoning_parser="gemma4",
        default_chat_template_kwargs={"enable_thinking": True},
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_loras=8,
        max_lora_rank=128,
        extra_args=["--served-model-name", COMPETITION_ALIAS, *extra],
    )
    adapters_dir = Path(args.submission_dir) / "adapters"
    manifest = discover_adapters(args.submission_dir) if adapters_dir.is_dir() else None
    cmd = VllmServer(config, adapter_manifest=manifest).build_cmd()
    cmd[0] = args.vllm_python
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
