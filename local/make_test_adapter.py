"""Write a zero-effect PEFT LoRA adapter for a Gemma 4 checkpoint, shaped like the sample submission's.

It mirrors sample_submission/adapters/main_lora (r=4, alpha=8, q_proj and o_proj, layer 0) but for the
given base model, with lora_B = 0 so outputs are unchanged. Use it to test adapter discovery and vLLM
LoRA serving on a base model that the 31B sample adapters do not fit. Run with the vLLM venv (torch):
    ~/venvs/vllm/bin/python local/make_test_adapter.py --base ~/models/gemma-4-E2B-it-qat-w4a16-ct \
        --hf-id google/gemma-4-E2B-it-qat-w4a16-ct --out local/lora_agent/adapters/main_lora
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

RANK, ALPHA, LAYER = 4, 8, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Local checkpoint directory (reads config.json)")
    p.add_argument("--hf-id", required=True, help="Recorded as base_model_name_or_path")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = json.loads((Path(os.path.expanduser(args.base)) / "config.json").read_text())["text_config"]
    sliding = cfg["layer_types"][LAYER] == "sliding_attention"
    head_dim = cfg["head_dim"] if sliding else cfg.get("global_head_dim") or cfg["head_dim"]
    hidden, attn = cfg["hidden_size"], cfg["num_attention_heads"] * head_dim
    prefix = f"base_model.model.model.language_model.layers.{LAYER}.self_attn"
    tensors = {
        f"{prefix}.q_proj.lora_A.weight": torch.randn(RANK, hidden) * 0.01,
        f"{prefix}.q_proj.lora_B.weight": torch.zeros(attn, RANK),
        f"{prefix}.o_proj.lora_A.weight": torch.randn(RANK, attn) * 0.01,
        f"{prefix}.o_proj.lora_B.weight": torch.zeros(hidden, RANK),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()}, out / "adapter_model.safetensors")
    sample = json.loads(Path("sample_submission/adapters/main_lora/adapter_config.json").read_text())
    sample.update(base_model_name_or_path=args.hf_id, r=RANK, lora_alpha=ALPHA, layers_to_transform=[LAYER])
    (out / "adapter_config.json").write_text(json.dumps(sample, indent=2) + "\n")
    print(f"wrote {out}: " + ", ".join(f"{k.split('self_attn.')[1]} {list(v.shape)}" for k, v in tensors.items()))


if __name__ == "__main__":
    main()
