"""Smoke-test the harness model path (swegemma registry -> LiteLLM -> Ollama) with ADK tool calls.

The tools here are in-memory fakes; no repository code runs. This checks that the model
emits parseable tool calls through the same ModelRegistry the swegemma CLI builds.

Usage:
    python ollama/smoke_ollama_agent.py --models-yaml ollama/models.yaml
"""

import argparse
import asyncio
import json
import time

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from swegemma.cli import _build_cli_model_registry

MODEL_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
FILES = {
    "calc/ops.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
    "tests/test_ops.py": "from calc.ops import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
}
CALLS = []


def read_file(filepath: str) -> str:
    """Read a file from the workspace."""
    CALLS.append(("read_file", filepath))
    if filepath not in FILES:
        return json.dumps({"status": "error", "error_message": f"No such file: {filepath}"})
    return json.dumps({"status": "ok", "filepath": filepath, "content": FILES[filepath]})


def edit_file(filepath: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in a workspace file."""
    CALLS.append(("edit_file", filepath))
    if old_string not in FILES.get(filepath, ""):
        return json.dumps({"status": "error", "error_message": "old_string not found"})
    FILES[filepath] = FILES[filepath].replace(old_string, new_string, 1)
    return json.dumps({"status": "ok", "filepath": filepath, "occurrences": 1})


def submit_patch() -> str:
    """Submit the current workspace changes as the final patch."""
    CALLS.append(("submit_patch", ""))
    return json.dumps({"status": "ok"})


async def run(models_yaml: str) -> None:
    model = _build_cli_model_registry(models_yaml_path=models_yaml).get(MODEL_ALIAS)
    agent = LlmAgent(
        name="smoke",
        model=model,
        instruction="You fix bugs. Workspace files: calc/ops.py, tests/test_ops.py. "
        "Use the tools, make the minimal fix, then call submit_patch.",
        tools=[read_file, edit_file, submit_patch],
    )
    runner = InMemoryRunner(agent=agent, app_name="smoke")
    session = await runner.session_service.create_session(app_name="smoke", user_id="u")
    msg = types.Content(role="user", parts=[types.Part(text="test_add fails. Fix it.")])
    start = time.time()
    async for event in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
        for part in (event.content.parts if event.content else []) or []:
            if part.function_call:
                print(f"[tool call] {part.function_call.name}({dict(part.function_call.args or {})})")
            elif part.text and not part.thought:
                print(f"[text] {part.text.strip()[:300]}")
    fixed = "return a + b" in FILES["calc/ops.py"]
    submitted = any(name == "submit_patch" for name, _ in CALLS)
    print(f"\nmodel={model.model} calls={len(CALLS)} fixed={fixed} submitted={submitted} "
          f"elapsed={time.time() - start:.1f}s")
    print("PASS" if fixed and submitted else "FAIL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-yaml", default="ollama/models.yaml")
    asyncio.run(run(parser.parse_args().models_yaml))
