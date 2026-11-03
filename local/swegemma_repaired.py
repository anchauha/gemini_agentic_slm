"""swegemma with the repaired environment: after the harness installs test dependencies in a container,
run /opt/repair/repair_env.py (local/repair/) for the current task. Needs --image swebench-sandbox:repaired.

Import it before using the harness (verify_gold.py --repaired does), or run it as the swegemma CLI:
    ~/venvs/harness/bin/python local/swegemma_repaired.py eval --image swebench-sandbox:repaired ...
"""

import contextvars
import functools
import shlex
import sys

import swegemma.evaluate as evaluate
import swegemma.harness.agent_runner as agent_runner
import swegemma.harness.container_setup as container_setup
import swegemma.harness.verification as verification

TASK_ID = contextvars.ContextVar("repair_task_id", default="")
_original = container_setup.install_test_dependencies


def install_test_dependencies(docker, container_id, repo="", **kwargs):
    _original(docker, container_id, repo, **kwargs)
    res = docker.exec(container_id, f"python3 /opt/repair/repair_env.py {shlex.quote(repo)} {shlex.quote(TASK_ID.get())}")
    if res.exit_code != 0:
        raise RuntimeError(f"repair_env.py failed for {TASK_ID.get()}:\n{res.stdout}\n{res.stderr}")


def _with_task(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        task = kwargs.get("task") or next(a for a in args if hasattr(a, "instance_id"))
        TASK_ID.set(task.instance_id)  # asyncio.to_thread copies the context into the setup thread
        return await fn(*args, **kwargs)
    return wrapper


for module in (container_setup, verification, agent_runner, evaluate):
    if hasattr(module, "install_test_dependencies"):
        module.install_test_dependencies = install_test_dependencies
verification.verify_task = evaluate.verify_task = _with_task(verification.verify_task)
evaluate.Evaluator.evaluate_task = _with_task(evaluate.Evaluator.evaluate_task)

if __name__ == "__main__":
    from swegemma.cli import main
    sys.exit(main())
