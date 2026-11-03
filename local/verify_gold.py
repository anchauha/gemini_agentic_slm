"""Check that each task's reference patch resolves in the local Container B (swegemma verify_task).

A task the gold patch cannot resolve locally is unsolvable here for environment reasons, so
exclude or flag it before attributing failures to the model. No model is called.

Run in the WSL harness venv, from the repository root:
    ~/venvs/harness/bin/python local/verify_gold.py --results-dir runs/gold --task-ids requests_6644
    ~/venvs/harness/bin/python local/verify_gold.py --results-dir runs/gold --concurrency 4      # all tasks
    ~/venvs/harness/bin/python local/verify_gold.py --results-dir runs/gold_repaired --repaired  # local/repair/
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from adk_submission.registry import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness import verification
from swegemma.models.task import load_tasks


async def verify_one(evaluator, config, task, sem, out):
    async with sem:
        snapshot, base_snapshot, patch_path = resolve_task_snapshot_paths(
            config.snapshots_dir, task.instance_id, task.repo
        )
        try:
            res = await verification.verify_task(
                evaluator.docker, config, task, snapshot,
                base_snapshot_path=base_snapshot, patch_path=patch_path, agent_patch=task.patch,
            )
            row = {"instance_id": task.instance_id, "repo": task.repo, "resolved": res.resolved,
                   "test_exit_code": res.test_exit_code, "error": res.error}
            (config.results_dir / "test_outputs").mkdir(parents=True, exist_ok=True)
            (config.results_dir / "test_outputs" / f"{task.instance_id}.log").write_text(res.test_output or "")
        except Exception as e:  # keep going; record harness failures per task
            row = {"instance_id": task.instance_id, "repo": task.repo, "resolved": False,
                   "test_exit_code": None, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(row), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


async def main(args):
    config = EvalConfig(
        tasks_path=Path(args.tasks), snapshots_dir=Path(args.snapshots_dir),
        results_dir=Path(args.results_dir), submission_dir=None, models=ModelRegistry(),
        image=args.image, skip_agent_patch=True, display_mode="quiet",
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(config)
    tasks = list(load_tasks(config.tasks_path))
    if args.task_ids:
        tasks = [t for t in tasks if t.instance_id in set(args.task_ids)]
    out = config.results_dir / "gold_results.jsonl"
    done = {json.loads(l)["instance_id"] for l in out.open()} if out.exists() else set()
    sem = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*(verify_one(evaluator, config, t, sem, out)
                           for t in tasks if t.instance_id not in done))
    rows = [json.loads(l) for l in out.open()]
    print(f"gold resolved {sum(r['resolved'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="tasks.jsonl")
    p.add_argument("--snapshots-dir", default="snapshots")
    p.add_argument("--results-dir", default="runs/gold")
    p.add_argument("--task-ids", nargs="*")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--image", default="swebench-sandbox:local",
                   help="swebench-sandbox:latest for the unmodified organizer image")
    p.add_argument("--repaired", action="store_true",
                   help="run local/repair/repair_env.py per task; implies --image swebench-sandbox:repaired")
    args = p.parse_args()
    if args.repaired:
        sys.path.insert(0, str(Path(__file__).parent))
        import swegemma_repaired  # noqa: F401  (patches the harness)
        args.image = "swebench-sandbox:repaired"
    asyncio.run(main(args))
