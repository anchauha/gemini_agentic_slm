"""Run the official harness task by task and keep a tracked log of every run in logs/.

Raw outputs (patches, traces, test logs) go to runs/<name>/<task>/ (not in Git). The tracked record is:
    logs/<name>/manifest.json   what was run: model checkpoint, vLLM flags, agent config, mode, budgets, versions
    logs/<name>/results.jsonl   one row per task: outcome, failure category, tool calls, tokens, first failing test
    logs/<name>/summary.md      totals by repo and failure category
    logs/ledger.csv             every task of every run, for comparing runs over time
    logs/runs.csv               one row per run

Budgets default to the scorer's (HARNESS_README 7.1: 100 tool calls, 60 min, 500 turns, 300 s per command).
Run in the WSL harness venv, from the repository root, with the vLLM server up:
    ~/venvs/harness/bin/python local/runlog.py run --name e2b_raw_baseline --mode repaired
    ~/venvs/harness/bin/python local/runlog.py summarize e2b_raw_baseline
    ~/venvs/harness/bin/python local/runlog.py compare e2b_raw_baseline other_run
A run resumes where it stopped: tasks already in results.jsonl are skipped.
"""

import argparse
import collections
import csv
import datetime
import hashlib
import importlib.metadata as md
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
HARNESS_PY = Path.home() / "venvs/harness/bin/python"
MODES = {  # mode -> (image, harness entry point, tasks whose gold patch resolves in this mode)
    "faithful": ("swebench-sandbox:latest", [str(Path.home() / "venvs/harness/bin/swegemma")],
                 "local/gold_resolvable_faithful.txt"),
    "local": ("swebench-sandbox:local", [str(Path.home() / "venvs/harness/bin/swegemma")],
              "local/gold_resolvable.txt"),
    "repaired": ("swebench-sandbox:repaired", [str(HARNESS_PY), "local/swegemma_repaired.py"],
                 "local/gold_resolvable_repaired.txt"),
}
LEDGER_FIELDS = ["run", "date", "model", "agent_sha", "mode", "instance_id", "repo", "gold_resolvable", "resolved",
                 "category", "submitted", "tool_calls", "tool_errors", "llm_calls", "total_tokens", "duration_s",
                 "test_exit_code", "first_failure"]


def read_ids(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(ROOT, path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def server_info(url: str) -> dict:
    """The checkpoint actually loaded; the server always answers under the competition alias."""
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
            data = json.load(r)["data"]
    except Exception as e:
        raise SystemExit(f"vLLM server not reachable at {url}: {e}")
    base = next(m for m in data if not m.get("parent"))
    return {"served_name": base["id"], "checkpoint": base.get("root"), "max_model_len": base.get("max_model_len"),
            "adapters": [m["id"] for m in data if m.get("parent")],
            "command": sh("pgrep -af 'vllm.entrypoints|vllm serve' | head -1")}


def write_manifest(args, run_dir: Path) -> dict:
    agent_yaml = Path(ROOT, args.submission_dir, "agent.yaml")
    image = MODES[args.mode][0]
    versions = {}
    for pkg in ("swegemma", "adk-submission", "adk-eval-core", "google-adk", "litellm"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    manifest = {
        "run": args.name, "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": args.notes, "mode": args.mode, "image": image,
        "image_id": sh(f"docker image inspect --format '{{{{.Id}}}}' {image}"),
        "server": server_info(args.server),
        "submission_dir": args.submission_dir,
        "agent_yaml_sha256": hashlib.sha256(agent_yaml.read_bytes()).hexdigest()[:12],
        "agent_yaml": agent_yaml.read_text(),
        "budgets": {"max_tool_calls": args.max_tool_calls, "max_time_minutes": args.max_time_minutes,
                    "max_turns": args.max_turns, "timeout_seconds": args.timeout_seconds},
        "task_set": args.tasks, "versions": versions,
        "repo_commit": sh(f"git -c safe.directory='*' -C {ROOT} rev-parse --short HEAD"),
        "repo_dirty": bool(sh(f"git -c safe.directory='*' -C {ROOT} status --porcelain -- local")),
    }
    path = run_dir / "manifest.json"
    if path.exists():  # resuming: refuse to mix configurations in one run
        old = json.loads(path.read_text())
        for key in ("mode", "agent_yaml_sha256", "budgets"):
            if old[key] != manifest[key]:
                raise SystemExit(f"{args.name} was started with a different {key}; use a new --name")
        if old["server"]["checkpoint"] != manifest["server"]["checkpoint"]:
            raise SystemExit(f"{args.name} was started with {old['server']['checkpoint']}; use a new --name")
        return old
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def first_failure(test_log: Path) -> str:
    if not test_log.exists():
        return ""
    text = test_log.read_text(errors="replace")
    where = re.search(r"^(?:FAILED|ERROR) (\S+)", text, re.M)
    what = re.search(r"^E +(\S.*)$", text, re.M) or re.search(r"^(\w*Error: .*)$", text, re.M)
    return " | ".join(m.group(1).strip() for m in (where, what) if m)[:200]


def classify(res: dict, submitted: bool, budget_hit: bool) -> str:
    error = res.get("error") or ""
    if res.get("resolved"):
        return "resolved"
    if not res.get("agent_patch_size"):
        return "budget_exhausted_no_patch" if budget_hit else "no_patch"
    if "apply" in error.lower():
        return "patch_not_applied"
    if error and "Required test node" not in error:
        return "harness_error"
    code = res.get("test_exit_code")
    if code in (2, 3, 4, 5):
        return "tests_not_collected"  # import or syntax error, usually introduced by the patch
    return "tests_failed"


def task_row(manifest: dict, task_dir: Path, task, gold: set[str]) -> dict:
    results = task_dir / "task_results.jsonl"
    res = json.loads(results.read_text().splitlines()[-1]) if results.exists() else {
        "resolved": False, "error": "no task_results.jsonl (harness crashed)"}
    trace_path = task_dir / "traces" / f"trace_{task.instance_id}.json"
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {"steps": [], "final_metrics": {}}
    calls = [c for s in trace["steps"] for c in s.get("tool_calls") or []]
    tool_errors = sum('"status": "error"' in str((s.get("observation") or {}).get("content", ""))
                      or "'error'" in str((s.get("observation") or {}).get("content", "")) for s in trace["steps"])
    submitted = any(c.get("function_name") == "submit_patch" for c in calls)
    budget = manifest["budgets"]
    budget_hit = (res.get("tool_calls") or 0) >= budget["max_tool_calls"] or \
        (res.get("duration_seconds") or 0) >= budget["max_time_minutes"] * 60
    return {
        "run": manifest["run"], "date": manifest["started"][:10],
        "model": Path(manifest["server"]["checkpoint"] or "?").name, "agent_sha": manifest["agent_yaml_sha256"],
        "mode": manifest["mode"], "instance_id": task.instance_id, "repo": task.repo,
        "gold_resolvable": task.instance_id in gold, "resolved": bool(res.get("resolved")),
        "category": classify(res, submitted, budget_hit), "submitted": submitted,
        "tool_calls": res.get("tool_calls"), "tool_errors": tool_errors, "llm_calls": res.get("total_llm_calls"),
        "total_tokens": trace["final_metrics"].get("total_tokens"), "duration_s": res.get("duration_seconds"),
        "test_exit_code": res.get("test_exit_code"),
        "first_failure": first_failure(task_dir / "test_outputs" / f"{task.instance_id}.log") or (res.get("error") or "")[:200],
    }


def append_ledger(row: dict) -> None:
    path = LOGS / "ledger.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, LEDGER_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def rebuild(name: str) -> None:
    """Recomputes a run's rows from runs/<name>/ (after a change to how rows are derived)."""
    from swegemma.models.task import load_tasks

    run_dir = LOGS / name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    gold = set(read_ids(MODES[manifest["mode"]][2]))
    tasks = {t.instance_id: t for t in load_tasks(ROOT / "tasks.jsonl")}
    ids = [json.loads(l)["instance_id"] for l in (run_dir / "results.jsonl").read_text().splitlines()]
    rows = [task_row(manifest, ROOT / "runs" / name / i, tasks[i], gold) for i in ids]
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ledger = LOGS / "ledger.csv"
    kept = [r for r in csv.DictReader(ledger.open()) if r["run"] != name] if ledger.exists() else []
    with ledger.open("w", newline="") as f:
        w = csv.DictWriter(f, LEDGER_FIELDS)
        w.writeheader()
        w.writerows(kept + rows)


def summarize(name: str, rebuild_rows: bool = False) -> None:
    if rebuild_rows:
        rebuild(name)
    run_dir = LOGS / name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    gold_rows = [r for r in rows if r["gold_resolvable"]]
    rate = lambda rs: f"{sum(r['resolved'] for r in rs)}/{len(rs)}"
    lines = [f"# {name}", "",
             f"- Model: `{manifest['server']['checkpoint']}` (served as `{manifest['server']['served_name']}`)",
             f"- Agent: `{manifest['submission_dir']}` (agent.yaml sha256 `{manifest['agent_yaml_sha256']}`)",
             f"- Mode: {manifest['mode']} (`{manifest['image']}`); budgets {manifest['budgets']}",
             f"- Started {manifest['started']}, repo commit `{manifest['repo_commit']}`"
             + (" (local/ had uncommitted changes)" if manifest["repo_dirty"] else ""),
             f"- Notes: {manifest['notes'] or '-'}", "",
             f"**Resolved: {rate(rows)} tasks run; {rate(gold_rows)} of tasks whose gold patch resolves in this mode.**",
             "", "| Repo | Resolved | Gold-resolvable resolved |", "| --- | --- | --- |"]
    for repo in sorted({r["repo"] for r in rows}):
        rs = [r for r in rows if r["repo"] == repo]
        lines.append(f"| {repo} | {rate(rs)} | {rate([r for r in rs if r['gold_resolvable']])} |")
    lines += ["", "| Outcome | Tasks |", "| --- | --- |"]
    for cat, n in collections.Counter(r["category"] for r in rows).most_common():
        lines.append(f"| {cat} | {n} |")
    n = len(rows) or 1
    lines += ["", f"Mean per task: {sum(r['tool_calls'] or 0 for r in rows) / n:.1f} tool calls, "
              f"{sum(r['tool_errors'] for r in rows) / n:.1f} tool errors, "
              f"{sum(r['total_tokens'] or 0 for r in rows) / n:,.0f} tokens, "
              f"{sum(r['duration_s'] or 0 for r in rows) / n / 60:.1f} min; "
              f"submitted a patch on {sum(r['submitted'] for r in rows)}/{len(rows)}."]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")

    path = LOGS / "runs.csv"
    fields = ["run", "started", "model", "agent_sha", "mode", "tasks_run", "resolved", "gold_resolvable_run",
              "gold_resolvable_resolved", "notes"]
    existing = [r for r in csv.DictReader(path.open())] if path.exists() else []
    existing = [r for r in existing if r["run"] != name] + [{
        "run": name, "started": manifest["started"], "model": Path(manifest["server"]["checkpoint"] or "?").name,
        "agent_sha": manifest["agent_yaml_sha256"], "mode": manifest["mode"], "tasks_run": len(rows),
        "resolved": sum(r["resolved"] for r in rows), "gold_resolvable_run": len(gold_rows),
        "gold_resolvable_resolved": sum(r["resolved"] for r in gold_rows), "notes": manifest["notes"]}]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(existing)
    print("\n".join(lines))


def run(args) -> None:
    from swegemma.models.task import load_tasks

    run_dir = LOGS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(args, run_dir)
    image, cli, gold_file = MODES[args.mode]
    gold = set(read_ids(gold_file))
    tasks = list(load_tasks(ROOT / "tasks.jsonl"))
    if args.tasks != "all":
        wanted = set(read_ids(args.tasks)) if Path(ROOT, args.tasks).is_file() else set(args.tasks.split(","))
        tasks = [t for t in tasks if t.instance_id in wanted]
    results_path = run_dir / "results.jsonl"
    done = {json.loads(l)["instance_id"] for l in results_path.read_text().splitlines()} if results_path.exists() else set()
    todo = [t for t in tasks if t.instance_id not in done]
    print(f"{args.name}: {len(done)} done, {len(todo)} to run", flush=True)
    for i, task in enumerate(todo, 1):
        task_dir = ROOT / "runs" / args.name / task.instance_id
        shutil.rmtree(task_dir, ignore_errors=True)  # a half-finished attempt from an interrupted run
        cmd = [*cli, "eval", "--tasks", "tasks.jsonl", "--snapshots-dir", "snapshots",
               "--submission-dir", args.submission_dir, "--results-dir", str(task_dir), "--task-id", task.instance_id,
               "--image", image, "--max-tool-calls", str(args.max_tool_calls),
               "--max-time-minutes", str(args.max_time_minutes), "--max-turns", str(args.max_turns),
               "--timeout-seconds", str(args.timeout_seconds), "--display", "quiet"]
        t0 = time.time()
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / "harness.log").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = task_row(manifest, task_dir, task, gold)
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        append_ledger(row)
        print(f"[{i}/{len(todo)}] {task.instance_id}: {row['category']} "
              f"({row['tool_calls']} calls, {time.time() - t0:.0f}s)", flush=True)
    summarize(args.name)


def compare(a: str, b: str) -> None:
    load = lambda n: {r["instance_id"]: r for r in map(json.loads, (LOGS / n / "results.jsonl").read_text().splitlines())}
    ra, rb = load(a), load(b)
    common = sorted(ra.keys() & rb.keys())
    print(f"{len(common)} tasks in both; {a}: {sum(ra[t]['resolved'] for t in common)} resolved, "
          f"{b}: {sum(rb[t]['resolved'] for t in common)} resolved")
    for label, keep in (("newly resolved", lambda x, y: not x and y), ("newly failing", lambda x, y: x and not y)):
        flips = [t for t in common if keep(ra[t]["resolved"], rb[t]["resolved"])]
        print(f"{label} in {b} ({len(flips)}):")
        for t in flips:
            print(f"  {t}: {ra[t]['category']} -> {rb[t]['category']}")
    changed = collections.Counter((ra[t]["category"], rb[t]["category"]) for t in common
                                  if ra[t]["category"] != rb[t]["category"])
    print("category changes:", dict(changed) or "none")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True, help="run name, e.g. e2b_raw_baseline; reuse it to resume")
    r.add_argument("--mode", choices=MODES, default="repaired")
    r.add_argument("--submission-dir", default="local/raw_agent")
    r.add_argument("--tasks", default="all", help="'all', comma-separated ids, or a file of ids")
    r.add_argument("--notes", default="")
    r.add_argument("--server", default="http://127.0.0.1:8000")
    r.add_argument("--max-tool-calls", type=int, default=100)
    r.add_argument("--max-time-minutes", type=float, default=60.0)
    r.add_argument("--max-turns", type=int, default=500)
    r.add_argument("--timeout-seconds", type=int, default=300)
    s = sub.add_parser("summarize")
    s.add_argument("name")
    s.add_argument("--rebuild", action="store_true", help="recompute rows from runs/<name>/ first")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args()
    {"run": lambda: run(args), "summarize": lambda: summarize(args.name, args.rebuild), "compare": lambda: compare(args.a, args.b)}[args.cmd]()
