#!/usr/bin/env bash
# Run one task through the official harness against the vLLM server on localhost:8000.
# Usage (repo root): bash local/run_task.sh <results-name> <task-id> [extra swegemma eval args]
# Example: bash local/run_task.sh e2b_rich_3061 rich_3061 --submission-dir local/lora_agent
# Default: swebench-sandbox:local (organizer image + missing wheels); pass --image swebench-sandbox:latest
# for the original. REPAIRED=1 uses the repaired environment (local/repair/, swebench-sandbox:repaired).
cd "$(dirname "$0")/.."
name=$1; task=$2; shift 2
if [ "${REPAIRED:-0}" = 1 ]; then
  cli=(~/venvs/harness/bin/python local/swegemma_repaired.py); image=swebench-sandbox:repaired
else
  cli=(~/venvs/harness/bin/swegemma); image=swebench-sandbox:local
fi
exec "${cli[@]}" eval --tasks tasks.jsonl --snapshots-dir snapshots \
  --submission-dir local/raw_agent --results-dir "runs/$name" --task-id "$task" --image "$image" \
  --max-tool-calls 50 --max-time-minutes 30 --display quiet "$@"
