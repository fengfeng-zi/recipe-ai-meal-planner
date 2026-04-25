# Agent Runtime Harness

Planning/runtime layer for turning recipe evidence and user constraints into multi-day meal plans.

## What It Does

- records `run -> step -> tool_call` traces
- persists checkpoints and supports resume
- injects user profile, meal-log, and vision context into planning payloads
- drafts meal plans and shopping lists
- ships with eval and report commands

## Quick Start

```powershell
cd <repo-root>
$root = (Get-Location).Path
$env:PYTHONPATH = "$root\agent_runtime_harness\src;$root\memory_rag_lab\src"

python -m agent_runtime_harness.cli run --task "Plan 3-day meals" --input-text "Need high-protein quick meals without peanuts" --goal "fat loss" --allergies peanut --dislikes lamb --planning-days 3 --required-terms breakfast,lunch,dinner
python -m agent_runtime_harness.cli eval
python -m agent_runtime_harness.cli report
python -m agent_runtime_harness.cli list-runs
```

Resume behavior:

```powershell
python -m agent_runtime_harness.cli resume <run_id>
python -m agent_runtime_harness.cli resume <run_id> --continue-execution
```

## Key Files

- `src/agent_runtime_harness/runtime.py`: orchestration and checkpoint flow
- `src/agent_runtime_harness/tools.py`: candidate picking, planning, shopping list
- `src/agent_runtime_harness/agents.py`: planner/executor/reviewer loop
- `src/agent_runtime_harness/evals.py`: behavior expectations

