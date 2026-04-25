from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from .evals import DEFAULT_EVAL_CASES


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _line_count(text: str) -> int:
    stripped = [line.strip() for line in text.splitlines() if line.strip()]
    return len(stripped)


def build_report(project_root: str | Path) -> dict:
    runs_dir = Path(project_root) / "data" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(runs_dir.glob("*.json"))]

    status_counts = Counter(payload.get("status", "unknown") for payload in payloads)
    failure_counts = Counter((payload.get("failure_type") or "none") for payload in payloads)
    checkpoint_stages = Counter((payload.get("checkpoint_state") or {}).get("stage", "unknown") for payload in payloads)
    policy_versions = Counter(payload.get("policy_version", "unknown") for payload in payloads)
    eval_versions = Counter(payload.get("eval_suite_version") or "none" for payload in payloads)
    tool_usage = Counter()

    step_counts: list[int] = []
    tool_call_counts: list[int] = []
    revision_runs = 0
    unique_tasks = set()

    meal_planning_runs = 0
    profile_aware_runs = 0
    shopping_signal_runs = 0
    goal_signal_runs = 0
    vision_enabled_runs = 0
    avg_output_lines: list[int] = []

    for payload in payloads:
        task = payload.get("task", "")
        unique_tasks.add(task)
        task_lower = task.lower()

        metadata = payload.get("metadata", {})
        profile = metadata.get("user_profile", {}) if isinstance(metadata, dict) else {}
        final_output = payload.get("final_output", "")
        output_lower = final_output.lower()

        if any(token in task_lower for token in ["meal", "recipe", "diet", "nutrition", "plan"]):
            meal_planning_runs += 1
        if isinstance(profile, dict) and any(profile.get(key) for key in ["allergies", "dislikes", "preferences"]):
            profile_aware_runs += 1
        if "shopping" in output_lower or "grocery" in output_lower:
            shopping_signal_runs += 1
        if any(token in output_lower for token in ["kcal", "calorie", "protein", "fat loss", "goal"]):
            goal_signal_runs += 1
        image_paths = metadata.get("image_paths", []) if isinstance(metadata, dict) else []
        if isinstance(image_paths, list) and image_paths:
            vision_enabled_runs += 1

        avg_output_lines.append(_line_count(final_output))

        steps = payload.get("steps", [])
        step_counts.append(len(steps))
        run_tool_calls = 0
        revised = False
        for step in steps:
            for call in step.get("tool_calls", []):
                tool_name = call.get("tool_name", "unknown")
                tool_usage[tool_name] += 1
                run_tool_calls += 1
                if tool_name == "revise_answer":
                    revised = True
        tool_call_counts.append(run_tool_calls)
        if revised:
            revision_runs += 1

    total_runs = len(payloads)
    return {
        "report_version": "meal-harness-report.v3",
        "total_runs": total_runs,
        "unique_tasks": len([task for task in unique_tasks if task]),
        "status_counts": dict(status_counts),
        "failure_counts": dict(failure_counts),
        "checkpoint_stage_counts": dict(checkpoint_stages),
        "policy_versions": dict(policy_versions),
        "eval_suite_versions": dict(eval_versions),
        "avg_steps_per_run": _average(step_counts),
        "avg_tool_calls_per_run": _average(tool_call_counts),
        "avg_output_lines": _average(avg_output_lines),
        "revision_run_count": revision_runs,
        "revision_rate": round(revision_runs / total_runs, 3) if total_runs else 0.0,
        "meal_planning_runs": meal_planning_runs,
        "profile_aware_runs": profile_aware_runs,
        "shopping_signal_runs": shopping_signal_runs,
        "goal_signal_runs": goal_signal_runs,
        "vision_enabled_runs": vision_enabled_runs,
        "tool_usage": dict(tool_usage),
        "structural_metrics": {
            "trace_levels": 3,
            "cli_workflows": 5,
            "checkpoint_snapshot_keys": 9,
            "default_eval_cases": len(DEFAULT_EVAL_CASES),
            "focus": "recipe_multi_agent_meal_planning",
            "vision_tool_enabled": 1,
        },
    }
