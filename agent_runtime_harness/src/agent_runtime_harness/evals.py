from __future__ import annotations

from pathlib import Path
import json
import re

from .models import DEFAULT_EVAL_SUITE_VERSION, EvalCase
from .runtime import AgentRuntime

DEFAULT_EVAL_CASES = [
    EvalCase(
        case_id="case_allergy_safe_plan",
        task="Build a 3-day meal plan for a busy student",
        input_text=(
            "Profile: allergic to peanuts, dislikes lamb, prefers high-protein meals. "
            "Need breakfast/lunch/dinner plan and shopping list suggestions."
        ),
        required_terms=["breakfast", "lunch", "dinner", "shopping list"],
        max_output_chars=2600,
        eval_suite_version="recipe-evals.v4",
    ),
    EvalCase(
        case_id="case_budget_goal_plan",
        task="Plan affordable meals for fat-loss",
        input_text=(
            "Goal: fat loss with 1900 kcal/day target. Budget level: low. "
            "Give practical meals and prep-friendly options."
        ),
        required_terms=["budget", "kcal"],
        max_output_chars=2600,
        eval_suite_version="recipe-evals.v4",
    ),
    EvalCase(
        case_id="case_meal_log_followup",
        task="Adjust weekly plan using recent meal logs",
        input_text=(
            "Recent logs: often skip breakfast and overeat at night. "
            "Design a correction plan and habit suggestions."
        ),
        required_terms=["habit", "breakfast"],
        max_output_chars=2600,
        eval_suite_version="recipe-evals.v4",
    ),
    EvalCase(
        case_id="case_visual_guided_plan",
        task="Use the uploaded meal photo to draft a matching healthy dinner plan",
        input_text=(
            "The user uploaded a food photo and wants a healthier but similar dinner plan. "
            "Keep it high-protein and concise."
        ),
        required_terms=["visual", "dinner", "ingredients"],
        image_paths=["data/vision_samples/salmon_tray_dinner.jpg"],
        vision_provider="sidecar",
        max_output_chars=3200,
        eval_suite_version="recipe-evals.v4",
    ),
    EvalCase(
        case_id="case_multi_day_structure",
        task="Plan 5-day meals with stable structure and shopping support",
        input_text=(
            "Need a 5-day high-protein plan with breakfast/lunch/dinner, clear day headers, "
            "and one consolidated shopping list."
        ),
        required_terms=["day 1", "day 5", "shopping list"],
        max_output_chars=4200,
        eval_suite_version="recipe-evals.v4",
    ),
    EvalCase(
        case_id="case_dinner_intent_focus",
        task="Create a dinner-focused plan only",
        input_text=(
            "Need quick high-protein dinners this week. Keep focus on dinner choices and avoid "
            "leading with breakfast/lunch recommendations."
        ),
        required_terms=["dinner", "protein"],
        max_output_chars=3200,
        eval_suite_version="recipe-evals.v5",
    ),
    EvalCase(
        case_id="case_retrieval_grounded_candidates",
        task="Use retrieval-grounded recipes for candidate ranking",
        input_text=(
            "Need a peanut-safe quick dinner and show retrieval-grounded candidates before planning."
        ),
        required_terms=["candidate recipes", "source"],
        max_output_chars=3200,
        eval_suite_version="recipe-evals.v5",
    ),
]

CASE_RUN_CONTEXT_OVERRIDES: dict[str, dict] = {
    "case_multi_day_structure": {"planning_days": 5},
}


def _ensure_vision_samples(project_root: str | Path) -> None:
    root = Path(project_root) / "data" / "vision_samples"
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "salmon_tray_dinner.jpg"
    if not image_path.exists():
        image_path.write_bytes(b"mock-salmon-image")
    sidecar = root / "salmon_tray_dinner.vision.json"
    if not sidecar.exists():
        sidecar.write_text(
            json.dumps(
                {
                    "dish_name": "Garlic Salmon Veggie Tray",
                    "meal_type": "dinner",
                    "visible_ingredients": ["salmon", "asparagus", "potatoes", "lemon"],
                    "cuisine_tags": ["tray-bake", "high-protein"],
                    "nutrition_signals": ["protein", "vegetable-forward"],
                    "caution_tags": [],
                    "summary": "The image shows a salmon tray dinner with vegetables and lemon.",
                    "confidence": 0.96,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _day_headers(output: str) -> list[int]:
    return [int(day) for day in re.findall(r"(?mi)^day\s+(\d+)\s*$", output)]


def _first_meal_slot(output: str) -> str:
    match = re.search(r"(?mi)^-\s*(breakfast|lunch|dinner|snack):", output)
    return match.group(1).lower() if match else ""


def _query_candidate_source(run) -> str:
    for step in reversed(run.steps):
        if not step.tool_calls:
            continue
        if step.tool_calls[0].tool_name != "query_recipe_candidates":
            continue
        result = step.payload.get("result", {})
        if isinstance(result, dict):
            source = str(result.get("candidate_source", "")).strip()
            if source:
                return source
    return ""


def score_output(output: str, required_terms: list[str], max_output_chars: int) -> dict:
    missing = [term for term in required_terms if term.lower() not in output.lower()]
    normalized = output.lower()
    has_meal_signal = any(token in normalized for token in ["meal", "recipe", "plan", "breakfast", "dinner"])
    has_structure = any(token in normalized for token in ["day 1", "shopping list", "planner rationale", "profile guardrails"])
    return {
        "missing_terms": missing,
        "within_length": len(output) <= max_output_chars,
        "has_meal_signal": has_meal_signal,
        "has_structure": has_structure,
        "passed": not missing and len(output) <= max_output_chars and has_meal_signal and has_structure,
    }


def _case_run_context(case: EvalCase, project_root: str | Path) -> dict:
    context = {
        "session_id": "eval_recipe",
        "goal": "fat loss",
        "planning_days": 3,
        "daily_kcal_target": 1900,
        "image_paths": [str(Path(project_root) / image_path) for image_path in case.image_paths],
        "vision_provider": case.vision_provider or "",
        "user_profile": {
            "allergies": ["peanut"],
            "dislikes": ["lamb"],
            "preferences": ["high-protein", "quick-cook"],
        },
        "meal_logs": [
            "2026-04-17: skipped breakfast, late-night snacking",
            "2026-04-18: high sugar drinks after lunch",
        ],
    }
    context.update(CASE_RUN_CONTEXT_OVERRIDES.get(case.case_id, {}))
    return context


def run_eval_suite(project_root: str | Path, cases: list[EvalCase] | None = None) -> dict:
    _ensure_vision_samples(project_root)
    runtime = AgentRuntime(project_root)
    cases = cases or DEFAULT_EVAL_CASES
    results = []
    for case in cases:
        run = runtime.run_task(
            case.task,
            case.input_text,
            required_terms=case.required_terms,
            max_output_chars=case.max_output_chars,
            eval_suite_version=case.eval_suite_version or DEFAULT_EVAL_SUITE_VERSION,
            run_context=_case_run_context(case, project_root),
        )
        verdict = score_output(run.final_output, case.required_terms, case.max_output_chars)
        day_headers = _day_headers(run.final_output)
        first_meal_slot = _first_meal_slot(run.final_output)
        multi_day_ok = True
        if case.case_id == "case_multi_day_structure":
            expected_days = int(CASE_RUN_CONTEXT_OVERRIDES["case_multi_day_structure"]["planning_days"])
            multi_day_ok = len(day_headers) >= expected_days and max(day_headers or [0]) >= expected_days
        visual_grounding_ok = True
        if case.case_id == "case_visual_guided_plan":
            visual_grounding_ok = bool(run.metadata.get("vision_analysis")) and (
                "visual reference" in run.final_output.lower() or "visual note" in run.final_output.lower()
            )
        dinner_focus_ok = True
        if case.case_id == "case_dinner_intent_focus":
            dinner_focus_ok = first_meal_slot in {"dinner", "snack"}
        retrieval_grounded_ok = True
        if case.case_id == "case_retrieval_grounded_candidates":
            retrieval_grounded_ok = _query_candidate_source(run) == "memory_rag_lab"
        passed = (
            verdict["passed"]
            and multi_day_ok
            and visual_grounding_ok
            and dinner_focus_ok
            and retrieval_grounded_ok
        )
        results.append(
            {
                "case_id": case.case_id,
                "run_id": run.id,
                "status": run.status,
                "policy_version": run.policy_version,
                "eval_suite_version": run.eval_suite_version,
                "checkpoint_stage": run.checkpoint_state.get("stage", ""),
                "failure_type": run.failure_type,
                "resumable_checkpoint": bool(run.checkpoint_state.get("pending_tools") is not None),
                "plan_length": len(run.checkpoint_state.get("plan", [])),
                "vision_used": bool(run.metadata.get("vision_analysis")),
                "day_headers": day_headers,
                "first_meal_slot": first_meal_slot,
                "multi_day_ok": multi_day_ok,
                "visual_grounding_ok": visual_grounding_ok,
                "dinner_focus_ok": dinner_focus_ok,
                "candidate_source": _query_candidate_source(run),
                "retrieval_grounded_ok": retrieval_grounded_ok,
                "passed": passed,
                **{k: v for k, v in verdict.items() if k != "passed"},
            }
        )
    passed = sum(1 for item in results if item["passed"])
    return {"total": len(results), "passed": passed, "results": results}
