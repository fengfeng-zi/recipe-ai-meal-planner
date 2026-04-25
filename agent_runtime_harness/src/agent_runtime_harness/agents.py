from __future__ import annotations

import re

from .models import DEFAULT_POLICY_VERSION, ReviewDecision, StepRecord, ToolCallRecord, make_id
from .tools import ToolRegistry


def classify_tool_failure(error: Exception) -> str:
    if isinstance(error, KeyError):
        return "tool_not_registered"
    if isinstance(error, TimeoutError):
        return "tool_timeout"
    if isinstance(error, ValueError):
        return "invalid_tool_input"
    return "tool_execution"


class PlannerAgent:
    def plan(self, task: str, input_text: str, run_context: dict | None = None) -> list[dict[str, str]]:
        combined = f"{task} {input_text}".lower()
        asks_for_plan = any(token in combined for token in ("plan", "meal", "recipe", "weekly", "day", "breakfast", "lunch", "dinner"))
        asks_for_shopping = any(token in combined for token in ("shopping", "grocery", "buy", "prep")) or bool(re.search(r"\b\d+\s*-\s*day|\b\d+\s*day", combined))
        run_context = run_context or {}
        image_paths = run_context.get("image_paths", []) if isinstance(run_context, dict) else []
        has_images = isinstance(image_paths, list) and bool(image_paths)
        mentions_images = any(token in combined for token in ("image", "photo", "picture", "upload"))

        plan = [
            {"role": "planner", "summary": f"Extract meal-planning signals for task: {task}", "tool": "extract_keywords"},
            {"role": "planner", "summary": "Analyze dietary profile, constraints, and habit risks", "tool": "analyze_profile"},
        ]
        if has_images or mentions_images:
            plan.append({"role": "planner", "summary": "Analyze uploaded meal images for ingredient cues", "tool": "analyze_meal_image"})
        plan.append({"role": "retriever", "summary": "Retrieve recipe candidates that fit the profile", "tool": "query_recipe_candidates"})
        if asks_for_plan:
            plan.append({"role": "planner", "summary": "Draft a multi-day meal plan", "tool": "draft_meal_plan"})
        else:
            plan.append({"role": "executor", "summary": "Draft a recipe answer", "tool": "compose_answer"})
        if asks_for_shopping or asks_for_plan:
            plan.append({"role": "executor", "summary": "Generate a grouped shopping list", "tool": "generate_shopping_list"})
        return plan


class ExecutorAgent:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, role: str, summary: str, tool_name: str, payload: dict, parent_step_id: str | None = None) -> tuple[StepRecord, dict]:
        step = StepRecord(id=make_id("step"), role=role, summary=summary, payload=payload, parent_step_id=parent_step_id)
        spec = self.registry.get_spec(tool_name)
        call = ToolCallRecord(id=make_id("tool"), tool_name=tool_name, input_payload=payload, tool_version=spec.version)
        step.tool_calls.append(call)
        try:
            output = self.registry.execute(tool_name, payload)
            call.complete(output)
            step.payload = {**step.payload, "result": output}
            step.complete()
            return step, output
        except Exception as error:  # pragma: no cover
            failure_type = classify_tool_failure(error)
            call.fail(str(error), failure_type=failure_type)
            step.payload = {**step.payload, "error": str(error)}
            step.fail(failure_type)
            return step, {"error": str(error), "failure_type": failure_type}


class ReviewerAgent:
    def __init__(self, policy_version: str = DEFAULT_POLICY_VERSION):
        self.policy_version = policy_version

    def review(
        self,
        final_output: str,
        required_terms: list[str] | None = None,
        max_output_chars: int = 2600,
        task: str = "",
        metadata: dict | None = None,
    ) -> ReviewDecision:
        required_terms = required_terms or []
        metadata = metadata or {}
        normalized = final_output.lower()
        missing_terms = [term for term in required_terms if term.lower() not in normalized]
        reasons: list[str] = []

        if missing_terms:
            reasons.append("Missing required terms: " + ", ".join(missing_terms))
        if len(final_output) > max_output_chars:
            reasons.append(f"Output exceeds max length {max_output_chars}.")
        if "missing required terms:" in normalized:
            reasons.append("Revision placeholder text leaked into the user-facing answer.")

        plan_like_task = any(token in f"{task} {normalized}" for token in ("meal plan", "weekly plan", "3-day", "breakfast", "lunch", "dinner"))
        if plan_like_task and not all(token in normalized for token in ("breakfast", "lunch", "dinner")):
            reasons.append("Meal-plan output should cover breakfast, lunch, and dinner explicitly.")

        planning_days = int(metadata.get("planning_days", 1) or 1)
        if planning_days > 1 and "day 1" not in normalized:
            reasons.append("Multi-day plans should enumerate at least the first day.")
        if planning_days > 1 and "shopping list" not in normalized:
            reasons.append("Multi-day plans should include a shopping list section.")

        image_paths = metadata.get("image_paths", []) if isinstance(metadata, dict) else []
        if image_paths and not any(token in normalized for token in ("visual", "image", "detected")):
            reasons.append("Image-guided runs should include a short visual analysis summary.")

        if reasons:
            return ReviewDecision(
                verdict="revise",
                reasons=reasons,
                revision_request="Return a concise but complete meal plan with natural section headings and no placeholder review text.",
                policy_version=self.policy_version,
            )
        return ReviewDecision(
            verdict="approve",
            reasons=["Output satisfies the configured checks."],
            policy_version=self.policy_version,
        )
