from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .agents import ExecutorAgent, PlannerAgent, ReviewerAgent
from .models import DEFAULT_EVAL_SUITE_VERSION, DEFAULT_POLICY_VERSION, RunRecord
from .store import JsonRunStore
from .tools import ToolRegistry, register_default_tools


class AgentRuntime:
    def __init__(self, project_root: str | Path, registry: ToolRegistry | None = None, policy_version: str = DEFAULT_POLICY_VERSION):
        self.project_root = Path(project_root)
        self.registry = register_default_tools(registry)
        self.store = JsonRunStore(self.project_root / "data")
        self.policy_version = policy_version
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(self.registry)
        self.reviewer = ReviewerAgent(policy_version=policy_version)

    @staticmethod
    def _normalize_profile(metadata: dict[str, Any]) -> dict[str, Any]:
        raw = metadata.get("user_profile", {}) if isinstance(metadata, dict) else {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _normalize_vision(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        raw = metadata.get("vision_analysis", []) if isinstance(metadata, dict) else []
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        return []

    def _build_payload(
        self,
        tool_name: str,
        task: str,
        input_text: str,
        keywords: list[str],
        final_output: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        profile = self._normalize_profile(metadata)
        vision_analysis = self._normalize_vision(metadata)
        session_id = str(metadata.get("session_id", "demo"))
        planning_days = int(metadata.get("planning_days", 7) or 7)
        daily_kcal_target = int(metadata.get("daily_kcal_target", 2000) or 2000)
        image_paths = metadata.get("image_paths", []) if isinstance(metadata.get("image_paths", []), list) else []
        base_payload = {
            "task": task,
            "input_text": input_text,
            "keywords": keywords,
            "session_id": session_id,
            "planning_days": planning_days,
            "daily_kcal_target": daily_kcal_target,
            "goal": metadata.get("goal", ""),
            "user_profile": profile,
            "meal_logs": metadata.get("meal_logs", []),
            "image_paths": image_paths,
            "vision_provider": metadata.get("vision_provider", ""),
            "vision_analysis": vision_analysis,
        }

        if tool_name == "extract_keywords":
            return {"input_text": input_text, "max_keywords": 12}

        if tool_name == "analyze_meal_image":
            return {
                "image_paths": image_paths,
                "vision_provider": metadata.get("vision_provider", ""),
            }

        if tool_name == "revise_answer":
            return {
                "current_output": final_output,
                "missing_terms": metadata.get("_missing_terms", []),
                "task": task,
                "session_id": session_id,
                "goal": metadata.get("goal", ""),
                "user_profile": profile,
                "meal_logs": metadata.get("meal_logs", []),
                "image_paths": image_paths,
                "vision_analysis": vision_analysis,
            }

        if tool_name in {"analyze_profile", "profile_analyzer", "query_recipe_candidates", "draft_meal_plan", "compose_answer"}:
            return base_payload

        if tool_name == "generate_shopping_list":
            return {
                **base_payload,
                "draft_plan": final_output,
            }

        return {**base_payload, "draft_output": final_output}

    def _checkpoint_state(
        self,
        stage: str,
        plan: list[dict[str, str]],
        completed_step_ids: list[str],
        last_step_id: str | None,
        keywords: list[str],
        final_output: str,
        next_index: int,
        stop_after_step: int = 0,
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "plan": [item["tool"] for item in plan],
            "pending_tools": [item["tool"] for item in plan[next_index:]],
            "completed_step_ids": completed_step_ids,
            "last_step_id": last_step_id,
            "keyword_cache": keywords,
            "draft_output": final_output,
            "next_index": next_index,
            "stop_after_step": stop_after_step,
        }

    def _finalize_failed_run(
        self,
        run: RunRecord,
        plan: list[dict[str, str]],
        completed_step_ids: list[str],
        last_step_id: str | None,
        keywords: list[str],
        final_output: str,
        next_index: int,
        failure_type: str,
        error: str,
    ) -> RunRecord:
        run.final_output = final_output
        run.status = "failed"
        run.failure_type = failure_type
        run.metadata["error"] = error
        run.checkpoint_state = self._checkpoint_state(
            "failed",
            plan,
            completed_step_ids,
            last_step_id,
            keywords,
            final_output,
            next_index,
            int(run.metadata.get("stop_after_step", 0) or 0),
        )
        run.touch()
        self.store.save_run(run)
        self.store.save_checkpoint(run)
        return run

    def _maybe_pause_run(
        self,
        run: RunRecord,
        plan: list[dict[str, str]],
        completed_step_ids: list[str],
        last_step_id: str | None,
        keywords: list[str],
        final_output: str,
        next_index: int,
    ) -> RunRecord | None:
        stop_after_step = int(run.metadata.get("stop_after_step", 0) or 0)
        if not stop_after_step:
            return None
        if len(completed_step_ids) < stop_after_step:
            return None
        if next_index >= len(plan):
            return None
        run.final_output = final_output
        run.status = "paused"
        run.failure_type = None
        run.checkpoint_state = self._checkpoint_state(
            "paused",
            plan,
            completed_step_ids,
            last_step_id,
            keywords,
            final_output,
            next_index,
            stop_after_step,
        )
        run.touch()
        self.store.save_run(run)
        self.store.save_checkpoint(run)
        return run

    def _execute_plan(
        self,
        run: RunRecord,
        plan: list[dict[str, str]],
        start_index: int,
        keywords: list[str],
        final_output: str,
        parent_step_id: str | None,
        completed_step_ids: list[str],
        required_terms: list[str] | None,
        max_output_chars: int,
    ) -> RunRecord:
        for index in range(start_index, len(plan)):
            item = plan[index]
            payload = self._build_payload(
                tool_name=item["tool"],
                task=run.task,
                input_text=run.input_text,
                keywords=keywords,
                final_output=final_output,
                metadata=run.metadata,
            )
            step, output = self.executor.execute(
                item.get("role", "executor"),
                item.get("summary", f"Execute {item['tool']}"),
                item["tool"],
                payload,
                parent_step_id,
            )

            run.add_step(step)
            completed_step_ids.append(step.id)
            parent_step_id = step.id

            if step.status == "failed":
                return self._finalize_failed_run(
                    run,
                    plan,
                    completed_step_ids,
                    parent_step_id,
                    keywords,
                    final_output,
                    index + 1,
                    step.failure_type or "runtime_error",
                    output.get("error", "Unknown execution error."),
                )

            if item["tool"] == "extract_keywords":
                keywords = output.get("keywords", [])
            elif item["tool"] == "analyze_meal_image":
                run.metadata["vision_analysis"] = output.get("vision_analysis", [])
                candidate_answer = output.get("answer")
                if candidate_answer:
                    final_output = candidate_answer
            else:
                candidate_answer = output.get("answer")
                if candidate_answer:
                    final_output = candidate_answer
                else:
                    final_output = output.get("plan", final_output) or final_output

            run.checkpoint_state = self._checkpoint_state(
                "executing",
                plan,
                completed_step_ids,
                parent_step_id,
                keywords,
                final_output,
                index + 1,
                int(run.metadata.get("stop_after_step", 0) or 0),
            )
            self.store.save_checkpoint(run)

            paused_run = self._maybe_pause_run(
                run,
                plan,
                completed_step_ids,
                parent_step_id,
                keywords,
                final_output,
                index + 1,
            )
            if paused_run is not None:
                return paused_run

        review = self.reviewer.review(
            final_output,
            required_terms=required_terms,
            max_output_chars=max_output_chars,
            task=run.task,
            metadata=run.metadata,
        )
        run.reviewer = review
        run.checkpoint_state = self._checkpoint_state(
            "reviewed",
            plan,
            completed_step_ids,
            parent_step_id,
            keywords,
            final_output,
            len(plan),
            int(run.metadata.get("stop_after_step", 0) or 0),
        )
        self.store.save_checkpoint(run)

        if review.verdict == "revise":
            missing_terms = [term for term in (required_terms or []) if term.lower() not in final_output.lower()]
            run.metadata["_missing_terms"] = missing_terms
            step, output = self.executor.execute(
                "reviewer",
                "Revise meal-planning output after review",
                "revise_answer",
                self._build_payload("revise_answer", run.task, run.input_text, keywords, final_output, run.metadata),
                parent_step_id,
            )
            run.add_step(step)
            completed_step_ids.append(step.id)
            parent_step_id = step.id

            if step.status == "failed":
                return self._finalize_failed_run(
                    run,
                    plan,
                    completed_step_ids,
                    parent_step_id,
                    keywords,
                    final_output,
                    len(plan),
                    step.failure_type or "runtime_error",
                    output.get("error", "Unknown execution error."),
                )

            final_output = output.get("answer", final_output)
            run.reviewer = self.reviewer.review(
                final_output,
                required_terms=required_terms,
                max_output_chars=max_output_chars,
                task=run.task,
                metadata=run.metadata,
            )
            run.checkpoint_state = self._checkpoint_state(
                "revised",
                plan,
                completed_step_ids,
                parent_step_id,
                keywords,
                final_output,
                len(plan),
                int(run.metadata.get("stop_after_step", 0) or 0),
            )
            self.store.save_checkpoint(run)

        run.final_output = final_output
        run.status = "completed" if run.reviewer and run.reviewer.verdict == "approve" else "needs_attention"
        run.failure_type = None
        run.checkpoint_state = self._checkpoint_state(
            run.status,
            plan,
            completed_step_ids,
            parent_step_id,
            keywords,
            final_output,
            len(plan),
            int(run.metadata.get("stop_after_step", 0) or 0),
        )
        run.metadata.pop("_missing_terms", None)
        run.touch()
        self.store.save_run(run)
        self.store.save_checkpoint(run)
        return run

    def run_task(
        self,
        task: str,
        input_text: str,
        required_terms: list[str] | None = None,
        max_output_chars: int = 2600,
        eval_suite_version: str | None = None,
        run_context: dict[str, Any] | None = None,
        resume_from_run_id: str | None = None,
    ) -> RunRecord:
        if resume_from_run_id:
            run = self.store.resume_run(resume_from_run_id)
            merged_metadata = deepcopy(run.metadata)
            if run_context:
                merged_metadata.update(run_context)
            run.metadata = merged_metadata
            run.status = "running"

            plan_names = run.checkpoint_state.get("plan", [])
            if plan_names:
                plan = [{"role": "resumed_executor", "summary": f"Resume tool: {name}", "tool": name} for name in plan_names]
            else:
                plan = self.planner.plan(run.task, run.input_text, run_context=run.metadata)

            pending = run.checkpoint_state.get("pending_tools", [])
            if pending:
                start_index = max(0, len(plan) - len(pending))
            else:
                start_index = int(run.checkpoint_state.get("next_index", len(run.checkpoint_state.get("completed_step_ids", []))))

            keywords = list(run.checkpoint_state.get("keyword_cache", []))
            final_output = str(run.checkpoint_state.get("draft_output", run.final_output))
            completed_step_ids = list(run.checkpoint_state.get("completed_step_ids", []))
            parent_step_id = run.checkpoint_state.get("last_step_id")

            return self._execute_plan(
                run=run,
                plan=plan,
                start_index=start_index,
                keywords=keywords,
                final_output=final_output,
                parent_step_id=parent_step_id,
                completed_step_ids=completed_step_ids,
                required_terms=required_terms or list(run.metadata.get("required_terms", [])),
                max_output_chars=max_output_chars,
            )

        metadata = {
            "required_terms": required_terms or [],
            "max_output_chars": max_output_chars,
        }
        if run_context:
            metadata.update(run_context)

        run = RunRecord.create(
            task=task,
            input_text=input_text,
            tool_versions=self.registry.versions(),
            metadata=metadata,
            policy_version=self.policy_version,
            eval_suite_version=eval_suite_version or DEFAULT_EVAL_SUITE_VERSION,
        )
        run.status = "running"

        plan = self.planner.plan(task, input_text, run_context=metadata)
        keywords: list[str] = []
        final_output = ""
        parent_step_id: str | None = None
        completed_step_ids: list[str] = []

        run.checkpoint_state = self._checkpoint_state(
            "planned",
            plan,
            completed_step_ids,
            parent_step_id,
            keywords,
            final_output,
            0,
            int(metadata.get("stop_after_step", 0) or 0),
        )
        self.store.save_checkpoint(run)

        return self._execute_plan(
            run=run,
            plan=plan,
            start_index=0,
            keywords=keywords,
            final_output=final_output,
            parent_step_id=parent_step_id,
            completed_step_ids=completed_step_ids,
            required_terms=required_terms,
            max_output_chars=max_output_chars,
        )

    def resume_task(self, run_id: str, continue_execution: bool = False) -> RunRecord:
        run = self.store.resume_run(run_id)
        if not continue_execution:
            return run

        stage = (run.checkpoint_state or {}).get("stage", "")
        if stage in {"completed", "needs_attention", "failed"} or run.status in {"completed", "needs_attention", "failed"}:
            return run

        resume_context = deepcopy(run.metadata)
        resume_context["stop_after_step"] = 0
        return self.run_task(
            task=run.task,
            input_text=run.input_text,
            required_terms=list(run.metadata.get("required_terms", [])),
            max_output_chars=int(run.metadata.get("max_output_chars", 2600)),
            eval_suite_version=run.eval_suite_version,
            run_context=resume_context,
            resume_from_run_id=run_id,
        )

    def list_runs(self) -> list[dict[str, str]]:
        return self.store.list_runs()
