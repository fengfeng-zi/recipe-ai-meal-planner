from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

DEFAULT_POLICY_VERSION = "recipe-review-policy.v2"
DEFAULT_EVAL_SUITE_VERSION = "recipe-evals.v2"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    side_effects: list[str] = field(default_factory=list)
    timeout_s: int = 10
    retry_limit: int = 0
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolSpec":
        return cls(**payload)


@dataclass(slots=True)
class ToolCallRecord:
    id: str
    tool_name: str
    input_payload: dict[str, Any]
    output_payload: dict[str, Any] | None = None
    status: str = "pending"
    error: str | None = None
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    attempt: int = 1
    tool_version: str = "unknown"
    failure_type: str | None = None

    def complete(self, output_payload: dict[str, Any]) -> None:
        self.output_payload = output_payload
        self.status = "completed"
        self.finished_at = now_iso()

    def fail(self, message: str, failure_type: str = "tool_execution") -> None:
        self.status = "failed"
        self.error = message
        self.failure_type = failure_type
        self.finished_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolCallRecord":
        return cls(**payload)


@dataclass(slots=True)
class StepRecord:
    id: str
    role: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    parent_step_id: str | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    failure_type: str | None = None

    def complete(self) -> None:
        self.status = "completed"
        self.finished_at = now_iso()

    def fail(self, failure_type: str = "step_execution") -> None:
        self.status = "failed"
        self.failure_type = failure_type
        self.finished_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StepRecord":
        calls = [ToolCallRecord.from_dict(item) for item in payload.get("tool_calls", [])]
        step = cls(
            id=payload["id"],
            role=payload["role"],
            summary=payload["summary"],
            payload=payload.get("payload", {}),
            status=payload.get("status", "pending"),
            started_at=payload.get("started_at", now_iso()),
            finished_at=payload.get("finished_at"),
            parent_step_id=payload.get("parent_step_id"),
            tool_calls=calls,
            failure_type=payload.get("failure_type"),
        )
        return step


@dataclass(slots=True)
class ReviewDecision:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    revision_request: str | None = None
    policy_version: str = DEFAULT_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ReviewDecision | None":
        if payload is None:
            return None
        return cls(**payload)


@dataclass(slots=True)
class EvalCase:
    case_id: str
    task: str
    input_text: str
    required_terms: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    vision_provider: str = ""
    max_output_chars: int = 2600
    eval_suite_version: str = DEFAULT_EVAL_SUITE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        return cls(**payload)


@dataclass(slots=True)
class RunRecord:
    id: str
    task: str
    input_text: str
    status: str = "pending"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    tool_versions: dict[str, str] = field(default_factory=dict)
    policy_version: str = DEFAULT_POLICY_VERSION
    eval_suite_version: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    reviewer: ReviewDecision | None = None
    final_output: str = ""
    checkpoint_state: dict[str, Any] = field(default_factory=dict)
    failure_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        task: str,
        input_text: str,
        tool_versions: dict[str, str],
        metadata: dict[str, Any] | None = None,
        policy_version: str = DEFAULT_POLICY_VERSION,
        eval_suite_version: str | None = None,
    ) -> "RunRecord":
        return cls(
            id=make_id("run"),
            task=task,
            input_text=input_text,
            tool_versions=tool_versions,
            metadata=metadata or {},
            policy_version=policy_version,
            eval_suite_version=eval_suite_version,
        )

    def touch(self) -> None:
        self.updated_at = now_iso()

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "input_text": self.input_text,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_versions": self.tool_versions,
            "policy_version": self.policy_version,
            "eval_suite_version": self.eval_suite_version,
            "steps": [step.to_dict() for step in self.steps],
            "reviewer": self.reviewer.to_dict() if self.reviewer else None,
            "final_output": self.final_output,
            "checkpoint_state": self.checkpoint_state,
            "failure_type": self.failure_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        run = cls(
            id=payload["id"],
            task=payload["task"],
            input_text=payload["input_text"],
            status=payload.get("status", "pending"),
            created_at=payload.get("created_at", now_iso()),
            updated_at=payload.get("updated_at", now_iso()),
            tool_versions=payload.get("tool_versions", {}),
            policy_version=payload.get("policy_version", DEFAULT_POLICY_VERSION),
            eval_suite_version=payload.get("eval_suite_version"),
            reviewer=ReviewDecision.from_dict(payload.get("reviewer")),
            final_output=payload.get("final_output", ""),
            checkpoint_state=payload.get("checkpoint_state", {}),
            failure_type=payload.get("failure_type"),
            metadata=payload.get("metadata", {}),
        )
        run.steps = [StepRecord.from_dict(step) for step in payload.get("steps", [])]
        return run
