from __future__ import annotations

from pathlib import Path
import json

from .models import RunRecord


class JsonRunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.checkpoints_dir = self.root / "checkpoints"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoints_dir / f"{run_id}.json"

    def save_run(self, run: RunRecord) -> None:
        self._run_path(run.id).write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")

    def save_checkpoint(self, run: RunRecord) -> None:
        self._checkpoint_path(run.id).write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")

    def load_run(self, run_id: str) -> RunRecord:
        payload = json.loads(self._run_path(run_id).read_text(encoding="utf-8"))
        return RunRecord.from_dict(payload)

    def resume_run(self, run_id: str) -> RunRecord:
        payload = json.loads(self._checkpoint_path(run_id).read_text(encoding="utf-8"))
        return RunRecord.from_dict(payload)

    def list_runs(self) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            items.append({
                "run_id": payload["id"],
                "task": payload["task"],
                "status": payload["status"],
                "failure_type": payload.get("failure_type") or "",
                "checkpoint_stage": payload.get("checkpoint_state", {}).get("stage", ""),
                "updated_at": payload["updated_at"],
            })
        return items
