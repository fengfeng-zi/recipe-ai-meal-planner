from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_indexable(path: Path) -> bool:
    if path.suffix.lower() not in {".txt", ".md", ".json"}:
        return False
    # Keep vision sidecars for multimodal runtime only, not retrieval corpus indexing.
    if path.name.lower().endswith(".vision.json"):
        return False
    return True


def _load_one(path: Path) -> Document:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        title = payload.get("title", path.stem)
        text = payload.get("text", json.dumps(payload, ensure_ascii=False))
        metadata = {k: v for k, v in payload.items() if k not in {"title", "text"}}
        return Document(doc_id=path.stem, title=title, text=text, source_path=str(path), metadata=metadata)
    text = path.read_text(encoding="utf-8")
    return Document(doc_id=path.stem, title=path.stem.replace("_", " "), text=text, source_path=str(path))


def load_documents(target: str | Path) -> list[Document]:
    path = Path(target)
    if path.is_file():
        return [_load_one(path)] if _is_indexable(path) else []
    docs: list[Document] = []
    for child in sorted(path.rglob("*")):
        if not _is_indexable(child):
            continue
        docs.append(_load_one(child))
    return docs
