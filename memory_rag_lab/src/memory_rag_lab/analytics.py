from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from .chunking import chunk_document
from .documents import load_documents
from .evals import DEFAULT_EVAL_CASES, ensure_sample_docs


def build_report(project_root: str | Path) -> dict:
    root = Path(project_root)
    docs_root = ensure_sample_docs(root)
    documents = load_documents(docs_root)

    chunk_counts = {}
    for strategy in ("paragraph", "sliding", "hybrid"):
        total = 0
        for document in documents:
            total += len(chunk_document(document, strategy=strategy))
        chunk_counts[strategy] = total

    memory_root = root / "data" / "memories"
    memory_root.mkdir(parents=True, exist_ok=True)
    session_files = sorted(memory_root.glob("*.json"))
    session_sizes = {}
    source_counts = Counter()
    memory_type_counts = Counter()
    total_memories = 0
    sessions_with_meal_logs = 0
    sessions_with_visual_memories = 0
    visual_confidence_values: list[float] = []
    visual_field_presence = Counter()

    for path in session_files:
        items = json.loads(path.read_text(encoding="utf-8-sig"))
        session_sizes[path.stem] = len(items)
        total_memories += len(items)
        has_meal_log = False
        has_visual_memory = False
        for item in items:
            source_counts[item.get("source", "conversation")] += 1
            memory_type = item.get("memory_type", "note")
            memory_type_counts[memory_type] += 1
            if memory_type == "meal_log":
                has_meal_log = True
            elif memory_type == "visual_analysis":
                has_visual_memory = True
                structured = item.get("structured", {})
                if isinstance(structured, dict):
                    for field in (
                        "dish_name",
                        "meal_type",
                        "visible_ingredients",
                        "cuisine_tags",
                        "nutrition_signals",
                        "caution_tags",
                        "recipe_cues",
                        "cooking_method",
                        "estimated_portions",
                    ):
                        value = structured.get(field)
                        if isinstance(value, str) and value.strip():
                            visual_field_presence[field] += 1
                        elif isinstance(value, (int, float)):
                            visual_field_presence[field] += 1
                        elif isinstance(value, list) and value:
                            visual_field_presence[field] += 1
                confidence = structured.get("confidence")
                if isinstance(confidence, (int, float)):
                    visual_confidence_values.append(float(confidence))
        if has_meal_log:
            sessions_with_meal_logs += 1
        if has_visual_memory:
            sessions_with_visual_memories += 1

    vision_assets = len(list((docs_root / "images").glob("*.*"))) if (docs_root / "images").exists() else 0
    avg_visual_confidence = (
        round(sum(visual_confidence_values) / len(visual_confidence_values), 3)
        if visual_confidence_values
        else 0.0
    )

    return {
        "report_version": "recipe-query-report.v3",
        "total_documents": len(documents),
        "source_extensions": sorted({Path(document.source_path).suffix.lower() for document in documents}),
        "chunk_counts": chunk_counts,
        "memory_sessions": len(session_files),
        "total_memories": total_memories,
        "memory_source_counts": dict(source_counts),
        "memory_type_counts": dict(memory_type_counts),
        "session_sizes": session_sizes,
        "sessions_with_meal_logs": sessions_with_meal_logs,
        "sessions_with_visual_memories": sessions_with_visual_memories,
        "visual_memory_count": memory_type_counts.get("visual_analysis", 0),
        "avg_visual_memory_confidence": avg_visual_confidence,
        "vision_assets": vision_assets,
        "visual_field_presence": dict(visual_field_presence),
        "structural_metrics": {
            "supported_document_types": 3,
            "chunking_strategies": 3,
            "retrieval_score_components": 8,
            "citation_fields": 8,
            "memory_record_fields": 8,
            "default_eval_cases": len(DEFAULT_EVAL_CASES),
            "default_top_k": 5,
            "query_loop_commands": 7,
            "vision_provider_modes": 3,
        },
    }
