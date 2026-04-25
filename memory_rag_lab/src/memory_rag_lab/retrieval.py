from __future__ import annotations

from dataclasses import dataclass, field
import re

from .index import SparseIndex, tokenize
from .memory import SessionMemoryStore
from .vision import analysis_tags, analysis_to_text

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    score: float
    sparse_score: float
    ingredient_overlap: float
    meal_type_match: float
    habit_alignment: float
    log_reuse_bonus: float
    vision_alignment: float
    visual_memory_alignment: float
    rerank_bonus: float = 0.0
    matched_memory_ids: list[str] = field(default_factory=list)
    conflicting_memory_ids: list[str] = field(default_factory=list)


def serialize_hits(hits: list[RetrievalHit]) -> list[dict]:
    return [
        {
            "chunk_id": hit.chunk_id,
            "score": round(hit.score, 3),
            "sparse_score": round(hit.sparse_score, 3),
            "ingredient_overlap": round(hit.ingredient_overlap, 3),
            "meal_type_match": round(hit.meal_type_match, 3),
            "habit_alignment": round(hit.habit_alignment, 3),
            "log_reuse_bonus": round(hit.log_reuse_bonus, 3),
            "vision_alignment": round(hit.vision_alignment, 3),
            "visual_memory_alignment": round(hit.visual_memory_alignment, 3),
            "rerank_bonus": round(hit.rerank_bonus, 3),
            "matched_memory_ids": hit.matched_memory_ids,
            "conflicting_memory_ids": hit.conflicting_memory_ids,
        }
        for hit in hits
    ]


def _query_meal_types(query_tokens: set[str]) -> set[str]:
    return {meal for meal in MEAL_TYPES if meal in query_tokens}


def _chunk_meal_type(text: str, chunk_tokens: set[str]) -> str | None:
    match = re.search(r"meal type:\s*(breakfast|lunch|dinner|snack)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    present = [meal for meal in MEAL_TYPES if meal in chunk_tokens]
    return present[0] if len(present) == 1 else None


def hybrid_retrieve(
    query: str,
    index: SparseIndex,
    top_k: int = 5,
    memory_store: SessionMemoryStore | None = None,
    session_id: str | None = None,
    image_analysis: dict | None = None,
) -> tuple[list[RetrievalHit], dict]:
    query_tokens = set(tokenize(query))
    query_meal_types = _query_meal_types(query_tokens)
    image_tokens = set(analysis_tags(image_analysis)) if image_analysis else set()
    image_meal_type = str(image_analysis.get("meal_type", "")).strip().lower() if image_analysis else ""
    memory_query = query
    if image_tokens:
        memory_query = f"{query} {' '.join(sorted(image_tokens))}".strip()
    memory_hits = memory_store.search(session_id, memory_query) if memory_store and session_id else []

    hits: list[RetrievalHit] = []
    filtered_conflicts: list[dict] = []
    for chunk_id, chunk in index.chunks.items():
        sparse = index.sparse_score(query, chunk_id)
        chunk_tokens = set(tokenize(chunk.text))
        chunk_meal_type = _chunk_meal_type(chunk.text, chunk_tokens)
        ingredient_overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
        meal_type_match = 0.0
        if query_meal_types:
            if chunk_meal_type and chunk_meal_type in query_meal_types:
                meal_type_match += 0.35
            elif chunk_meal_type and chunk_meal_type not in query_meal_types:
                meal_type_match -= 0.45
            elif query_meal_types & chunk_tokens:
                meal_type_match += 0.2

        habit_alignment = 0.0
        log_reuse_bonus = 0.0
        vision_alignment = 0.0
        visual_memory_alignment = 0.0
        matched_memory_ids: list[str] = []
        conflicting_memory_ids: list[str] = []
        for item in memory_hits:
            memory_tokens = set(tokenize(item["memory_text"])) | set(item.get("tags", []))
            overlaps_chunk = bool(memory_tokens & chunk_tokens) or (
                item.get("structured", {}).get("meal_type") in chunk_tokens
            )
            if not overlaps_chunk:
                continue
            matched_memory_ids.append(item["memory_id"])
            memory_type = item["memory_type"]
            if memory_type in {"preference", "goal"}:
                habit_alignment += min(0.28, 0.05 * item["overlap"] + 0.03 * item["meal_type_match"])
            elif memory_type in {"allergy", "dislike"}:
                penalty = min(0.46, 0.14 * max(1, item["overlap"]))
                habit_alignment -= penalty
                conflicting_memory_ids.append(item["memory_id"])
            elif memory_type == "meal_log":
                log_reuse_bonus += min(0.18, 0.05 * max(1, item["overlap"]) + 0.04 * item["meal_type_match"])
            elif memory_type == "visual_analysis":
                structured = item.get("structured", {})
                visual_tags = set(structured.get("analysis_tags", [])) | set(item.get("tags", []))
                overlap = len(visual_tags & chunk_tokens)
                confidence = float(structured.get("confidence", item.get("visual_confidence", 0.6)) or 0.6)
                confidence = max(0.0, min(1.0, confidence))
                if overlap:
                    visual_memory_alignment += min(0.32, 0.07 * overlap * max(0.35, confidence))
                visual_meal_type = str(structured.get("meal_type", "")).strip().lower()
                if visual_meal_type and visual_meal_type in chunk_tokens:
                    visual_memory_alignment += 0.1 * max(0.35, confidence)
            else:
                habit_alignment += min(0.1, 0.03 * max(1, item["overlap"]))

        if image_analysis:
            overlap = len(image_tokens & chunk_tokens)
            if overlap:
                vision_alignment += min(0.42, 0.08 * overlap)
            if image_meal_type and image_meal_type in chunk_tokens:
                vision_alignment += 0.18

        total = (
            sparse
            + ingredient_overlap
            + meal_type_match
            + habit_alignment
            + log_reuse_bonus
            + vision_alignment
            + visual_memory_alignment
        )
        if conflicting_memory_ids and total < 0.25:
            filtered_conflicts.append(
                {
                    "chunk_id": chunk_id,
                    "conflicting_memory_ids": conflicting_memory_ids,
                }
            )
            continue
        if total <= 0:
            continue
        hits.append(
            RetrievalHit(
                chunk_id=chunk_id,
                score=total,
                sparse_score=sparse,
                ingredient_overlap=ingredient_overlap,
                meal_type_match=meal_type_match,
                habit_alignment=habit_alignment,
                log_reuse_bonus=log_reuse_bonus,
                vision_alignment=vision_alignment,
                visual_memory_alignment=visual_memory_alignment,
                matched_memory_ids=matched_memory_ids,
                conflicting_memory_ids=conflicting_memory_ids,
            )
        )
    hits.sort(key=lambda item: item.score, reverse=True)
    ranked_hits = hits[:top_k]
    trace = {
        "trace_version": "recipe-query-trace.v5",
        "query": query,
        "session_id": session_id,
        "image_analysis": {
            **image_analysis,
            "analysis_text": analysis_to_text(image_analysis),
        } if image_analysis else None,
        "image_tags": sorted(image_tokens),
        "image_meal_type": image_meal_type,
        "memory_hits": [
            {
                "memory_id": item["memory_id"],
                "memory_type": item["memory_type"],
                "memory_text": item["memory_text"],
                "source": item.get("source", "conversation"),
                "tags": item.get("tags", []),
                "overlap": item["overlap"],
                "meal_type_match": item["meal_type_match"],
                "visual_overlap": item.get("visual_overlap", 0),
                "visual_confidence": item.get("visual_confidence", 0.0),
            }
            for item in memory_hits[:8]
        ],
        "filtered_conflicts": filtered_conflicts,
        "ranked_hits": serialize_hits(ranked_hits),
    }
    return ranked_hits, trace
