from __future__ import annotations

from dataclasses import replace
import re

from .index import SparseIndex
from .retrieval import RetrievalHit

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


def _query_meal_types(query: str) -> set[str]:
    query_lower = query.lower()
    return {meal for meal in MEAL_TYPES if meal in query_lower}


def _chunk_meal_type(text: str) -> str | None:
    match = re.search(r"meal type:\s*(breakfast|lunch|dinner|snack)", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def rerank_hits(query: str, hits: list[RetrievalHit], index: SparseIndex, top_k: int = 3) -> list[RetrievalHit]:
    query_lower = query.lower()
    query_meal_types = _query_meal_types(query)
    strict_intent = len(query_meal_types) == 1
    rescored: list[tuple[float, RetrievalHit]] = []
    for hit in hits:
        chunk = index.chunks[hit.chunk_id]
        chunk_lower = chunk.text.lower()
        chunk_meal_type = _chunk_meal_type(chunk.text)
        phrase_bonus = 0.35 if query_lower in chunk_lower else 0.0
        concise_bonus = 0.1 if len(chunk.text) < 550 else 0.0
        protective_bonus = 0.08 if not hit.conflicting_memory_ids else -0.05

        meal_intent_bonus = 0.0
        if query_meal_types and chunk_meal_type:
            if chunk_meal_type in query_meal_types:
                meal_intent_bonus += 0.55 if strict_intent else 0.4
            else:
                meal_intent_bonus -= 1.25 if strict_intent else 0.9
        elif query_meal_types and not chunk_meal_type:
            meal_intent_bonus -= 0.25 if strict_intent else 0.1

        rerank_bonus = phrase_bonus + concise_bonus + protective_bonus + meal_intent_bonus
        rescored_hit = replace(hit, rerank_bonus=rerank_bonus, score=hit.score + rerank_bonus)
        rescored.append((rescored_hit.score, rescored_hit))
    rescored.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in rescored[:top_k]]
