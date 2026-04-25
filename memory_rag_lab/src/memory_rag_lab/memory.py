from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re
import uuid

PROJECT_TRACK = "recipe_query_loop"
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")
TYPE_WEIGHTS = {
    "preference": 0.16,
    "goal": 0.14,
    "allergy": -0.32,
    "dislike": -0.22,
    "meal_log": 0.1,
    "visual_analysis": 0.12,
    "note": 0.06,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _dedupe_tags(tags: list[str] | None) -> list[str]:
    ordered: list[str] = []
    for tag in tags or []:
        tag_clean = tag.strip().lower()
        if tag_clean and tag_clean not in ordered:
            ordered.append(tag_clean)
    return ordered


def _normalize_str_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        items = [raw]
    else:
        items = []
    return _dedupe_tags([str(item).strip().lower() for item in items if str(item).strip()])


def _clamp_confidence(value: Any, default: float = 0.6) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


class SessionMemoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def _normalize_item(self, item: dict[str, Any], index: int) -> dict[str, Any]:
        memory_type = item.get("memory_type")
        if not memory_type:
            source = item.get("source", "conversation")
            memory_type = "meal_log" if source == "meal_log" else "note"
        return {
            "memory_id": item.get("memory_id", f"legacy_{index}"),
            "memory_text": item["memory_text"],
            "source": item.get("source", "conversation"),
            "created_at": item.get("created_at", now_iso()),
            "memory_type": memory_type,
            "tags": _dedupe_tags(item.get("tags", [])),
            "structured": item.get("structured", {}),
            "project_track": item.get("project_track", PROJECT_TRACK),
        }

    def _load(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        items = json.loads(path.read_text(encoding="utf-8-sig"))
        return [self._normalize_item(item, index) for index, item in enumerate(items)]

    def _write(self, session_id: str, items: list[dict[str, Any]]) -> None:
        self._path(session_id).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_memories(self, session_id: str) -> list[dict[str, Any]]:
        return self._load(session_id)

    def clear_memories(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            path.unlink()

    def save_memory(
        self,
        session_id: str,
        memory_text: str,
        source: str = "conversation",
        memory_type: str = "note",
        tags: list[str] | None = None,
        structured: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        items = self._load(session_id)
        clean_tags = _dedupe_tags(tags)
        for item in items:
            if item["memory_text"] == memory_text and item["memory_type"] == memory_type and item["tags"] == clean_tags:
                return item
        payload = {
            "memory_id": f"mem_{uuid.uuid4().hex[:8]}",
            "memory_text": memory_text,
            "source": source,
            "created_at": now_iso(),
            "memory_type": memory_type,
            "tags": clean_tags,
            "structured": structured or {},
            "project_track": PROJECT_TRACK,
        }
        items.append(payload)
        self._write(session_id, items)
        return payload

    def save_preference(
        self,
        session_id: str,
        memory_text: str,
        memory_type: str = "preference",
        tags: list[str] | None = None,
        source: str = "profile",
    ) -> dict[str, Any]:
        return self.save_memory(
            session_id,
            memory_text=memory_text,
            source=source,
            memory_type=memory_type,
            tags=tags,
        )

    def save_visual_analysis(
        self,
        session_id: str,
        analysis: dict[str, Any],
        source: str = "vision",
    ) -> dict[str, Any]:
        dish_name = str(analysis.get("dish_name", "meal photo")).strip()
        meal_type = str(analysis.get("meal_type", "")).strip().lower()
        visible_ingredients = _normalize_str_list(analysis.get("visible_ingredients", []))
        cuisine_tags = _normalize_str_list(analysis.get("cuisine_tags", []))
        nutrition_signals = _normalize_str_list(analysis.get("nutrition_signals", []))
        caution_tags = _normalize_str_list(analysis.get("caution_tags", []))
        confidence = _clamp_confidence(analysis.get("confidence", 0.6))
        image_path = str(analysis.get("image_path", "")).strip()
        analysis_text = str(analysis.get("summary", "")).strip()

        analysis_tags = _dedupe_tags(
            [meal_type] + visible_ingredients + cuisine_tags + nutrition_signals + caution_tags
        )
        summary_bits = [
            f"Visual analysis observed {dish_name}.",
            f"Meal type: {meal_type}." if meal_type else "",
            f"Visible ingredients: {', '.join(visible_ingredients)}." if visible_ingredients else "",
            f"Cuisine tags: {', '.join(cuisine_tags)}." if cuisine_tags else "",
            f"Caution tags: {', '.join(caution_tags)}." if caution_tags else "",
            f"Summary: {analysis_text}." if analysis_text else "",
        ]
        memory_text = " ".join(bit for bit in summary_bits if bit)
        return self.save_memory(
            session_id,
            memory_text=memory_text,
            source=source,
            memory_type="visual_analysis",
            tags=analysis_tags,
            structured={
                "dish_name": dish_name,
                "meal_type": meal_type,
                "visible_ingredients": visible_ingredients,
                "cuisine_tags": cuisine_tags,
                "nutrition_signals": nutrition_signals,
                "caution_tags": caution_tags,
                "confidence": confidence,
                "image_path": image_path,
                "summary": analysis_text,
                "analysis_tags": analysis_tags,
            },
        )

    def log_meal(
        self,
        session_id: str,
        meal_name: str,
        meal_type: str,
        calories: int | None = None,
        ingredients: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        ingredient_list = _dedupe_tags(ingredients)
        detail_bits = [
            f"Logged {meal_type}: {meal_name}.",
            f"Calories: {calories}." if calories is not None else "",
            f"Ingredients: {', '.join(ingredient_list)}." if ingredient_list else "",
            f"Notes: {notes.strip()}." if notes.strip() else "",
        ]
        memory_text = " ".join(bit for bit in detail_bits if bit)
        return self.save_memory(
            session_id,
            memory_text=memory_text,
            source="meal_log",
            memory_type="meal_log",
            tags=[meal_type, *ingredient_list],
            structured={
                "meal_name": meal_name,
                "meal_type": meal_type,
                "calories": calories,
                "ingredients": ingredient_list,
                "notes": notes.strip(),
            },
        )

    def build_profile_summary(self, session_id: str) -> dict[str, Any]:
        memories = self._load(session_id)
        summary = {
            "preferences": [],
            "goals": [],
            "allergies": [],
            "dislikes": [],
            "recent_meal_types": [],
            "meal_log_count": 0,
            "visual_analysis_count": 0,
            "visual_caution_tags": [],
        }
        for item in memories:
            memory_type = item["memory_type"]
            if memory_type == "preference":
                summary["preferences"].append(item["memory_text"])
            elif memory_type == "goal":
                summary["goals"].append(item["memory_text"])
            elif memory_type == "allergy":
                summary["allergies"].append(item["memory_text"])
            elif memory_type == "dislike":
                summary["dislikes"].append(item["memory_text"])
            elif memory_type == "meal_log":
                summary["meal_log_count"] += 1
                meal_type = item.get("structured", {}).get("meal_type")
                if meal_type and meal_type not in summary["recent_meal_types"]:
                    summary["recent_meal_types"].append(meal_type)
            elif memory_type == "visual_analysis":
                summary["visual_analysis_count"] += 1
                for caution in item.get("structured", {}).get("caution_tags", []):
                    if caution and caution not in summary["visual_caution_tags"]:
                        summary["visual_caution_tags"].append(caution)
        return summary

    def extract_memory_candidates(self, text: str) -> list[str]:
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        keepers = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(
                keyword in sentence_lower
                for keyword in ("prefer", "avoid", "allergy", "goal", "usually", "breakfast", "dinner")
            ):
                keepers.append(sentence)
        return keepers[:3]

    def search(self, session_id: str, query: str) -> list[dict[str, Any]]:
        query_terms = _tokenize(query)
        query_meal_types = {meal for meal in MEAL_TYPES if meal in query_terms}
        matches = []
        for item in self._load(session_id):
            base_terms = _tokenize(item["memory_text"])
            tag_terms = set(item.get("tags", []))
            terms = base_terms | tag_terms
            overlap = len(query_terms & terms)
            structured = item.get("structured", {})
            meal_type = structured.get("meal_type")
            meal_type_match = 1 if meal_type and meal_type in query_meal_types else 0
            visual_overlap = 0
            visual_confidence = 0.0
            if item.get("memory_type") == "visual_analysis":
                visual_tags = set(structured.get("analysis_tags", []))
                visual_overlap = len(query_terms & visual_tags)
                visual_confidence = _clamp_confidence(structured.get("confidence", 0.6))
            if not overlap and not meal_type_match and not visual_overlap:
                continue
            type_weight = TYPE_WEIGHTS.get(item["memory_type"], 0.05)
            recency_boost = 0.05 if item["memory_type"] == "meal_log" else 0.0
            visual_boost = visual_overlap * 0.1 * max(0.35, visual_confidence) if visual_overlap else 0.0
            score = overlap * abs(type_weight) + meal_type_match * 0.4 + recency_boost + visual_boost
            matches.append({
                **item,
                "overlap": overlap,
                "meal_type_match": meal_type_match,
                "visual_overlap": visual_overlap,
                "visual_confidence": round(visual_confidence, 3),
                "type_weight": type_weight,
                "recency_boost": recency_boost,
                "visual_boost": round(visual_boost, 3),
                "score": round(score, 3),
            })
        matches.sort(key=lambda entry: (entry["score"], entry["overlap"]), reverse=True)
        return matches
