from __future__ import annotations

import re

from .index import SparseIndex
from .retrieval import RetrievalHit
from .vision import analysis_to_text

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")
MEAL_LABELS = {
    "breakfast": "\u65e9\u9910",
    "lunch": "\u5348\u9910",
    "dinner": "\u665a\u9910",
    "snack": "\u52a0\u9910",
    "meal": "\u6b63\u9910",
}
COMPONENT_LABELS = {
    "sparse_score": "\u7a00\u758f\u53ec\u56de",
    "ingredient_overlap": "\u98df\u6750\u91cd\u5408",
    "meal_type_match": "\u9910\u522b\u5339\u914d",
    "habit_alignment": "\u4e60\u60ef\u5339\u914d",
    "log_reuse_bonus": "\u8bb0\u5f55\u590d\u7528",
    "vision_alignment": "\u56fe\u50cf\u5bf9\u9f50",
    "visual_memory_alignment": "\u89c6\u89c9\u8bb0\u5fc6\u5bf9\u9f50",
    "rerank_bonus": "\u91cd\u6392\u52a0\u5206",
}


def _meal_label(value: str) -> str:
    return MEAL_LABELS.get(str(value).strip().lower(), str(value).strip() or "\u672a\u77e5\u9910\u522b")


def _line_value(text: str, prefix: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(prefix)}\s*(.+)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _recipe_snapshot(text: str, chunk_id: str) -> dict[str, str]:
    title = _line_value(text, "Recipe:") or _line_value(text, "Name:") or chunk_id
    meal_type = _line_value(text, "Meal type:") or "meal"
    prep_time = _line_value(text, "Prep time:") or "\u672a\u6807\u6ce8\u8017\u65f6"
    calories = _line_value(text, "Calories:") or "\u672a\u6807\u6ce8\u70ed\u91cf"
    protein = _line_value(text, "Protein:") or "\u672a\u6807\u6ce8\u86cb\u767d\u8d28"
    ingredients = _line_value(text, "Ingredients:") or "\u672a\u6807\u6ce8\u98df\u6750"
    benefits = _line_value(text, "Benefits:") or _line_value(text, "Use when:") or text[:120].strip()
    return {
        "title": title,
        "meal_type": meal_type,
        "prep_time": prep_time,
        "calories": calories,
        "protein": protein,
        "ingredients": ingredients,
        "benefits": benefits,
    }


def _memory_sections(trace: dict | None) -> dict[str, list[str]]:
    sections = {
        "preference": [],
        "goal": [],
        "allergy": [],
        "dislike": [],
        "meal_log": [],
        "visual_analysis": [],
        "note": [],
    }
    for item in (trace or {}).get("memory_hits", []):
        memory_type = item.get("memory_type", "note")
        sections.setdefault(memory_type, []).append(item.get("memory_text", ""))
    return sections


def _clean_memory_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_terms(texts: list[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower()))
    return tokens


def _query_meal_intent(query: str) -> list[str]:
    lowered = query.lower()
    intents = [meal for meal in MEAL_TYPES if meal in lowered]
    chinese_map = {
        "\u65e9\u9910": "breakfast",
        "\u5348\u9910": "lunch",
        "\u665a\u9910": "dinner",
        "\u52a0\u9910": "snack",
    }
    for zh, en in chinese_map.items():
        if zh in query and en not in intents:
            intents.append(en)
    return intents


def _non_zero_components(hit: RetrievalHit) -> list[str]:
    components = {
        "sparse_score": hit.sparse_score,
        "ingredient_overlap": hit.ingredient_overlap,
        "meal_type_match": hit.meal_type_match,
        "habit_alignment": hit.habit_alignment,
        "log_reuse_bonus": hit.log_reuse_bonus,
        "vision_alignment": hit.vision_alignment,
        "visual_memory_alignment": hit.visual_memory_alignment,
        "rerank_bonus": hit.rerank_bonus,
    }
    return [COMPONENT_LABELS.get(name, name) for name, value in components.items() if abs(value) > 1e-9]


def _visual_recipe_cues(analysis: dict) -> list[str]:
    cues: list[str] = []
    dish_name = str(analysis.get("dish_name", "")).strip()
    meal_type = str(analysis.get("meal_type", "")).strip().lower()
    ingredients = [str(item).strip() for item in analysis.get("visible_ingredients", []) if str(item).strip()]
    nutrition = [str(item).strip() for item in analysis.get("nutrition_signals", []) if str(item).strip()]
    cautions = [str(item).strip() for item in analysis.get("caution_tags", []) if str(item).strip()]
    method = str(analysis.get("cooking_method", "")).strip()
    portions = analysis.get("estimated_portions")
    recipe_cues = [str(item).strip() for item in analysis.get("recipe_cues", []) if str(item).strip()]

    if dish_name:
        cues.append(f"- \u8bc6\u522b\u83dc\u540d\uff1a{dish_name}\u3002")
    if meal_type and meal_type in MEAL_TYPES:
        cues.append(f"- \u8bc6\u522b\u9910\u522b\uff1a{_meal_label(meal_type)}\u3002")
    if ingredients:
        cues.append(f"- \u53ef\u89c1\u98df\u6750\uff1a{'\u3001'.join(ingredients[:6])}\u3002")
    if method:
        cues.append(f"- \u70f9\u996a\u65b9\u5f0f\uff1a{method}\u3002")
    if isinstance(portions, (int, float)) and portions > 0:
        cues.append(f"- \u4f30\u8ba1\u4efd\u91cf\uff1a{portions:g} \u4efd\u3002")
    if recipe_cues:
        cues.append(f"- \u505a\u6cd5\u7ebf\u7d22\uff1a{'\u3001'.join(recipe_cues[:5])}\u3002")
    if nutrition:
        cues.append(f"- \u8425\u517b\u4fe1\u53f7\uff1a{'\u3001'.join(nutrition[:4])}\u3002")
    if cautions:
        cues.append(f"- \u6ce8\u610f\u4e8b\u9879\uff1a{'\u3001'.join(cautions[:4])}\u3002")
    reconstruction = analysis.get("recipe_reconstruction", {})
    if isinstance(reconstruction, dict):
        style = str(reconstruction.get("dish_style", "")).strip()
        prep_summary = str(reconstruction.get("prep_summary", "")).strip()
        step_outline = [str(item).strip() for item in reconstruction.get("step_outline", []) if str(item).strip()]
        substitutions = [str(item).strip() for item in reconstruction.get("substitutions", []) if str(item).strip()]
        pantry_staples = [str(item).strip() for item in reconstruction.get("pantry_staples", []) if str(item).strip()]
        if style:
            cues.append(f"- \u83dc\u7cfb\u98ce\u683c\uff1a{style}\u3002")
        if prep_summary:
            cues.append(f"- \u5236\u4f5c\u6458\u8981\uff1a{prep_summary}")
        if step_outline:
            cues.append(f"- \u6b65\u9aa4\u63d0\u8981\uff1a{' | '.join(step_outline[:3])}\u3002")
        if substitutions:
            cues.append(f"- \u53ef\u66ff\u6362\u98df\u6750\uff1a{' | '.join(substitutions[:3])}\u3002")
        if pantry_staples:
            cues.append(f"- \u5e38\u5907\u98df\u6750\uff1a{'\u3001'.join(pantry_staples[:6])}\u3002")
    return cues


def build_grounded_answer(query: str, hits: list[RetrievalHit], index: SparseIndex, trace: dict | None = None) -> dict:
    recommendations = []
    citations = []
    conflict_guard_count = 0
    image_analysis = (trace or {}).get("image_analysis") or {}
    image_signal_applied = bool(image_analysis)

    for hit in hits:
        chunk = index.chunks[hit.chunk_id]
        snapshot = _recipe_snapshot(chunk.text, chunk.chunk_id)
        recommendations.append(
            f"- {snapshot['title']} [{chunk.chunk_id}]\uff1b\u9910\u522b\uff1a{_meal_label(snapshot['meal_type'])}\uff1b\u51c6\u5907\u65f6\u957f\uff1a{snapshot['prep_time']}\uff1b\u70ed\u91cf\uff1a{snapshot['calories']}\uff1b\u86cb\u767d\u8d28\uff1a{snapshot['protein']}\uff1b\u9002\u7528\u573a\u666f\uff1a{snapshot['benefits']}\uff1b\u6838\u5fc3\u98df\u6750\uff1a{snapshot['ingredients']}\u3002"
        )
        if hit.conflicting_memory_ids:
            conflict_guard_count += 1
        citations.append({
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source_path": chunk.source_path,
            "score": round(hit.score, 3),
            "components": {
                "sparse_score": round(hit.sparse_score, 3),
                "ingredient_overlap": round(hit.ingredient_overlap, 3),
                "meal_type_match": round(hit.meal_type_match, 3),
                "habit_alignment": round(hit.habit_alignment, 3),
                "log_reuse_bonus": round(hit.log_reuse_bonus, 3),
                "vision_alignment": round(hit.vision_alignment, 3),
                "visual_memory_alignment": round(hit.visual_memory_alignment, 3),
                "rerank_bonus": round(hit.rerank_bonus, 3),
            },
            "applied_signals": _non_zero_components(hit),
            "matched_memory_ids": hit.matched_memory_ids,
            "conflicting_memory_ids": hit.conflicting_memory_ids,
        })

    memory_sections = _memory_sections(trace)
    query_lower = query.lower()
    answer_lines: list[str] = []

    if memory_sections["allergy"] or memory_sections["dislike"]:
        constraints = [_clean_memory_text(item) for item in memory_sections["allergy"][:2] + memory_sections["dislike"][:2]]
        answer_lines.append("\u996e\u98df\u9650\u5236\u4e0e\u5fcc\u53e3")
        answer_lines.extend(f"- \u5fcc\u53e3/\u8fc7\u654f\uff1a{item}" for item in constraints)
    if memory_sections["goal"] or memory_sections["preference"]:
        preferences = [_clean_memory_text(item) for item in memory_sections["goal"][:2] + memory_sections["preference"][:2]]
        if not any(line == "\u76ee\u6807\u4e0e\u504f\u597d" for line in answer_lines):
            answer_lines.append("\u76ee\u6807\u4e0e\u504f\u597d")
        answer_lines.extend(f"- \u76ee\u6807/\u504f\u597d\uff1a{item}" for item in preferences)

    if "what did i log" in query_lower or "adjust" in query_lower or "log" in query_lower or "\u8bb0\u5f55" in query:
        if memory_sections["meal_log"]:
            meal_note = _clean_memory_text(memory_sections["meal_log"][0])
            answer_lines.append(f"\u7ed3\u5408\u8fd1\u671f\u996e\u98df\u8bb0\u5f55\uff1a{meal_note}")
            answer_lines.append("\u5df2\u636e\u6b64\u8c03\u6574\u63a8\u8350\u4f18\u5148\u7ea7\uff0c\u907f\u514d\u91cd\u590d\u4e0e\u51b2\u7a81\u642d\u914d\u3002")

    visual_memory_count = len(memory_sections.get("visual_analysis", []))
    if image_signal_applied or visual_memory_count:
        answer_lines.append("\u56fe\u50cf\u4e0e\u89c6\u89c9\u8bb0\u5fc6\u4fe1\u53f7")
        if image_signal_applied:
            answer_lines.append(f"- {analysis_to_text(image_analysis)}")
            image_tags = (trace or {}).get("image_tags", [])
            if image_tags:
                answer_lines.append(f"- \u56fe\u50cf\u6807\u7b7e\uff1a{'\u3001'.join(image_tags[:8])}\u3002")
            answer_lines.extend(_visual_recipe_cues(image_analysis))
        if visual_memory_count:
            answer_lines.append(f"- \u547d\u4e2d\u5386\u53f2\u89c6\u89c9\u8bb0\u5fc6\uff1a{visual_memory_count} \u6761\u3002")

    safety_terms = _extract_terms(memory_sections["allergy"] + memory_sections["dislike"])
    visual_terms = _extract_terms(memory_sections.get("visual_analysis", []))
    overlap_terms = sorted(safety_terms & visual_terms)
    if overlap_terms:
        answer_lines.append("\u68c0\u6d4b\u5230\u8fc7\u654f/\u5fcc\u53e3\u4e0e\u89c6\u89c9\u4fe1\u53f7\u5b58\u5728\u91cd\u53e0\uff1a" + "\u3001".join(overlap_terms[:5]) + "\u3002")

    if recommendations:
        answer_lines.append("\u68c0\u7d22\u8bc1\u636e\u4e0e\u5019\u9009\u83dc\u8c31")
        answer_lines.extend(recommendations[:3])
        if conflict_guard_count:
            answer_lines.append("\u5df2\u89e6\u53d1\u51b2\u7a81\u9632\u62a4\uff0c\u81ea\u52a8\u4e0b\u8c03\u5b58\u5728\u8bb0\u5fc6\u51b2\u7a81\u7684\u5019\u9009\u3002")
    else:
        answer_lines.append("\u5f53\u524d\u672a\u68c0\u7d22\u5230\u53ef\u76f4\u63a5\u63a8\u8350\u7684\u83dc\u8c31\uff0c\u8bf7\u8865\u5145\u504f\u597d\u3001\u9910\u522b\u6216\u4e0a\u4f20\u66f4\u6e05\u6670\u56fe\u7247\u3002")

    if hits:
        top_hit = hits[0]
        top_signals = _non_zero_components(top_hit)
        if top_signals:
            answer_lines.append("\u672c\u6b21\u6392\u5e8f\u4e3b\u8981\u4f9d\u636e\uff1a" + "\u3001".join(top_signals[:6]) + "\u3002")
        query_intent = _query_meal_intent(query)
        if query_intent:
            answer_lines.append("\u8bc6\u522b\u5230\u4f60\u7684\u9910\u522b\u610f\u56fe\uff1a" + "\u3001".join(_meal_label(item) for item in query_intent) + "\u3002")

    if "trace" in query_lower or "ranking" in query_lower or "\u6392\u5e8f" in query or "\u8bc1\u636e" in query:
        answer_lines.append("\u5982\u9700\u67e5\u770b\u8be6\u7ec6\u6392\u5e8f\u8fc7\u7a0b\uff0c\u8bf7\u5728\u95ee\u9898\u4e2d\u52a0\u5165\u201ctrace/\u6392\u5e8f/\u8bc1\u636e\u201d\u7b49\u5173\u952e\u8bcd\uff0c\u7cfb\u7edf\u4f1a\u8f93\u51fa\u5b8c\u6574\u6eaf\u6e90\u4fe1\u606f\u3002")

    query_intent = _query_meal_intent(query)
    trace_payload = {
        "evidence_count": len(citations),
        "memory_augmented_hits": sum(1 for hit in hits if hit.matched_memory_ids),
        "top_chunk_ids": [hit.chunk_id for hit in hits],
        "conflict_guard_count": conflict_guard_count,
        "memory_signal_count": sum(len(values) for values in memory_sections.values()),
        "visual_signal_count": len(memory_sections.get("visual_analysis", [])) + (1 if image_signal_applied else 0),
        "visual_augmented_hits": sum(1 for hit in hits if hit.vision_alignment > 0 or hit.visual_memory_alignment > 0),
        "image_signal_applied": image_signal_applied,
        "query_meal_intent": query_intent,
        "top_hit_applied_signals": _non_zero_components(hits[0]) if hits else [],
    }
    return {"query": query, "answer": "\n".join(answer_lines), "citations": citations, "trace": trace_payload}
