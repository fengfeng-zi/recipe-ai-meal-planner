from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import sys

from .models import ToolSpec
from .vision import analysis_tags, analysis_to_text, analyze_meal_image

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "be", "this", "that",
    "it", "as", "by", "from", "at", "you", "your", "we", "our", "can", "will", "about",
}
MEAL_SEQUENCE = ("breakfast", "lunch", "dinner", "snack")

MEAL_LABELS = {
    "breakfast": "??",
    "lunch": "??",
    "dinner": "??",
    "snack": "??",
    "meal": "??",
}
FIT_REASON_LABELS = {
    "filtered_by_constraint": "??????",
    "protein_target": "???????",
    "quick_cook": "??????",
    "calorie_aligned": "???????",
    "budget_friendly": "????",
    "habit_breakfast_gap": "??????",
    "habit_satiety_support": "?????",
    "safe_default": "??????",
    "keyword_overlap": "?????",
    "visual_ingredient_match": "??????",
    "visual_meal_type": "??????",
    "visual_dish_match": "??????",
    "retrieval_score": "?????",
}


def _meal_label(value: str) -> str:
    return MEAL_LABELS.get(str(value).strip().lower(), str(value).strip() or "??")


def _fit_reason_label(reason: str) -> str:
    raw = str(reason).strip()
    if raw in FIT_REASON_LABELS:
        return FIT_REASON_LABELS[raw]
    if raw.startswith("matches_"):
        return f"??{_meal_label(raw.removeprefix('matches_'))}??"
    return raw.replace("_", " ")

RECIPE_CATALOG = [
    {
        "name": "Greek Yogurt Berry Oats",
        "meal_type": "breakfast",
        "kcal": 410,
        "protein_g": 31,
        "prep_minutes": 8,
        "ingredients": ["greek yogurt", "oats", "berries", "chia seeds", "cinnamon"],
        "tags": ["high-protein", "quick-cook", "fat-loss", "breakfast"],
        "fit_notes": "High-satiety breakfast that helps prevent late-night snacking.",
    },
    {
        "name": "Egg White Spinach Wrap",
        "meal_type": "breakfast",
        "kcal": 360,
        "protein_g": 28,
        "prep_minutes": 10,
        "ingredients": ["egg whites", "whole wheat wrap", "spinach", "tomato", "feta"],
        "tags": ["high-protein", "quick-cook", "breakfast"],
        "fit_notes": "Portable breakfast for busy mornings with solid protein density.",
    },
    {
        "name": "Chicken Quinoa Power Bowl",
        "meal_type": "lunch",
        "kcal": 560,
        "protein_g": 44,
        "prep_minutes": 20,
        "ingredients": ["chicken breast", "quinoa", "broccoli", "carrot", "olive oil"],
        "tags": ["high-protein", "meal-prep", "fat-loss", "lunch"],
        "fit_notes": "Meal-prep friendly lunch with vegetables and stable energy.",
    },
    {
        "name": "Tofu Edamame Grain Bowl",
        "meal_type": "lunch",
        "kcal": 520,
        "protein_g": 32,
        "prep_minutes": 18,
        "ingredients": ["firm tofu", "edamame", "brown rice", "cucumber", "sesame"],
        "tags": ["vegetarian", "high-protein", "lunch"],
        "fit_notes": "Plant-forward lunch that still hits protein targets.",
    },
    {
        "name": "Turkey Chili Lettuce Bowl",
        "meal_type": "dinner",
        "kcal": 510,
        "protein_g": 40,
        "prep_minutes": 25,
        "ingredients": ["lean turkey", "kidney beans", "tomato", "lettuce", "bell pepper"],
        "tags": ["high-protein", "batch-cook", "dinner", "fat-loss"],
        "fit_notes": "Dense protein dinner with fiber for appetite control.",
    },
    {
        "name": "Garlic Salmon Veggie Tray",
        "meal_type": "dinner",
        "kcal": 540,
        "protein_g": 38,
        "prep_minutes": 22,
        "ingredients": ["salmon", "asparagus", "baby potatoes", "garlic", "lemon"],
        "tags": ["high-protein", "quick-cook", "dinner"],
        "fit_notes": "Balanced sheet-pan dinner with minimal cleanup.",
    },
    {
        "name": "Chicken Stir-Fry Rice Bowl",
        "meal_type": "dinner",
        "kcal": 560,
        "protein_g": 42,
        "prep_minutes": 20,
        "ingredients": ["chicken breast", "rice", "broccoli", "mushroom", "soy sauce"],
        "tags": ["high-protein", "quick-cook", "dinner"],
        "fit_notes": "Fast weeknight dinner with easy ingredient swaps.",
    },
    {
        "name": "Cottage Cheese Fruit Cup",
        "meal_type": "snack",
        "kcal": 220,
        "protein_g": 20,
        "prep_minutes": 5,
        "ingredients": ["cottage cheese", "apple", "walnut", "cinnamon"],
        "tags": ["high-protein", "snack", "quick-cook"],
        "fit_notes": "Useful snack when the user tends to overeat at night.",
    },
]


@dataclass(slots=True)
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[[dict], dict]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: Callable[[dict], dict]) -> None:
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def get_spec(self, name: str) -> ToolSpec:
        return self._tools[name].spec

    def versions(self) -> dict[str, str]:
        return {name: tool.spec.version for name, tool in self._tools.items()}

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, payload: dict) -> dict:
        return self._tools[name].handler(payload)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _token_set(*parts: str) -> set[str]:
    merged: list[str] = []
    for part in parts:
        merged.extend(_tokens(part))
    return set(merged)


def _line_value(text: str, prefix: str) -> str:
    match = re.search(rf"^{re.escape(prefix)}\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _first_number(text: str, default: int) -> int:
    match = re.search(r"(\d+)", text or "")
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _as_ingredient_list(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _normalize_terms(raw: object) -> list[str]:
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = [str(item) for item in raw]
    else:
        items = []
    values: list[str] = []
    for item in items:
        clean = item.strip().lower()
        if clean and clean not in values:
            values.append(clean)
    return values


def _profile(payload: dict) -> dict:
    profile = payload.get("user_profile", {})
    return profile if isinstance(profile, dict) else {}


def _meal_logs(payload: dict) -> list[str]:
    raw = payload.get("meal_logs", [])
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _vision_analyses(payload: dict) -> list[dict]:
    raw = payload.get("vision_analysis", [])
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def _detect_habit_flags(logs: list[str], input_text: str) -> list[str]:
    combined = " ".join(logs + [input_text]).lower()
    flags: list[str] = []
    if "skip breakfast" in combined or "skipped breakfast" in combined:
        flags.append("breakfast_gap")
    if "late-night" in combined or "night snacking" in combined or "overeat at night" in combined:
        flags.append("late_night_snacking")
    if "sugar" in combined:
        flags.append("high_sugar_drinks")
    return flags


def _recipe_is_safe(recipe: dict, allergies: list[str], dislikes: list[str]) -> bool:
    haystack = _token_set(recipe["name"], " ".join(recipe["ingredients"]), " ".join(recipe["tags"]))
    blocked = set(allergies) | set(dislikes)
    return not bool(haystack & blocked)


def _vision_alignment(recipe: dict, payload: dict) -> tuple[float, list[str]]:
    analyses = _vision_analyses(payload)
    if not analyses:
        return 0.0, []
    recipe_tokens = _token_set(recipe["name"], recipe["meal_type"], " ".join(recipe["ingredients"]), " ".join(recipe["tags"]))
    score = 0.0
    reasons: list[str] = []
    for analysis in analyses:
        tags = set(analysis_tags(analysis))
        overlap = len(tags & recipe_tokens)
        if overlap:
            score += min(1.1, 0.22 * overlap)
            reasons.append("visual_ingredient_match")
        meal_type = str(analysis.get("meal_type", "")).strip().lower()
        if meal_type and meal_type == recipe["meal_type"]:
            score += 0.55
            reasons.append("visual_meal_type")
        dish_name = str(analysis.get("dish_name", "")).lower()
        if dish_name and any(token in dish_name for token in _tokens(recipe["name"])):
            score += 0.45
            reasons.append("visual_dish_match")
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return round(score, 3), deduped


def _score_recipe(recipe: dict, payload: dict) -> tuple[float, list[str]]:
    profile = _profile(payload)
    query_tokens = _token_set(payload.get("task", ""), payload.get("input_text", ""), " ".join(payload.get("keywords", [])))
    allergies = _normalize_terms(profile.get("allergies", []))
    dislikes = _normalize_terms(profile.get("dislikes", []))
    preferences = _normalize_terms(profile.get("preferences", []))
    logs = _meal_logs(payload)
    habit_flags = _detect_habit_flags(logs, str(payload.get("input_text", "")))

    if not _recipe_is_safe(recipe, allergies, dislikes):
        return -999.0, ["filtered_by_constraint"]

    recipe_tokens = _token_set(recipe["name"], recipe["meal_type"], " ".join(recipe["ingredients"]), " ".join(recipe["tags"]), recipe["fit_notes"])
    reasons: list[str] = []
    score = 0.0

    meal_type = recipe["meal_type"]
    if meal_type in query_tokens:
        score += 1.6
        reasons.append(f"matches_{meal_type}")
    if (("high" in query_tokens and "protein" in query_tokens) or ("high-protein" in preferences)):
        score += recipe["protein_g"] / 25.0
        reasons.append("protein_target")
    if ("quick" in query_tokens or "busy" in query_tokens or "quick-cook" in preferences) and recipe["prep_minutes"] <= 20:
        score += 1.0
        reasons.append("quick_cook")
    if (("fat" in query_tokens and "loss" in query_tokens) or ("fat-loss" in recipe["tags"])) and recipe["kcal"] <= 560:
        score += 0.8
        reasons.append("calorie_aligned")
    if any(token in query_tokens for token in ("budget", "affordable", "cheap", "low")):
        if any(token in recipe_tokens for token in ("oats", "egg", "turkey", "beans", "rice", "chicken")):
            score += 0.7
            reasons.append("budget_friendly")
    overlap = len(query_tokens & recipe_tokens)
    if overlap:
        score += overlap * 0.18
    if "breakfast_gap" in habit_flags and meal_type == "breakfast":
        score += 0.9
        reasons.append("habit_breakfast_gap")
    if "late_night_snacking" in habit_flags and recipe["protein_g"] >= 28:
        score += 0.6
        reasons.append("habit_satiety_support")
    visual_score, visual_reasons = _vision_alignment(recipe, payload)
    if visual_score:
        score += visual_score
        reasons.extend(visual_reasons)
    return round(score, 3), reasons


@lru_cache(maxsize=1)
def _load_memory_rag_runtime() -> dict | None:
    repo_root = Path(__file__).resolve().parents[3]
    rag_root = repo_root / "memory_rag_lab"
    rag_src = rag_root / "src"
    if not rag_src.exists():
        return None
    if str(rag_src) not in sys.path:
        sys.path.insert(0, str(rag_src))
    try:
        from memory_rag_lab.chunking import chunk_document
        from memory_rag_lab.documents import load_documents
        from memory_rag_lab.evals import ensure_sample_docs
        from memory_rag_lab.index import SparseIndex
        from memory_rag_lab.memory import SessionMemoryStore
        from memory_rag_lab.rerank import rerank_hits
        from memory_rag_lab.retrieval import hybrid_retrieve
    except Exception:
        return None

    docs_root = ensure_sample_docs(rag_root)
    documents = load_documents(docs_root)
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy="hybrid"))
    return {
        "index": SparseIndex(chunks),
        "memory_store": SessionMemoryStore(rag_root / "data" / "memories"),
        "hybrid_retrieve": hybrid_retrieve,
        "rerank_hits": rerank_hits,
    }


def _rag_chunk_to_candidate(chunk, hit) -> dict:
    text = chunk.text
    meal_type = (_line_value(text, "Meal type:") or "meal").lower()
    components = {
        "sparse": round(float(getattr(hit, "sparse_score", 0.0)), 3),
        "meal_intent": round(float(getattr(hit, "meal_type_match", 0.0)), 3),
        "habit": round(float(getattr(hit, "habit_alignment", 0.0)), 3),
        "vision": round(float(getattr(hit, "vision_alignment", 0.0) + getattr(hit, "visual_memory_alignment", 0.0)), 3),
    }
    fit_reasons = [f"{name}:{value}" for name, value in components.items() if abs(value) > 0.0] or ["retrieval_score"]
    return {
        "name": _line_value(text, "Recipe:") or _line_value(text, "Name:") or chunk.chunk_id,
        "meal_type": meal_type,
        "kcal": _first_number(_line_value(text, "Calories:"), 520),
        "protein_g": _first_number(_line_value(text, "Protein:"), 30),
        "prep_minutes": _first_number(_line_value(text, "Prep time:"), 20),
        "ingredients": _as_ingredient_list(_line_value(text, "Ingredients:")),
        "tags": [meal_type, "??????"],
        "fit_notes": _line_value(text, "Benefits:") or _line_value(text, "Use when:") or "???????????",
        "fit_reasons": fit_reasons,
        "score": round(float(getattr(hit, "score", 0.0)), 3),
        "candidate_source": "memory_rag_lab",
    }


def _pick_candidates_from_memory_rag(payload: dict, limit: int = 6, meal_type: str | None = None) -> list[dict]:
    runtime = _load_memory_rag_runtime()
    if not runtime:
        return []

    query = " ".join(
        item
        for item in [
            str(payload.get("task", "")).strip(),
            str(payload.get("input_text", "")).strip(),
            " ".join(str(token).strip() for token in payload.get("keywords", [])),
        ]
        if item
    ).strip()
    if not query:
        return []

    image_analyses = _vision_analyses(payload)
    image_analysis = image_analyses[0] if image_analyses else None

    try:
        hits, _ = runtime["hybrid_retrieve"](
            query,
            runtime["index"],
            top_k=max(8, limit * 2),
            memory_store=runtime["memory_store"],
            session_id=str(payload.get("session_id", "demo")),
            image_analysis=image_analysis,
        )
        hits = runtime["rerank_hits"](query, hits, runtime["index"], top_k=max(8, limit * 2))
    except Exception:
        return []

    profile = _profile(payload)
    allergies = _normalize_terms(profile.get("allergies", []))
    dislikes = _normalize_terms(profile.get("dislikes", []))
    candidates: list[dict] = []
    for hit in hits:
        chunk = runtime["index"].chunks.get(hit.chunk_id)
        if chunk is None:
            continue
        candidate = _rag_chunk_to_candidate(chunk, hit)
        if meal_type and candidate.get("meal_type") != meal_type:
            continue
        if not _recipe_is_safe(candidate, allergies, dislikes):
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def _pick_candidates(payload: dict, limit: int = 6, meal_type: str | None = None) -> list[dict]:
    rag_candidates = _pick_candidates_from_memory_rag(payload, limit=limit, meal_type=meal_type)
    scored: list[tuple[float, dict]] = []
    safe_fallback: list[dict] = []
    profile = _profile(payload)
    allergies = _normalize_terms(profile.get("allergies", []))
    dislikes = _normalize_terms(profile.get("dislikes", []))
    for recipe in RECIPE_CATALOG:
        if meal_type and recipe["meal_type"] != meal_type:
            continue
        if not _recipe_is_safe(recipe, allergies, dislikes):
            continue
        safe_fallback.append({**recipe, "score": 0.1, "fit_reasons": ["safe_default"], "candidate_source": "catalog"})
        score, reasons = _score_recipe(recipe, payload)
        if score <= 0:
            continue
        scored.append((score, {**recipe, "score": score, "fit_reasons": reasons or ["keyword_overlap"], "candidate_source": "catalog"}))
    scored.sort(key=lambda item: (-item[0], item[1]["prep_minutes"], -item[1]["protein_g"]))
    fallback = [item for _, item in scored] if scored else safe_fallback

    merged: list[dict] = []
    seen_names: set[str] = set()
    for recipe in rag_candidates + fallback:
        name = str(recipe.get("name", "")).strip().lower()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        merged.append(recipe)
        if len(merged) >= limit:
            break
    return merged[:limit]


def _profile_summary_lines(payload: dict) -> list[str]:
    profile = _profile(payload)
    goal = str(payload.get("goal", "")).strip() or str(profile.get("goal", "")).strip()
    allergies = _normalize_terms(profile.get("allergies", []))
    dislikes = _normalize_terms(profile.get("dislikes", []))
    preferences = _normalize_terms(profile.get("preferences", []))
    habit_flags = _detect_habit_flags(_meal_logs(payload), str(payload.get("input_text", "")))
    lines = [
        f"???{goal or '????'}",
        f"????{', '.join(allergies) if allergies else '?'}",
        f"?????{', '.join(dislikes) if dislikes else '?'}",
        f"???{', '.join(preferences) if preferences else '?'}",
    ]
    for analysis in _vision_analyses(payload)[:2]:
        lines.append("?????" + analysis_to_text(analysis))
    if habit_flags:
        lines.append("???????" + "?".join(habit_flags))
    return lines


def _format_candidate(recipe: dict) -> str:
    source = str(recipe.get("candidate_source", "catalog")).replace("_", " ")
    source_label = {"memory rag lab": "?????", "catalog": "?????", "catalog fallback": "???????"}.get(source, source)
    fit_labels = "?".join(_fit_reason_label(item) for item in recipe["fit_reasons"])
    return (
        f"- {_meal_label(recipe['meal_type'])}?{recipe['name']} "
        f"?{recipe['kcal']} kcal?{recipe['protein_g']}g ????{recipe['prep_minutes']} ??? | "
        f"?????{fit_labels} | ???{source_label}?"
    )


def _plan_header(payload: dict) -> str:
    goal = str(payload.get("goal", "")).strip() or str(_profile(payload).get("goal", "")).strip() or "????"
    days = max(1, int(payload.get("planning_days", 3) or 3))
    kcal_target = int(payload.get("daily_kcal_target", 2000) or 2000)
    return f"?????????{goal}??? {days} ??????? {kcal_target} kcal/??"


def _requested_meal_sequence(payload: dict) -> list[str]:
    combined = " ".join(
        [
            str(payload.get("task", "")),
            str(payload.get("input_text", "")),
        ]
    ).lower()
    avoided = {
        meal
        for meal in MEAL_SEQUENCE
        if re.search(rf"(avoid|without|skip|exclude)[^.\n]{{0,40}}\b{meal}s?\b", combined)
    }
    intents = [
        meal
        for meal in MEAL_SEQUENCE
        if re.search(rf"\b{meal}s?\b", combined) and meal not in avoided
    ]
    if not intents:
        return ["breakfast", "lunch", "dinner"]

    remaining = [meal for meal in ("breakfast", "lunch", "dinner") if meal not in intents]
    strict_focus = (
        len(intents) == 1
        and any(
            phrase in combined
            for phrase in (
                "focus on",
                "focused",
                "focus only",
                "only",
                "avoid breakfast",
                "avoid lunch",
                "avoid snack",
                "avoid leading with",
            )
        )
    )
    if strict_focus:
        return intents
    return intents + remaining


def _build_day_plan(day_index: int, payload: dict) -> list[str]:
    sections: list[str] = [f"? {day_index} ?"]
    plan_state = payload.get("_plan_state", {})
    usage = plan_state.setdefault("usage", {})
    last_by_meal = plan_state.setdefault("last_by_meal", {})

    def choose_recipe(candidates: list[dict], meal_type: str) -> dict | None:
        if not candidates:
            return None
        best_recipe: dict | None = None
        best_score = -9999.0
        last_name = last_by_meal.get(meal_type)
        for idx, recipe in enumerate(candidates):
            name = recipe["name"]
            usage_penalty = 0.45 * int(usage.get(name, 0))
            repeat_penalty = 0.8 if name == last_name else 0.0
            rotation_bonus = 0.03 * ((day_index + idx) % 3)
            adjusted = float(recipe.get("score", 0.0)) - usage_penalty - repeat_penalty + rotation_bonus
            if adjusted > best_score:
                best_recipe = recipe
                best_score = adjusted
        if not best_recipe:
            return None
        chosen_name = best_recipe["name"]
        usage[chosen_name] = int(usage.get(chosen_name, 0)) + 1
        last_by_meal[meal_type] = chosen_name
        return best_recipe

    for meal_type in _requested_meal_sequence(payload):
        candidate_payload = {**payload, "keywords": list(dict.fromkeys([*payload.get("keywords", []), meal_type]))}
        candidates = _pick_candidates(candidate_payload, limit=4, meal_type=meal_type)
        recipe = choose_recipe(candidates, meal_type)
        if not recipe:
            continue
        sections.append(
            f"- {_meal_label(meal_type)}?{recipe['name']}?{recipe['kcal']} kcal?{recipe['protein_g']}g ????{recipe['prep_minutes']} ???"
            f" | ???{', '.join(recipe.get('ingredients', [])[:6])}"
        )
    snack_candidates = _pick_candidates(payload, limit=3, meal_type="snack")
    snack = choose_recipe(snack_candidates, "snack")
    if snack:
        sections.append(
            f"- ???{snack['name']}?{snack['kcal']} kcal?{snack['protein_g']}g ????"
            f" | ???{', '.join(snack.get('ingredients', [])[:6])}"
        )
    return sections


def _bucket_for_ingredient(ingredient: str) -> str:
    lower = ingredient.lower()
    if any(token in lower for token in ("chicken", "turkey", "salmon", "tofu", "yogurt", "egg", "cottage cheese", "beans")):
        return "???"
    if any(token in lower for token in ("spinach", "tomato", "berries", "broccoli", "carrot", "asparagus", "potatoes", "cucumber", "lettuce", "pepper", "apple", "lemon")):
        return "????"
    return "????"


def _shopping_groups_from_text(draft_plan: str) -> dict[str, list[str]]:
    groups = {"???": [], "????": [], "????": []}
    selected_names = []
    for line in draft_plan.splitlines():
        if "| ???" in line:
            ingredient_blob = line.split("| ???", 1)[1].strip()
            for ingredient in [item.strip() for item in ingredient_blob.split(",") if item.strip()]:
                bucket = _bucket_for_ingredient(ingredient)
                if ingredient not in groups[bucket]:
                    groups[bucket].append(ingredient)
        if ":" not in line:
            continue
        if not any(prefix in line for prefix in ("??", "??", "??", "??")):
            continue
        selected_names.append(line.split(":", 1)[1].split("(")[0].strip())
    selected = [recipe for recipe in RECIPE_CATALOG if recipe["name"] in selected_names]
    for recipe in selected:
        for ingredient in recipe["ingredients"]:
            bucket = _bucket_for_ingredient(ingredient)
            if ingredient not in groups[bucket]:
                groups[bucket].append(ingredient)
    return groups


def extract_keywords(payload: dict) -> dict:
    text = payload.get("input_text", "")
    limit = int(payload.get("max_keywords", 8))
    scores: dict[str, int] = {}
    for token in _tokens(text):
        if token in STOPWORDS or len(token) < 3:
            continue
        scores[token] = scores.get(token, 0) + 1
    keywords = [token for token, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]
    return {"keywords": keywords}


def analyze_profile(payload: dict) -> dict:
    lines = _profile_summary_lines(payload)
    return {
        "profile_summary": {
            "goal": str(payload.get("goal", "")).strip() or str(_profile(payload).get("goal", "")).strip(),
            "allergies": _normalize_terms(_profile(payload).get("allergies", [])),
            "dislikes": _normalize_terms(_profile(payload).get("dislikes", [])),
            "preferences": _normalize_terms(_profile(payload).get("preferences", [])),
            "habit_flags": _detect_habit_flags(_meal_logs(payload), str(payload.get("input_text", ""))),
        },
        "answer": "Profile summary:\n" + "\n".join(f"- {line}" for line in lines),
    }


def analyze_meal_image_tool(payload: dict) -> dict:
    image_paths = payload.get("image_paths", [])
    provider = str(payload.get("vision_provider", "")).strip() or None
    analyses = [analyze_meal_image(image_path, provider=provider) for image_path in image_paths]
    if not analyses:
        return {"vision_analysis": [], "answer": "???????????"}
    lines = ["???????"] + [f"- {analysis_to_text(item)}" for item in analyses]
    return {"vision_analysis": analyses, "answer": "\n".join(lines)}


def query_recipe_candidates(payload: dict) -> dict:
    candidates = _pick_candidates(payload, limit=6)
    source = "?????" if any(item.get("candidate_source") == "memory_rag_lab" for item in candidates) else "???????"
    answer_lines = [f"?????{source}??"] + [_format_candidate(recipe) for recipe in candidates]
    return {"candidates": candidates, "candidate_source": source, "answer": "\n".join(answer_lines)}


def draft_meal_plan(payload: dict) -> dict:
    days = max(1, int(payload.get("planning_days", 3) or 3))
    lines = [_plan_header(payload), "????????"] + [f"- {line}" for line in _profile_summary_lines(payload)]
    lines.append("?????")
    lines.append("- ???????????????????????????")
    if _vision_analyses(payload):
        lines.append("- ?????????????????????????")
        lines.append("?????")
        for analysis in _vision_analyses(payload)[:2]:
            lines.append(f"- {analysis_to_text(analysis)}")
    habit_flags = _detect_habit_flags(_meal_logs(payload), str(payload.get("input_text", "")))
    if habit_flags:
        lines.append("?????")
        if "breakfast_gap" in habit_flags:
            lines.append("- ????????????????????")
        if "late_night_snacking" in habit_flags:
            lines.append("- ??????????????????????")
        if "high_sugar_drinks" in habit_flags:
            lines.append("- ????????????????????")
    query_tokens = _token_set(payload.get("task", ""), payload.get("input_text", ""), " ".join(payload.get("keywords", [])))
    if any(token in query_tokens for token in ("budget", "affordable", "cheap", "low")):
        lines.append("???????????????????????????????")
    if days > 3:
        lines.append("???????????????????????????")
    plan_state = {"usage": {}, "last_by_meal": {}}
    for day_index in range(1, days + 1):
        lines.extend(_build_day_plan(day_index, {**payload, "_plan_state": plan_state}))
        if day_index % 3 == 0 and day_index < days:
            lines.append("- ??????????????????????????????")
    return {"answer": "\n".join(lines)}


def generate_shopping_list(payload: dict) -> dict:
    draft_plan = str(payload.get("draft_plan", "")).strip()
    if not draft_plan:
        draft_plan = draft_meal_plan(payload)["answer"]
    groups = _shopping_groups_from_text(draft_plan)
    lines = [draft_plan, "?????"]
    for group_name, items in groups.items():
        if items:
            lines.append(f"- {group_name}: {', '.join(items)}")
    return {"answer": "\n".join(lines)}


def compose_answer(payload: dict) -> dict:
    return draft_meal_plan(payload)


def revise_answer(payload: dict) -> dict:
    current_output = payload.get("current_output", "")
    missing_terms = payload.get("missing_terms", [])
    if not missing_terms:
        return {"answer": current_output}
    additions: list[str] = []
    for term in missing_terms:
        normalized = term.lower()
        if normalized in {"breakfast", "lunch", "dinner", "snack"}:
            additions.append(f"- ???{_meal_label(normalized)}?????")
        elif normalized in {"budget", "affordable"}:
            additions.append("- ????????????????????????????????")
        elif normalized in {"kcal", "calorie"}:
            additions.append("- ?????????????????????")
        elif normalized == "protein":
            additions.append("- ???????????????????????")
        elif normalized == "habit":
            additions.append("- ?????????????????????????????")
        elif normalized == "plan":
            additions.append("- ?????????????????????????")
        elif normalized in {"visual", "image", "detected"}:
            additions.append("- ??????????????????????????????")
        else:
            additions.append(f"- ??????{term}?")
    revised = current_output.rstrip()
    if additions:
        revised = revised + "\nCoverage check:\n" + "\n".join(additions)
    return {"answer": revised}


def register_default_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    registry = registry or ToolRegistry()
    registry.register(
        ToolSpec(
            name="extract_keywords",
            description="Extract salient keywords from text.",
            input_schema={"input_text": "str", "max_keywords": "int"},
            output_schema={"keywords": "list[str]"},
            version="1.2.0",
        ),
        extract_keywords,
    )
    registry.register(
        ToolSpec(
            name="analyze_profile",
            description="Summarize dietary profile, constraints, meal-log habits, and visual cues.",
            input_schema={"task": "str", "input_text": "str", "user_profile": "dict", "meal_logs": "list[str]", "vision_analysis": "list[dict]"},
            output_schema={"profile_summary": "dict", "answer": "str"},
            version="1.2.0",
        ),
        analyze_profile,
    )
    registry.register(
        ToolSpec(
            name="analyze_meal_image",
            description="Analyze uploaded meal images into structured dish, ingredient, and nutrition cues.",
            input_schema={"image_paths": "list[str]", "vision_provider": "str"},
            output_schema={"vision_analysis": "list[dict]", "answer": "str"},
            version="1.2.0",
        ),
        analyze_meal_image_tool,
    )
    registry.register(
        ToolSpec(
            name="query_recipe_candidates",
            description="Retrieve safe recipe candidates for the current meal-planning request.",
            input_schema={"task": "str", "input_text": "str", "keywords": "list[str]", "user_profile": "dict", "vision_analysis": "list[dict]"},
            output_schema={"candidates": "list[dict]", "answer": "str"},
            version="1.2.0",
        ),
        query_recipe_candidates,
    )
    registry.register(
        ToolSpec(
            name="draft_meal_plan",
            description="Draft a concise multi-day meal plan from recipe candidates, user profile, and image cues.",
            input_schema={"task": "str", "input_text": "str", "keywords": "list[str]", "planning_days": "int", "user_profile": "dict", "vision_analysis": "list[dict]"},
            output_schema={"answer": "str"},
            version="1.2.0",
        ),
        draft_meal_plan,
    )
    registry.register(
        ToolSpec(
            name="generate_shopping_list",
            description="Append a grouped shopping list to the draft meal plan.",
            input_schema={"draft_plan": "str", "planning_days": "int", "user_profile": "dict"},
            output_schema={"answer": "str"},
            version="1.2.0",
        ),
        generate_shopping_list,
    )
    registry.register(
        ToolSpec(
            name="compose_answer",
            description="Backwards-compatible wrapper that drafts a meal plan.",
            input_schema={"task": "str", "input_text": "str", "keywords": "list[str]"},
            output_schema={"answer": "str"},
            version="1.2.0",
        ),
        compose_answer,
    )
    registry.register(
        ToolSpec(
            name="revise_answer",
            description="Revise a meal-planning answer to satisfy reviewer-required terms.",
            input_schema={"current_output": "str", "missing_terms": "list[str]"},
            output_schema={"answer": "str"},
            version="1.2.0",
        ),
        revise_answer,
    )
    return registry
