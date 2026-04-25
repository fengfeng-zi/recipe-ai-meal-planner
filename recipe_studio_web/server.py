from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import uuid
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MEMORY_SRC = ROOT / "memory_rag_lab" / "src"
HARNESS_SRC = ROOT / "agent_runtime_harness" / "src"
WEB_ROOT = ROOT / "recipe_studio_web"
STATIC_ROOT = WEB_ROOT / "static"
TEMPLATE_ROOT = WEB_ROOT / "templates"
UPLOAD_ROOT = WEB_ROOT / "uploads"
MAX_BODY_BYTES = 8 * 1024 * 1024
DAY_HEADER_RE = re.compile("^(?:Day\\s+(\\d+)|\\u7b2c\\s*(\\d+)\\s*\\u5929)\\s*$", re.IGNORECASE)
MEAL_LINE_RE = re.compile(r"^-\s*([^:?]+)[:?]\s*(.+)$")
MEAL_LABELS = {
    "breakfast": "\u65e9\u9910",
    "lunch": "\u5348\u9910",
    "dinner": "\u665a\u9910",
    "snack": "\u52a0\u9910",
    "meal": "\u9910\u98df",
}

for src in (MEMORY_SRC, HARNESS_SRC):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from memory_rag_lab.llm import chat_complete_with_model, llm_enabled, llm_model, parse_json_text  # noqa: E402
from memory_rag_lab.memory import SessionMemoryStore  # noqa: E402
from memory_rag_lab.service import RecipeQueryService  # noqa: E402
from memory_rag_lab.vision import analysis_to_text, analyze_meal_image  # noqa: E402
from agent_runtime_harness.tools import _pick_candidates  # noqa: E402

UPLOAD_ALIASES = {"/api/upload-image", "/api/upload", "/upload", "/upload-image"}
ANALYZE_ALIASES = {"/api/analyze-image", "/analyze-image", "/api/analyze_image"}
QUERY_ALIASES = {"/api/query", "/query", "/api/ask"}
PLAN_ALIASES = {"/api/meal-plan", "/api/plan", "/plan"}


def _project_roots() -> dict[str, Path]:
    return {
        "memory": ROOT / "memory_rag_lab",
        "harness": ROOT / "agent_runtime_harness",
    }


def _memory_store() -> SessionMemoryStore:
    return SessionMemoryStore(_project_roots()["memory"] / "data" / "memories")


def _recipe_query_service() -> RecipeQueryService:
    return RecipeQueryService(project_root_path=_project_roots()["memory"])


def _llm_provider_label(model_name: str | None = None) -> str:
    return f"openai-compatible:{model_name or llm_model()}"


def _llm_query_answer(payload: dict[str, Any], result: dict[str, Any]) -> tuple[str | None, str | None]:
    if not llm_enabled():
        return None, None

    citations = []
    for item in result.get("citations", [])[:4]:
        citations.append(
            {
                "title": item.get("title", ""),
                "score": item.get("score", 0),
                "applied_signals": item.get("applied_signals", []),
            }
        )

    prompt_payload = {
        "query": result.get("query", ""),
        "goal": str(payload.get("goal", "")).strip(),
        "allergies": _parse_csv_or_list(payload.get("allergies", [])),
        "dislikes": _parse_csv_or_list(payload.get("dislikes", [])),
        "preferences": _parse_csv_or_list(payload.get("preferences", [])),
        "image_analysis": result.get("image_analysis") or {},
        "candidates": result.get("candidates", [])[:3],
        "citations": citations,
    }
    raw, used_model = chat_complete_with_model(
        [
            {
                "role": "system",
                "content": (
                    "You are a grounded meal-planning assistant. "
                    "Write natural simplified Chinese only. "
                    "Use only the provided evidence. "
                    "Be concrete, helpful, and briefly explain why the top options fit. "
                    "Return JSON with key answer."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False),
            },
        ],
        temperature=0.35,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    if not raw:
        return None, None
    parsed = parse_json_text(raw)
    if not parsed:
        return None, None
    answer = str(parsed.get("answer", "")).strip()
    return (answer or None), used_model


def _normalize_llm_plan_days(raw_days: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_days, list):
        return normalized
    for idx, day in enumerate(raw_days, 1):
        if not isinstance(day, dict):
            continue
        meals: list[dict[str, Any]] = []
        for meal in day.get("meals", []):
            if not isinstance(meal, dict):
                continue
            ingredients = meal.get("ingredients", [])
            meals.append(
                {
                    "meal": str(meal.get("meal", "")).strip() or f"\u7b2c {idx} \u9910",
                    "description": str(meal.get("description", "")).strip(),
                    "kcal": str(meal.get("kcal", "")).strip(),
                    "protein": str(meal.get("protein", "")).strip(),
                    "time": str(meal.get("time", "")).strip(),
                    "ingredients": [str(item).strip() for item in ingredients if str(item).strip()] if isinstance(ingredients, list) else [],
                }
            )
        normalized.append(
            {
                "day": int(day.get("day", idx) or idx),
                "title": str(day.get("title", "")).strip() or f"\u7b2c {idx} \u5929",
                "notes": [str(item).strip() for item in day.get("notes", []) if str(item).strip()] if isinstance(day.get("notes", []), list) else [],
                "meals": meals,
            }
        )
    return normalized


def _llm_plan_override(
    payload: dict[str, Any],
    *,
    plan_days: list[dict[str, Any]],
    shopping_list: list[str],
    image_analysis: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not llm_enabled():
        return None, None

    prompt_payload = {
        "goal": str(payload.get("goal", "")).strip(),
        "query": str(payload.get("query", "")).strip(),
        "task": str(payload.get("task", "")).strip(),
        "input_text": str(payload.get("input_text", "")).strip(),
        "planning_days": int(payload.get("planning_days", 3) or 3),
        "allergies": _parse_csv_or_list(payload.get("allergies", [])),
        "dislikes": _parse_csv_or_list(payload.get("dislikes", [])),
        "preferences": _parse_csv_or_list(payload.get("preferences", [])),
        "image_analysis": image_analysis or {},
        "draft_plan_days": plan_days,
        "shopping_list": shopping_list,
    }
    raw, used_model = chat_complete_with_model(
        [
            {
                "role": "system",
                "content": (
                    "You are a meal planner for a Chinese UI. "
                    "Rewrite the draft plan into natural simplified Chinese. "
                    "Keep the original calories, protein, and time constraints intact. "
                    "Return JSON only with keys final_output, plan_days, shopping_list. "
                    "Each plan_days item must contain day, title, notes, meals. "
                    "Each meal must contain meal, description, kcal, protein, time, ingredients. "
                    "Descriptions should be concise and human, not robotic."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False),
            },
        ],
        temperature=0.45,
        max_tokens=2200,
        response_format={"type": "json_object"},
    )
    if not raw:
        return None, None
    parsed = parse_json_text(raw)
    if not parsed:
        return None, None

    final_output = str(parsed.get("final_output", "")).strip()
    parsed_days = _normalize_llm_plan_days(parsed.get("plan_days", []))
    parsed_shopping = parsed.get("shopping_list", [])
    shopping = [str(item).strip() for item in parsed_shopping if str(item).strip()] if isinstance(parsed_shopping, list) else []
    if not final_output or not parsed_days:
        return None, None
    return {
        "final_output": final_output,
        "plan_days": parsed_days,
        "shopping_list": shopping or shopping_list,
    }, used_model


def _json_error(message: str, status: int = 400) -> tuple[int, dict[str, Any]]:
    return status, {"ok": False, "error": message}


def _parse_csv_or_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _meal_label(value: str) -> str:
    return MEAL_LABELS.get(str(value).strip().lower(), str(value).strip() or "\u9910\u98df")


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "upload")
    cleaned = cleaned.strip("._") or "upload"
    return cleaned[:120]


def _save_upload_bytes(filename: str, payload: bytes) -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    original = _safe_filename(filename or "upload.bin")
    suffix = Path(original).suffix or ".bin"
    stem = Path(original).stem or "upload"
    target = UPLOAD_ROOT / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    with target.open("wb") as fh:
        fh.write(payload)
    return target


def _coerce_form_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _parse_multipart_form(body: bytes, content_type_header: str) -> dict[str, Any] | None:
    message = BytesParser(policy=email_policy).parsebytes(
        (
            f"Content-Type: {content_type_header}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("utf-8")
        + body
    )
    if not message.is_multipart():
        return None

    payload: dict[str, Any] = {}
    uploads: list[str] = []
    seen_files: set[tuple[str, str, bytes]] = set()
    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            continue

        filename = part.get_filename()
        raw = part.get_payload(decode=True) or b""
        if filename:
            signature = (field_name, filename, raw)
            if signature in seen_files:
                continue
            seen_files.add(signature)
            saved = _save_upload_bytes(filename, raw)
            uploads.append(str(saved))
            continue

        charset = part.get_content_charset() or "utf-8"
        value = _coerce_form_value(raw.decode(charset, errors="replace"))
        existing = payload.get(field_name)
        if existing is None:
            payload[field_name] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            payload[field_name] = [existing, value]

    if uploads:
        payload["uploaded_paths"] = uploads
        payload.setdefault("image_path", uploads[0])
        payload.setdefault("image_paths", uploads)
    return payload


def _load_request_payload(handler: BaseHTTPRequestHandler) -> tuple[dict[str, Any] | None, bool]:
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    if content_length > MAX_BODY_BYTES:
        return None, False

    content_type = handler.headers.get_content_type()
    if content_type == "multipart/form-data":
        if content_length <= 0:
            return None, True
        raw = handler.rfile.read(content_length)
        return _parse_multipart_form(raw, handler.headers.get("Content-Type", "")), True

    if content_type in {"application/json", "text/json"}:
        if content_length <= 0:
            return {}, False
        raw = handler.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, False
        return payload if isinstance(payload, dict) else None, False

    if content_length <= 0:
        return {}, False
    return None, False


def _apply_profile_to_session(session_id: str, payload: dict[str, Any]) -> None:
    raw_profile = payload.get("user_profile", {})
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    store = _memory_store()

    allergies = _parse_csv_or_list(payload.get("allergies", profile.get("allergies", [])))
    dislikes = _parse_csv_or_list(payload.get("dislikes", profile.get("dislikes", [])))
    preferences = _parse_csv_or_list(payload.get("preferences", profile.get("preferences", [])))
    goal = str(payload.get("goal", profile.get("goal", ""))).strip()

    for item in allergies:
        store.save_preference(session_id, memory_text=f"Avoid {item}.", memory_type="allergy", tags=[item])
    for item in dislikes:
        store.save_preference(session_id, memory_text=f"Dislike {item}.", memory_type="dislike", tags=[item])
    for item in preferences:
        store.save_preference(session_id, memory_text=f"Prefer {item} meals.", memory_type="preference", tags=[item])
    if goal:
        store.save_preference(session_id, memory_text=f"Goal: {goal}.", memory_type="goal", tags=[goal])


def _extract_parenthetical_metrics(text: str) -> dict[str, str]:
    match = re.search(r"[?(]([^?)]*)[?)]", text)
    if not match:
        return {}
    metrics: dict[str, str] = {}
    for token in [part.strip() for part in match.group(1).split(",") if part.strip()]:
        lower = token.lower()
        if "kcal" in lower:
            metrics["kcal"] = token
        elif "protein" in lower or "\u86cb\u767d\u8d28" in token:
            metrics["protein"] = token
        elif "min" in lower or "\u5206\u949f" in token:
            metrics["time"] = token
    return metrics


def _extract_plan_days(plan_text: str) -> list[dict[str, Any]]:
    days: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        day_match = DAY_HEADER_RE.match(line)
        if day_match:
            if current:
                days.append(current)
            day_number = int(day_match.group(1) or day_match.group(2))
            current = {"day": day_number, "title": f"\u7b2c {day_number} \u5929", "meals": [], "notes": []}
            continue
        if line.lower().startswith("shopping list") or line.startswith("\u8d2d\u7269\u6e05\u5355") or line.startswith("\u91c7\u8d2d\u6e05\u5355"):
            break
        if current is None:
            continue
        meal_match = MEAL_LINE_RE.match(line)
        if meal_match:
            meal_name = meal_match.group(1).strip()
            description = meal_match.group(2).strip()
            meal = {"meal": meal_name, "description": description, **_extract_parenthetical_metrics(description)}
            ingredients_match = re.search(r"\|\s*(?:ingredients|\u98df\u6750)[:\uff1a]\s*(.+)$", description, re.IGNORECASE)
            if ingredients_match:
                meal["ingredients"] = [item.strip() for item in ingredients_match.group(1).split(",") if item.strip()]
            current["meals"].append(meal)
        else:
            current["notes"].append(line)
    if current:
        days.append(current)
    return days


def _extract_shopping_list(plan_text: str) -> list[str]:
    lines = [line.strip() for line in plan_text.splitlines()]
    shopping: list[str] = []
    collecting = False
    for line in lines:
        if not line:
            continue
        if line.lower().startswith("shopping list") or line.startswith("\u8d2d\u7269\u6e05\u5355") or line.startswith("\u91c7\u8d2d\u6e05\u5355"):
            collecting = True
            continue
        if collecting and line.startswith("-"):
            shopping.append(line.removeprefix("-").strip())
    return shopping


def _shopping_bucket(ingredient: str) -> str:
    lower = ingredient.lower()
    if any(token in lower for token in ("chicken", "turkey", "salmon", "tofu", "yogurt", "egg", "cottage cheese", "beans")):
        return "\u86cb\u767d\u8d28"
    if any(token in lower for token in ("spinach", "tomato", "berries", "broccoli", "carrot", "asparagus", "potatoes", "cucumber", "lettuce", "pepper", "apple", "lemon")):
        return "\u8536\u83dc\u6c34\u679c"
    return "\u53a8\u623f\u5e38\u5907"


def _build_local_plan(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[str]]:
    raw_profile = payload.get("user_profile", {}) if isinstance(payload.get("user_profile", {}), dict) else {}
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    planning_days = max(1, int(payload.get("planning_days", 3) or 3))
    kcal_target = int(payload.get("daily_kcal_target", 2000) or 2000)
    goal = str(payload.get("goal", profile.get("goal", ""))).strip() or "\u5747\u8861\u996e\u98df"
    allergies = _parse_csv_or_list(payload.get("allergies", profile.get("allergies", [])))
    dislikes = _parse_csv_or_list(payload.get("dislikes", profile.get("dislikes", [])))
    preferences = _parse_csv_or_list(payload.get("preferences", profile.get("preferences", [])))
    image_paths = _parse_csv_or_list(payload.get("image_paths", []))
    image_path = str(payload.get("image_path", "")).strip()
    if image_path and image_path not in image_paths:
        image_paths.append(image_path)
    image_analysis = payload.get("image_analysis") if isinstance(payload.get("image_analysis"), dict) else None
    if image_analysis is None and image_paths:
        try:
            image_analysis = analyze_meal_image(image_paths[0])
        except Exception:
            image_analysis = None

    base_payload = {
        "task": payload.get("task", "\u751f\u6210\u81b3\u98df\u8ba1\u5212"),
        "input_text": payload.get("input_text", payload.get("query", "")),
        "keywords": [],
        "user_profile": {
            "allergies": allergies,
            "dislikes": dislikes,
            "preferences": preferences,
            "goal": goal,
        },
        "meal_logs": _parse_csv_or_list(payload.get("meal_logs", [])),
        "vision_analysis": [image_analysis] if image_analysis else [],
        "session_id": str(payload.get("session_id", "web")),
        "goal": goal,
        "planning_days": planning_days,
        "daily_kcal_target": kcal_target,
    }

    usage: dict[str, int] = {}
    last_by_meal: dict[str, str] = {}
    meal_order = ["breakfast", "lunch", "dinner", "snack"]
    plan_days: list[dict[str, Any]] = []

    def choose_recipe(candidates: list[dict[str, Any]], meal_type: str, day_index: int) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = -10**9
        for idx, recipe in enumerate(candidates):
            name = str(recipe.get("name", ""))
            score = float(recipe.get("score", 0.0))
            score -= 0.45 * usage.get(name, 0)
            if last_by_meal.get(meal_type) == name:
                score -= 0.8
            score += 0.03 * ((day_index + idx) % 3)
            if score > best_score:
                best_score = score
                best = recipe
        if best is not None:
            name = str(best.get("name", ""))
            usage[name] = usage.get(name, 0) + 1
            last_by_meal[meal_type] = name
        return best

    for day_index in range(1, planning_days + 1):
        day = {"day": day_index, "title": f"\u7b2c {day_index} \u5929", "meals": [], "notes": []}
        for meal_type in meal_order:
            candidate_payload = {**base_payload, "keywords": [meal_type]}
            candidates = _pick_candidates(candidate_payload, limit=4 if meal_type != "snack" else 3, meal_type=meal_type)
            recipe = choose_recipe(candidates, meal_type, day_index)
            if not recipe:
                continue
            description = f"{recipe['name']}\uff08{recipe['kcal']} kcal\uff0c{recipe['protein_g']}g \u86cb\u767d\u8d28\uff0c{recipe['prep_minutes']} \u5206\u949f\uff09 | \u98df\u6750\uff1a{', '.join(recipe.get('ingredients', [])[:6])}"
            day["meals"].append({
                "meal": _meal_label(meal_type),
                "description": description,
                "kcal": f"{recipe['kcal']} kcal",
                "protein": f"{recipe['protein_g']}g \u86cb\u767d\u8d28",
                "time": f"{recipe['prep_minutes']} \u5206\u949f",
                "ingredients": recipe.get("ingredients", []),
            })
        if image_analysis and day_index == 1:
            day["notes"].append("\u56fe\u50cf\u53c2\u8003\uff1a" + analysis_to_text(image_analysis))
        plan_days.append(day)

    grouped: dict[str, list[str]] = {"\u86cb\u767d\u8d28": [], "\u8536\u83dc\u6c34\u679c": [], "\u53a8\u623f\u5e38\u5907": []}
    for day in plan_days:
        for meal in day["meals"]:
            for ingredient in meal.get("ingredients", []):
                bucket = _shopping_bucket(str(ingredient))
                if ingredient not in grouped[bucket]:
                    grouped[bucket].append(str(ingredient))

    shopping_lines = [f"{group}: {', '.join(items)}" for group, items in grouped.items() if items]
    lines = [
        f"\u81b3\u98df\u8ba1\u5212\u6982\u89c8\uff1a\u4e3a\u201c{goal}\u201d\u751f\u6210 {planning_days} \u5929\u8ba1\u5212\uff0c\u76ee\u6807\u7ea6 {kcal_target} kcal/\u5929\u3002",
        "\u7528\u6237\u7ea6\u675f\u4e0e\u504f\u597d\uff1a",
        f"- \u76ee\u6807\uff1a{goal}",
        f"- \u8fc7\u654f\u539f\uff1a{', '.join(allergies) if allergies else '\u65e0'}",
        f"- \u4e0d\u559c\u6b22\u5403\uff1a{', '.join(dislikes) if dislikes else '\u65e0'}",
        f"- \u504f\u597d\uff1a{', '.join(preferences) if preferences else '\u65e0'}",
        "\u89c4\u5212\u539f\u5219\uff1a",
        "- \u4f18\u5148\u9009\u62e9\u9ad8\u86cb\u767d\u3001\u51c6\u5907\u65f6\u95f4\u77ed\u3001\u5e76\u4e14\u907f\u5f00\u8fc7\u654f\u539f\u7684\u5019\u9009\u83dc\u8c31\u3002",
    ]
    if image_analysis:
        lines.append("- \u5df2\u7ed3\u5408\u56fe\u7247\u8bc6\u522b\u7ed3\u679c\u4f5c\u4e3a\u83dc\u5f0f\u53c2\u8003\u3002")
    for day in plan_days:
        lines.append(day["title"])
        for meal in day["meals"]:
            lines.append(f"- {meal['meal']}\uff1a{meal['description']}")
        for note in day["notes"]:
            lines.append(note)
    if shopping_lines:
        lines.append("\u8d2d\u7269\u6e05\u5355\uff1a")
        for item in shopping_lines:
            lines.append(f"- {item}")
    return "\n".join(lines), plan_days, shopping_lines


def _analyze_image(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    image_path = str(payload.get("image_path", "")).strip()
    if not image_path:
        return _json_error("`image_path` is required.", 400)

    provider = str(payload.get("vision_provider", "")).strip() or None
    analysis = analyze_meal_image(image_path, provider=provider)
    result: dict[str, Any] = {
        "ok": True,
        "image_path": image_path,
        "analysis": analysis,
        "analysis_text": analysis_to_text(analysis),
    }
    if bool(payload.get("remember", False)):
        session_id = str(payload.get("session_id", "demo")).strip() or "demo"
        result["saved_memory"] = _memory_store().save_visual_analysis(session_id, analysis)
    return 200, result


def _upload_image(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    image_path = str(payload.get("image_path", "")).strip()
    if not image_path:
        return _json_error("No uploaded image found in multipart form data.", 400)

    provider = str(payload.get("vision_provider", "")).strip() or None
    analysis = analyze_meal_image(image_path, provider=provider)
    response: dict[str, Any] = {
        "ok": True,
        "image_path": image_path,
        "uploaded_path": image_path,
        "uploaded_image_path": image_path,
        "analysis": analysis,
        "analysis_text": analysis_to_text(analysis),
    }
    if bool(payload.get("remember", False)):
        session_id = str(payload.get("session_id", "demo")).strip() or "demo"
        response["saved_memory"] = _memory_store().save_visual_analysis(session_id, analysis)
    return 200, response


def _run_query(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    query = str(payload.get("query", "")).strip()
    if not query:
        return _json_error("`query` is required.", 400)

    session_id = str(payload.get("session_id", "demo")).strip() or "demo"
    _apply_profile_to_session(session_id, payload)
    top_k = max(1, int(payload.get("top_k", 5) or 5))
    rerank_top_k = max(1, int(payload.get("rerank_top_k", 3) or 3))
    image_path = str(payload.get("image_path", "")).strip() or None
    provider = str(payload.get("vision_provider", "")).strip() or None
    image_analysis = payload.get("image_analysis")
    if not isinstance(image_analysis, dict):
        image_analysis = None

    result = _recipe_query_service().query(
        query=query,
        session_id=session_id,
        image_path=image_path,
        image_analysis=image_analysis,
        vision_provider=provider,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
    )
    answer_provider = "local-rag"
    llm_answer, used_model = _llm_query_answer(payload, result)
    if llm_answer:
        result["answer"] = llm_answer
        answer_provider = _llm_provider_label(used_model)
    return 200, {"ok": True, "answer_provider": answer_provider, **result}


def _run_meal_plan(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    task = str(payload.get("task", "")).strip()
    input_text = str(payload.get("input_text", "")).strip()
    if not task or not input_text:
        return _json_error("`task` and `input_text` are required.", 400)

    final_output, plan_days, shopping_list = _build_local_plan(payload)
    image_analysis = payload.get("image_analysis") if isinstance(payload.get("image_analysis"), dict) else None
    if image_analysis is None:
        image_path = str(payload.get("image_path", "")).strip()
        if image_path:
            try:
                image_analysis = analyze_meal_image(image_path, provider=str(payload.get("vision_provider", "")).strip() or None)
            except Exception:
                image_analysis = None

    plan_provider = "local-template"
    llm_plan, used_model = _llm_plan_override(
        payload,
        plan_days=plan_days,
        shopping_list=shopping_list,
        image_analysis=image_analysis,
    )
    if llm_plan:
        final_output = llm_plan["final_output"]
        plan_days = llm_plan["plan_days"]
        shopping_list = llm_plan["shopping_list"]
        plan_provider = _llm_provider_label(used_model)
    return 200, {
        "ok": True,
        "run_id": "local-plan",
        "status": "completed",
        "checkpoint_stage": "localized_plan_ready",
        "final_output": final_output,
        "plan_days": plan_days,
        "shopping_list": shopping_list,
        "plan_provider": plan_provider,
        "reviewer": None,
        "metadata": {
            "planning_days": int(payload.get("planning_days", 3) or 3),
            "daily_kcal_target": int(payload.get("daily_kcal_target", 2000) or 2000),
            "goal": str(payload.get("goal", "")).strip(),
        },
    }


def _static_response(path: str) -> tuple[int, bytes, str] | None:
    if path in {"/", "/index.html"}:
        target = TEMPLATE_ROOT / "index.html"
    elif path.startswith("/static/"):
        target = STATIC_ROOT / path.removeprefix("/static/")
    else:
        return None

    try:
        resolved = target.resolve(strict=True)
    except FileNotFoundError:
        return None

    base = TEMPLATE_ROOT.resolve() if path in {"/", "/index.html"} else STATIC_ROOT.resolve()
    if resolved != base and base not in resolved.parents:
        return None

    mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return 200, resolved.read_bytes(), mime_type


class RecipeStudioHandler(BaseHTTPRequestHandler):
    server_version = "RecipeStudioHTTP/0.5"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {"ok": True, "service": "recipe-studio-web-backend"})
            return

        static_response = _static_response(path)
        if static_response is not None:
            status, body, mime_type = static_response
            self.send_response(status)
            content_type = f"{mime_type}; charset=utf-8" if mime_type.startswith("text/") else mime_type
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload, is_multipart = _load_request_payload(self)
        if payload is None:
            self._send_json(400, {"ok": False, "error": "Invalid or unsupported request body."})
            return

        if path in UPLOAD_ALIASES:
            status, response = _upload_image(payload)
            self._send_json(status, response)
            return

        if path in ANALYZE_ALIASES:
            if is_multipart and payload.get("image_path"):
                status, response = _upload_image(payload)
            else:
                status, response = _analyze_image(payload)
            self._send_json(status, response)
            return

        if path in QUERY_ALIASES:
            status, response = _run_query(payload)
            self._send_json(status, response)
            return

        if path in PLAN_ALIASES:
            status, response = _run_meal_plan(payload)
            self._send_json(status, response)
            return

        self._send_json(404, {"ok": False, "error": "Not found"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Recipe Studio web backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("RECIPE_STUDIO_WEB_PORT") or os.getenv("PORT") or 8787))
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RecipeStudioHandler)
    print(f"recipe-studio backend listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
