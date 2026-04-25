from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request

DEFAULT_PROVIDER = "sidecar"
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")
MEAL_TYPE_ALIASES = {
    "brunch": "lunch",
    "supper": "dinner",
    "late-night": "snack",
    "late_night": "snack",
}
KNOWN_TOKENS = {
    "salmon", "chicken", "turkey", "tofu", "egg", "eggs", "rice", "quinoa", "oats", "berries", "broccoli",
    "carrot", "lettuce", "tomato", "cucumber", "potato", "potatoes", "banana", "yogurt", "wrap", "bowl",
    "salad", "tray", "stir", "fried", "grilled", "peanut", "bean", "beans", "avocado", "apple", "mushroom",
}
COOKING_METHOD_HINTS = (
    ("stir-fry", {"stir", "stirfry", "wok"}),
    ("grilled", {"grilled", "grill"}),
    ("baked", {"baked", "roasted", "tray", "sheet-pan", "sheetpan"}),
    ("fried", {"fried", "pan-fried", "panfried"}),
    ("boiled", {"boiled", "steamed", "poached"}),
)


def _tokenize_filename(path: Path) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", path.stem.lower())


def _normalize_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        items = [raw]
    else:
        items = []
    output: list[str] = []
    for item in items:
        clean = str(item).strip().lower()
        if clean and clean not in output:
            output.append(clean)
    return output


def _guess_meal_type(tokens: list[str]) -> str:
    for meal_type in MEAL_TYPES:
        if meal_type in tokens:
            return meal_type
    if any(token in tokens for token in ("oats", "yogurt", "egg", "wrap")):
        return "breakfast"
    if any(token in tokens for token in ("salad", "bowl", "quinoa")):
        return "lunch"
    if any(token in tokens for token in ("salmon", "tray", "stir", "grilled")):
        return "dinner"
    return "meal"


def _normalize_meal_type(raw: Any, tokens: list[str]) -> str:
    value = str(raw or "").strip().lower().replace(" ", "-")
    if value in MEAL_TYPES:
        return value
    if value in MEAL_TYPE_ALIASES:
        return MEAL_TYPE_ALIASES[value]
    guessed = _guess_meal_type(tokens)
    return guessed or "meal"


def _sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".vision.json")


def _cooking_method(tokens: list[str], parsed: dict[str, Any] | None = None) -> str:
    if parsed:
        raw = str(parsed.get("cooking_method", "")).strip().lower()
        if raw:
            return raw
    token_set = set(tokens)
    for method, hints in COOKING_METHOD_HINTS:
        if token_set & hints:
            return method
    return "unknown"


def _estimated_portions(parsed: dict[str, Any] | None = None) -> float:
    candidates: list[Any] = []
    if parsed:
        candidates = [
            parsed.get("estimated_portions"),
            parsed.get("servings"),
            parsed.get("portion_count"),
        ]
    for raw in candidates:
        if isinstance(raw, (int, float)):
            return max(1.0, min(float(raw), 8.0))
        value = str(raw or "").strip().lower()
        if not value:
            continue
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            try:
                return max(1.0, min(float(match.group(1)), 8.0))
            except ValueError:
                continue
    return 1.0


def _recipe_cues(tokens: list[str], parsed: dict[str, Any] | None = None) -> list[str]:
    cues = _normalize_list((parsed or {}).get("recipe_cues", []))
    token_set = set(tokens)
    if "quick" in token_set or "quick-cook" in token_set:
        cues.append("quick-cook")
    if "tray" in token_set or "sheet-pan" in token_set:
        cues.append("sheet-pan-friendly")
    if "bowl" in token_set:
        cues.append("bowl-style")
    if "salad" in token_set:
        cues.append("cold-assembly")
    if "stir" in token_set or "stirfry" in token_set:
        cues.append("wok-friendly")
    deduped: list[str] = []
    for cue in cues:
        if cue and cue not in deduped:
            deduped.append(cue)
    return deduped


def _build_analysis(parsed: dict[str, Any], image_path: Path, provider_name: str) -> dict[str, Any]:
    filename_tokens = _tokenize_filename(image_path)
    dish_name = str(parsed.get("dish_name", "")).strip() or image_path.stem.replace("_", " ").title()
    meal_type = _normalize_meal_type(parsed.get("meal_type", "meal"), filename_tokens + _tokenize_filename(Path(dish_name)))
    visible_ingredients = _normalize_list(parsed.get("visible_ingredients", []))
    cuisine_tags = _normalize_list(parsed.get("cuisine_tags", []))
    nutrition_signals = _normalize_list(parsed.get("nutrition_signals", []))
    caution_tags = _normalize_list(parsed.get("caution_tags", []))
    summary = str(parsed.get("summary", "")).strip()
    tokens_for_cues = filename_tokens + visible_ingredients + cuisine_tags + nutrition_signals
    return {
        "image_path": str(image_path),
        "provider": provider_name,
        "dish_name": dish_name,
        "meal_type": meal_type,
        "visible_ingredients": visible_ingredients,
        "cuisine_tags": cuisine_tags,
        "nutrition_signals": nutrition_signals,
        "caution_tags": caution_tags,
        "summary": summary,
        "confidence": _confidence_value(parsed.get("confidence", 0.75)),
        "cooking_method": _cooking_method(tokens_for_cues, parsed=parsed),
        "estimated_portions": _estimated_portions(parsed),
        "recipe_cues": _recipe_cues(tokens_for_cues, parsed=parsed),
    }


def _filename_fallback(image_path: Path) -> dict[str, Any]:
    tokens = _tokenize_filename(image_path)
    ingredients = [token for token in tokens if token in KNOWN_TOKENS and token not in {"bowl", "salad", "tray", "wrap"}]
    tags = [token for token in tokens if token in KNOWN_TOKENS and token not in ingredients]
    meal_type = _normalize_meal_type("", tokens)
    dish_name = " ".join(word.capitalize() for word in tokens[:4]) or image_path.stem
    cooking_method = _cooking_method(tokens)
    cues = _recipe_cues(tokens)
    summary_bits = [
        f"Observed dish: {dish_name}.",
        f"Likely meal type: {meal_type}." if meal_type else "",
        f"Visible ingredients: {', '.join(ingredients)}." if ingredients else "",
        f"Visual tags: {', '.join(tags)}." if tags else "",
        f"Likely cooking method: {cooking_method}." if cooking_method != "unknown" else "",
    ]
    return {
        "image_path": str(image_path),
        "provider": "filename-fallback",
        "dish_name": dish_name,
        "meal_type": meal_type,
        "visible_ingredients": ingredients,
        "cuisine_tags": tags,
        "nutrition_signals": [],
        "caution_tags": ["peanut"] if "peanut" in tokens else [],
        "summary": " ".join(bit for bit in summary_bits if bit),
        "confidence": 0.35,
        "cooking_method": cooking_method,
        "estimated_portions": 1.0,
        "recipe_cues": cues,
    }


def _load_sidecar(image_path: Path) -> dict[str, Any] | None:
    sidecar = _sidecar_path(image_path)
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    return _build_analysis(payload, image_path, "sidecar")


def _env_flag(*names: str) -> bool:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _image_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _analysis_instructions() -> str:
    return (
        "You analyze meal photos and return raw JSON only with keys "
        "dish_name, meal_type, visible_ingredients, cuisine_tags, nutrition_signals, "
        "caution_tags, summary, confidence, cooking_method, estimated_portions, recipe_cues."
    )


def _build_chat_completions_payload(model: str, image_path: Path, disable_storage: bool) -> bytes:
    body = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": _analysis_instructions(),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this food image for recipe retrieval and meal planning."},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            },
        ],
    }
    if disable_storage:
        body["store"] = False
    return json.dumps(body).encode("utf-8")


def _build_responses_payload(model: str, image_path: Path, disable_storage: bool) -> bytes:
    body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": _analysis_instructions()},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Analyze this food image for recipe retrieval and meal planning. Return JSON only.",
                    },
                    {"type": "input_image", "image_url": _image_data_url(image_path)},
                ],
            },
        ],
    }
    if disable_storage:
        body["store"] = False
    effort = (os.getenv("VISION_REASONING_EFFORT") or os.getenv("OPENAI_REASONING_EFFORT") or "").strip().lower()
    if effort:
        body["reasoning"] = {"effort": effort}
    return json.dumps(body).encode("utf-8")


def _read_json_response(request: urllib.request.Request) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _extract_text_payload(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    outputs = payload.get("output")
    if isinstance(outputs, list):
        parts: list[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            if parts:
                return "".join(parts)

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""

    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return str(content)


def _parse_json_text(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return _parse_json_text(value)
    return None


def _is_analysis_dict(payload: dict[str, Any]) -> bool:
    keys = {"dish_name", "meal_type", "visible_ingredients", "summary"}
    return bool(keys & set(payload.keys()))


def _extract_structured_analysis(payload: dict[str, Any]) -> dict[str, Any] | None:
    if _is_analysis_dict(payload):
        return payload

    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                for key in ("json", "value", "arguments", "text"):
                    parsed = _json_from_value(content.get(key))
                    if parsed and _is_analysis_dict(parsed):
                        return parsed

    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        message = None
    if isinstance(message, dict):
        parsed = _json_from_value(message.get("parsed"))
        if parsed and _is_analysis_dict(parsed):
            return parsed
        tool_calls = message.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for tool in tool_calls:
                if not isinstance(tool, dict):
                    continue
                fn_payload = tool.get("function", {})
                if not isinstance(fn_payload, dict):
                    continue
                parsed = _json_from_value(fn_payload.get("arguments"))
                if parsed and _is_analysis_dict(parsed):
                    return parsed

    parsed = _parse_json_text(_extract_text_payload(payload))
    if parsed and _is_analysis_dict(parsed):
        return parsed
    return None



def _confidence_value(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return max(0.0, min(float(raw), 1.0))
    value = str(raw).strip().lower()
    mapping = {
        "very low": 0.1,
        "low": 0.25,
        "medium": 0.5,
        "high": 0.8,
        "very high": 0.95,
    }
    if value in mapping:
        return mapping[value]
    try:
        return max(0.0, min(float(value), 1.0))
    except ValueError:
        return 0.75


def _request_openai_compatible(
    *,
    image_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    wire_api: str,
    disable_storage: bool,
) -> dict[str, Any] | None:
    endpoint = "/responses" if wire_api == "responses" else "/chat/completions"
    body = (
        _build_responses_payload(model, image_path, disable_storage)
        if wire_api == "responses"
        else _build_chat_completions_payload(model, image_path, disable_storage)
    )
    request = urllib.request.Request(
        url=f"{base_url}{endpoint}",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    payload = _read_json_response(request)
    if not payload:
        return None

    parsed = _extract_structured_analysis(payload)
    if not parsed:
        return None
    provider_name = "openai-compatible-responses" if wire_api == "responses" else "openai-compatible-chat-completions"
    return _build_analysis(parsed, image_path, provider_name)


def _call_openai_compatible(image_path: Path) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VISION_API_KEY")
    if not api_key:
        return None
    base_url = (os.getenv("VISION_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("VISION_MODEL") or os.getenv("OPENAI_VISION_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    wire_api = (os.getenv("VISION_WIRE_API") or os.getenv("OPENAI_WIRE_API") or "chat_completions").strip().lower()
    disable_storage = _env_flag("VISION_DISABLE_RESPONSE_STORAGE", "OPENAI_DISABLE_RESPONSE_STORAGE")
    preferred_wire_api = "responses" if wire_api in {"responses", "response"} else "chat_completions"
    live = _request_openai_compatible(
        image_path=image_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
        wire_api=preferred_wire_api,
        disable_storage=disable_storage,
    )
    if live or preferred_wire_api == "chat_completions":
        return live
    return _request_openai_compatible(
        image_path=image_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
        wire_api="chat_completions",
        disable_storage=disable_storage,
    )


def analyze_meal_image(image_path: str | Path, provider: str | None = None) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        fallback = _filename_fallback(path)
        fallback["provider"] = "missing-file-fallback"
        fallback["summary"] = (
            f"Image file not found at {path}. "
            + fallback.get("summary", "Using filename-based hints only.")
        )
        fallback["confidence"] = min(float(fallback.get("confidence", 0.35)), 0.2)
        fallback.setdefault("caution_tags", [])
        if "file_not_found" not in fallback["caution_tags"]:
            fallback["caution_tags"].append("file_not_found")
        return fallback

    resolved_provider = (provider or os.getenv("VISION_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if resolved_provider in {"openai", "openai-compatible", "api", "live"}:
        live = _call_openai_compatible(path)
        if live:
            return live

    sidecar = _load_sidecar(path)
    if sidecar:
        return sidecar
    return _filename_fallback(path)


def analysis_to_text(analysis: dict[str, Any]) -> str:
    ingredients = ", ".join(analysis.get("visible_ingredients", [])) or "not listed"
    tags = ", ".join(analysis.get("cuisine_tags", [])) or "none"
    nutrition = ", ".join(analysis.get("nutrition_signals", [])) or "none"
    cautions = ", ".join(analysis.get("caution_tags", [])) or "none"
    method = str(analysis.get("cooking_method", "")).strip().lower() or "unknown"
    portions = analysis.get("estimated_portions", 1.0)
    cues = ", ".join(_normalize_list(analysis.get("recipe_cues", []))) or "none"
    return (
        f"Visual analysis suggests {analysis.get('dish_name', 'an unknown dish')} as a {analysis.get('meal_type', 'meal')}. "
        f"Visible ingredients: {ingredients}. Tags: {tags}. Nutrition signals: {nutrition}. "
        f"Caution tags: {cautions}. Cooking method: {method}. Estimated portions: {portions}. Recipe cues: {cues}."
    )


def analysis_tags(analysis: dict[str, Any]) -> list[str]:
    tags = []
    value = str(analysis.get("meal_type", "")).strip().lower()
    if value:
        tags.append(value)
    tags.extend(_normalize_list(analysis.get("visible_ingredients", [])))
    tags.extend(_normalize_list(analysis.get("cuisine_tags", [])))
    tags.extend(_normalize_list(analysis.get("nutrition_signals", [])))
    tags.extend(_normalize_list(analysis.get("caution_tags", [])))
    tags.extend(_normalize_list(analysis.get("recipe_cues", [])))
    method = str(analysis.get("cooking_method", "")).strip().lower()
    if method and method != "unknown":
        tags.append(method)
    deduped: list[str] = []
    for tag in tags:
        if tag and tag not in deduped:
            deduped.append(tag)
    return deduped
