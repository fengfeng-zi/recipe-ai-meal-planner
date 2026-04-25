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

DEFAULT_PROVIDER = "openai-compatible"
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")
KNOWN_TOKENS = {
    "salmon", "chicken", "turkey", "tofu", "egg", "eggs", "rice", "quinoa", "oats", "berries", "broccoli",
    "carrot", "lettuce", "tomato", "cucumber", "potato", "potatoes", "banana", "yogurt", "wrap", "bowl",
    "salad", "tray", "stir", "fried", "grilled", "peanut", "bean", "beans", "avocado", "apple", "mushroom",
}
COOKING_METHOD_HINTS = ("grilled", "fried", "stir-fry", "stir", "baked", "roasted", "steamed", "raw")
PROTEIN_OPTIONS = ("chicken", "salmon", "turkey", "tofu", "egg", "eggs", "beans", "yogurt")
CARB_OPTIONS = ("rice", "quinoa", "oats", "potato", "potatoes", "wrap")
VEG_OPTIONS = ("broccoli", "carrot", "spinach", "lettuce", "tomato", "cucumber", "mushroom", "asparagus")


def _tokenize_filename(path: Path) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", path.stem.lower())


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


def _normalize_meal_type(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in MEAL_TYPES else "meal"


def _infer_cooking_method(tokens: list[str], tags: list[str]) -> str:
    candidates = set(tokens) | set(tags)
    if "stir" in candidates or "stir-fry" in candidates:
        return "stir-fry"
    for method in COOKING_METHOD_HINTS:
        if method in candidates:
            return method
    return "unknown"


def _normalize_portions(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return max(1.0, min(float(raw), 8.0))
    text = str(raw or "").strip().lower()
    if not text:
        return 1.0
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 1.0
    return max(1.0, min(float(match.group(0)), 8.0))


def _build_recipe_cues(
    *,
    meal_type: str,
    visible_ingredients: list[str],
    cuisine_tags: list[str],
    nutrition_signals: list[str],
    caution_tags: list[str],
    cooking_method: str,
) -> list[str]:
    cues: list[str] = []
    if meal_type in MEAL_TYPES:
        cues.append(f"{meal_type}-candidate")
    if "high-protein" in cuisine_tags or any("protein" in item for item in nutrition_signals):
        cues.append("high-protein")
    if any(item in visible_ingredients for item in ("chicken", "salmon", "turkey", "tofu", "egg", "eggs")):
        cues.append("protein-anchor")
    if any(item in visible_ingredients for item in ("broccoli", "carrot", "spinach", "lettuce", "tomato", "cucumber")):
        cues.append("vegetable-forward")
    if cooking_method != "unknown":
        cues.append(f"method:{cooking_method}")
    if "peanut" in caution_tags:
        cues.append("allergy-check-peanut")
    deduped: list[str] = []
    for cue in cues:
        if cue and cue not in deduped:
            deduped.append(cue)
    return deduped


def _dish_style(
    *,
    meal_type: str,
    visible_ingredients: list[str],
    cuisine_tags: list[str],
    cooking_method: str,
) -> str:
    if "tray" in cuisine_tags or cooking_method in {"baked", "roasted"}:
        return "sheet-pan"
    if "bowl" in cuisine_tags:
        return "bowl"
    if "salad" in cuisine_tags:
        return "salad"
    if cooking_method == "stir-fry":
        return "stir-fry"
    if meal_type in {"breakfast", "snack"} and any(item in visible_ingredients for item in ("oats", "yogurt", "egg", "eggs")):
        return "quick-assembly"
    return "mixed-plate"


def _prep_summary(meal_type: str, style: str, cooking_method: str, portions: float) -> str:
    base = f"{meal_type.title() if meal_type in MEAL_TYPES else 'Meal'} likely follows a {style} format."
    if cooking_method and cooking_method != "unknown":
        base += f" Primary method: {cooking_method}."
    base += f" Estimated yield: about {portions:.1f} serving(s)."
    base += " Keep seasoning simple and adjust salt at the end."
    return base


def _step_outline(
    *,
    visible_ingredients: list[str],
    cooking_method: str,
    style: str,
) -> list[str]:
    proteins = [item for item in visible_ingredients if item in PROTEIN_OPTIONS]
    carbs = [item for item in visible_ingredients if item in CARB_OPTIONS]
    vegetables = [item for item in visible_ingredients if item in VEG_OPTIONS]
    protein_text = proteins[0] if proteins else "protein"
    carb_text = carbs[0] if carbs else "carb base"
    veg_text = ", ".join(vegetables[:3]) if vegetables else "vegetables"
    if style == "quick-assembly":
        return [
            f"Prepare base ingredients and portion the {protein_text}.",
            f"Assemble with {carb_text} and {veg_text}.",
            "Finish with a light sauce or seasoning, then serve immediately.",
        ]
    method = cooking_method if cooking_method and cooking_method != "unknown" else "cook"
    return [
        f"Prep and season the {protein_text}; cut {veg_text} into bite-size pieces.",
        f"{method.title()} the protein and vegetables until just cooked.",
        f"Plate with {carb_text} and adjust seasoning for final balance.",
    ]


def _substitutions(visible_ingredients: list[str], caution_tags: list[str]) -> list[str]:
    swaps: list[str] = []
    if "salmon" in visible_ingredients:
        swaps.append("Swap salmon with chicken breast or tofu if fish is unavailable.")
    if "chicken" in visible_ingredients:
        swaps.append("Swap chicken with turkey or firm tofu for variety.")
    if "rice" in visible_ingredients:
        swaps.append("Swap rice with quinoa for extra fiber and protein.")
    if "peanut" in caution_tags:
        swaps.append("Use sesame or sunflower seeds instead of peanuts.")
    if not swaps:
        swaps.append("Swap one protein and one vegetable based on availability while keeping meal type stable.")
    deduped: list[str] = []
    for item in swaps:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4]


def _pantry_staples(visible_ingredients: list[str], cooking_method: str) -> list[str]:
    staples = ["salt", "black pepper", "olive oil"]
    if any(item in visible_ingredients for item in ("rice", "quinoa", "oats")):
        staples.append("stock or water")
    if any(item in visible_ingredients for item in ("salmon", "chicken", "turkey", "tofu")):
        staples.append("garlic")
    if cooking_method in {"stir-fry", "fried"}:
        staples.append("soy sauce")
    if any(item in visible_ingredients for item in ("salad", "lettuce", "cucumber", "tomato")):
        staples.append("lemon or vinegar")
    deduped: list[str] = []
    for item in staples:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _recipe_reconstruction(
    *,
    meal_type: str,
    visible_ingredients: list[str],
    cuisine_tags: list[str],
    caution_tags: list[str],
    cooking_method: str,
    estimated_portions: float,
) -> dict[str, Any]:
    style = _dish_style(
        meal_type=meal_type,
        visible_ingredients=visible_ingredients,
        cuisine_tags=cuisine_tags,
        cooking_method=cooking_method,
    )
    return {
        "dish_style": style,
        "prep_summary": _prep_summary(meal_type, style, cooking_method, estimated_portions),
        "step_outline": _step_outline(
            visible_ingredients=visible_ingredients,
            cooking_method=cooking_method,
            style=style,
        ),
        "substitutions": _substitutions(visible_ingredients, caution_tags),
        "pantry_staples": _pantry_staples(visible_ingredients, cooking_method),
    }


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


def _sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".vision.json")


def _filename_fallback(image_path: Path) -> dict[str, Any]:
    tokens = _tokenize_filename(image_path)
    ingredients = [token for token in tokens if token in KNOWN_TOKENS and token not in {"bowl", "salad", "tray", "wrap"}]
    tags = [token for token in tokens if token in KNOWN_TOKENS and token not in ingredients]
    meal_type = _normalize_meal_type(_guess_meal_type(tokens))
    cooking_method = _infer_cooking_method(tokens, tags)
    caution_tags = ["peanut"] if "peanut" in tokens else []
    recipe_cues = _build_recipe_cues(
        meal_type=meal_type,
        visible_ingredients=ingredients,
        cuisine_tags=tags,
        nutrition_signals=[],
        caution_tags=caution_tags,
        cooking_method=cooking_method,
    )
    dish_name = " ".join(word.capitalize() for word in tokens[:4]) or image_path.stem
    summary_bits = [
        f"Observed dish: {dish_name}.",
        f"Likely meal type: {meal_type}." if meal_type else "",
        f"Visible ingredients: {', '.join(ingredients)}." if ingredients else "",
        f"Visual tags: {', '.join(tags)}." if tags else "",
    ]
    estimated_portions = 1.0
    reconstruction = _recipe_reconstruction(
        meal_type=meal_type,
        visible_ingredients=ingredients,
        cuisine_tags=tags,
        caution_tags=caution_tags,
        cooking_method=cooking_method,
        estimated_portions=estimated_portions,
    )
    return {
        "image_path": str(image_path),
        "provider": "filename-fallback",
        "dish_name": dish_name,
        "meal_type": meal_type,
        "visible_ingredients": ingredients,
        "cuisine_tags": tags,
        "nutrition_signals": [],
        "caution_tags": caution_tags,
        "summary": " ".join(bit for bit in summary_bits if bit),
        "confidence": 0.35,
        "cooking_method": cooking_method,
        "estimated_portions": estimated_portions,
        "recipe_cues": recipe_cues,
        "recipe_reconstruction": reconstruction,
    }


def _load_sidecar(image_path: Path) -> dict[str, Any] | None:
    sidecar = _sidecar_path(image_path)
    if not sidecar.exists():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    meal_type = _normalize_meal_type(payload.get("meal_type", "meal"))
    visible_ingredients = _normalize_list(payload.get("visible_ingredients", []))
    cuisine_tags = _normalize_list(payload.get("cuisine_tags", []))
    nutrition_signals = _normalize_list(payload.get("nutrition_signals", []))
    caution_tags = _normalize_list(payload.get("caution_tags", []))
    cooking_method = _infer_cooking_method(_tokenize_filename(image_path), cuisine_tags)
    if payload.get("cooking_method"):
        cooking_method = str(payload.get("cooking_method", "")).strip().lower() or cooking_method
    recipe_cues = _normalize_list(payload.get("recipe_cues", []))
    if not recipe_cues:
        recipe_cues = _build_recipe_cues(
            meal_type=meal_type,
            visible_ingredients=visible_ingredients,
            cuisine_tags=cuisine_tags,
            nutrition_signals=nutrition_signals,
            caution_tags=caution_tags,
            cooking_method=cooking_method,
        )

    estimated_portions = _normalize_portions(payload.get("estimated_portions", 1.0))
    reconstruction = payload.get("recipe_reconstruction", {})
    if not isinstance(reconstruction, dict):
        reconstruction = {}
    if not reconstruction:
        reconstruction = _recipe_reconstruction(
            meal_type=meal_type,
            visible_ingredients=visible_ingredients,
            cuisine_tags=cuisine_tags,
            caution_tags=caution_tags,
            cooking_method=cooking_method,
            estimated_portions=estimated_portions,
        )
    return {
        "image_path": str(image_path),
        "provider": "sidecar",
        "dish_name": payload.get("dish_name", image_path.stem.replace("_", " ").title()),
        "meal_type": meal_type,
        "visible_ingredients": visible_ingredients,
        "cuisine_tags": cuisine_tags,
        "nutrition_signals": nutrition_signals,
        "caution_tags": caution_tags,
        "summary": str(payload.get("summary", "")).strip(),
        "confidence": _confidence_value(payload.get("confidence", 0.9)),
        "cooking_method": cooking_method,
        "estimated_portions": estimated_portions,
        "recipe_cues": recipe_cues,
        "recipe_reconstruction": reconstruction,
    }


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
        "caution_tags, summary, confidence, cooking_method, estimated_portions, recipe_cues, "
        "recipe_reconstruction."
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


def _read_json_response(request: urllib.request.Request) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


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


def _build_analysis(parsed: dict[str, Any], image_path: Path, provider_name: str) -> dict[str, Any]:
    meal_type = _normalize_meal_type(parsed.get("meal_type", "meal"))
    visible_ingredients = _normalize_list(parsed.get("visible_ingredients", []))
    cuisine_tags = _normalize_list(parsed.get("cuisine_tags", []))
    nutrition_signals = _normalize_list(parsed.get("nutrition_signals", []))
    caution_tags = _normalize_list(parsed.get("caution_tags", []))
    cooking_method = str(parsed.get("cooking_method", "")).strip().lower()
    if not cooking_method:
        cooking_method = _infer_cooking_method(_tokenize_filename(image_path), cuisine_tags)
    recipe_cues = _normalize_list(parsed.get("recipe_cues", []))
    if not recipe_cues:
        recipe_cues = _build_recipe_cues(
            meal_type=meal_type,
            visible_ingredients=visible_ingredients,
            cuisine_tags=cuisine_tags,
            nutrition_signals=nutrition_signals,
            caution_tags=caution_tags,
            cooking_method=cooking_method,
        )
    estimated_portions = _normalize_portions(parsed.get("estimated_portions", 1.0))
    reconstruction = parsed.get("recipe_reconstruction", {})
    if not isinstance(reconstruction, dict):
        reconstruction = {}
    if not reconstruction:
        reconstruction = _recipe_reconstruction(
            meal_type=meal_type,
            visible_ingredients=visible_ingredients,
            cuisine_tags=cuisine_tags,
            caution_tags=caution_tags,
            cooking_method=cooking_method,
            estimated_portions=estimated_portions,
        )
    return {
        "image_path": str(image_path),
        "provider": provider_name,
        "dish_name": parsed.get("dish_name", image_path.stem.replace("_", " ").title()),
        "meal_type": meal_type,
        "visible_ingredients": visible_ingredients,
        "cuisine_tags": cuisine_tags,
        "nutrition_signals": nutrition_signals,
        "caution_tags": caution_tags,
        "summary": str(parsed.get("summary", "")).strip(),
        "confidence": _confidence_value(parsed.get("confidence", 0.75)),
        "cooking_method": cooking_method,
        "estimated_portions": estimated_portions,
        "recipe_cues": recipe_cues,
        "recipe_reconstruction": reconstruction,
    }


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
    payload, error_text = _read_json_response(request)
    if not payload:
        return None, error_text

    parsed = _parse_json_text(_extract_text_payload(payload))
    if not parsed:
        return None, "Vision response was not valid JSON."
    provider_name = "openai-compatible-responses" if wire_api == "responses" else "openai-compatible-chat-completions"
    return _build_analysis(parsed, image_path, provider_name), None


def _vision_model_candidates() -> list[str]:
    raw_candidates = [
        os.getenv("VISION_MODEL"),
        os.getenv("OPENAI_VISION_MODEL"),
        os.getenv("OPENAI_MODEL"),
    ]

    candidates: list[str] = []
    for raw in raw_candidates:
        if not raw:
            continue
        for token in str(raw).split(","):
            model = token.strip()
            if model and model not in candidates:
                candidates.append(model)

    expanded: list[str] = []
    for model in candidates:
        lowered = model.lower()
        if lowered in {"qwen-coder", "qwen_coder", "qwen coder"}:
            for alias in ("qwen3-coder", "[限时]qwen3-coder"):
                if alias not in expanded:
                    expanded.append(alias)
        if model not in expanded:
            expanded.append(model)

    if not expanded:
        expanded.extend(["qwen3-coder", "[限时]qwen3-coder", "qwen-3.5-plus", "gpt-4.1-mini"])

    if "qwen-3.5-plus" not in expanded:
        expanded.append("qwen-3.5-plus")
    return expanded


def _call_openai_compatible(image_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VISION_API_KEY")
    if not api_key:
        return None, {"reason": "Missing OPENAI_API_KEY / VISION_API_KEY.", "attempts": []}
    base_url = (os.getenv("VISION_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model_candidates = _vision_model_candidates()
    wire_api = (os.getenv("VISION_WIRE_API") or os.getenv("OPENAI_WIRE_API") or "chat_completions").strip().lower()
    disable_storage = _env_flag("VISION_DISABLE_RESPONSE_STORAGE", "OPENAI_DISABLE_RESPONSE_STORAGE")
    preferred_wire_api = "responses" if wire_api in {"responses", "response"} else "chat_completions"
    attempts: list[dict[str, str]] = []
    for model in model_candidates:
        live, error_text = _request_openai_compatible(
            image_path=image_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            wire_api=preferred_wire_api,
            disable_storage=disable_storage,
        )
        attempts.append({"model": model, "wire_api": preferred_wire_api, "error": error_text or "unknown error"})
        if live:
            return live, None
        if preferred_wire_api != "chat_completions":
            live, error_text = _request_openai_compatible(
                image_path=image_path,
                api_key=api_key,
                base_url=base_url,
                model=model,
                wire_api="chat_completions",
                disable_storage=disable_storage,
            )
            attempts.append({"model": model, "wire_api": "chat_completions", "error": error_text or "unknown error"})
            if live:
                return live, None
    return None, {"reason": "All configured vision models failed.", "attempts": attempts}


def analyze_meal_image(image_path: str | Path, provider: str | None = None) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    diagnostics = None
    resolved_provider = (provider or os.getenv("VISION_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if resolved_provider in {"openai", "openai-compatible", "api", "live"}:
        live, diagnostics = _call_openai_compatible(path)
        if live:
            return live

    sidecar = _load_sidecar(path)
    if sidecar:
        if diagnostics:
            sidecar["vision_diagnostics"] = diagnostics
            sidecar["vision_live_failed"] = True
        return sidecar

    fallback = _filename_fallback(path)
    if diagnostics:
        fallback["vision_diagnostics"] = diagnostics
        fallback["vision_live_failed"] = True
    return fallback


def analysis_to_text(analysis: dict[str, Any]) -> str:
    meal_labels = {
        "breakfast": "\u65e9\u9910",
        "lunch": "\u5348\u9910",
        "dinner": "\u665a\u9910",
        "snack": "\u52a0\u9910",
        "meal": "\u6b63\u9910",
    }
    ingredients = "\u3001".join(analysis.get("visible_ingredients", [])) or "\u672a\u8bc6\u522b"
    tags = "\u3001".join(analysis.get("cuisine_tags", [])) or "\u65e0"
    nutrition = "\u3001".join(analysis.get("nutrition_signals", [])) or "\u65e0"
    cautions = "\u3001".join(analysis.get("caution_tags", [])) or "\u65e0"
    method = str(analysis.get("cooking_method", "")).strip().lower() or "\u672a\u77e5"
    portions = _normalize_portions(analysis.get("estimated_portions", 1.0))
    recipe_cues = "\u3001".join(_normalize_list(analysis.get("recipe_cues", []))) or "\u65e0"
    dish_name = str(analysis.get("dish_name", "")).strip() or "\u672a\u77e5\u83dc\u54c1"
    meal_type = meal_labels.get(str(analysis.get("meal_type", "meal")).strip().lower(), "\u6b63\u9910")
    reconstruction = analysis.get("recipe_reconstruction", {})
    style = ""
    prep_summary = ""
    if isinstance(reconstruction, dict):
        style = str(reconstruction.get("dish_style", "")).strip().lower()
        prep_summary = str(reconstruction.get("prep_summary", "")).strip()
    extras: list[str] = []
    if style:
        extras.append(f"\u98ce\u683c\uff1a{style}")
    if prep_summary:
        extras.append(f"\u5236\u4f5c\u6458\u8981\uff1a{prep_summary}")
    diagnostics = analysis.get("vision_diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("reason"):
        extras.append(f"\u5728\u7ebf\u89c6\u89c9\u5931\u8d25\uff1a{diagnostics.get("reason")}")
    extra_text = f"\uff1b{'\uff1b'.join(extras)}" if extras else ""
    return (
        f"\u56fe\u50cf\u8bc6\u522b\u7ed3\u679c\uff1a{dish_name}\uff1b\u9910\u522b\uff1a{meal_type}\uff1b\u53ef\u89c1\u98df\u6750\uff1a{ingredients}\uff1b"
        f"\u6807\u7b7e\uff1a{tags}\uff1b\u8425\u517b\u4fe1\u53f7\uff1a{nutrition}\uff1b\u6ce8\u610f\u4e8b\u9879\uff1a{cautions}\uff1b"
        f"\u70f9\u996a\u65b9\u5f0f\uff1a{method}\uff1b\u4f30\u8ba1\u4efd\u91cf\uff1a{portions:.1f} \u4efd\uff1b\u505a\u6cd5\u7ebf\u7d22\uff1a{recipe_cues}{extra_text}\u3002"
    )


def analysis_tags(analysis: dict[str, Any]) -> list[str]:
    tags = []
    for key in ("meal_type",):
        value = str(analysis.get(key, "")).strip().lower()
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
