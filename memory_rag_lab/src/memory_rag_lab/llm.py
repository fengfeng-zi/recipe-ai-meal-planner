from __future__ import annotations

import json
import os
from typing import Any
import urllib.error
import urllib.request


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def llm_base_url() -> str:
    return _env_first("LLM_BASE_URL", "OPENAI_BASE_URL", "VISION_BASE_URL", default="https://once.novai.su/v1").rstrip("/")


def llm_api_key() -> str:
    return _env_first("LLM_API_KEY", "OPENAI_API_KEY", "VISION_API_KEY")


def llm_model() -> str:
    return llm_model_candidates()[0]


def llm_model_candidates(model: str | None = None) -> list[str]:
    raw = model or _env_first("LLM_MODEL", "TEXT_MODEL", "OPENAI_MODEL", default="qwen-3.5-plus")
    candidates: list[str] = []
    for token in str(raw).split(","):
        item = token.strip()
        if item and item not in candidates:
            candidates.append(item)

    expanded: list[str] = []
    for item in candidates:
        lowered = item.lower()
        if lowered in {"qwen-coder", "qwen_coder", "qwen coder"}:
            for alias in ("qwen3-coder", "[\u9650\u65f6]qwen3-coder"):
                if alias not in expanded:
                    expanded.append(alias)
        if item not in expanded:
            expanded.append(item)

    if not expanded:
        expanded.extend(["qwen3-coder", "qwen-3.5-plus"])

    if "qwen-3.5-plus" not in expanded:
        expanded.append("qwen-3.5-plus")
    return expanded


def llm_enabled() -> bool:
    return bool(llm_api_key())


def _read_json_response(request: urllib.request.Request) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _extract_chat_text(payload: dict[str, Any]) -> str:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text")
        ).strip()
    return str(content).strip()


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped.removeprefix("```json").removeprefix("```JSON").removeprefix("```").strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


def parse_json_text(raw_text: str) -> dict[str, Any] | None:
    text = _strip_code_fence(raw_text)
    if not text:
        return None
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


def chat_complete_with_model(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1600,
    response_format: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    api_key = llm_api_key()
    if not api_key:
        return None, None

    for candidate in llm_model_candidates(model):
        body: dict[str, Any] = {
            "model": candidate,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        request = urllib.request.Request(
            url=f"{llm_base_url()}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = _read_json_response(request)
        if not payload:
            continue
        text = _extract_chat_text(payload)
        if text:
            return text, candidate

    return None, None


def chat_complete(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1600,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    text, _used_model = chat_complete_with_model(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    return text
