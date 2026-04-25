from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

WEB_DIR = Path(__file__).resolve().parent
ENTRYPOINT_CANDIDATES = ("server.py", "app.py", "main.py")
UPLOAD_PATH_CANDIDATES = ("/api/upload-image", "/api/upload", "/upload", "/api/analyze-image")
ENDPOINT_CASES = [
    ("/api/analyze-image", {"image_path": "valid_meal_like.png", "session_id": "webtest"}),
    ("/api/query", {"query": "quick high-protein dinner", "session_id": "webtest"}),
    (
        "/api/plan",
        {
            "task": "Plan 3-day meals",
            "input_text": "Need high-protein dinners without peanuts",
            "planning_days": 3,
            "session_id": "webtest",
        },
    ),
]


def _load_backend_module() -> tuple[Any, Path]:
    for name in ENTRYPOINT_CANDIDATES:
        path = WEB_DIR / name
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(f"recipe_studio_web_{path.stem}", path)
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, path
    raise FileNotFoundError("No backend entrypoint found in recipe_studio_web (server.py/app.py/main.py).")


def _resolve_wsgi_app(module: Any) -> Callable | None:
    create_app = getattr(module, "create_app", None)
    if callable(create_app):
        app = create_app()
        if callable(app):
            return app
    for attr in ("application", "app", "wsgi_app"):
        candidate = getattr(module, attr, None)
        if callable(candidate):
            return candidate
    return None


def _call_wsgi(app: Callable, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], str]:
    body = json.dumps(payload).encode("utf-8")
    return _call_wsgi_raw(app, path, body, "application/json")


def _call_wsgi_raw(app: Callable, path: str, body: bytes, content_type: str) -> tuple[int, dict[str, str], str]:
    environ = {
        "REQUEST_METHOD": "POST",
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "SERVER_NAME": "127.0.0.1",
        "SERVER_PORT": "0",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": "http",
        "wsgi.input": io.BytesIO(body),
        "wsgi.errors": io.StringIO(),
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }
    status_line = "500 Internal Server Error"
    headers: list[tuple[str, str]] = []

    def start_response(status: str, response_headers: list[tuple[str, str]], _exc_info=None):
        nonlocal status_line, headers
        status_line = status
        headers = response_headers

    result = app(environ, start_response)
    try:
        chunks: list[bytes] = []
        for part in result:
            if isinstance(part, bytes):
                chunks.append(part)
            else:
                chunks.append(str(part).encode("utf-8"))
    finally:
        close_fn = getattr(result, "close", None)
        if callable(close_fn):
            close_fn()

    code = int(status_line.split(" ", 1)[0])
    header_map = {k.lower(): v for k, v in headers}
    text = b"".join(chunks).decode("utf-8", errors="replace")
    return code, header_map, text


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, timeout_s: float = 8.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            request.urlopen(url, timeout=1.0)
            return True
        except Exception:
            time.sleep(0.2)
    return False


class WebLayerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.module, cls.entrypoint = _load_backend_module()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))

        cls.wsgi_app = _resolve_wsgi_app(cls.module)
        cls.proc: subprocess.Popen | None = None
        cls.base_url = ""

        if cls.wsgi_app is not None:
            return

        port = _pick_free_port()
        env = dict(os.environ)
        env["PORT"] = str(port)
        env["RECIPE_STUDIO_WEB_PORT"] = str(port)
        cmd = [sys.executable, str(cls.entrypoint)]
        cls.proc = subprocess.Popen(
            cmd,
            cwd=str(WEB_DIR.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.base_url = f"http://127.0.0.1:{port}"
        if not _wait_for_http(cls.base_url + "/", timeout_s=10.0):
            raise RuntimeError("Backend process started but HTTP endpoint did not become reachable.")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.proc is not None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    def _request_case(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], str]:
        if self.wsgi_app is not None:
            return _call_wsgi(self.wsgi_app, path, payload)

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5.0) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return int(resp.status), dict(resp.headers.items()), body
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), dict(exc.headers.items()) if exc.headers else {}, body

    def _request_raw(self, path: str, body: bytes, content_type: str) -> tuple[int, dict[str, str], str]:
        if self.wsgi_app is not None:
            return _call_wsgi_raw(self.wsgi_app, path, body, content_type)

        req = request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5.0) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                return int(resp.status), dict(resp.headers.items()), response_body
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), dict(exc.headers.items()) if exc.headers else {}, response_body

    @staticmethod
    def _index_markup() -> str:
        index_path = WEB_DIR / "templates" / "index.html"
        return index_path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _multipart_form(parts: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
        boundary = "----recipe-studio-test-boundary"
        chunks: list[bytes] = []
        for field_name, filename, payload, mime in parts:
            header = (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {mime}\r\n\r\n"
            ).encode("utf-8")
            chunks.append(header)
            chunks.append(payload)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def test_core_endpoints_are_healthy(self) -> None:
        for path, payload in ENDPOINT_CASES:
            with self.subTest(path=path):
                code, headers, body = self._request_case(path, payload)
                self.assertEqual(code, 200, f"Endpoint {path} should return HTTP 200, got {code} with body: {body}")
                content_type = next((value for key, value in headers.items() if key.lower() == "content-type"), "")
                self.assertIn("application/json", content_type.lower())
                payload_json = json.loads(body)
                self.assertTrue(payload_json.get("ok"), f"Endpoint {path} returned non-ok payload: {payload_json}")

    def test_upload_endpoint_returns_saved_path_and_analysis(self) -> None:
        sample = WEB_DIR.parent / "valid_meal_like.png"
        self.assertTrue(sample.exists(), "Expected sample upload file at repo root.")
        body, content_type = self._multipart_form(
            [("image", "valid_meal_like.png", sample.read_bytes(), "image/png")]
        )
        success_payload = None
        for path in UPLOAD_PATH_CANDIDATES:
            code, _headers, response_body = self._request_raw(path, body, content_type)
            if code == 200:
                success_payload = json.loads(response_body)
                break
        self.assertIsNotNone(success_payload, "No upload-capable endpoint returned HTTP 200.")
        self.assertTrue(success_payload.get("image_path") or success_payload.get("uploaded_image_path"))
        self.assertTrue(isinstance(success_payload.get("analysis"), dict), "Upload response should include image analysis.")

    def test_query_returns_card_ready_evidence_structure(self) -> None:
        code, _headers, body = self._request_case(
            "/api/query",
            {
                "query": "quick high-protein dinner",
                "session_id": "webtest",
                "top_k": 5,
                "rerank_top_k": 3,
            },
        )
        self.assertEqual(code, 200, f"/api/query should return 200, got {code}")
        payload = json.loads(body)
        self.assertIn("answer_provider", payload, "Query response should include `answer_provider`.")
        self.assertTrue(str(payload.get("answer_provider", "")).strip(), "`answer_provider` should be non-empty.")
        self.assertIn("answer", payload, "Query response should include `answer` text for UI.")

        candidates = payload.get("candidates", [])
        hits = payload.get("hits", [])
        self.assertTrue(
            isinstance(candidates, list) or isinstance(hits, list),
            "Query response must include list-like `candidates` or `hits` for evidence cards.",
        )
        evidence_items = candidates if isinstance(candidates, list) and candidates else hits
        self.assertTrue(evidence_items, "Expected non-empty evidence items for card rendering.")
        first = evidence_items[0]
        self.assertTrue(
            any(key in first for key in ("title", "chunk_id", "name")),
            "Evidence item must include an identity field (`title`, `chunk_id`, or `name`).",
        )
        self.assertIn("score", first, "Evidence item should include `score` for card ranking display.")

        trace = payload.get("trace", {})
        self.assertTrue(isinstance(trace, dict), "Expected `trace` object in query response.")
        self.assertTrue(
            isinstance(trace.get("reranked_hits", []), list) or isinstance(trace.get("ranked_hits", []), list),
            "Trace should include `reranked_hits` or `ranked_hits` list for detailed evidence panels.",
        )

    def test_plan_returns_structured_days(self) -> None:
        code, _headers, body = self._request_case(
            "/api/plan",
            {
                "task": "Generate meal plan",
                "input_text": "Need a practical high-protein 3-day dinner plan without peanuts",
                "planning_days": 3,
                "session_id": "webtest",
                "goal": "fat loss",
                "allergies": ["peanut"],
                "preferences": ["high-protein", "quick-cook"],
            },
        )
        self.assertEqual(code, 200, f"/api/plan should return 200, got {code}")
        payload = json.loads(body)
        self.assertIn("plan_provider", payload, "Plan response should include `plan_provider`.")
        self.assertTrue(str(payload.get("plan_provider", "")).strip(), "`plan_provider` should be non-empty.")
        self.assertIn("shopping_list", payload, "Plan response should include `shopping_list` for UI.")
        self.assertTrue(payload.get("final_output"), "Plan response should include final_output text.")
        self.assertTrue(isinstance(payload.get("plan_days"), list), "Plan response should expose list-like `plan_days`.")
        self.assertTrue(payload.get("plan_days"), "Expected at least one structured plan day.")
        first_day = payload["plan_days"][0]
        self.assertIn("title", first_day)
        self.assertTrue(isinstance(first_day.get("meals"), list), "Each plan day should include a `meals` list.")
        if first_day.get("meals"):
            first_meal = first_day["meals"][0]
            for key in ("meal", "description", "kcal", "protein", "time", "ingredients"):
                self.assertIn(key, first_meal, f"Plan meal should include `{key}` for card rendering.")

    def test_page_exposes_preview_progress_and_plan_card_containers(self) -> None:
        html = self._index_markup()

        preview_signals = (
            'id="imagePreview"',
            'id="previewImage"',
            'id="previewPane"',
            'class="image-preview"',
            'class="preview-pane"',
        )
        progress_signals = (
            'id="uploadProgress"',
            'id="uploadProgressBar"',
            'id="uploadProgressText"',
            'id="uploadState"',
            'class="upload-progress"',
        )
        plan_card_signals = (
            'id="planCards"',
            'id="mealPlanCards"',
            'id="planCardGrid"',
            'class="plan-cards"',
            'class="plan-card-grid"',
        )

        self.assertTrue(
            any(token in html for token in preview_signals),
            "index.html should expose an image preview container or equivalent preview UI hook.",
        )
        self.assertTrue(
            any(token in html for token in progress_signals),
            "index.html should expose an upload progress/status container or equivalent progress UI hook.",
        )
        self.assertTrue(
            any(token in html for token in plan_card_signals) or re.search(r'class="[^\"]*plan-card', html),
            "index.html should expose a day-based meal-plan card container or equivalent card UI hook.",
        )

    def test_query_llm_override_returns_answer_when_enabled(self) -> None:
        module = self.module
        original_llm_enabled = module.llm_enabled
        original_chat_complete = module.chat_complete_with_model
        try:
            module.llm_enabled = lambda: True
            module.chat_complete_with_model = lambda *args, **kwargs: (
                json.dumps(
                    {"answer": "???? LLM ??????????"},
                    ensure_ascii=False,
                ),
                "qwen3-coder",
            )
            answer, used_model = module._llm_query_answer(
                {"goal": "??", "allergies": ["??"]},
                {
                    "query": "?????",
                    "citations": [{"title": "x", "score": 0.9, "applied_signals": []}],
                    "candidates": [{"title": "?????"}],
                    "image_analysis": {"meal_type": "dinner"},
                },
            )
            self.assertEqual(answer, "???? LLM ??????????")
            self.assertEqual(used_model, "qwen3-coder")
        finally:
            module.llm_enabled = original_llm_enabled
            module.chat_complete_with_model = original_chat_complete

    def test_plan_llm_override_returns_structured_payload_when_enabled(self) -> None:
        module = self.module
        original_llm_enabled = module.llm_enabled
        original_chat_complete = module.chat_complete_with_model
        try:
            module.llm_enabled = lambda: True
            module.chat_complete_with_model = lambda *args, **kwargs: (
                json.dumps(
                    {
                        "final_output": "???????",
                        "plan_days": [
                            {
                                "day": 1,
                                "title": "? 1 ?",
                                "notes": ["???"],
                                "meals": [
                                    {
                                        "meal": "??",
                                        "description": "?????????",
                                        "kcal": "520 kcal",
                                        "protein": "42g ???",
                                        "time": "20 ??",
                                        "ingredients": ["???", "???", "?"],
                                    }
                                ],
                            }
                        ],
                        "shopping_list": ["???: ???", "????: ???"],
                    },
                    ensure_ascii=False,
                ),
                "qwen3-coder",
            )
            result, used_model = module._llm_plan_override(
                {"goal": "??", "planning_days": 1},
                plan_days=[
                    {"day": 1, "title": "? 1 ?", "notes": [], "meals": []},
                ],
                shopping_list=["???: ???"],
                image_analysis={"meal_type": "dinner"},
            )
            self.assertIsInstance(result, dict)
            self.assertEqual(used_model, "qwen3-coder")
            self.assertEqual(result["final_output"], "???????")
            self.assertTrue(isinstance(result.get("plan_days"), list) and result["plan_days"])
            self.assertIn("shopping_list", result)
        finally:
            module.llm_enabled = original_llm_enabled
            module.chat_complete_with_model = original_chat_complete


if __name__ == "__main__":
    unittest.main(verbosity=2)


