from __future__ import annotations

import io
import json
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from runeconomy.backends import (
    BackendCapabilities,
    BackendProbeError,
    Capability,
    _post_probe,
    _render_capability,
    discover_vllm_capabilities,
    sanitized_target,
)


class _DoctorHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/openapi.json":
            self._json(
                {
                    "paths": {
                        "/v1/chat/completions": {},
                        "/v1/chat/completions/render": {},
                        "/tokenize": {},
                    },
                    "components": {
                        "schemas": {
                            "ChatCompletionRequest": {
                                "properties": {"cache_salt": {"type": "string"}}
                            }
                        }
                    },
                }
            )
        elif self.path == "/version":
            self._json({"version": "0.33.0"})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": "example/model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions/render":
            self._json({"prompt_token_ids": [1, 2, 3]})
        else:
            self.send_error(404)

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class BackendCapabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _DoctorHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://user:password@127.0.0.1:{self.server.server_port}/private?token=x"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_doctor_reports_advertised_and_unknown_capabilities(self) -> None:
        base = f"http://127.0.0.1:{self.server.server_port}"
        result = discover_vllm_capabilities(base).to_dict()
        self.assertEqual(result["backend_version"], "0.33.0")
        self.assertEqual(result["model_ids"], ["example/model"])
        self.assertEqual(result["chat_completions"]["status"], "supported")
        self.assertEqual(result["chat_completions_render"]["evidence"], "successful non-inference render probe")
        self.assertEqual(result["responses"]["status"], "unsupported")
        self.assertEqual(result["cache_salt"]["status"], "supported")
        self.assertEqual(result["cache_telemetry"]["status"], "unknown")
        salt_mode = next(m for m in result["experiment_modes"] if m["mode"] == "salt-isolated")
        self.assertTrue(salt_mode["supported"])
        self.assertEqual(salt_mode["physical_cache_state"], "unchanged")
        self.assertEqual(result["readiness"]["status"], "ready")

    def test_sanitized_target_removes_credentials_path_and_query(self) -> None:
        display = sanitized_target(self.url)
        self.assertNotIn("password", display)
        self.assertNotIn("token", display)
        self.assertNotIn("private", display)

    def test_advertised_but_runtime_disabled_render_is_not_supported(self) -> None:
        original = _DoctorHandler.do_POST

        def disabled(handler: _DoctorHandler) -> None:
            length = int(handler.headers.get("Content-Length", "0"))
            json.loads(handler.rfile.read(length) or b"{}")
            body = json.dumps(
                {"error": {"message": "Scale-out endpoints are disabled."}}
            ).encode()
            handler.send_response(403)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

        _DoctorHandler.do_POST = disabled
        try:
            base = f"http://127.0.0.1:{self.server.server_port}"
            result = discover_vllm_capabilities(base).to_dict()
        finally:
            _DoctorHandler.do_POST = original
        self.assertEqual(result["chat_completions_render"]["status"], "unsupported")
        salt_mode = next(m for m in result["experiment_modes"] if m["mode"] == "salt-isolated")
        self.assertTrue(salt_mode["supported"])

    def test_authentication_failure_has_safe_category_and_action(self) -> None:
        error = urllib.error.HTTPError(
            "https://private.example/openapi.json", 401, "unauthorized", {}, None
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(BackendProbeError) as caught:
                discover_vllm_capabilities("https://private.example")
        self.assertEqual(caught.exception.category, "authentication")
        self.assertIn("--api-key-env", caught.exception.next_action)
        self.assertNotIn("private.example", str(caught.exception))

    def test_invalid_base_url_is_configuration_failure(self) -> None:
        with self.assertRaises(BackendProbeError) as caught:
            discover_vllm_capabilities("localhost:8000")
        self.assertEqual(caught.exception.category, "configuration")
        self.assertIn("--base-url", caught.exception.next_action)

    def test_server_error_is_not_downgraded_to_unknown(self) -> None:
        error = urllib.error.HTTPError(
            "https://private.example/openapi.json", 503, "unavailable", {}, None
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(BackendProbeError) as caught:
                discover_vllm_capabilities("https://private.example")
        self.assertEqual(caught.exception.category, "server_error")
        self.assertIn("server error", caught.exception.next_action)

    def test_connection_and_timeout_failures_are_distinct(self) -> None:
        cases = (
            (urllib.error.URLError("refused"), "connection"),
            (urllib.error.URLError(TimeoutError("slow")), "timeout"),
        )
        for error, category in cases:
            with self.subTest(category=category):
                with patch("urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(BackendProbeError) as caught:
                        discover_vllm_capabilities("https://private.example")
                self.assertEqual(caught.exception.category, category)
                self.assertNotIn("private.example", str(caught.exception))

    def test_malformed_openapi_response_is_actionable(self) -> None:
        class Response(io.BytesIO):
            status = 200

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

        with patch("urllib.request.urlopen", return_value=Response(b"not-json")):
            with self.assertRaises(BackendProbeError) as caught:
                discover_vllm_capabilities("https://private.example")
        self.assertEqual(caught.exception.category, "malformed_response")
        self.assertIn("OpenAI-compatible", caught.exception.next_action)

    def test_incompatible_dialect_is_a_completed_blocked_diagnosis(self) -> None:
        absent = Capability("unsupported", "absent from /openapi.json")
        result = BackendCapabilities(
            backend="vllm",
            backend_version="0.33.0",
            model_ids=(),
            responses=absent,
            chat_completions=absent,
            responses_render=absent,
            chat_completions_render=absent,
            tokenize=absent,
            cache_telemetry=Capability("unknown", "requires generation"),
            per_request_timing=Capability("unknown", "requires generation"),
            cache_reset=absent,
            cache_salt=absent,
        ).to_dict()
        self.assertEqual(result["readiness"]["status"], "blocked")
        self.assertIn("No supported Responses", result["readiness"]["blockers"][0])
        self.assertTrue(result["readiness"]["next_actions"])

    def test_render_path_must_match_the_supported_generation_dialect(self) -> None:
        supported = Capability("supported", "advertised")
        absent = Capability("unsupported", "absent")
        result = BackendCapabilities(
            backend="vllm",
            backend_version="0.33.0",
            model_ids=("example/model",),
            responses=absent,
            chat_completions=supported,
            responses_render=supported,
            chat_completions_render=absent,
            tokenize=absent,
            cache_telemetry=Capability("unknown", "requires generation"),
            per_request_timing=Capability("unknown", "requires generation"),
            cache_reset=absent,
            cache_salt=absent,
        ).to_dict()
        self.assertEqual(result["readiness"]["status"], "blocked")
        warm = next(
            mode for mode in result["experiment_modes"] if mode["mode"] == "warm-cache"
        )
        self.assertFalse(warm["supported"])

    def test_http_200_without_rendered_token_ids_is_unknown(self) -> None:
        original = _DoctorHandler.do_POST

        def incomplete(handler: _DoctorHandler) -> None:
            length = int(handler.headers.get("Content-Length", "0"))
            json.loads(handler.rfile.read(length) or b"{}")
            handler._json({"token_ids": []})

        _DoctorHandler.do_POST = incomplete
        try:
            base = f"http://127.0.0.1:{self.server.server_port}"
            result = discover_vllm_capabilities(base).to_dict()
        finally:
            _DoctorHandler.do_POST = original
        self.assertEqual(result["chat_completions_render"]["status"], "unknown")
        self.assertIn(
            "without integer token IDs",
            result["chat_completions_render"]["evidence"],
        )

    def test_advertised_render_without_model_id_remains_unknown(self) -> None:
        capability = _render_capability(
            Capability("supported", "advertised"),
            base_url="https://private.example",
            path="/v1/chat/completions/render",
            payload=None,
            api_key=None,
            timeout_seconds=0.1,
        )
        self.assertEqual(capability.status, "unknown")
        self.assertIn("no model ID", capability.evidence)

    def test_render_authentication_failure_has_specific_remediation(self) -> None:
        original = _DoctorHandler.do_POST

        def unauthorized(handler: _DoctorHandler) -> None:
            length = int(handler.headers.get("Content-Length", "0"))
            handler.rfile.read(length)
            handler.send_error(401)

        _DoctorHandler.do_POST = unauthorized
        try:
            base = f"http://127.0.0.1:{self.server.server_port}"
            result = discover_vllm_capabilities(base).to_dict()
        finally:
            _DoctorHandler.do_POST = original
        capability = result["chat_completions_render"]
        self.assertEqual(capability["status"], "unknown")
        self.assertIn("--api-key-env", capability["next_action"])

    def test_render_probe_timeout_is_not_mislabeled_as_connection(self) -> None:
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(TimeoutError("slow")),
        ):
            with self.assertRaises(BackendProbeError) as caught:
                _post_probe(
                    "https://private.example",
                    "/v1/chat/completions/render",
                    {"model": "example/model", "messages": []},
                    api_key=None,
                    timeout_seconds=0.1,
                )
        self.assertEqual(caught.exception.category, "timeout")

    def test_sensitive_or_ambiguous_url_components_are_rejected(self) -> None:
        cases = (
            "https://user:password@private.example",
            "https://private.example?token=secret",
            "https://private.example#fragment",
        )
        for url in cases:
            with self.subTest(url=url):
                with self.assertRaises(BackendProbeError) as caught:
                    discover_vllm_capabilities(url)
                self.assertEqual(caught.exception.category, "configuration")
                self.assertNotIn("private.example", str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))
