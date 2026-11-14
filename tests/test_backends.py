from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from runeconomy.backends import discover_vllm_capabilities, sanitized_target


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
