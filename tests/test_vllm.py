from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from runeconomy.vllm import VLLMClient


class _Handler(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict, str | None]] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.requests.append((self.path, payload, self.headers.get("Authorization")))
        if self.path == "/v1/responses/render":
            self._json({"token_ids": [11, 12, 13]})
            return
        if self.path.startswith("/reset_prefix_cache"):
            self._json({"success": True})
            return
        if self.path == "/v1/responses":
            events = [
                {"type": "response.created", "response": {"id": "resp_1"}},
                {"type": "response.output_text.delta", "delta": "hello"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "hello"}
                                ],
                            }
                        ],
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 1,
                            "input_tokens_details": {"cached_tokens": 2},
                        },
                        "metrics": {
                            "time_to_first_token_ms": 4.5,
                            "queue_time_ms": 1.5,
                            "generation_time_ms": 2.0,
                            "mean_itl_ms": None,
                            "tokens_per_second": 150.0,
                        },
                    },
                },
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body.encode())))
            self.end_headers()
            self.wfile.write(body.encode())
            return
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


class VLLMClientTests(unittest.TestCase):
    def setUp(self) -> None:
        _Handler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_capture_uses_render_and_extracts_telemetry(self) -> None:
        ticks = iter([1.0, 2.0, 3.0, 4.0])
        client = VLLMClient(
            f"http://127.0.0.1:{self.server.server_port}",
            api_key="secret",
            clock=lambda: next(ticks),
        )
        result = client.capture({"model": "test", "input": "hello"})

        self.assertEqual(result.prompt_token_ids, (11, 12, 13))
        self.assertEqual(result.cached_tokens, 2)
        self.assertEqual(result.client_ttft_ms, 2000.0)
        self.assertEqual(result.total_latency_ms, 3000.0)
        self.assertEqual(result.server_queue_ms, 1.5)
        self.assertEqual(result.output_text, "hello")
        self.assertNotIn("output_text", result.to_dict())
        self.assertAlmostEqual(result.cache_coverage or 0, 2 / 3)
        self.assertEqual([item[0] for item in _Handler.requests[:2]], [
            "/v1/responses/render",
            "/v1/responses",
        ])
        self.assertNotIn("stream", _Handler.requests[0][1])
        self.assertIs(_Handler.requests[1][1]["stream"], True)
        self.assertEqual(_Handler.requests[0][2], "Bearer secret")

    def test_reset_external_uses_dev_endpoint_query(self) -> None:
        client = VLLMClient(f"http://127.0.0.1:{self.server.server_port}")
        client.reset_prefix_cache(reset_external=True)
        self.assertEqual(_Handler.requests[0][0], "/reset_prefix_cache?reset_external=true")

    def test_incomplete_response_is_a_captured_terminal_event(self) -> None:
        original = _Handler.do_POST

        def incomplete(handler: _Handler) -> None:
            if handler.path != "/v1/responses":
                return original(handler)
            length = int(handler.headers.get("Content-Length", "0"))
            json.loads(handler.rfile.read(length) or b"{}")
            event = {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_incomplete",
                    "status": "incomplete",
                    "output": [],
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 8,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            }
            body = f"data: {json.dumps(event)}\n\n".encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)

        _Handler.do_POST = incomplete
        try:
            client = VLLMClient(f"http://127.0.0.1:{self.server.server_port}")
            result = client.capture({"model": "test", "input": "hello"})
        finally:
            _Handler.do_POST = original
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.output_tokens, 8)


if __name__ == "__main__":
    unittest.main()