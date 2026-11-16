"""Small deterministic vLLM-compatible server for the public golden path."""

from __future__ import annotations

import json
import hashlib
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "127.0.0.1"
PORT = 8765


def _candidate(payload: dict) -> bool:
    messages = payload.get("messages", [])
    return bool(
        isinstance(messages, list)
        and messages
        and isinstance(messages[0], dict)
        and messages[0].get("content") == "Stable instructions"
    )


def _dynamic_token(payload: dict) -> int:
    messages = payload.get("messages", [])
    metadata = next(
        (
            str(item.get("content"))
            for item in messages
            if isinstance(item, dict)
            and str(item.get("content", "")).startswith("Run metadata ")
        ),
        "Run metadata unknown",
    )
    digest = hashlib.sha256(metadata.encode("utf-8")).digest()
    return 1000 + int.from_bytes(digest[:2], "big")


class Handler(BaseHTTPRequestHandler):
    server_version = "RunEconomyGoldenMock/0.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        return value if isinstance(value, dict) else {}

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        payload = self._payload()
        candidate = _candidate(payload)
        if self.path == "/reset_prefix_cache":
            self._json(200, {"success": True})
            return
        if self.path == "/v1/chat/completions/render":
            self._json(404, {"error": "render endpoint intentionally unavailable"})
            return
        if self.path == "/tokenize":
            dynamic = _dynamic_token(payload)
            tokens = list(range(10, 18)) + [dynamic] if candidate else [dynamic] + list(range(10, 18))
            self._json(200, {"tokens": tokens})
            return
        if self.path == "/v1/chat/completions":
            time.sleep(0.001 if candidate else 0.008)
            cached = 8 if candidate else 0
            ttft = 2.0 if candidate else 8.0
            events = [
                {
                    "id": "golden-response",
                    "choices": [{"delta": {"content": "PASS"}, "finish_reason": None}],
                    "metrics": {
                        "time_to_first_token_ms": ttft,
                        "queue_time_ms": 0.1,
                        "generation_time_ms": 1.0,
                    },
                },
                {
                    "id": "golden-response",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": cached},
                    },
                },
            ]
            body = "".join(
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                for event in events
            ) + "data: [DONE]\n\n"
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self._json(404, {"error": "not found"})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"mock vLLM listening on http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
