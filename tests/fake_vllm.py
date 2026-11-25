"""A deterministic fake vLLM chat server for lab-agent tests (no GPU).

Tokens are stable word hashes over a ChatML-like rendering, so prefix caching
and per-message token boundaries behave like a real prefix-stable template.
Outputs follow the request's response_format: an enum is answered with the
value the evidence supports, and an unconstrained plan writes a long
explanation before its JSON.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _render(messages: list[dict[str, Any]], generation: bool) -> list[int]:
    text = "".join(f"<im_start> {m['role']} {m['content']} <im_end> " for m in messages)
    if generation:
        text += "<im_start> assistant "
    return [int(hashlib.sha256(word.encode()).hexdigest()[:6], 16) for word in text.split()]


def _answer(request: dict[str, Any]) -> str:
    messages = request["messages"]
    joined = "\n".join(str(m["content"]) for m in messages)
    response_format = request.get("response_format")
    if response_format is None:
        question = next(m["content"] for m in messages if str(m["content"]).startswith("Question:"))
        marker = "exact string literal " if "exact string literal " in question else "exact call expression "
        phrase = ast.literal_eval(question.split(marker, 1)[1][:-1])
        reasoning = " ".join(["I should search for the exact phrase because the question quotes it."] * 25)
        return reasoning + "\n" + json.dumps({"query": phrase})
    schema = response_format["json_schema"]["schema"]
    properties = schema["properties"]
    identifier = re.search(r"ENCLOSING_IDENTIFIER: (\w+)", joined)
    if "query" in properties:
        return json.dumps({"query": properties["query"]["enum"][0]})
    if "verified" in properties:
        return json.dumps({"verified": True})
    if "choice" in properties:
        options = properties["choice"]["enum"]
        return json.dumps({"choice": identifier.group(1) if identifier and identifier.group(1) in options else options[0]})
    options = properties["answer"]["enum"]
    value = identifier.group(1) if identifier and identifier.group(1) in options else sorted(options)[0]
    result: dict[str, Any] = {"answer": value}
    if "explanation" in properties:
        result = {"explanation": "The search evidence names the enclosing identifier. " * 6, "answer": value}
    return json.dumps(result)


class FakeVLLM(BaseHTTPRequestHandler):
    model = "runeconomy-qwen2.5-14b-bf16"
    seen_prefixes: set[tuple[int, ...]] = set()
    delay_seconds = 0.0

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/render"):
            self.send_error(404)
            return
        if self.path == "/tokenize":
            self._json({"tokens": _render(request["messages"], request.get("add_generation_prompt", True))})
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        prompt = _render(request["messages"], True)
        if type(self).delay_seconds:
            time.sleep(type(self).delay_seconds)
        cached = max((len(p) for p in type(self).seen_prefixes if tuple(prompt[: len(p)]) == p), default=0)
        type(self).seen_prefixes.add(tuple(prompt))
        text = _answer(request)
        events = [
            {"id": "c", "model": type(self).model, "choices": [{"delta": {"content": text}, "finish_reason": None}]},
            {"id": "c", "choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": len(prompt), "completion_tokens": max(1, len(text) // 4),
                       "prompt_tokens_details": {"cached_tokens": cached // 16 * 16}}},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format: str, *args: object) -> None:
        pass


def serve() -> ThreadingHTTPServer:
    FakeVLLM.seen_prefixes = set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
