"""Installed, deterministic tutorial fixture and local vLLM-shaped server."""

from __future__ import annotations

import errno
import hashlib
import json
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from . import __version__


TUTORIAL_SCHEMA_VERSION = "0.1"
DEFAULT_TUTORIAL_HOST = "127.0.0.1"
DEFAULT_TUTORIAL_PORT = 8765


class TutorialError(ValueError):
    """Raised when a tutorial cannot be materialized or served safely."""


def _validate_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise TutorialError("tutorial port must be between 1 and 65535")


def _fixture_payload(port: int) -> dict[str, Any]:
    _validate_port(port)
    resource = files("runeconomy").joinpath("data", "tutorial-evidence.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    context = payload["runeconomy"]["experiment_context"]
    context["backend"]["base_url"] = f"http://{DEFAULT_TUTORIAL_HOST}:{port}"
    return payload


def _tutorial_readme(directory_name: str, port: int) -> str:
    serve_command = "runeconomy tutorial serve"
    if port != DEFAULT_TUTORIAL_PORT:
        serve_command += f" --port {port}"
    return f"""# RunEconomy local tutorial

Tutorial schema: `{TUTORIAL_SCHEMA_VERSION}`

This deterministic, local tutorial validates RunEconomy's product plumbing. It
does not run a model, measure real inference performance, or provide a complete
vLLM implementation.

The fixture is a RunEconomy-enriched OpenInference artifact. It includes
explicit prompt tokens, prompt segments, workflow dependencies, and experiment
context for this controlled tutorial. Ordinary OpenTelemetry or OpenInference
evidence may support diagnosis while still producing an unresolved experiment
workspace.

In a separate terminal, start the localhost-only mock server:

```bash
{serve_command}
```

From the directory that contains `{directory_name}`, run:

```bash
runeconomy import {directory_name}/evidence.json \
  --output evidence.json

runeconomy experiment init \
  --from-evidence evidence.json \
  --output experiment/

runeconomy experiment validate experiment/

runeconomy experiment run experiment/ \
  --output results/

runeconomy experiment report results/result.json \
  --output report.html \
  --open
```

The report is generated from the newly executed controlled mock trials. It is
not a relabelled historical trace and must not be presented as real-vLLM
validation or a performance benchmark.
"""


def materialize_tutorial(output: Path, *, port: int = DEFAULT_TUTORIAL_PORT) -> list[Path]:
    """Create a new sanitized tutorial directory without overwriting files."""

    _validate_port(port)
    if output.exists():
        raise TutorialError(
            f"tutorial output already exists: {output}. "
            "Choose a new directory. RunEconomy will not overwrite tutorial files."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".runeconomy-tmp")
    if temporary.exists():
        raise TutorialError(
            f"temporary tutorial path already exists: {temporary}. Remove it after review or choose a new output directory."
        )
    payload = _fixture_payload(port)
    evidence = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    metadata = {
        "artifact_type": "runeconomy.tutorial",
        "schema_version": TUTORIAL_SCHEMA_VERSION,
        "runeconomy_version": __version__,
        "host": DEFAULT_TUTORIAL_HOST,
        "port": port,
        "evidence_sha256": "sha256:"
        + hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
        "claim_boundary": "Deterministic product-plumbing tutorial, not real inference or a performance benchmark.",
    }
    try:
        temporary.mkdir()
        (temporary / "evidence.json").write_text(evidence, encoding="utf-8")
        (temporary / "README.md").write_text(
            _tutorial_readme(output.name, port), encoding="utf-8"
        )
        (temporary / "tutorial.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except OSError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise TutorialError(f"cannot create tutorial at {output}: {exc}") from exc
    return [output / "evidence.json", output / "README.md", output / "tutorial.json"]


def _candidate(payload: dict[str, Any]) -> bool:
    messages = payload.get("messages", [])
    return bool(
        isinstance(messages, list)
        and messages
        and isinstance(messages[0], dict)
        and messages[0].get("content") == "Stable instructions"
    )


def _dynamic_token(payload: dict[str, Any]) -> int:
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


class TutorialRequestHandler(BaseHTTPRequestHandler):
    """Narrow deterministic HTTP behavior needed by the tutorial only."""

    server_version = "RunEconomyTutorialMock/0.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        return value if isinstance(value, dict) else {}

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json(
                200,
                {
                    "status": "ok",
                    "tutorial_schema_version": TUTORIAL_SCHEMA_VERSION,
                },
            )
            return
        self._json(404, {"error": "not found"})

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
            tokens = (
                list(range(10, 18)) + [dynamic]
                if candidate
                else [dynamic] + list(range(10, 18))
            )
            self._json(200, {"tokens": tokens})
            return
        if self.path == "/v1/chat/completions":
            time.sleep(0.001 if candidate else 0.008)
            events = [
                {
                    "id": "tutorial-response",
                    "choices": [
                        {"delta": {"content": "PASS"}, "finish_reason": None}
                    ],
                    "metrics": {
                        "time_to_first_token_ms": 2.0 if candidate else 8.0,
                        "queue_time_ms": 0.1,
                        "generation_time_ms": 1.0,
                    },
                },
                {
                    "id": "tutorial-response",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {
                            "cached_tokens": 8 if candidate else 0
                        },
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


def create_tutorial_server(
    *, port: int = DEFAULT_TUTORIAL_PORT
) -> ThreadingHTTPServer:
    """Create the localhost-only tutorial server or raise a helpful error."""

    _validate_port(port)
    try:
        return ThreadingHTTPServer(
            (DEFAULT_TUTORIAL_HOST, port), TutorialRequestHandler
        )
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise TutorialError(
                f"cannot start the tutorial server on {DEFAULT_TUTORIAL_HOST}:{port}: address already in use. "
                "Choose another port, then create a matching tutorial with "
                f"`runeconomy tutorial init ./runeconomy-tutorial-{port + 1} --port {port + 1}`."
            ) from exc
        raise TutorialError(
            f"cannot start the tutorial server on {DEFAULT_TUTORIAL_HOST}:{port}: {exc}"
        ) from exc


def serve_tutorial(*, port: int = DEFAULT_TUTORIAL_PORT) -> None:
    """Serve the deterministic tutorial until interrupted."""

    server = create_tutorial_server(port=port)
    endpoint = f"http://{DEFAULT_TUTORIAL_HOST}:{port}"
    print("RunEconomy tutorial server · Deterministic local mock")
    print(f"Endpoint: {endpoint}")
    print("This server is not vLLM and does not run a model or benchmark inference.")
    print(
        "What to do next: in another terminal, run "
        "`runeconomy import runeconomy-tutorial/evidence.json --output evidence.json`."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRunEconomy tutorial server stopped.")
    finally:
        server.server_close()
