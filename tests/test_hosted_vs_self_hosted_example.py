"""End-to-end tests for the hosted-API versus self-hosted vLLM example."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from runeconomy.cli import main


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "hosted_vs_self_hosted"
AGENT = EXAMPLE / "agent.py"
SECRET_ANSWER_TEXT = "Parcelway"


def _answer_for(messages: list[dict[str, Any]]) -> str:
    """Behave like a perfect model: plan the lookup, then copy the field."""

    last = str(messages[-1].get("content", ""))
    question = str(messages[1].get("content", ""))
    if last.startswith("Tool lookup_order returned:"):
        record = json.loads(last.splitlines()[1])
        field = re.search(r"Requested field: (\w+)", question).group(1)
        return str(record[field])
    order_id = re.search(r"\b[A-Z]-\d{4}\b", question).group(0)
    return json.dumps({"order_id": order_id})


class _BaseHandler(BaseHTTPRequestHandler):
    paths: list[str] = []
    authorizations: list[str | None] = []
    delay_seconds = 0.0

    def _sse(self, events: list[dict[str, Any]]) -> None:
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        encoded = (body + "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _json(self, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict[str, Any]:
        type(self).paths.append(self.path)
        type(self).authorizations.append(self.headers.get("Authorization"))
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def log_message(self, format: str, *args: object) -> None:
        pass


class _HostedHandler(_BaseHandler):
    """A hosted API: Chat Completions only, no tokenizer or server metrics."""

    paths: list[str] = []
    authorizations: list[str | None] = []
    delay_seconds = 0.0
    status_code = 200
    models = ["fake-hosted-model-2026-01-01"]
    fingerprints = ["fp_test"]
    request_count = 0

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        payload = self._payload()
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if self.status_code != 200:
            self.send_error(self.status_code)
            return
        if "cache_salt" in payload or not payload.get("stream"):
            self.send_error(400)
            return
        cls = type(self)
        model = cls.models[cls.request_count % len(cls.models)]
        fingerprint = cls.fingerprints[cls.request_count % len(cls.fingerprints)]
        cls.request_count += 1
        time.sleep(self.delay_seconds)
        common = {"model": model, "system_fingerprint": fingerprint}
        self._sse(
            [
                {
                    **common,
                    "choices": [
                        {
                            "delta": {"content": _answer_for(payload["messages"])},
                            "finish_reason": None,
                        }
                    ],
                },
                {**common, "choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    **common,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 6,
                        "prompt_tokens_details": {"cached_tokens": 64},
                    },
                },
            ]
        )


class _VLLMHandler(_BaseHandler):
    """A vLLM server: /tokenize plus Chat Completions with server timing."""

    paths: list[str] = []
    authorizations: list[str | None] = []
    delay_seconds = 0.0

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        payload = self._payload()
        if self.path == "/v1/chat/completions/render":
            self.send_error(404)
            return
        if self.path == "/tokenize":
            self._json({"count": 5, "tokens": [1, 2, 3, 4, 5]})
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        time.sleep(self.delay_seconds)
        self._sse(
            [
                {
                    "id": "chat-1",
                    "choices": [
                        {
                            "delta": {"content": _answer_for(payload["messages"])},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chat-1",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 6,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    },
                    "metrics": {"time_to_first_token_ms": 3.0, "queue_time_ms": 0.4},
                },
            ]
        )


def _serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class HostedVersusSelfHostedExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        for handler in (_HostedHandler, _VLLMHandler):
            handler.paths = []
            handler.authorizations = []
            handler.delay_seconds = 0.0
        _HostedHandler.status_code = 200
        _HostedHandler.models = ["fake-hosted-model-2026-01-01"]
        _HostedHandler.fingerprints = ["fp_test"]
        _HostedHandler.request_count = 0
        self.hosted = _serve(_HostedHandler)
        self.vllm = _serve(_VLLMHandler)
        self.environment = {
            "HOSTED_BASE_URL": f"http://127.0.0.1:{self.hosted.server_port}/v1",
            "HOSTED_API_KEY": "hosted-test-key",
            "HOSTED_MODEL": "fake-hosted-model",
            "HOSTED_PROVIDER": "fake-hosted",
            "SELF_HOSTED_BASE_URL": f"http://127.0.0.1:{self.vllm.server_port}/v1",
            "SELF_HOSTED_MODEL": "example/open-weight-model",
            "SELF_HOSTED_MODEL_REVISION": "0123456789abcdef",
            "SELF_HOSTED_BACKEND_VERSION": "0.0.0-test",
            "SELF_HOSTED_CHAT_TEMPLATE_DIGEST": "sha256:" + "0" * 64,
        }

    def tearDown(self) -> None:
        for server in (self.hosted, self.vllm):
            server.shutdown()
            server.server_close()

    def _run(
        self,
        directory: Path,
        *,
        environment: dict[str, str] | None = None,
        tasks: int | None = None,
        min_identity_grade: str | None = "provider-declared",
    ) -> dict[str, Any]:
        dataset = EXAMPLE / "tasks.json"
        if tasks is not None:
            payload = json.loads(dataset.read_text(encoding="utf-8"))
            dataset = directory / "tasks.json"
            dataset.write_text(
                json.dumps({"tasks": payload["tasks"][:tasks]}), encoding="utf-8"
            )
        workspace = directory / "experiment"
        results = directory / "results"
        quiet = contextlib.redirect_stdout(io.StringIO())
        with patch.dict(os.environ, environment or self.environment), quiet:
            commands = [
                [
                    "experiment", "init", "--mode", "agent-process",
                    "--baseline", f"{sys.executable} {AGENT} --arm hosted",
                    "--candidate", f"{sys.executable} {AGENT} --arm self-hosted",
                    "--dataset", str(dataset),
                    "--evaluator", str(EXAMPLE / "evaluator.py"),
                    "--optimization-target", "engine-change-latency",
                    "--latency-semantics", "superiority",
                    "--min-task-success-rate", "0.5",
                    "--max-task-success-regression", "0.5",
                    "--order", "alternating",
                    "--timeout", "60",
                    "--output", str(workspace),
                    *(
                        []
                        if min_identity_grade is None
                        else ["--min-identity-grade", min_identity_grade]
                    ),
                ],
                ["experiment", "validate", str(workspace)],
                ["experiment", "run", str(workspace), "--output", str(results)],
            ]
            for command in commands:
                self.assertEqual(main(command), 0, command)
        return json.loads((results / "result.json").read_text(encoding="utf-8"))

    def _attempts(self, result: dict[str, Any], variant: str) -> list[dict[str, Any]]:
        return [item for item in result["attempts"] if item["variant"] == variant]

    def _check(self, result: dict[str, Any], name: str) -> dict[str, Any]:
        return next(
            check for check in result["decision"]["checks"] if check["name"] == name
        )

    def test_hosted_baseline_versus_vllm_candidate_produces_valid_decision(
        self,
    ) -> None:
        _HostedHandler.delay_seconds = 0.15
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            result = self._run(directory)
            self.assertEqual(result["decision"]["status"], "verified")
            self.assertTrue(
                all(item["status"] == "completed" for item in result["attempts"])
            )
            self.assertTrue(all(item["task_success"] for item in result["attempts"]))
            self.assertTrue(
                self._check(result, "intentional_backend_identity_difference")["passed"]
            )
            self.assertTrue(
                self._check(result, "trace_and_environment_identity")["passed"]
            )

            hosted = self._attempts(result, "baseline")
            self_hosted = self._attempts(result, "candidate")
            self.assertEqual(len(hosted), 6)
            self.assertEqual(len(self_hosted), 6)
            for attempt in hosted:
                metrics = attempt["metrics"]
                # Server-side metrics are not observable on a hosted API.
                self.assertIsNone(metrics["server_ttft_ms"])
                self.assertIsNone(metrics["queue_ms"])
                self.assertIsNone(metrics["prefill_ms"])
                self.assertIsNone(metrics["decode_ms"])
                self.assertIsNone(metrics["cost_usd"])
                self.assertIsNone(metrics["gpu_time_ms"])
                # Client timing and provider-reported usage are observed.
                self.assertIsNotNone(metrics["client_ttft_ms"])
                self.assertEqual(metrics["input_tokens"], 240)
                self.assertEqual(metrics["cached_input_tokens"], 128)
                self.assertEqual(metrics["dependency_evidence"], "complete_explicit")
                quality = metrics["evidence_quality"]["fields"]
                self.assertEqual(quality["server_ttft"]["grade"], "unresolved")
                self.assertEqual(quality["prompt_tokens"]["grade"], "reliably_derived")
                identity = metrics["environment_identity"][0]
                self.assertEqual(identity["backend"], "fake-hosted")
                self.assertEqual(identity["model"], "fake-hosted-model-2026-01-01")
                self.assertEqual(
                    identity["model_revision"], "fake-hosted-model-2026-01-01"
                )
                # Undisclosed fields are null and declared, never placeholders.
                for field in (
                    "backend_version",
                    "tokenizer",
                    "tokenizer_revision",
                    "chat_template_digest",
                ):
                    self.assertIsNone(identity[field])
                self.assertFalse(metrics["environment_identity_complete"])
                declaration = metrics["identity_declarations"][0]
                self.assertEqual(declaration["grade"], "provider_declared")
                self.assertTrue(
                    declaration["provider_fingerprint_sha256"].startswith("sha256:")
                )
            for attempt in self_hosted:
                metrics = attempt["metrics"]
                self.assertEqual(metrics["server_ttft_ms"], 6.0)
                self.assertEqual(metrics["queue_ms"], 0.8)
                quality = metrics["evidence_quality"]["fields"]
                self.assertEqual(quality["prompt_tokens"]["grade"], "directly_observed")
                self.assertEqual(metrics["environment_identity"][0]["backend"], "vllm")

            assurance = result["decision"]["identity_assurance"]
            self.assertEqual(
                assurance["grades"],
                {"baseline": "provider_declared", "candidate": "operator_verified"},
            )
            self.assertEqual(assurance["claim_scope"], "observed_service_window")
            self.assertEqual(
                assurance["variants"]["baseline"]["undisclosed_fields"],
                [
                    "backend_version",
                    "chat_template_digest",
                    "tokenizer",
                    "tokenizer_revision",
                ],
            )
            self.assertEqual(
                [limit["code"] for limit in assurance["claim_limits"]],
                ["identity.provider_declared", "identity.undisclosed_fields"],
            )
            self.assertEqual(
                assurance["variants"]["baseline"]["provider_fingerprint"],
                {
                    "request_count": 12,
                    "reported_count": 12,
                    "distinct_count": 1,
                    "changed_during_run": False,
                    "affects_identity": False,
                    "affects_validity": False,
                },
            )
            self.assertEqual(
                assurance["variants"]["candidate"]["provider_fingerprint"][
                    "reported_count"
                ],
                0,
            )
            self.assertEqual(
                result["design"]["acceptance"]["identity_policy"],
                {"schema_version": "0.1", "minimum_grade": "provider_declared"},
            )
            self.assertNotIn("fp_test", json.dumps(result))

            # The hosted arm never touches vLLM-only endpoints; credentials
            # are sent only to the arm they belong to.
            self.assertEqual(set(_HostedHandler.paths), {"/v1/chat/completions"})
            self.assertEqual(
                set(_HostedHandler.authorizations), {"Bearer hosted-test-key"}
            )
            self.assertIn("/tokenize", _VLLMHandler.paths)
            self.assertEqual(set(_VLLMHandler.authorizations), {None})

            # Generated answers stay in private attempt artifacts.
            self.assertNotIn(SECRET_ANSWER_TEXT, json.dumps(result))

            report = directory / "report.html"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "experiment", "report",
                            str(directory / "results" / "result.json"),
                            "--output", str(report),
                        ]
                    ),
                    0,
                )
            html = report.read_text(encoding="utf-8")
            self.assertIn("Verified", html)
            self.assertIn("Identity assurance", html)
            self.assertIn("provider_declared", html)

    def test_fingerprint_drift_is_reported_but_never_invalidates(self) -> None:
        _HostedHandler.delay_seconds = 0.15
        _HostedHandler.fingerprints = ["fp_a", "fp_b", "fp_c"]
        with TemporaryDirectory() as raw:
            result = self._run(Path(raw))
        self.assertNotEqual(result["decision"]["status"], "invalid")
        self.assertTrue(
            self._check(result, "trace_and_environment_identity")["passed"]
        )
        assurance = result["decision"]["identity_assurance"]
        fingerprint = assurance["variants"]["baseline"]["provider_fingerprint"]
        self.assertEqual(fingerprint["distinct_count"], 3)
        self.assertTrue(fingerprint["changed_during_run"])
        self.assertIn(
            "identity.provider_fingerprint_changed",
            [limit["code"] for limit in assurance["claim_limits"]],
        )
        for value in ("fp_a", "fp_b", "fp_c"):
            self.assertNotIn(value, json.dumps(result))

    def test_legacy_contract_without_policy_rejects_undisclosed_identity(
        self,
    ) -> None:
        with TemporaryDirectory() as raw:
            result = self._run(Path(raw), tasks=2, min_identity_grade=None)
        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertNotIn("identity_assurance", result["decision"])
        self.assertNotIn("identity_policy", result["design"]["acceptance"])
        self.assertFalse(
            self._check(result, "trace_and_environment_identity")["passed"]
        )

    def test_operator_verified_minimum_rejects_hosted_arm(self) -> None:
        with TemporaryDirectory() as raw:
            result = self._run(
                Path(raw), tasks=2, min_identity_grade="operator-verified"
            )
        self.assertEqual(result["decision"]["status"], "invalid")
        assurance = result["decision"]["identity_assurance"]
        self.assertFalse(assurance["variants"]["baseline"]["meets_minimum_grade"])
        self.assertTrue(assurance["variants"]["candidate"]["valid"])

    def test_missing_self_hosted_identity_is_invalid_not_guessed(self) -> None:
        environment = dict(self.environment)
        del environment["SELF_HOSTED_CHAT_TEMPLATE_DIGEST"]
        with TemporaryDirectory() as raw:
            result = self._run(Path(raw), environment=environment, tasks=2)
        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertFalse(
            self._check(result, "trace_and_environment_identity")["passed"]
        )
        for attempt in self._attempts(result, "candidate"):
            self.assertFalse(attempt["metrics"]["environment_identity_complete"])

    def test_hosted_model_drift_within_variant_is_invalid(self) -> None:
        _HostedHandler.models = [
            "fake-hosted-model-2026-01-01",
            "fake-hosted-model-2026-02-01",
        ]
        with TemporaryDirectory() as raw:
            result = self._run(Path(raw), tasks=2)
        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertFalse(
            self._check(result, "trace_and_environment_identity")["passed"]
        )

    def test_hosted_outage_is_infrastructure_failure(self) -> None:
        _HostedHandler.status_code = 503
        with TemporaryDirectory() as raw:
            result = self._run(Path(raw), tasks=2)
        self.assertEqual(result["decision"]["status"], "invalid")
        for attempt in self._attempts(result, "baseline"):
            self.assertEqual(attempt["status"], "infrastructure_failure")
            self.assertEqual(
                attempt["agent_status"]["provenance"],
                {
                    "component": "infrastructure",
                    "stage": "model_request",
                    "code": "hosted_request_failed",
                },
            )
        for attempt in self._attempts(result, "candidate"):
            self.assertEqual(attempt["status"], "completed")


if __name__ == "__main__":
    unittest.main()
