"""Backend adapter protocol: vLLM port, SGLang and OpenAI-compatible APIs."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from runeconomy.adapters import (
    BACKEND_KINDS,
    BackendAdapter,
    BackendRequestError,
    CapabilityUnavailableError,
    OpenAICompatAdapter,
    SGLangAdapter,
    VLLMAdapter,
    create_adapter,
    profile_for,
)
from runeconomy.adapters.prometheus import SGLANG_METRICS, VLLM_METRICS, select_metrics
from runeconomy.cli import main
from runeconomy.experiments import ExperimentManifest, ExperimentValidationError
from runeconomy.vllm import CapturedResponse


ROOT = Path(__file__).parents[1]


def _sse(handler: BaseHTTPRequestHandler, events: list[dict[str, Any]]) -> None:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    encoded = (body + "data: [DONE]\n\n").encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _json(handler: BaseHTTPRequestHandler, value: Any, status: int = 200) -> None:
    body = json.dumps(value).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text(handler: BaseHTTPRequestHandler, value: str, status: int = 200) -> None:
    body = value.encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _Recording(BaseHTTPRequestHandler):
    calls: list[tuple[str, str]] = []

    def _record(self) -> dict[str, Any]:
        type(self).calls.append((self.command, self.path))
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def log_message(self, format: str, *args: object) -> None:
        pass


SGLANG_METRICS_TEXT = """# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="org/model"} 2
sglang:num_queue_reqs{model_name="org/model"} 5
sglang:token_usage{model_name="org/model"} 0.25
sglang:cache_hit_rate{model_name="org/model"} 0.6
sglang:gen_throughput{model_name="org/model"} 123.5
"""


class _SGLangHandler(_Recording):
    calls: list[tuple[str, str]] = []
    flush_status = 200

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/openapi.json":
            _json(
                self,
                {
                    "paths": {
                        "/v1/chat/completions": {},
                        "/tokenize": {},
                        "/flush_cache": {},
                        "/get_model_info": {},
                    }
                },
            )
        elif self.path == "/get_model_info":
            _json(self, {"model_path": "org/model", "tokenizer_path": "org/model", "is_generation": True})
        elif self.path == "/get_server_info":
            _json(
                self,
                {
                    "version": "0.5.0",
                    "enable_cache_report": True,
                    "enable_metrics": True,
                    "disable_radix_cache": False,
                },
            )
        elif self.path == "/v1/models":
            _json(self, {"data": [{"id": "org/model"}]})
        elif self.path == "/metrics":
            _text(self, SGLANG_METRICS_TEXT)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._record()
        if self.path == "/tokenize":
            self.assertion_payload = payload
            _json(self, {"tokens": [7, 8, 9, 10], "count": 4})
        elif self.path == "/flush_cache":
            _text(self, "Cache flushed.", type(self).flush_status)
        elif self.path == "/v1/chat/completions":
            if "cache_salt" in payload:
                self.send_error(400)
                return
            _sse(
                self,
                [
                    {"id": "c1", "choices": [{"delta": {"content": "rollback"}, "finish_reason": None}]},
                    {
                        "id": "c1",
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 2},
                        },
                    },
                ],
            )
        else:
            self.send_error(404)


class _HostedHandler(_Recording):
    calls: list[tuple[str, str]] = []
    models = ["hosted-model-2026-01-01"]
    fingerprints = ["fp_one"]
    count = 0

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/v1/models":
            _json(self, {"data": [{"id": "hosted-model-2026-01-01"}, {"id": "other"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        payload = self._record()
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not payload.get("stream") or payload.get("stream_options") != {"include_usage": True}:
            self.send_error(400)
            return
        cls = type(self)
        model = cls.models[cls.count % len(cls.models)]
        fingerprint = cls.fingerprints[cls.count % len(cls.fingerprints)]
        cls.count += 1
        common = {"id": "h1", "model": model, "system_fingerprint": fingerprint}
        _sse(
            self,
            [
                {**common, "choices": [{"delta": {"content": "rollback"}, "finish_reason": None}]},
                {**common, "choices": [{"delta": {}, "finish_reason": "stop"}]},
                {
                    **common,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 3,
                        "prompt_tokens_details": {"cached_tokens": 32},
                    },
                },
            ],
        )


def _serve(handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _ServerCase(unittest.TestCase):
    handler: type[_Recording]

    def setUp(self) -> None:
        self.handler.calls = []
        self.server = _serve(self.handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class ProtocolTests(unittest.TestCase):
    def test_every_kind_implements_the_protocol(self) -> None:
        self.assertEqual(BACKEND_KINDS, ("vllm", "sglang", "openai_compat"))
        for kind in BACKEND_KINDS:
            adapter = create_adapter(kind, "http://127.0.0.1:9", api_dialect="chat_completions")
            self.assertIsInstance(adapter, BackendAdapter)
            self.assertEqual(adapter.kind, kind)
            self.assertEqual(adapter.profile, profile_for(kind))
        with self.assertRaises(ValueError):
            profile_for("tgi")

    def test_vllm_adapter_delegates_capture_and_reset_unchanged(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[Any] = []

            def capture(self, request: Any) -> str:
                self.calls.append(("capture", request))
                return "captured"

            def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
                self.calls.append(("reset", reset_external))

        client = FakeClient()
        adapter = VLLMAdapter("http://127.0.0.1:9", client=client)  # type: ignore[arg-type]
        self.assertEqual(adapter.capture({"x": 1}), "captured")
        adapter.reset_prefix_cache(reset_external=True)
        self.assertEqual(client.calls, [("capture", {"x": 1}), ("reset", True)])
        self.assertIsNone(adapter.capability_record())
        self.assertTrue(adapter.supports_cache_reset)
        # The static runner reads no live identity from the vLLM adapter, as before.
        self.assertFalse(hasattr(adapter, "environment_identity"))


class PrometheusTests(unittest.TestCase):
    def test_known_metrics_are_selected_and_labels_dropped(self) -> None:
        selected = select_metrics(SGLANG_METRICS_TEXT, SGLANG_METRICS)
        self.assertEqual(selected["waiting_requests"], 5.0)
        self.assertEqual(selected["prefix_cache_hit_rate"], 0.6)
        self.assertNotIn("org/model", json.dumps(selected))
        vllm = select_metrics(
            "vllm:num_requests_waiting 3\nvllm:prefix_cache_hits_total 30\n"
            "vllm:prefix_cache_queries_total 120\nvllm:kv_cache_usage_perc 0.5\n",
            VLLM_METRICS,
        )
        self.assertEqual(vllm["waiting_requests"], 3.0)
        self.assertEqual(vllm["prefix_cache_hit_rate"], 0.25)
        self.assertIsNone(vllm["preemptions_total"])


class SGLangAdapterTests(_ServerCase):
    handler = _SGLangHandler

    def setUp(self) -> None:
        _SGLangHandler.flush_status = 200
        super().setUp()

    def test_discovery_reports_capability_table_and_identity(self) -> None:
        report = SGLangAdapter(self.url).discover()
        self.assertEqual(report["schema_version"], "0.3")
        self.assertEqual(report["backend_kind"], "sglang")
        self.assertEqual(report["backend_version"], "0.5.0")
        statuses = {item["name"]: item["status"] for item in report["capability_table"]["capabilities"]}
        self.assertEqual(
            statuses,
            {
                "chat_completions": "supported",
                "responses": "unsupported",
                "exact_prompt_tokens": "supported",
                "cache_telemetry": "supported",
                "server_timing": "unsupported",
                "cache_reset": "supported",
                "cache_salt": "unsupported",
                "model_identity": "supported",
                "prometheus_metrics": "supported",
            },
        )
        self.assertEqual(report["identity"]["grade"], "operator_verified")
        self.assertEqual(report["identity"]["model_path"], "org/model")
        self.assertEqual(report["readiness"]["status"], "ready")
        self.assertFalse(report["probe_safety"]["cache_reset_invoked"])
        self.assertNotIn(("POST", "/flush_cache"), _SGLangHandler.calls)
        for key in ("chat_completions", "tokenize", "cache_reset", "responses"):
            self.assertIn("status", report[key])

    def test_capture_uses_tokenize_for_exact_tokens_and_no_server_timing(self) -> None:
        adapter = SGLangAdapter(self.url)
        captured = adapter.capture(
            {"model": "org/model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        )
        self.assertEqual(captured.prompt_token_ids, (7, 8, 9, 10))
        self.assertEqual(captured.cached_tokens, 2)
        self.assertIsNone(captured.server_ttft_ms)
        self.assertIsNotNone(captured.client_ttft_ms)
        paths = [path for _, path in _SGLangHandler.calls]
        self.assertIn("/tokenize", paths)
        self.assertNotIn("/v1/chat/completions/render", paths)
        with self.assertRaises(CapabilityUnavailableError):
            adapter.capture({"model": "org/model", "messages": [], "cache_salt": "x"})

    def test_flush_cache_reset_and_metrics(self) -> None:
        adapter = SGLangAdapter(self.url)
        adapter.reset_prefix_cache()
        self.assertIn(("POST", "/flush_cache"), _SGLangHandler.calls)
        with self.assertRaises(CapabilityUnavailableError):
            adapter.reset_prefix_cache(reset_external=True)
        _SGLangHandler.flush_status = 400
        with self.assertRaises(BackendRequestError):
            adapter.reset_prefix_cache()
        metrics = adapter.scrape_metrics()
        self.assertEqual(metrics["running_requests"], 2.0)

    def test_doctor_cli_json_and_human_output(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["doctor", "--backend", "sglang", "--base-url", self.url, "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["output_schema"], "runeconomy.cli.doctor.v1")
        self.assertEqual(payload["capability_table"]["backend_kind"], "sglang")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            main(["doctor", "--backend", "sglang", "--base-url", self.url])
        text = stdout.getvalue()
        self.assertIn("Backend kind: sglang", text)
        self.assertIn("Downgraded capabilities", text)
        self.assertIn("Server timing", text)


class OpenAICompatAdapterTests(_ServerCase):
    handler = _HostedHandler

    def setUp(self) -> None:
        _HostedHandler.models = ["hosted-model-2026-01-01"]
        _HostedHandler.fingerprints = ["fp_one"]
        _HostedHandler.count = 0
        super().setUp()

    def test_discovery_is_honest_about_what_a_hosted_api_shows(self) -> None:
        report = OpenAICompatAdapter(f"{self.url}/v1", provider="openai").discover()
        statuses = {item["name"]: item["status"] for item in report["capability_table"]["capabilities"]}
        self.assertEqual(statuses["chat_completions"], "unknown")
        self.assertEqual(statuses["exact_prompt_tokens"], "unsupported")
        self.assertEqual(statuses["server_timing"], "unsupported")
        self.assertEqual(statuses["cache_reset"], "unsupported")
        self.assertEqual(statuses["model_identity"], "supported")
        self.assertEqual(report["identity"]["grade"], "provider_declared")
        self.assertEqual(report["model_ids"], ["hosted-model-2026-01-01", "other"])
        self.assertEqual(report["readiness"]["supported_modes"], ["warm-cache"])
        self.assertEqual(_HostedHandler.calls, [("GET", "/v1/models")])

    def test_capture_records_provider_declarations_and_unknown_tokens(self) -> None:
        adapter = OpenAICompatAdapter(f"{self.url}/v1", provider="openai")
        captured = adapter.capture(
            {"model": "hosted-model-2026-01-01", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertIsInstance(captured, CapturedResponse)
        self.assertIsNone(captured.prompt_token_ids)
        self.assertEqual(captured.response_model, "hosted-model-2026-01-01")
        self.assertTrue(captured.provider_fingerprint_sha256.startswith("sha256:"))
        self.assertEqual(captured.cached_tokens, 32)
        record = captured.to_dict()
        self.assertIsNone(record["prompt_token_ids"])
        self.assertNotIn("fp_one", json.dumps(record))
        self.assertNotIn("output_text", record)
        self.assertEqual([path for _, path in _HostedHandler.calls], ["/v1/chat/completions"])
        self.assertFalse(adapter.supports_cache_reset)
        with self.assertRaises(CapabilityUnavailableError):
            adapter.reset_prefix_cache()

    def test_http_failure_is_a_backend_request_error(self) -> None:
        adapter = OpenAICompatAdapter(f"{self.url}/missing", provider="openai")
        with self.assertRaises(BackendRequestError):
            adapter.capture({"model": "m", "messages": []})


def _static_manifest(
    *, kind: str, base_url: str, **overrides: Any
) -> dict[str, Any]:
    backend: dict[str, Any] = {
        "kind": kind,
        "base_url": base_url,
        "model": "hosted-model-2026-01-01" if kind == "openai_compat" else "org/model",
        "block_size": 1,
        "api_dialect": "chat_completions",
    }
    acceptance: dict[str, Any] = {
        "primary_metric": "total_latency_ms",
        "max_task_success_regression": 1.0,
        "min_task_success_rate": 0.0,
        "min_completed_rate": 1.0,
        "require_latency_improvement": False,
    }
    if kind == "openai_compat":
        backend.update(
            {
                "provider": "openai",
                "model_revision": "hosted-model-2026-01-01",
                "undisclosed_identity": [
                    "backend_version",
                    "chat_template_digest",
                    "engine_config",
                    "hardware",
                    "tokenizer_revision",
                ],
            }
        )
        acceptance["identity_policy"] = {
            "schema_version": "0.1",
            "minimum_grade": "provider_declared",
        }
    else:
        backend.update(
            {
                "model_revision": "abc123",
                "tokenizer_revision": "abc123",
                "chat_template_digest": "sha256:" + "2" * 64,
                "backend_version": "0.5.0",
                "hardware": "1x test GPU",
                "engine_config": {"disable_radix_cache": False, "page_size": 1},
            }
        )
    manifest = {
        "schema_version": "0.3",
        "experiment_id": f"{kind}-static-test",
        "backend": backend,
        "sampling": {"temperature": 0, "max_tokens": 8},
        "execution": {
            "repetitions": 2,
            "order": "alternating",
            "seed": 3,
            "cache_treatment": "warm",
        },
        "acceptance": acceptance,
        "cases": [
            {
                "case_id": "triage-001",
                "evaluator": {"kind": "regex", "pattern": ".*rollback.*", "full_match": True},
                "variants": {
                    "baseline": {
                        "request": {"messages": [{"role": "user", "content": "T {{RUNECONOMY_TRIAL_ID}} what first?"}]}
                    },
                    "candidate": {
                        "request": {"messages": [{"role": "user", "content": "what first? T {{RUNECONOMY_TRIAL_ID}}"}]}
                    },
                },
            }
        ],
    }
    for path, value in overrides.items():
        target = manifest
        *parents, leaf = path.split(".")
        for name in parents:
            target = target[name]
        if value is None:
            target.pop(leaf, None)
        else:
            target[leaf] = value
    return manifest


class StaticManifestValidationTests(unittest.TestCase):
    def _invalid(self, manifest: dict[str, Any], message: str) -> None:
        with self.assertRaises(ExperimentValidationError) as raised:
            ExperimentManifest.from_dict(manifest)
        self.assertIn(message, str(raised.exception))

    def test_adapter_kinds_parse_with_profiles(self) -> None:
        hosted = ExperimentManifest.from_dict(_static_manifest(kind="openai_compat", base_url="https://api.example.invalid/v1"))
        self.assertEqual(hosted.backend.kind, "openai_compat")
        self.assertIsNone(hosted.backend.tokenizer_revision)
        self.assertEqual(hosted.identity_policy["minimum_grade"], "provider_declared")
        resolved = hosted.to_resolved_dict()
        self.assertEqual(resolved["backend"]["kind"], "openai_compat")
        self.assertEqual(resolved["acceptance"]["identity_policy"]["minimum_grade"], "provider_declared")
        self.assertEqual(ExperimentManifest.from_dict(resolved).fingerprint(), hosted.fingerprint())
        sglang = ExperimentManifest.from_dict(_static_manifest(kind="sglang", base_url="http://127.0.0.1:9"))
        self.assertEqual(sglang.backend.kind, "sglang")
        self.assertNotEqual(sglang.fingerprint(), hosted.fingerprint())

    def test_legacy_vllm_manifest_shape_and_fingerprint_are_unchanged(self) -> None:
        path = ROOT / "examples" / "experiments" / "timestamp-placement.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = ExperimentManifest.from_dict(payload)
        resolved = manifest.to_resolved_dict()
        self.assertEqual(list(resolved["backend"])[0], "kind")
        for key in ("provider", "undisclosed_identity"):
            self.assertNotIn(key, resolved["backend"])
        self.assertNotIn("identity_policy", resolved["acceptance"])
        self.assertNotIn("kind", manifest._fingerprint_payload()["backend"])
        self.assertNotIn("identity_policy", manifest._fingerprint_payload())

    def test_capabilities_and_identity_rules_are_enforced(self) -> None:
        url = "https://api.example.invalid/v1"
        self._invalid(
            _static_manifest(kind="openai_compat", base_url=url, **{"acceptance.identity_policy": None}),
            "declare acceptance.identity_policy",
        )
        self._invalid(
            _static_manifest(
                kind="openai_compat",
                base_url=url,
                **{"acceptance.identity_policy": {"schema_version": "0.1", "minimum_grade": "operator_verified"}},
            ),
            "can never be met",
        )
        self._invalid(
            _static_manifest(
                kind="openai_compat",
                base_url=url,
                **{"execution.cache_treatment": "cold", "execution.cache_isolation": "physical_reset"},
            ),
            "unavailable on openai_compat",
        )
        self._invalid(
            _static_manifest(kind="openai_compat", base_url=url, **{"backend.tokenizer_revision": "x"}),
            "listed as undisclosed",
        )
        self._invalid(
            _static_manifest(kind="openai_compat", base_url=url, **{"backend.undisclosed_identity": ["model"]}),
            "must always be declared",
        )
        self._invalid(
            _static_manifest(kind="openai_compat", base_url=url, **{"backend.model_revision": "provider-managed"}),
            "placeholder",
        )
        self._invalid(
            _static_manifest(kind="openai_compat", base_url=url, schema_version="0.2"),
            "requires schema_version '0.3'",
        )
        self._invalid(
            _static_manifest(
                kind="sglang",
                base_url="http://127.0.0.1:9",
                **{"execution.cache_treatment": "cold", "execution.cache_isolation": "salt_isolated"},
            ),
            "unavailable on sglang",
        )
        self._invalid(
            _static_manifest(kind="sglang", base_url="http://127.0.0.1:9", **{"backend.engine_config": {"page_size": 1}}),
            "disable_radix_cache",
        )
        self._invalid(
            _static_manifest(kind="sglang", base_url="http://127.0.0.1:9", **{"backend.undisclosed_identity": ["hardware"]}),
            "allowed only for openai_compat",
        )
        self._invalid(
            _static_manifest(kind="tgi", base_url="http://127.0.0.1:9"),
            "backend.kind must be",
        )


class StaticRunnerTests(unittest.TestCase):
    def _run(self, manifest: dict[str, Any]) -> dict[str, Any]:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            quiet = contextlib.redirect_stdout(io.StringIO())
            with quiet, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["experiment", "validate", str(path)]), 0)
                self.assertEqual(
                    main(["experiment", "run", str(path), "--output", str(directory / "result.json")]),
                    0,
                )
                report = directory / "report.html"
                self.assertEqual(
                    main(["experiment", "report", str(directory / "result.json"), "--output", str(report)]),
                    0,
                )
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            result["_report_html"] = report.read_text(encoding="utf-8")
            return result

    def test_openai_compat_static_run_is_valid_with_explicit_downgrades(self) -> None:
        _HostedHandler.calls = []
        _HostedHandler.models = ["hosted-model-2026-01-01"]
        _HostedHandler.fingerprints = ["fp_one", "fp_two"]
        _HostedHandler.count = 0
        server = _serve(_HostedHandler)
        try:
            result = self._run(
                _static_manifest(
                    kind="openai_compat",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                )
            )
        finally:
            server.shutdown()
            server.server_close()
        decision = result["decision"]
        self.assertNotEqual(decision["status"], "invalid", decision["reason"])
        check = next(item for item in decision["checks"] if item["name"] == "serving_identity_grade")
        self.assertTrue(check["passed"])
        assurance = decision["identity_assurance"]
        self.assertEqual(assurance["claim_scope"], "observed_service_window")
        self.assertEqual(assurance["grades"], {"baseline": "provider_declared", "candidate": "provider_declared"})
        self.assertTrue(assurance["variants"]["baseline"]["provider_fingerprint"]["changed_during_run"])
        self.assertEqual(result["provenance"]["backend_kind"], "openai_compat")
        self.assertEqual(result["serving_capabilities"]["backend_kind"], "openai_compat")
        downgraded = {item["capability"] for item in result["serving_capabilities"]["downgrades"]}
        self.assertIn("exact_prompt_tokens", downgraded)
        self.assertIn("cache_reset", downgraded)
        for trial in result["trials"]:
            self.assertIsNone(trial["prompt_token_ids"])
            self.assertIsNone(trial["predicted_eligible_cached_tokens"])
            self.assertIsNone(trial["server_ttft_ms"])
        self.assertEqual({path for _, path in _HostedHandler.calls}, {"/v1/chat/completions"})
        self.assertNotIn("fp_one", json.dumps({k: v for k, v in result.items() if k != "_report_html"}))
        self.assertIn("Identity assurance", result["_report_html"])
        self.assertIn("Serving capabilities", result["_report_html"])

    def test_openai_compat_served_model_drift_invalidates(self) -> None:
        _HostedHandler.calls = []
        _HostedHandler.models = ["hosted-model-2026-01-01", "hosted-model-2026-02-01"]
        _HostedHandler.fingerprints = ["fp_one"]
        _HostedHandler.count = 0
        server = _serve(_HostedHandler)
        try:
            result = self._run(
                _static_manifest(
                    kind="openai_compat",
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                )
            )
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result["decision"]["status"], "invalid")
        check = next(item for item in result["decision"]["checks"] if item["name"] == "serving_identity_grade")
        self.assertFalse(check["passed"])

    def test_sglang_static_run_with_physical_reset(self) -> None:
        _SGLangHandler.calls = []
        _SGLangHandler.flush_status = 200
        server = _serve(_SGLangHandler)
        try:
            manifest = _static_manifest(
                kind="sglang",
                base_url=f"http://127.0.0.1:{server.server_port}",
                **{"execution.cache_treatment": "cold", "execution.cache_isolation": "physical_reset"},
            )
            result = self._run(manifest)
        finally:
            server.shutdown()
            server.server_close()
        flushes = [call for call in _SGLangHandler.calls if call == ("POST", "/flush_cache")]
        self.assertEqual(len(flushes), 4)
        self.assertEqual(result["provenance"]["backend_kind"], "sglang")
        self.assertEqual(result["serving_capabilities"]["identity_grade"], "operator_verified")
        self.assertNotIn("identity_assurance", result["decision"])
        for trial in result["trials"]:
            self.assertEqual(trial["prompt_token_ids"], [7, 8, 9, 10])

    def test_vllm_static_run_records_no_adapter_sections(self) -> None:
        # The vLLM path is the pre-protocol path: same client, no new keys.
        manifest = copy.deepcopy(_static_manifest(kind="sglang", base_url="http://127.0.0.1:9"))
        manifest["backend"]["kind"] = "vllm"
        manifest["backend"]["engine_config"] = {"enable_prefix_caching": True}
        parsed = ExperimentManifest.from_dict(manifest)
        self.assertEqual(parsed.backend.kind, "vllm")
        with patch.dict(os.environ, {}):
            adapter = create_adapter("vllm", "http://127.0.0.1:9", api_dialect="chat_completions")
        self.assertIsNone(adapter.capability_record())


if __name__ == "__main__":
    unittest.main()
