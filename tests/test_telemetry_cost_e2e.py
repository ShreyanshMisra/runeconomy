"""Server telemetry sampling and cost models through real concurrent CLI runs."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from runeconomy.cli import main
from runeconomy.experiments import ExperimentManifest, ExperimentValidationError
from runeconomy.telemetry import ServerMetricsSampler, prometheus_url_scraper

from tests.test_adapters import _HostedHandler, _serve, _SGLangHandler, _static_manifest


GPU_COST_MODEL = {
    "schema_version": "0.1",
    "kind": "gpu_hourly",
    "usd_per_gpu_hour": 1.95,
    "gpus_per_variant": 1,
    "source": "test price",
}
TOKEN_COST_MODEL = {
    "schema_version": "0.1",
    "kind": "per_token",
    "usd_per_1m_input_tokens": 2.0,
    "usd_per_1m_cached_input_tokens": 0.5,
    "usd_per_1m_output_tokens": 8.0,
}


def _concurrent(kind: str, base_url: str, cost_model: dict[str, Any]) -> dict[str, Any]:
    manifest = _static_manifest(kind=kind, base_url=base_url)
    manifest["schema_version"] = "0.4"
    manifest["cases"] = [
        {**manifest["cases"][0], "case_id": f"triage-{index:03d}"} for index in range(24)
    ]
    manifest["execution"].update(
        {"repetitions": 1, "arrival_pattern": "closed_loop", "load": {"concurrency": 4, "tail": "sustain"}}
    )
    manifest["acceptance"]["metric_gates"] = [
        {"metric": "usd_per_1k_successful_workflows", "semantics": "noninferiority", "relative_margin": 0.5}
    ]
    manifest["cost_model"] = cost_model
    return manifest


class SamplerTests(unittest.TestCase):
    def test_block_summary_reports_gauges_counter_deltas_and_hit_rate(self) -> None:
        state = {"step": 0}

        def scrape() -> dict[str, Any]:
            state["step"] += 1
            step = state["step"]
            return {
                "running_requests": float(step),
                "waiting_requests": 0.0,
                "kv_cache_usage_ratio": 0.1 * step,
                "preemptions_total": 2.0,
                "prefix_cache_hits_total": 10.0 * step,
                "prefix_cache_queries_total": 20.0 * step,
            }

        sampler = ServerMetricsSampler({"server": scrape}, interval_seconds=60)
        sampler.on_block_start({})
        trial = sampler.on_trial_end(None, {"variant": "baseline"})
        summary = sampler.on_block_end({})["server_metrics"]["targets"]["server"]
        self.assertEqual(trial["metrics_target"], "server")
        self.assertEqual(summary["status"], "measured")
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["gauges"]["running_requests"]["max"], 2.0)
        self.assertEqual(summary["counters"]["preemptions_total_delta"], 0.0)
        self.assertEqual(summary["prefix_cache_hit_rate_in_block"], 0.5)

    def test_failing_scrape_is_unknown_not_fatal(self) -> None:
        def scrape() -> dict[str, Any]:
            raise OSError("down")

        sampler = ServerMetricsSampler({"baseline": scrape}, interval_seconds=60)
        sampler.on_block_start({})
        self.assertIsNone(sampler.on_trial_end(None, {"variant": "baseline"}))
        summary = sampler.on_block_end({})["server_metrics"]["targets"]["baseline"]
        self.assertEqual(summary, {"status": "unavailable", "samples": 0, "scrape_errors": 2})


# Exposition excerpts with the metric names vLLM 0.34.0 and SGLang publish.
VLLM_0_34_SCRAPE = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 7.0
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.41
vllm:num_preemptions_total{engine="0",model_name="m"} 3.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 50.0
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 200.0
"""
SGLANG_SCRAPE = """sglang:num_running_reqs{model_name="m"} 5.0
sglang:num_queue_reqs{model_name="m"} 1.0
sglang:token_usage{model_name="m"} 0.3
sglang:cache_hit_rate{model_name="m"} 0.6
sglang:gen_throughput{model_name="m"} 900.0
"""


def _metrics_handler(body: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return None

    return Handler


class PrometheusScraperTests(unittest.TestCase):
    def _scrape(self, body: str) -> dict[str, Any]:
        server = _serve(_metrics_handler(body))
        try:
            return dict(prometheus_url_scraper(f"http://127.0.0.1:{server.server_port}/metrics")())
        finally:
            server.shutdown()

    def test_vllm_gauges_are_read_alongside_sglang_names(self) -> None:
        # Regression: a dict merge let SGLang's source names replace vLLM's
        # for shared keys, so every vLLM gauge was recorded as unknown.
        values = self._scrape(VLLM_0_34_SCRAPE)
        self.assertEqual(values["running_requests"], 7.0)
        self.assertEqual(values["waiting_requests"], 2.0)
        self.assertEqual(values["kv_cache_usage_ratio"], 0.41)
        self.assertEqual(values["preemptions_total"], 3.0)
        self.assertEqual(values["prefix_cache_hit_rate"], 0.25)

    def test_sglang_gauges_still_resolve(self) -> None:
        values = self._scrape(SGLANG_SCRAPE)
        self.assertEqual(values["running_requests"], 5.0)
        self.assertEqual(values["waiting_requests"], 1.0)
        self.assertEqual(values["kv_cache_usage_ratio"], 0.3)
        self.assertEqual(values["generation_throughput_tokens_per_s"], 900.0)
        self.assertEqual(values["prefix_cache_hit_rate"], 0.6)
        self.assertIsNone(values["preemptions_total"])


class CostModelManifestTests(unittest.TestCase):
    def test_cost_model_is_prospective_and_fingerprinted(self) -> None:
        manifest = _concurrent("sglang", "http://127.0.0.1:9", GPU_COST_MODEL)
        parsed = ExperimentManifest.from_dict(manifest)
        self.assertEqual(parsed.to_resolved_dict()["cost_model"], GPU_COST_MODEL)
        changed = json.loads(json.dumps(manifest))
        changed["cost_model"]["usd_per_gpu_hour"] = 2.5
        self.assertNotEqual(ExperimentManifest.from_dict(changed).fingerprint(), parsed.fingerprint())
        legacy = json.loads(json.dumps(manifest))
        legacy["schema_version"] = "0.3"
        legacy["execution"].pop("arrival_pattern")
        legacy["execution"].pop("load")
        legacy["acceptance"].pop("metric_gates")
        with self.assertRaisesRegex(ExperimentValidationError, "cost_model requires"):
            ExperimentManifest.from_dict(legacy)


class ConcurrentAdapterRunTests(unittest.TestCase):
    def _run(self, manifest: dict[str, Any], *flags: str) -> dict[str, Any]:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    ["experiment", "run", str(path), "--output", str(root / "result.json"), *flags, "--json"]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            return json.loads((root / "result.json").read_text(encoding="utf-8"))

    def test_sglang_concurrent_run_with_gpu_cost_gate_and_metrics(self) -> None:
        _SGLangHandler.calls = []
        server = _serve(_SGLangHandler)
        try:
            result = self._run(
                _concurrent("sglang", f"http://127.0.0.1:{server.server_port}", GPU_COST_MODEL),
                "--server-metrics",
                "--server-metrics-interval",
                "0.05",
            )
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result["provenance"]["cost_model"]["model"], GPU_COST_MODEL)
        population = [trial for trial in result["trials"] if not trial.get("warmup")]
        self.assertTrue(population)
        for trial in population:
            self.assertGreater(trial["economics"]["cost_usd"], 0)
            self.assertEqual(trial["economics"]["attribution"], "fair_share_reserved_gpu_time")
            self.assertEqual(trial["server_telemetry"]["metrics_target"], "server")
        blocks = result["load_blocks"]
        self.assertTrue(blocks)
        targets = blocks[0]["telemetry"]["server_metrics"]["targets"]
        self.assertEqual(targets["server"]["status"], "measured")
        self.assertEqual(targets["server"]["gauges"]["waiting_requests"]["max"], 5.0)
        gate = next(
            check
            for check in result["decision"]["checks"]
            if check["name"] == "usd_per_1k_successful_workflows_noninferiority"
        )
        self.assertEqual(gate["observed"]["clusters"], 24)
        self.assertNotEqual(gate["observed"]["status"], "insufficient_sample")
        self.assertIsNotNone(gate["observed"]["baseline"])

    def test_openai_compat_concurrent_run_prices_tokens(self) -> None:
        _HostedHandler.calls = []
        _HostedHandler.models = ["hosted-model-2026-01-01"]
        _HostedHandler.fingerprints = ["fp_one"]
        _HostedHandler.count = 0
        server = _serve(_HostedHandler)
        try:
            result = self._run(
                _concurrent("openai_compat", f"http://127.0.0.1:{server.server_port}/v1", TOKEN_COST_MODEL)
            )
        finally:
            server.shutdown()
            server.server_close()
        self.assertNotEqual(result["decision"]["status"], "invalid", result["decision"]["reason"])
        expected = (8 * 2.0 + 32 * 0.5 + 3 * 8.0) / 1_000_000
        for trial in (trial for trial in result["trials"] if not trial.get("warmup")):
            self.assertAlmostEqual(trial["economics"]["cost_usd"], expected)
        self.assertEqual({path for _, path in _HostedHandler.calls}, {"/v1/chat/completions"})


if __name__ == "__main__":
    unittest.main()


class AgentCostModelTests(unittest.TestCase):
    def test_agent_manifest_cost_model_prices_attempts(self) -> None:
        import sys
        import textwrap

        from runeconomy.manifest_patch import PatchOperation, patch_workspace
        from runeconomy.workspaces import create_agent_workspace
        from tests.test_comparison import AGENT, EVALUATOR

        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
            (root / "evaluator.py").write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
            (root / "tasks.json").write_text(
                json.dumps({"tasks": [{"task_id": f"t{i}", "expected": f"v{i}"} for i in range(3)]}),
                encoding="utf-8",
            )
            workspace = root / "experiment"
            agent = f"{sys.executable} {root / 'agent.py'}"
            create_agent_workspace(
                output=workspace,
                baseline_command=agent,
                candidate_command=agent,
                dataset_path=root / "tasks.json",
                evaluator_path=root / "evaluator.py",
                order="alternating",
                min_task_success_rate=0.0,
                max_task_success_regression=1.0,
            )
            with self.assertRaises(ExperimentValidationError):
                patch_workspace(workspace, [PatchOperation("set", "cost_model", GPU_COST_MODEL)])
            patch_workspace(
                workspace,
                [
                    PatchOperation("set", "schema_version", "0.4"),
                    PatchOperation("set", "cost_model", GPU_COST_MODEL),
                ],
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["experiment", "run", str(workspace), "--output", str(root / "results"), "--json"])
            self.assertEqual(code, 0, stderr.getvalue())
            result = json.loads((root / "results" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["design"]["cost_model"]["model"], GPU_COST_MODEL)
        for attempt in result["attempts"]:
            self.assertEqual(attempt["economics"]["attribution"], "reserved_gpu_time")
            self.assertGreater(attempt["economics"]["cost_usd"], 0)


class PreregistrationIdentityTests(unittest.TestCase):
    def test_planner_draft_lists_identity_policy_gate(self) -> None:
        from runeconomy.planning import _identity_gate_rows

        rows = _identity_gate_rows(
            {
                "acceptance": {
                    "identity_policy": {"schema_version": "0.1", "minimum_grade": "provider_declared"},
                    "require_environment_identity": True,
                }
            }
        )
        self.assertEqual([row["name"] for row in rows], ["serving_identity_grade", "environment_identity"])
        self.assertEqual(rows[0]["minimum_grade"], "provider_declared")
        self.assertEqual(_identity_gate_rows({"acceptance": {}}), [])
