from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.cli import main
from runeconomy.experiments import ExperimentManifest
from runeconomy.reports import (
    load_demo_result,
    render_demo_html,
    render_experiment_html,
    render_experiment_markdown,
)
from scripts.rebuild_demo_artifact import build


class ReportTests(unittest.TestCase):
    def test_verified_report_contains_claim_and_metrics(self) -> None:
        metrics = {
            name: {"mean": value}
            for name, value in {
                "input_tokens": 100.0,
                "cached_tokens": 10.0,
                "cache_coverage": 0.5,
                "server_ttft_ms": 20.0,
                "client_ttft_ms": 30.0,
                "total_latency_ms": 50.0,
                "output_tokens": 4.0,
                "task_success": 1.0,
            }.items()
        }
        effects = {
            name: {
                "candidate_minus_baseline_mean": -1.0,
                "bootstrap_95_ci": [-2.0, -0.5],
            }
            for name in metrics
        }
        result = {
            "experiment_id": "demo",
            "provenance": {"model": "model", "backend_version": "1"},
            "summary": {
                "variants": {
                    "baseline": {
                        **metrics,
                        "outcomes": {
                            "completed_rate": 1.0,
                            "task_success_rate": 1.0,
                            "evaluated_count": 2,
                        },
                    },
                    "candidate": {
                        **metrics,
                        "outcomes": {
                            "completed_rate": 1.0,
                            "task_success_rate": 1.0,
                            "evaluated_count": 2,
                        },
                    },
                },
                "effects": effects,
                "prediction_vs_observation": {
                    "predicted_candidate_minus_baseline_cached_tokens": 16,
                    "observed_candidate_minus_baseline_cached_tokens": 16,
                    "realization_ratio": 1.0,
                },
            },
            "decision": {
                "status": "verified",
                "reason": "all checks passed",
                "checks": [{"name": "quality", "passed": True}],
            },
        }
        rendered = render_experiment_markdown(result)
        self.assertIn("**Decision: VERIFIED**", rendered)
        self.assertIn("Prediction versus observation", rendered)
        self.assertIn("Generated output text was evaluated locally", rendered)

    def test_bundled_demo_renders_self_contained_html(self) -> None:
        result = load_demo_result()
        rendered = render_demo_html(result)
        self.assertIn("RunEconomy", rendered)
        self.assertIn("NegativeSplit", rendered)
        self.assertIn("37.2%", rendered)
        self.assertIn("inconclusive", rendered)
        self.assertIn("exact CI", rendered)
        self.assertIn("No remote scripts", rendered)
        self.assertIn("background: #fff", rendered)
        self.assertNotIn("radial-gradient", rendered)
        self.assertNotIn("border-radius", rendered)
        self.assertNotIn('class="status"', rendered)
        self.assertNotIn("<script", rendered)

    def test_generic_html_handles_unevaluated_result_without_demo_claims(self) -> None:
        result = load_demo_result()
        result["summary"]["workflow"]["variants"]["baseline"]["outcomes"].update(
            {"task_success_rate": None, "evaluated_count": 0}
        )
        result["summary"]["workflow"]["variants"]["candidate"]["outcomes"].update(
            {"task_success_rate": None, "evaluated_count": 0}
        )
        result["summary"]["workflow"]["effects"]["task_success"].update(
            {"candidate_minus_baseline_mean": None, "bootstrap_95_ci": None}
        )
        rendered = render_experiment_html(result)
        self.assertIn("Task success was not evaluated", rendered)
        self.assertNotIn("tool definitions per request", rendered)
        self.assertNotIn("summed over two inference calls", rendered)

    def test_generic_reports_show_predeclared_prediction_assessment(self) -> None:
        result = load_demo_result()
        result["summary"]["predeclared_prediction"] = {
            "status": "not_established",
            "metric": "total_latency_ms",
            "direction": "decrease",
            "causal_mechanism": "less prefill work",
            "candidate_minus_baseline_mean": -3.0,
            "confidence_interval": [-8.0, 2.0],
        }
        html_report = render_experiment_html(result)
        markdown_report = render_experiment_markdown(result)
        self.assertIn("Predeclared hypothesis", html_report)
        self.assertIn("not_established", html_report)
        self.assertIn("## Predeclared hypothesis", markdown_report)
        self.assertIn("NOT_ESTABLISHED", markdown_report)

    def test_demo_command_writes_html_without_network(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "demo.html"
            self.assertEqual(main(["demo", "--output", str(output)]), 0)
            self.assertIn(
                "<strong>Runner decision:</strong> inconclusive.",
                output.read_text(encoding="utf-8"),
            )

    def test_bundled_demo_context_is_internally_consistent(self) -> None:
        result = load_demo_result()
        context = result["measurement_context"]
        workflow = result["summary"]["workflow"]
        fixture_root = Path(__file__).parents[1] / "examples" / "fixtures"
        broad_tools = json.loads(
            (fixture_root / "coding-tools-broad.json").read_text(encoding="utf-8")
        )
        scoped_tools = json.loads(
            (fixture_root / "coding-tools-scoped.json").read_text(encoding="utf-8")
        )
        self.assertEqual(context["workflow_step_count"], 2)
        self.assertEqual(context["measured_workflow_pair_count"], 15)
        self.assertEqual(context["measured_inference_call_count"], 60)
        self.assertEqual(context["warmup_inference_call_count"], 8)
        self.assertEqual(context["baseline_tool_count"], 12)
        self.assertEqual(context["candidate_tool_count"], 3)
        self.assertEqual(context["baseline_tool_count"], len(broad_tools))
        self.assertEqual(context["candidate_tool_count"], len(scoped_tools))
        self.assertEqual(
            context["candidate_tool_names"],
            [tool["name"] for tool in scoped_tools],
        )
        manifest = ExperimentManifest.from_path(
            Path(__file__).parents[1]
            / "examples"
            / "experiments"
            / "repository-agent-tool-pruning-recorded.json"
        )
        self.assertEqual(
            manifest.reproducibility_fingerprint(),
            result["provenance"]["reproducibility_manifest_sha256"],
        )
        self.assertEqual(result["decision"]["status"], "inconclusive")
        for case in manifest.cases:
            differing_fields = {
                key
                for key in set(case.baseline) | set(case.candidate)
                if case.baseline.get(key) != case.candidate.get(key)
            }
            self.assertEqual(differing_fields, {"tools"})
        self.assertEqual(
            workflow["effects"]["input_tokens"]["paired_count"],
            context["measured_workflow_pair_count"],
        )
        self.assertEqual(
            context["measured_inference_call_count"],
            context["measured_workflow_pair_count"]
            * context["workflow_step_count"]
            * 2,
        )

    def test_sanitized_demo_matches_local_raw_result_when_available(self) -> None:
        root = Path(__file__).parents[1]
        raw_path = root / "reports" / "repository-agent-tool-pruning-modal.json"
        if not raw_path.exists():
            self.skipTest("local raw result is intentionally not distributed")
        raw_bytes = raw_path.read_bytes()
        sanitized = load_demo_result()
        self.assertEqual(
            hashlib.sha256(raw_bytes).hexdigest(),
            sanitized["sanitization"]["raw_result_sha256"],
        )
        rebuilt = build(
            raw_path,
            root
            / "examples"
            / "experiments"
            / "repository-agent-tool-pruning-recorded.json",
        )
        self.assertEqual(sanitized, rebuilt)


if __name__ == "__main__":
    unittest.main()
