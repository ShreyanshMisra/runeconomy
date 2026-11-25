from __future__ import annotations

import copy
import hashlib
import json
import unittest
from html.parser import HTMLParser
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


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"style", "script", "code", "pre"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"style", "script", "code", "pre"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return " ".join(self.parts)


def _assert_json_equivalent(
    testcase: unittest.TestCase,
    expected: object,
    actual: object,
    path: str = "$",
) -> None:
    """Compare JSON values while tolerating cross-Python float rounding."""

    testcase.assertIs(type(actual), type(expected), path)
    if isinstance(expected, dict):
        testcase.assertEqual(expected.keys(), actual.keys(), path)
        for key in expected:
            _assert_json_equivalent(
                testcase, expected[key], actual[key], f"{path}.{key}"
            )
    elif isinstance(expected, list):
        testcase.assertEqual(len(expected), len(actual), path)
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _assert_json_equivalent(
                testcase, expected_item, actual_item, f"{path}[{index}]"
            )
    elif isinstance(expected, float):
        testcase.assertAlmostEqual(expected, actual, places=14, msg=path)
    else:
        testcase.assertEqual(expected, actual, path)


class ReportTests(unittest.TestCase):
    def test_public_report_uses_required_order_and_plain_verdict(self) -> None:
        rendered = render_demo_html(load_demo_result())
        headings = [
            rendered.index(f">{heading}<")
            for heading in (
                "Intervention Tested",
                "Results",
                "Verdict",
                "Experiment Details",
            )
        ]
        self.assertEqual(headings, sorted(headings))
        self.assertIn("Result not verified", rendered)
        self.assertIn("Verdict: Invalid", rendered)
        self.assertIn("Invalid does not mean the candidate was slower", rendered)
        self.assertNotIn("INVALID", rendered)
        self.assertNotIn("<footer", rendered)

    def test_favorable_numbers_are_qualified_in_results(self) -> None:
        rendered = render_demo_html(load_demo_result())
        results = rendered[rendered.index(">Results<") : rendered.index(">Verdict<")]
        self.assertIn("+105", results)
        self.assertIn("9 ms lower", results)
        self.assertIn("62 ms lower", results)
        self.assertIn("36 task pairs where both variants succeeded", results)
        self.assertIn("not a verified improvement", results)

    def test_report_prose_punctuation_and_offline_contract(self) -> None:
        rendered = render_demo_html(load_demo_result())
        parser = _VisibleText()
        parser.feed(rendered)
        self.assertNotIn("—", parser.text)
        self.assertNotIn(";", parser.text)
        self.assertNotIn("<script", rendered)
        self.assertNotRegex(rendered, r"(?:src|href)=[\"']https?://")
        self.assertNotIn("@import", rendered)

    def test_html_is_deterministic_and_responsive(self) -> None:
        result = load_demo_result()
        first = render_demo_html(result)
        second = render_demo_html(result)
        self.assertEqual(first, second)
        self.assertEqual(first.splitlines(), [line.rstrip() for line in first.splitlines()])
        self.assertIn("@media (max-width: 700px)", first)
        self.assertIn('name="viewport"', first)
        self.assertIn('aria-label="Decision path"', first)
        self.assertIn("Synthetic labels show ordering only", first)

    def test_all_verdicts_render_coherently(self) -> None:
        base = load_demo_result("historical")
        expected = {
            "verified": "Improvement verified",
            "regression": "Candidate regressed",
            "inconclusive": "Result inconclusive",
            "invalid": "Result not verified",
        }
        for status, message in expected.items():
            with self.subTest(status=status):
                result = copy.deepcopy(base)
                result["decision"]["status"] = status
                rendered = render_experiment_html(result)
                markdown = render_experiment_markdown(result)
                self.assertIn(message, rendered)
                self.assertIn(f"Verdict: {status.title()}", rendered)
                self.assertIn("## Intervention Tested", markdown)
                self.assertLess(
                    markdown.index("## Intervention Tested"),
                    markdown.index("## Results"),
                )

    def test_generic_html_handles_unevaluated_result_without_demo_claims(self) -> None:
        result = load_demo_result("historical")
        result["summary"]["workflow"]["variants"]["baseline"]["outcomes"].update(
            {"task_success_rate": None, "evaluated_count": 0}
        )
        result["summary"]["workflow"]["variants"]["candidate"]["outcomes"].update(
            {"task_success_rate": None, "evaluated_count": 0}
        )
        rendered = render_experiment_html(result)
        self.assertIn("Task success was not evaluated", rendered)
        self.assertNotIn("36 task pairs", rendered)

    def test_predeclared_prediction_remains_in_technical_details(self) -> None:
        result = load_demo_result("historical")
        result["summary"]["predeclared_prediction"] = {
            "status": "not_established",
            "metric": "total_latency_ms",
            "direction": "decrease",
            "causal_mechanism": "less prefill work",
        }
        html_report = render_experiment_html(result)
        markdown_report = render_experiment_markdown(result)
        self.assertIn("Predeclared hypothesis", html_report)
        self.assertIn("not_established", html_report)
        self.assertIn("Predeclared hypothesis", markdown_report)

    def test_evidence_quality_is_visible_in_human_renderings(self) -> None:
        result = load_demo_result("historical")
        result["evidence_quality"] = {
            "schema_version": "0.1",
            "overall_grade": "weakly_inferred",
            "fields": {
                "server_ttft": {
                    "grade": "weakly_inferred",
                    "source": "generic framework TTFT attribute",
                }
            },
        }
        html_report = render_experiment_html(result)
        markdown_report = render_experiment_markdown(result)
        self.assertIn("Evidence quality", html_report)
        self.assertIn("weakly_inferred", html_report)
        self.assertIn("Evidence quality", markdown_report)

    def test_demo_command_writes_unified_report(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "demo.html"
            self.assertEqual(main(["demo", "--output", str(output)]), 0)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("RunEconomy Experiment Report", rendered)
            self.assertIn("Result not verified", rendered)

    def test_bundled_historical_context_is_internally_consistent(self) -> None:
        result = load_demo_result("historical")
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
        self.assertEqual(context["baseline_tool_count"], len(broad_tools))
        self.assertEqual(context["candidate_tool_count"], len(scoped_tools))
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
        self.assertEqual(
            workflow["effects"]["input_tokens"]["paired_count"],
            context["measured_workflow_pair_count"],
        )

    def test_sanitized_historical_demo_matches_local_raw_result_when_available(
        self,
    ) -> None:
        root = Path(__file__).parents[1]
        raw_path = root / "reports" / "repository-agent-tool-pruning-modal.json"
        if not raw_path.exists():
            self.skipTest("local raw result is intentionally not distributed")
        raw_bytes = raw_path.read_bytes()
        sanitized = load_demo_result("historical")
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
        _assert_json_equivalent(self, sanitized, rebuilt)


if __name__ == "__main__":
    unittest.main()
