from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.analysis import analyze_trace
from runeconomy.cli import _build_parser, main
from runeconomy.experiment_init import initialize_experiment, select_finding
from runeconomy.experiments import ExperimentManifest, ExperimentValidationError
from runeconomy.models import Trace
from tests.test_experiments import _manifest


ROOT = Path(__file__).parents[1]


def _unused_tool_report() -> dict:
    payload = json.loads(
        (ROOT / "examples" / "traces" / "unstable_timestamp.json").read_text()
    )
    request = payload["requests"][1]
    request["available_tools"] = [
        "read_file",
        "search_repository",
        "run_tests",
        "write_file",
    ]
    request["invoked_tools"] = ["read_file"]
    return analyze_trace(Trace.from_dict(payload))


class ExperimentInitTests(unittest.TestCase):
    def test_analysis_finding_produces_explicit_unresolved_skeleton(self) -> None:
        report = _unused_tool_report()
        finding = next(
            item
            for item in report["findings"]
            if item["rule_id"] == "tools.unused_exposure"
        )
        artifact = initialize_experiment(finding)

        self.assertEqual(artifact["generation_status"], "unresolved")
        self.assertEqual(
            artifact["artifact_type"], "runeconomy.experiment_skeleton"
        )
        paths = {item["path"] for item in artifact["unresolved_fields"]}
        self.assertIn("finding.unsupported_reason", paths)
        self.assertIn("experiment_parameters.baseline_request", paths)
        self.assertEqual(
            artifact["manifest_draft"]["design"]["hypothesis"]["prediction"],
            {"metric": "input_tokens", "direction": "decrease"},
        )

    def test_complete_safe_tool_finding_produces_valid_manifest(self) -> None:
        report = _unused_tool_report()
        finding = next(
            item
            for item in report["findings"]
            if item["rule_id"] == "tools.unused_exposure"
        )
        base = _manifest()
        base["acceptance"].update(
            {
                "task_success_method": "paired_exact",
                "optimization_target": "input_tokens",
                "min_input_token_reduction": 0,
            }
        )
        finding["unsupported_reason"] = None
        finding["experiment_parameters"].update(
            {
                "backend": base["backend"],
                "sampling": base["sampling"],
                "execution": base["execution"],
                "acceptance": base["acceptance"],
                "evaluator": base["cases"][0]["evaluator"],
                "baseline_request": {
                    "input": "inspect repository",
                    "tools": [
                        {"type": "function", "name": name, "parameters": {}}
                        for name in (
                            "read_file",
                            "search_repository",
                            "run_tests",
                            "write_file",
                        )
                    ],
                },
                "confirmed_capabilities": list(finding["required_capabilities"]),
            }
        )

        artifact = initialize_experiment(finding)
        parsed = ExperimentManifest.from_dict(artifact)

        self.assertEqual(parsed.experiment_id, f"from-{finding['finding_id']}")
        self.assertEqual(
            [tool["name"] for tool in parsed.cases[0].candidate["tools"]],
            ["read_file"],
        )
        hypothesis = artifact["design"]["hypothesis"]
        self.assertEqual(hypothesis["prediction"]["direction"], "decrease")
        self.assertEqual(
            artifact["design"]["source_finding"]["finding_id"],
            finding["finding_id"],
        )

    def test_report_selection_requires_finding_id_when_ambiguous(self) -> None:
        report = _unused_tool_report()
        self.assertGreater(len(report["findings"]), 1)
        with self.assertRaisesRegex(ExperimentValidationError, "multiple findings"):
            select_finding(report)
        expected = report["findings"][0]
        selected = select_finding(report, expected["finding_id"])
        self.assertEqual(selected["finding_id"], expected["finding_id"])

    def test_cli_init_writes_selected_skeleton(self) -> None:
        report = _unused_tool_report()
        finding = next(
            item
            for item in report["findings"]
            if item["rule_id"] == "tools.unused_exposure"
        )
        with TemporaryDirectory() as directory:
            source = Path(directory) / "report.json"
            output = Path(directory) / "experiment.json"
            source.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "experiment",
                        "init",
                        "--from-finding",
                        str(source),
                        "--finding-id",
                        finding["finding_id"],
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            artifact = json.loads(output.read_text())
            self.assertEqual(artifact["generation_status"], "unresolved")

    def test_experiment_is_primary_command_and_legacy_alias_remains(self) -> None:
        help_text = _build_parser().format_help()
        self.assertIn("experiment (negative-split)", help_text)
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
            self.assertEqual(
                main(["negative-split", "validate", str(manifest)]), 0
            )


if __name__ == "__main__":
    unittest.main()
