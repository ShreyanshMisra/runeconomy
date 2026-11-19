from __future__ import annotations

import json
import shutil
import unittest
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.cli import main
from runeconomy.evidence import import_evidence, load_evidence_bundle
from runeconomy.workspaces import (
    create_agent_workspace,
    create_workspace_from_evidence,
    validate_experiment_artifact,
)


class EvidenceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture_directory = TemporaryDirectory()
        cls.fixture = Path(cls._fixture_directory.name) / "tutorial-evidence.json"
        resource = files("runeconomy").joinpath("data", "tutorial-evidence.json")
        cls.fixture.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._fixture_directory.cleanup()

    def test_realistic_openinference_bundle_is_runnable_without_editing(self) -> None:
        bundle = import_evidence(self.fixture)
        assessment = bundle["capability_assessment"]

        self.assertTrue(assessment["runnable_experiment"])
        self.assertEqual(assessment["recommended_mode"], "static_request")
        self.assertEqual(
            assessment["capabilities"]["exact_token_cache_analysis"]["status"],
            "available",
        )
        self.assertEqual(
            assessment["capabilities"]["public_verification_claim"]["status"],
            "missing",
        )
        self.assertEqual(len(bundle["analysis"]["findings"]), 1)
        finding = bundle["analysis"]["findings"][0]
        self.assertEqual(
            finding["experiment_template_id"], "prompt.move_dynamic_metadata"
        )
        self.assertIsNone(finding["unsupported_reason"])

        with TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            workspace = Path(directory) / "experiment"
            evidence_path.write_text(json.dumps(bundle), encoding="utf-8")
            create_workspace_from_evidence(
                bundle, evidence_path=evidence_path, output=workspace
            )
            report = validate_experiment_artifact(workspace)

        self.assertEqual(report["status"], "runnable")
        self.assertEqual(report["mode"], "static_request")

    def test_imported_count_only_trace_does_not_claim_exact_analysis(self) -> None:
        payload = {
            "spans": [
                {
                    "trace_id": "count-only",
                    "span_id": "llm-1",
                    "start_time_unix_nano": 0,
                    "end_time_unix_nano": 1_000_000,
                    "attributes": {
                        "openinference.span.kind": "LLM",
                        "llm.token_count.prompt": 12,
                    },
                }
            ]
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "trace.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            bundle = import_evidence(source)

        capabilities = bundle["capability_assessment"]["capabilities"]
        self.assertEqual(capabilities["exact_token_cache_analysis"]["status"], "missing")
        self.assertEqual(capabilities["critical_path_analysis"]["status"], "missing")
        self.assertFalse(bundle["capability_assessment"]["runnable_experiment"])

    def test_openinference_directory_preserves_multiple_runs_without_auto_selection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(self.fixture, root / "run-a.json")
            second = json.loads(self.fixture.read_text())
            spans = second["resourceSpans"][0]["scopeSpans"][0]["spans"]
            for span in spans:
                span["traceId"] = "golden-trace-two"
            (root / "run-b.json").write_text(json.dumps(second), encoding="utf-8")
            bundle = import_evidence(root)

        self.assertEqual(len(bundle["runs"]), 2)
        self.assertEqual(bundle["source"]["kind"], "trace_directory")
        self.assertFalse(bundle["capability_assessment"]["runnable_experiment"])
        self.assertIn("select one run", bundle["capability_assessment"]["next_action"])

    def test_unresolved_workspace_has_corrective_validation_issues(self) -> None:
        bundle = import_evidence(self.fixture)
        finding = bundle["analysis"]["findings"][0]
        finding["experiment_parameters"].pop("evaluator")
        with TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            workspace = Path(directory) / "experiment"
            evidence_path.write_text(json.dumps(bundle), encoding="utf-8")
            create_workspace_from_evidence(
                bundle, evidence_path=evidence_path, output=workspace
            )
            report = validate_experiment_artifact(workspace)

        self.assertEqual(report["status"], "unresolved")
        self.assertTrue(
            all(issue["code"] == "runeconomy.workspace.unresolved" for issue in report["issues"])
        )
        self.assertFalse(report["runnable"])
        self.assertTrue(report["issues"][0]["execution_blocking"])
        self.assertIn("evaluator", {item["field_path"].split(".")[-1] for item in report["issues"]})

    def test_agent_process_workspace_uses_same_validate_and_run_contract(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tasks.json"
            evaluator = root / "evaluator.json"
            dataset.write_text('{"tasks":[{"task_id":"one"}]}', encoding="utf-8")
            evaluator.write_text('{"kind":"exact","expected":"PASS"}', encoding="utf-8")
            workspace = root / "experiment"
            create_agent_workspace(
                output=workspace,
                baseline_command="python baseline.py",
                candidate_command="python candidate.py",
                dataset_path=dataset,
                evaluator_path=evaluator,
            )

            report = validate_experiment_artifact(workspace)

        self.assertEqual(report["status"], "runnable")
        self.assertEqual(report["mode"], "agent_process")

    def test_saved_doctor_report_can_mark_backend_unsupported(self) -> None:
        bundle = import_evidence(self.fixture)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.json"
            workspace = root / "experiment"
            evidence_path.write_text(json.dumps(bundle), encoding="utf-8")
            create_workspace_from_evidence(
                bundle, evidence_path=evidence_path, output=workspace
            )
            report = validate_experiment_artifact(
                workspace,
                doctor_report={"chat_completions": {"status": "unsupported"}},
            )

        self.assertEqual(report["status"], "unsupported_backend")
        self.assertFalse(report["runnable"])
        self.assertEqual(report["issues"][0]["field_path"], "chat_completions")
        self.assertEqual(
            report["issues"][0]["code"], "runeconomy.backend.unsupported"
        )

    def test_public_claim_without_identity_is_reported_but_remains_runnable(self) -> None:
        bundle = import_evidence(self.fixture)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.json"
            workspace = root / "experiment"
            evidence_path.write_text(json.dumps(bundle), encoding="utf-8")
            create_workspace_from_evidence(
                bundle, evidence_path=evidence_path, output=workspace
            )
            descriptor_path = workspace / "workspace.json"
            descriptor = json.loads(descriptor_path.read_text())
            descriptor["intended_claim"] = "public_verification"
            descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
            report = validate_experiment_artifact(workspace)

        self.assertEqual(report["status"], "claim_evidence_incomplete")
        self.assertTrue(report["runnable"])
        self.assertTrue(report["issues"][0]["claim_weakening"])

    def test_cli_import_json_output_is_machine_readable(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            self.assertEqual(
                main(["import", str(self.fixture), "--output", str(output), "--json"]),
                0,
            )
            loaded = load_evidence_bundle(output)
        self.assertEqual(loaded["artifact_type"], "runeconomy.evidence_bundle")


if __name__ == "__main__":
    unittest.main()
