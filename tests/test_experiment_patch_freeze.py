from __future__ import annotations

import contextlib
import io
import json
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from runeconomy.cli import main
from runeconomy.freeze import FreezeError, freeze_workspace, verify_freeze
from runeconomy.manifest_patch import (
    ManifestPatchError,
    PatchOperation,
    parse_assignment,
    parse_path,
    patch_workspace,
    read_provenance,
)
from runeconomy.planning import plan_experiment
from runeconomy.workspaces import create_agent_workspace, validate_experiment_artifact

from tests.test_comparison import AGENT, EVALUATOR
from tests.test_planning import _complete_planning


def _cli(*arguments: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(list(arguments))
    return code, stdout.getvalue(), stderr.getvalue()


def _workspace(root: Path, *, tasks: int = 40) -> Path:
    (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
    (root / "evaluator.py").write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
    (root / "tasks.json").write_text(
        json.dumps({"tasks": [{"task_id": f"t{i}", "expected": f"v{i}"} for i in range(tasks)]}),
        encoding="utf-8",
    )
    workspace = root / "experiment"
    create_agent_workspace(
        output=workspace,
        baseline_command=f"{sys.executable} {root / 'agent.py'}",
        candidate_command=f"{sys.executable} {root / 'agent.py'}",
        dataset_path=root / "tasks.json",
        evaluator_path=root / "evaluator.py",
        order="alternating",
        seed=17,
        max_task_success_regression=0.10,
        min_task_success_rate=0.80,
    )
    return workspace


def _manifest(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))


def _ready(workspace: Path) -> None:
    patch_workspace(
        workspace, [PatchOperation("set", "design.planning", _complete_planning())], reason="design"
    )


class PatchPathTests(unittest.TestCase):
    def test_dot_bracket_and_pointer_paths_parse_identically(self) -> None:
        expected = ("cases", 0, "request", "model")
        self.assertEqual(parse_path("cases[0].request.model"), expected)
        self.assertEqual(parse_path("cases.0.request.model"), expected)
        self.assertEqual(parse_path("/cases/0/request/model"), expected)
        for bad in ("", "a..b", "a.[0]x", "a.b[x]", "/a//b"):
            with self.assertRaises(ManifestPatchError):
                parse_path(bad)

    def test_assignment_values_are_json_unless_raw(self) -> None:
        self.assertEqual(parse_assignment("a.b=0.9").value, 0.9)
        self.assertEqual(parse_assignment('a="x=y"').value, "x=y")
        self.assertEqual(parse_assignment("a=plain text", raw_string=True).value, "plain text")
        with self.assertRaisesRegex(ManifestPatchError, "--set-string"):
            parse_assignment("a=plain text")


class PatchTests(unittest.TestCase):
    def test_valid_patch_writes_manifest_and_chained_provenance(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw))
            first = patch_workspace(
                workspace, [PatchOperation("set", "acceptance.min_task_success_rate", 0.9)], reason="tighten"
            )
            second = patch_workspace(workspace, [PatchOperation("set", "execution.seed", 3)])
            history = read_provenance(workspace)
            manifest = _manifest(workspace)

        self.assertEqual(first["status"], "patched")
        self.assertEqual(manifest["acceptance"]["min_task_success_rate"], 0.9)
        self.assertEqual(manifest["execution"]["seed"], 3)
        self.assertEqual([entry["sequence"] for entry, _ in history], [1, 2])
        self.assertIsNone(history[0][0]["previous_entry_sha256"])
        self.assertEqual(history[1][0]["previous_entry_sha256"], history[0][1])
        self.assertEqual(history[0][0]["operations"][0]["previous"], 0.8)
        self.assertEqual(history[0][0]["reason"], "tighten")
        self.assertEqual(first["manifest_sha256_after"], second["manifest_sha256_before"])

    def test_invalid_result_is_refused_and_nothing_is_written(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw))
            before = (workspace / "manifest.json").read_bytes()
            cases = (
                [PatchOperation("set", "acceptance.min_task_success_rate", 2)],
                [PatchOperation("set", "execution.order", "random")],
                # New semantics need a prospective schema version.
                [PatchOperation("set", "acceptance.task_success_method", "paired_score_exact")],
                [PatchOperation("set", "artifact_type", "x")],
                [PatchOperation("unset", "acceptance.not_there")],
            )
            for operations in cases:
                with self.subTest(operations=operations), self.assertRaises(ManifestPatchError):
                    patch_workspace(workspace, operations)
            self.assertEqual((workspace / "manifest.json").read_bytes(), before)
            self.assertFalse((workspace / "provenance.jsonl").exists())

    def test_schema_bump_and_new_semantics_apply_together(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw))
            patch_workspace(
                workspace,
                [
                    PatchOperation("set", "schema_version", "0.3"),
                    PatchOperation("set", "acceptance.task_success_method", "paired_score_exact"),
                ],
            )
            self.assertEqual(_manifest(workspace)["acceptance"]["task_success_method"], "paired_score_exact")

    def test_dataset_path_change_repins_digest_and_repin_tracks_content(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = _workspace(root)
            replacement = root / "tasks-v2.json"
            replacement.write_text('{"tasks":[{"task_id":"only","expected":"x"}]}', encoding="utf-8")
            moved = patch_workspace(workspace, [PatchOperation("set", "dataset.path", str(replacement))])
            derived = [change for change in moved["operations"] if change["source"] == "derived_digest"]
            self.assertEqual([change["path"] for change in derived], ["dataset.sha256"])
            self.assertEqual(validate_experiment_artifact(workspace)["status"], "runnable")

            replacement.write_text('{"tasks":[{"task_id":"other","expected":"y"}]}', encoding="utf-8")
            self.assertEqual(validate_experiment_artifact(workspace)["status"], "invalid")
            repinned = patch_workspace(workspace, [], repin=True)
            self.assertEqual(repinned["operations"][0]["path"], "dataset.sha256")
            self.assertEqual(validate_experiment_artifact(workspace)["status"], "runnable")

    def test_dry_run_unset_and_untracked_edit_detection(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw))
            dry = patch_workspace(
                workspace, [PatchOperation("set", "execution.seed", 99)], dry_run=True
            )
            self.assertEqual(dry["status"], "dry_run")
            self.assertEqual(_manifest(workspace)["execution"]["seed"], 17)

            patch_workspace(workspace, [PatchOperation("unset", "cost_ceiling_usd")])
            self.assertNotIn("cost_ceiling_usd", _manifest(workspace))

            manifest = _manifest(workspace)
            manifest["execution"]["seed"] = 5
            (workspace / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            outcome = patch_workspace(workspace, [PatchOperation("set", "execution.seed", 6)])
            self.assertTrue(outcome["untracked_change_before"])

    def test_tampered_provenance_chain_is_detected(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw))
            patch_workspace(workspace, [PatchOperation("set", "execution.seed", 1)])
            patch_workspace(workspace, [PatchOperation("set", "execution.seed", 2)])
            log = workspace / "provenance.jsonl"
            lines = log.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace('"value":1', '"value":7')
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ManifestPatchError) as caught:
                patch_workspace(workspace, [PatchOperation("set", "execution.seed", 3)])
            self.assertEqual(caught.exception.error_code, "runeconomy.provenance.broken")

    def test_cli_patch_file_and_flags_produce_contract_output(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = _workspace(root)
            patch_file = root / "patch.json"
            patch_file.write_text(
                json.dumps(
                    {
                        "patch_version": "0.1",
                        "operations": [{"op": "replace", "path": "/execution/repetitions", "value": 2}],
                    }
                ),
                encoding="utf-8",
            )
            code, stdout, stderr = _cli(
                "experiment", "patch", str(workspace),
                "--patch-file", str(patch_file),
                "--set", "acceptance.min_task_success_rate=0.85",
                "--set-string", "baseline.command=python other.py",
                "--json",
            )
            manifest = _manifest(workspace)

        self.assertEqual(code, 0, stderr)
        payload = json.loads(stdout)
        self.assertEqual(payload["output_schema"], "runeconomy.cli.experiment-patch.v1")
        self.assertEqual(
            [item["path"] for item in payload["operations"]],
            ["execution.repetitions", "acceptance.min_task_success_rate", "baseline.command"],
        )
        self.assertEqual(manifest["execution"]["repetitions"], 2)
        self.assertEqual(manifest["baseline"]["command"], "python other.py")

    def test_cli_invalid_patch_reports_reason_without_writing(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw))
            code, stdout, stderr = _cli(
                "experiment", "patch", str(workspace), "--set", "execution.repetitions=0", "--json"
            )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        error = json.loads(stderr)["error"]
        self.assertEqual(error["code"], "runeconomy.patch.invalid")
        self.assertIn("execution.repetitions", error["details"]["reason"])


class FreezeTests(unittest.TestCase):
    def test_incomplete_design_is_refused_unless_explicitly_allowed(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw))
            with self.assertRaises(FreezeError) as caught:
                freeze_workspace(workspace)
            self.assertEqual(caught.exception.error_code, "runeconomy.freeze.not_ready")
            self.assertEqual(caught.exception.public_details["planner_state"], "needs_design_input")
            self.assertFalse((workspace / "preregistration.json").exists())

            outcome = freeze_workspace(workspace, allow_incomplete_design=True)
            self.assertEqual(outcome["design_completeness"], "incomplete")

    def test_blocked_design_cannot_be_forced(self) -> None:
        with TemporaryDirectory() as raw:
            workspace = _workspace(Path(raw), tasks=4)
            _ready(workspace)
            self.assertEqual(plan_experiment(workspace)["state"], "blocked")
            with self.assertRaises(FreezeError):
                freeze_workspace(workspace, allow_incomplete_design=True)

    def test_freeze_locks_digests_gates_and_preregistration(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = _workspace(root)
            _ready(workspace)
            outcome = freeze_workspace(workspace, lock_files=[root / "agent.py"], reason="held-out")
            record = json.loads((workspace / "preregistration.json").read_text(encoding="utf-8"))
            history = read_provenance(workspace)
            manifest = _manifest(workspace)
            again = freeze_workspace(workspace)

        self.assertEqual(outcome["status"], "frozen")
        self.assertEqual(outcome["design_completeness"], "complete")
        self.assertEqual(record["artifact_type"], "runeconomy.preregistration_record")
        self.assertEqual(record["locks"]["gates"]["value"], manifest["acceptance"])
        self.assertEqual(record["locks"]["execution"]["value"]["seed"], 17)
        self.assertEqual(set(record["locks"]["inputs"]), {"dataset", "evaluator"})
        self.assertEqual(len(record["locks"]["additional_files"]), 1)
        for field in ("claim", "population", "gates", "method", "n", "tolerance", "cost_estimate", "stopping_rule"):
            self.assertIn(field, record["preregistration"])
        self.assertEqual(record["plan"]["state"], "ready_to_freeze")
        self.assertFalse(record["heldout_executed"])
        self.assertEqual([entry["event"] for entry, _ in history], ["patch", "freeze"])
        self.assertEqual(history[-1][0]["record_sha256"], outcome["record_sha256"])
        self.assertEqual(record["provenance"]["head_entry_sha256"], history[0][1])
        self.assertEqual(again["status"], "already_frozen")
        self.assertEqual(again["record_sha256"], outcome["record_sha256"])

    def test_frozen_workspace_refuses_patches_and_detects_every_drift(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = _workspace(root)
            _ready(workspace)
            freeze_workspace(workspace, lock_files=[root / "agent.py"])
            with self.assertRaises(ManifestPatchError) as caught:
                patch_workspace(workspace, [PatchOperation("set", "execution.seed", 1)])
            self.assertEqual(caught.exception.error_code, "runeconomy.workspace.frozen")
            self.assertEqual(verify_freeze(workspace)["status"], "intact")

            original = (root / "agent.py").read_text(encoding="utf-8")
            (root / "agent.py").write_text(original + "\n# changed\n", encoding="utf-8")
            state = verify_freeze(workspace)
            self.assertEqual(state["status"], "drifted")
            self.assertEqual([item["lock"] for item in state["mismatches"]], ["additional_files[0]"])
            report = validate_experiment_artifact(workspace)
            self.assertEqual(report["status"], "invalid")
            self.assertEqual(report["issues"][0]["code"], "runeconomy.freeze.drift")
            (root / "agent.py").write_text(original, encoding="utf-8")

            manifest = _manifest(workspace)
            manifest["acceptance"]["min_task_success_rate"] = 0.1
            (workspace / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            locks = {item["lock"] for item in verify_freeze(workspace)["mismatches"]}
            self.assertEqual(locks, {"manifest", "gates"})
            code, stdout, stderr = _cli(
                "experiment", "run", str(workspace), "--output", str(root / "results"), "--json"
            )
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            error = json.loads(stderr)["error"]
            self.assertEqual(error["code"], "runeconomy.freeze.drift")
            self.assertEqual(sorted(error["details"]["mismatched_locks"]), ["gates", "manifest"])
            self.assertFalse((root / "results" / "result.json").exists())

    def test_run_require_frozen_refuses_unfrozen_workspace(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            workspace = _workspace(root, tasks=1)
            code, _, stderr = _cli(
                "experiment", "run", str(workspace), "--output", str(root / "results"),
                "--require-frozen", "--json",
            )
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(stderr)["error"]["code"], "runeconomy.freeze.required")


class ZeroJsonEditGoldenPathTests(unittest.TestCase):
    """init -> patch -> freeze -> plan -> run -> report with CLI flags only."""

    def test_agent_drives_frozen_experiment_without_editing_json(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
            (root / "evaluator.py").write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
            (root / "tasks.json").write_text(
                json.dumps({"tasks": [{"task_id": f"t{i}", "expected": f"v{i}"} for i in range(40)]}),
                encoding="utf-8",
            )
            workspace = root / "experiment"
            agent = f"{sys.executable} {root / 'agent.py'}"
            manifest_before: bytes | None = None
            steps: list[tuple[str, ...]] = [
                (
                    "experiment", "init", "--mode", "agent-process",
                    "--baseline", agent, "--candidate", agent,
                    "--dataset", str(root / "tasks.json"), "--evaluator", str(root / "evaluator.py"),
                    "--order", "alternating", "--seed", "17",
                    "--max-task-success-regression", "0.1", "--min-task-success-rate", "0.8",
                    "--output", str(workspace),
                ),
                (
                    "experiment", "patch", str(workspace),
                    "--set", "design.planning=" + json.dumps(_complete_planning()),
                    "--reason", "declare preregistered design",
                ),
                ("experiment", "freeze", str(workspace), "--lock", str(root / "agent.py")),
                ("experiment", "plan", str(workspace)),
                (
                    "experiment", "run", str(workspace), "--output", str(root / "results"),
                    "--require-frozen",
                ),
                (
                    "experiment", "report", str(root / "results" / "result.json"),
                    "--output", str(root / "report.md"),
                ),
            ]
            outputs = []
            for step in steps:
                if step[1] == "run":
                    manifest_before = (workspace / "manifest.json").read_bytes()
                code, stdout, stderr = _cli(*step, "--json")
                self.assertEqual(code, 0, (step[:2], stderr))
                outputs.append(json.loads(stdout))
            report = (root / "report.md").read_text(encoding="utf-8")
            stored = json.loads((root / "results" / "result.json").read_text(encoding="utf-8"))
            manifest_after = (workspace / "manifest.json").read_bytes()

        self.assertEqual(
            [item["output_schema"] for item in outputs],
            [
                "runeconomy.cli.experiment-init.v1",
                "runeconomy.cli.experiment-patch.v1",
                "runeconomy.cli.experiment-freeze.v1",
                "runeconomy.cli.experiment-plan.v1",
                "runeconomy.cli.experiment-run.v1",
                "runeconomy.cli.experiment-report.v1",
            ],
        )
        self.assertEqual(outputs[2]["status"], "frozen")
        self.assertEqual(outputs[3]["state"], "ready_to_freeze")
        self.assertEqual(outputs[4]["freeze"]["status"], "intact")
        self.assertEqual(outputs[4]["freeze"]["record_sha256"], outputs[2]["record_sha256"])
        self.assertIn(outputs[4]["verdict"], {"verified", "regression", "inconclusive", "invalid"})
        self.assertEqual(manifest_before, manifest_after)
        self.assertIn("Verdict", report)
        self.assertEqual(stored["preregistration"]["record_sha256"], outputs[2]["record_sha256"])
        self.assertEqual(stored["preregistration"]["status"], "intact")
        self.assertIn(outputs[2]["record_sha256"], report)


if __name__ == "__main__":
    unittest.main()
