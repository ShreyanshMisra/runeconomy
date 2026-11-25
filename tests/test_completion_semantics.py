"""Preregistered task-success method and common-mode completion policy."""

from __future__ import annotations

import json
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.comparison import _decision, _summarize, run_comparison
from runeconomy.completion_policy import (
    CompletionPolicy,
    CompletionPolicyError,
    common_mode_failure_pairs,
)
from runeconomy.experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    run_experiment,
)
from runeconomy.workspaces import (
    agent_run_arguments,
    create_agent_workspace,
    validate_agent_manifest,
)
from tests.test_comparison import AGENT, EVALUATOR
from tests.test_experiments import _FakeClient, _manifest


IDENTITY = {
    "backend": "vllm",
    "backend_version": "0.33.0",
    "model": "fixture/model",
    "model_revision": "model-revision",
    "tokenizer": "fixture/tokenizer",
    "tokenizer_revision": "tokenizer-revision",
    "chat_template_digest": "sha256:template",
}
TASK_OUTCOME_POLICY = {
    "schema_version": "0.1",
    "common_mode_paired_failures": "task_outcome",
}


def _attempt(
    variant: str,
    task: int,
    *,
    status: str = "completed",
    code: str = "tool_contract_failed",
    latency: float = 100.0,
) -> dict:
    failed = status != "completed"
    return {
        "variant": variant,
        "task_id": f"sha256:task-{task}",
        "repetition": 0,
        "status": status,
        "task_success": (
            None
            if status in {"infrastructure_failure", "timeout"}
            else not failed
        ),
        "agent_status": (
            {
                "schema_version": "0.1",
                "status": status,
                "provenance": {
                    "component": "application",
                    "stage": "tool_dispatch",
                    "code": code,
                },
            }
            if failed and status not in {"infrastructure_failure", "timeout"}
            else None
        ),
        "workflow_latency_ms": latency,
        "metrics": {
            "dependency_evidence": "complete_explicit",
            "environment_identity": [IDENTITY],
            "environment_identity_complete": True,
        },
    }


def _proof_2_shaped_attempts(*, candidate_code: str = "tool_contract_failed") -> list[dict]:
    """180 pairs: 179 shared successes and one shared application failure."""

    attempts = []
    for task in range(180):
        if task == 17:
            attempts.append(_attempt("baseline", task, status="application_failure"))
            attempts.append(
                _attempt(
                    "candidate", task, status="application_failure", code=candidate_code
                )
            )
            continue
        attempts.append(_attempt("baseline", task, latency=1300.0 + task % 7))
        attempts.append(_attempt("candidate", task, latency=910.0 + task % 5))
    return attempts


def _decide(attempts: list[dict], *, policy: dict | None, method: str = "paired_exact") -> dict:
    parsed = None if policy is None else CompletionPolicy.from_value(policy)
    summary = _summarize(
        attempts,
        task_success_method=method,
        max_task_success_regression=0.05,
        completion_policy=parsed,
    )
    return {
        "summary": summary,
        "decision": _decision(
            summary,
            max_task_success_regression=0.05,
            min_task_success_rate=0.90,
            min_completed_rate=1.0,
            task_success_method=method,
        ),
    }


class CompletionPolicyParsingTests(unittest.TestCase):
    def test_absent_policy_is_the_legacy_classification(self) -> None:
        policy = CompletionPolicy.from_value(None)
        self.assertEqual(policy.common_mode_paired_failures, "variant_non_completion")
        self.assertFalse(policy.reclassifies_common_mode)

    def test_malformed_policies_are_rejected(self) -> None:
        for value in (
            "task_outcome",
            {"common_mode_paired_failures": "task_outcome"},
            {"schema_version": "0.2", "common_mode_paired_failures": "task_outcome"},
            {"schema_version": "0.1", "common_mode_paired_failures": "ignore"},
            {**TASK_OUTCOME_POLICY, "cap": 3},
        ):
            with self.assertRaises(CompletionPolicyError):
                CompletionPolicy.from_value(value)

    def test_only_identical_declared_task_failures_are_common_mode(self) -> None:
        attempts = [
            _attempt("baseline", 1, status="application_failure"),
            _attempt("candidate", 1, status="application_failure"),
            # Same status, different provenance: not the same way.
            _attempt("baseline", 2, status="application_failure"),
            _attempt("candidate", 2, status="application_failure", code="other"),
            # Different declared status.
            _attempt("baseline", 3, status="application_failure"),
            _attempt("candidate", 3, status="model_output_failure"),
            # Infrastructure failures carry no task outcome.
            _attempt("baseline", 4, status="infrastructure_failure"),
            _attempt("candidate", 4, status="infrastructure_failure"),
            _attempt("baseline", 5, status="timeout"),
            _attempt("candidate", 5, status="timeout"),
            # One-sided failure.
            _attempt("baseline", 6, status="application_failure"),
            _attempt("candidate", 6),
        ]
        pairs = common_mode_failure_pairs(attempts)
        self.assertEqual([pair["task_id"] for pair in pairs], ["sha256:task-1"])
        self.assertEqual(pairs[0]["status"], "application_failure")
        self.assertEqual(pairs[0]["provenance"]["code"], "tool_contract_failed")


class CommonModeDecisionTests(unittest.TestCase):
    def test_legacy_contract_keeps_proof_2_shaped_result_invalid(self) -> None:
        result = _decide(_proof_2_shaped_attempts(), policy=None)
        decision = result["decision"]
        self.assertEqual(decision["status"], "invalid")
        self.assertEqual(decision["decisive_gate"]["name"], "baseline_completion")
        # Legacy artifacts keep their exact shape.
        self.assertNotIn("completion_policy", result["summary"])
        self.assertNotIn("completion_gate", result["summary"]["variants"]["baseline"])
        self.assertNotIn("policy", decision["common_mode_paired_failures"])
        self.assertEqual(decision["failure_origin"]["category"], "common_mode_paired_failure")

    def test_declared_task_outcome_policy_classifies_the_shared_failure(self) -> None:
        result = _decide(_proof_2_shaped_attempts(), policy=TASK_OUTCOME_POLICY)
        decision = result["decision"]
        summary = result["summary"]
        self.assertEqual(decision["status"], "verified")
        checks = {check["name"]: check for check in decision["checks"]}
        for name in ("baseline_completion", "candidate_completion"):
            self.assertTrue(checks[name]["passed"])
            self.assertEqual(checks[name]["observed"], 1.0)
            gate = checks[name]["completion_gate"]
            self.assertEqual(gate["completed_count"], 179)
            self.assertEqual(gate["common_mode_task_outcome_count"], 1)
        # The shared failure stays in the task-success population.
        self.assertEqual(
            summary["effects"]["task_success"]["outcome_counts"]["both_failure"], 1
        )
        self.assertAlmostEqual(
            summary["variants"]["baseline"]["task_success_rate"], 179 / 180
        )
        self.assertAlmostEqual(summary["variants"]["baseline"]["completed_rate"], 179 / 180)
        common_mode = decision["common_mode_paired_failures"]
        self.assertEqual(common_mode["classification"], "task_outcome")
        self.assertEqual(common_mode["same_failure_pair_count"], 1)
        self.assertEqual(common_mode["reclassified_pair_count"], 1)
        self.assertEqual(common_mode["both_failure_count"], 1)

    def test_policy_does_not_reclassify_failures_that_differ(self) -> None:
        result = _decide(
            _proof_2_shaped_attempts(candidate_code="different_failure"),
            policy=TASK_OUTCOME_POLICY,
        )
        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertEqual(
            result["decision"]["common_mode_paired_failures"]["same_failure_pair_count"], 0
        )

    def test_explicit_legacy_policy_records_but_does_not_reclassify(self) -> None:
        result = _decide(
            _proof_2_shaped_attempts(),
            policy={"schema_version": "0.1", "common_mode_paired_failures": "variant_non_completion"},
        )
        self.assertEqual(result["decision"]["status"], "invalid")
        common_mode = result["decision"]["common_mode_paired_failures"]
        self.assertEqual(common_mode["same_failure_pair_count"], 1)
        self.assertEqual(common_mode["reclassified_pair_count"], 0)

    def test_score_method_gate_uses_preregistered_margin(self) -> None:
        result = _decide(
            _proof_2_shaped_attempts(),
            policy=TASK_OUTCOME_POLICY,
            method="paired_score_exact",
        )
        effect = result["summary"]["effects"]["task_success"]
        self.assertEqual(effect["method_id"], "paired_score_exact")
        self.assertTrue(effect["rejects_inferiority"])
        self.assertGreater(effect["one_sided_lower_bound"], -0.0203)
        noninferiority = next(
            check
            for check in result["decision"]["checks"]
            if check["name"] == "task_success_noninferiority"
        )
        self.assertEqual(noninferiority["method"], "paired_score_exact")
        self.assertTrue(noninferiority["passed"])
        self.assertEqual(result["decision"]["status"], "verified")

    def test_score_gate_fails_closed_without_matching_statistics(self) -> None:
        summary = _summarize(_proof_2_shaped_attempts())
        decision = _decision(
            summary,
            max_task_success_regression=0.05,
            min_task_success_rate=0.90,
            min_completed_rate=0.99,
            task_success_method="paired_score_exact",
        )
        noninferiority = next(
            check for check in decision["checks"] if check["name"] == "task_success_noninferiority"
        )
        self.assertFalse(noninferiority["passed"])
        self.assertEqual(decision["status"], "inconclusive")


class AgentManifestVersioningTests(unittest.TestCase):
    def _workspace(self, root: Path, **options: object) -> Path:
        dataset = root / "tasks.json"
        evaluator = root / "evaluator.py"
        dataset.write_text(json.dumps({"tasks": [{"task_id": "a"}]}), encoding="utf-8")
        evaluator.write_text("# evaluator\n", encoding="utf-8")
        workspace = root / "experiment"
        create_agent_workspace(
            output=workspace,
            baseline_command="python baseline.py",
            candidate_command="python candidate.py",
            dataset_path=dataset,
            evaluator_path=evaluator,
            **options,
        )
        return workspace

    def test_default_workspace_keeps_schema_0_2_and_legacy_arguments(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = self._workspace(Path(raw_directory))
            manifest = json.loads((workspace / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "0.2")
            self.assertNotIn("task_success_method", manifest["acceptance"])
            arguments = agent_run_arguments(manifest, workspace)
            self.assertEqual(arguments["task_success_method"], "paired_exact")
            self.assertIsNone(arguments["completion_policy"])

    def test_inputs_outside_the_workspace_are_referenced_relatively(self) -> None:
        # A shared or frozen workspace must not embed the local home path.
        with TemporaryDirectory() as raw_directory:
            workspace = self._workspace(Path(raw_directory))
            manifest = json.loads((workspace / "manifest.json").read_text())
            self.assertEqual(manifest["dataset"]["path"], "../tasks.json")
            self.assertEqual(manifest["evaluator"]["path"], "../evaluator.py")
            self.assertNotIn(str(Path(raw_directory).resolve()), (workspace / "manifest.json").read_text())
            arguments = agent_run_arguments(manifest, workspace)
            self.assertTrue(arguments["dataset_path"].is_file())

    def test_new_semantics_are_written_as_schema_0_3(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = self._workspace(
                Path(raw_directory),
                task_success_method="paired_score_exact",
                completion_policy=TASK_OUTCOME_POLICY,
            )
            manifest = json.loads((workspace / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "0.3")
            arguments = agent_run_arguments(manifest, workspace)
            self.assertEqual(arguments["task_success_method"], "paired_score_exact")
            self.assertEqual(arguments["completion_policy"], TASK_OUTCOME_POLICY)

    def test_older_manifests_cannot_adopt_the_new_semantics(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = self._workspace(Path(raw_directory))
            manifest = json.loads((workspace / "manifest.json").read_text())
            for field, value in (
                ("task_success_method", "paired_score_exact"),
                ("completion_policy", TASK_OUTCOME_POLICY),
            ):
                for version in ("0.1", "0.2"):
                    candidate = json.loads(json.dumps(manifest))
                    candidate["schema_version"] = version
                    candidate["acceptance"][field] = value
                    with self.assertRaisesRegex(ExperimentValidationError, "0.3"):
                        validate_agent_manifest(candidate, workspace)
            invalid = json.loads(json.dumps(manifest))
            invalid["schema_version"] = "0.3"
            invalid["acceptance"]["task_success_method"] = "point_estimate"
            with self.assertRaises(ExperimentValidationError):
                validate_agent_manifest(invalid, workspace)


class EndToEndCommonModeTests(unittest.TestCase):
    def _run(self, directory: Path, name: str, **options: object) -> dict:
        agent = directory / "agent.py"
        agent.write_text(
            textwrap.dedent(AGENT).replace(
                'behavior = task.get("candidate_behavior") if variant == "candidate" else None',
                'behavior = task.get(variant + "_behavior")',
            ),
            encoding="utf-8",
        )
        evaluator = directory / "evaluator.py"
        evaluator.write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
        dataset = directory / "tasks.json"
        tasks = [{"task_id": f"task-{index}", "expected": "ok"} for index in range(3)]
        tasks.append(
            {
                "task_id": "shared",
                "expected": "ok",
                "baseline_behavior": "application_failure",
                "candidate_behavior": "application_failure",
            }
        )
        dataset.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        return run_comparison(
            baseline_command=f"{sys.executable} {agent}",
            candidate_command=f"{sys.executable} {agent}",
            dataset_path=dataset,
            evaluator_path=evaluator,
            output_path=directory / f"{name}.json",
            artifacts_dir=directory / name,
            workdir=directory,
            order="alternating",
            min_task_success_rate=0.0,
            max_task_success_regression=0.6,
            latency_semantics="noninferiority",
            max_latency_regression_ms=60000.0,
            **options,
        )

    def test_real_processes_honour_the_declared_policy(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            legacy = self._run(directory, "legacy")
            declared = self._run(
                directory,
                "declared",
                task_success_method="paired_score_exact",
                completion_policy=TASK_OUTCOME_POLICY,
            )
        self.assertEqual(legacy["decision"]["status"], "invalid")
        self.assertNotIn("completion_policy", legacy["design"]["acceptance"])
        self.assertNotIn("task_success_method", legacy["design"]["acceptance"])
        self.assertEqual(
            declared["design"]["acceptance"]["completion_policy"], TASK_OUTCOME_POLICY
        )
        self.assertEqual(
            declared["design"]["acceptance"]["task_success_method"], "paired_score_exact"
        )
        self.assertNotEqual(legacy["design"]["fingerprint"], declared["design"]["fingerprint"])
        checks = {check["name"]: check for check in declared["decision"]["checks"]}
        self.assertTrue(checks["baseline_completion"]["passed"])
        self.assertTrue(checks["candidate_completion"]["passed"])
        self.assertEqual(declared["summary"]["completion_policy"]["common_mode_pair_count"], 1)
        self.assertNotEqual(declared["decision"]["status"], "invalid")

    def test_invalid_policy_is_rejected_before_execution(self) -> None:
        with TemporaryDirectory() as raw_directory:
            with self.assertRaises(ExperimentValidationError):
                self._run(Path(raw_directory), "bad", completion_policy={"schema_version": "9"})
            with self.assertRaises(ExperimentValidationError):
                self._run(Path(raw_directory), "bad2", task_success_method="point_estimate")


class StaticManifestMethodTests(unittest.TestCase):
    def test_score_method_requires_a_fingerprinted_schema(self) -> None:
        payload = _manifest()
        payload["acceptance"] = {"task_success_method": "paired_score_exact"}
        with self.assertRaisesRegex(ExperimentValidationError, "0.2"):
            ExperimentManifest.from_dict(payload)
        payload["schema_version"] = "0.2"
        manifest = ExperimentManifest.from_dict(payload)
        self.assertEqual(manifest.acceptance.task_success_method, "paired_score_exact")
        legacy = _manifest()
        legacy["schema_version"] = "0.2"
        legacy["acceptance"] = {"task_success_method": "paired_exact"}
        self.assertNotEqual(
            manifest.fingerprint(), ExperimentManifest.from_dict(legacy).fingerprint()
        )

    def test_static_manifests_reject_completion_policy(self) -> None:
        payload = _manifest()
        payload["schema_version"] = "0.3"
        payload["acceptance"] = {"completion_policy": TASK_OUTCOME_POLICY}
        with self.assertRaisesRegex(ExperimentValidationError, "agent-process"):
            ExperimentManifest.from_dict(payload)

    def test_score_method_verifies_where_paired_exact_cannot(self) -> None:
        """60 agreeing pairs: below the 72 paired_exact needs, above the 59 score needs."""

        results = {}
        for method in ("paired_exact", "paired_score_exact"):
            payload = _manifest()
            payload["schema_version"] = "0.3"
            payload["execution"].update({"repetitions": 60, "warmup_repetitions": 0})
            payload["acceptance"] = {
                "task_success_method": method,
                "max_task_success_regression": 0.05,
                "min_task_success_rate": 0.80,
            }
            results[method] = run_experiment(
                ExperimentManifest.from_dict(payload), _FakeClient()
            )
        self.assertEqual(results["paired_exact"]["decision"]["status"], "inconclusive")
        self.assertEqual(results["paired_score_exact"]["decision"]["status"], "verified")
        effect = results["paired_score_exact"]["summary"]["effects"]["task_success"]
        self.assertEqual(effect["method_id"], "paired_score_exact")
        self.assertGreaterEqual(effect["one_sided_lower_bound"], -0.05)
        self.assertLess(
            results["paired_exact"]["summary"]["effects"]["task_success"][
                "one_sided_lower_bound"
            ],
            -0.05,
        )


if __name__ == "__main__":
    unittest.main()
