from __future__ import annotations

import hashlib
import json
import socket
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy.cli import main
from runeconomy.experiments import (
    _clopper_pearson_interval,
    _paired_success_statistics,
)
from runeconomy.planning import plan_experiment
from runeconomy.workspaces import create_agent_workspace


PRIVATE_SENTINEL = "PRIVATE_TASK_PROMPT_AND_ANSWER_SENTINEL"


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_planning() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "hypothesis": "A stable prefix reduces successful-task workflow latency.",
        "intervention": "Move changing metadata after stable instructions.",
        "predicted_direction": "decrease",
        "claim": {
            "level": "workflow",
            "primary_metric": "workflow_latency_ms",
        },
        "calibration": {"artifact_sha256": "sha256:" + "a" * 64},
        "heldout": {
            "artifact_sha256": "sha256:" + "b" * 64,
            "operator_declared_unseen": True,
        },
        "evaluator_frozen": True,
        "gates_frozen": True,
        "stopping_rule": {
            "kind": "fixed_sample",
            "favorable_result_dependent": False,
        },
        "claim_boundary": "One local agent workflow under the declared environment.",
        "intervention_variables": ["prompt_order"],
        "measurement_scope": {
            "tool_time": True,
            "concurrency": False,
            "routing": False,
            "production_behavior": False,
            "live_agent": True,
        },
        "latency_assumptions": {
            "expected_difference_ms": -50.0,
            "paired_standard_deviation_ms": 100.0,
        },
    }


def _agent_workspace(
    root: Path,
    *,
    task_count: int = 40,
    repetitions: int = 1,
    duplicate: bool = False,
    planning: dict[str, object] | None = None,
) -> Path:
    dataset = root / "tasks.json"
    evaluator = root / "evaluator.py"
    tasks = [
        {
            "task_id": "same" if duplicate else f"task-{index}",
            "prompt": PRIVATE_SENTINEL,
            "answer": PRIVATE_SENTINEL,
        }
        for index in range(task_count)
    ]
    dataset.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    evaluator.write_text("# deterministic test evaluator\n", encoding="utf-8")
    workspace = root / "experiment"
    create_agent_workspace(
        output=workspace,
        baseline_command="python private-baseline-agent.py",
        candidate_command="python private-candidate-agent.py",
        dataset_path=dataset,
        evaluator_path=evaluator,
        repetitions=repetitions,
        order="alternating",
        seed=17,
        max_task_success_regression=0.10,
        min_task_success_rate=0.80,
    )
    if planning is not None:
        manifest_path = workspace / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["design"] = {"planning": planning}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return workspace


class ExperimentPlanningTests(unittest.TestCase):
    def test_complete_agent_design_is_ready_to_freeze(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = _agent_workspace(
                Path(raw_directory), planning=_complete_planning()
            )
            report = plan_experiment(workspace)

        self.assertEqual(report["state"], "ready_to_freeze")
        self.assertEqual(report["sample"]["unique_task_count"], 40)
        self.assertEqual(report["sample"]["planned_pairs"], 40)
        self.assertEqual(report["sample"]["total_attempts"], 80)
        self.assertEqual(report["task_success"]["minimum_successful_units"], 38)
        self.assertTrue(
            report["task_success"]["paired_noninferiority"][
                "feasible_in_best_case"
            ]
        )
        exact = _paired_success_statistics([(True, True)] * 40)
        self.assertEqual(
            report["task_success"]["paired_noninferiority"][
                "best_case_one_sided_lower_bound"
            ],
            exact["one_sided_lower_bound"],
        )
        scenarios = report["task_success"]["paired_noninferiority"][
            "scenario_table"
        ]
        self.assertLess(
            scenarios[1]["one_sided_lower_bound"],
            scenarios[0]["one_sided_lower_bound"],
        )
        self.assertTrue(
            report["performance"]["task_failures_reduce_analyzable_pairs"]
        )
        self.assertEqual(
            report["design_declarations"]["heldout_status"],
            "operator_declared_unseen",
        )

    def test_absolute_success_minimum_uses_existing_exact_interval(self) -> None:
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(
                    Path(raw_directory), planning=_complete_planning()
                )
            )
        minimum = report["task_success"]["minimum_successful_units"]
        assert minimum is not None
        passing = _clopper_pearson_interval(minimum, 40)
        failing = _clopper_pearson_interval(minimum - 1, 40)
        assert passing is not None and failing is not None
        self.assertGreaterEqual(passing[0], 0.8)
        self.assertLess(failing[0], 0.8)
        self.assertLess(report["task_success"]["all_success_lower_bound"], 1.0)

    def test_completion_warning_reports_exact_integer_failure_tolerance(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = _agent_workspace(
                Path(raw_directory), planning=_complete_planning()
            )
            report = plan_experiment(workspace)

        self.assertEqual(
            report["completion"],
            {
                "threshold": 1.0,
                "planned_attempts_per_variant": 40,
                "minimum_completed_attempts_per_variant": 40,
                "maximum_failed_or_incomplete_attempts_per_variant": 0,
                "warning": (
                    "A completion threshold of 1 requires at least 40 of 40 "
                    "completed attempts per variant and tolerates exactly 0 failed "
                    "or incomplete attempts per variant."
                ),
            },
        )
        warning = next(
            issue
            for issue in report["issues"]
            if issue["code"] == "runeconomy.plan.completion_failure_tolerance"
        )
        self.assertIn("exactly 0 failed or incomplete attempts", warning["why"])

    def test_completion_tolerance_rounds_threshold_to_an_exact_count(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = _agent_workspace(
                Path(raw_directory), planning=_complete_planning()
            )
            manifest_path = workspace / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["acceptance"]["min_completed_rate"] = 0.95
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = plan_experiment(workspace)

        completion = report["completion"]
        self.assertEqual(completion["minimum_completed_attempts_per_variant"], 38)
        self.assertEqual(
            completion["maximum_failed_or_incomplete_attempts_per_variant"], 2
        )

    def test_missing_design_inputs_and_power_are_reported(self) -> None:
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(_agent_workspace(Path(raw_directory)))
        self.assertEqual(report["state"], "needs_design_input")
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("runeconomy.plan.design_input_missing", codes)
        self.assertIn("runeconomy.plan.latency_power_unknown", codes)
        self.assertEqual(report["performance"]["power"]["status"], "unknown")

    def test_optional_latency_assumptions_produce_transparent_scenario(self) -> None:
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(
                    Path(raw_directory), planning=_complete_planning()
                )
            )
        power = report["performance"]["power"]
        self.assertEqual(power["status"], "scenario_estimate")
        self.assertIn("paired normal approximation", power["method"])
        self.assertEqual(power["planned_pairs"], 40)
        self.assertIsInstance(power["estimated_required_pairs"], int)

    def test_declared_latency_scenario_can_block_an_underpowered_sample(self) -> None:
        planning = _complete_planning()
        planning["latency_assumptions"] = {
            "expected_difference_ms": -10.0,
            "paired_standard_deviation_ms": 100.0,
        }
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(Path(raw_directory), planning=planning)
            )
        self.assertEqual(report["state"], "blocked")
        self.assertIn(
            "runeconomy.plan.latency_sample_below_assumption",
            {issue["code"] for issue in report["issues"]},
        )

    def test_multiple_repetitions_are_not_described_as_independent_tasks(self) -> None:
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(
                    Path(raw_directory),
                    repetitions=2,
                    planning=_complete_planning(),
                )
            )
        self.assertEqual(report["sample"]["unique_task_count"], 40)
        self.assertEqual(report["sample"]["planned_pairs"], 80)
        self.assertFalse(
            report["experimental_unit"]["repetitions_are_independent_tasks"]
        )
        self.assertIn(
            "runeconomy.plan.repetitions_not_independent_tasks",
            {issue["code"] for issue in report["issues"]},
        )

    def test_small_sample_blocks_absolute_and_paired_gates(self) -> None:
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(
                    Path(raw_directory),
                    task_count=5,
                    planning=_complete_planning(),
                )
            )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["state"], "blocked")
        self.assertIn("runeconomy.plan.absolute_gate_infeasible", codes)
        self.assertIn("runeconomy.plan.paired_gate_infeasible", codes)

    def test_forty_pair_failure_mode_is_reported_with_larger_sizing_scenario(self) -> None:
        historical_pairs = (
            [(True, True)] * 36
            + [(True, False)]
            + [(False, True)] * 2
            + [(False, False)]
        )
        observed = _paired_success_statistics(historical_pairs)
        self.assertLess(observed["one_sided_lower_bound"], -0.10)

        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(
                    Path(raw_directory), planning=_complete_planning()
                )
            )
        paired = report["task_success"]["paired_noninferiority"]
        required_for_one = paired[
            "minimum_pairs_by_baseline_only_with_zero_candidate_only"
        ]["1"]
        self.assertGreater(required_for_one, 40)
        self.assertIn(
            "runeconomy.plan.paired_gate_fragile",
            {issue["code"] for issue in report["issues"]},
        )

    def test_duplicate_task_ids_block_pairing_without_echoing_them(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            report = plan_experiment(
                _agent_workspace(
                    root,
                    duplicate=True,
                    planning=_complete_planning(),
                )
            )
            serialized = json.dumps(report)
        self.assertEqual(report["state"], "blocked")
        self.assertFalse(report["sample"]["task_ids_unique"])
        self.assertIn(
            "runeconomy.plan.duplicate_task_id",
            {issue["code"] for issue in report["issues"]},
        )
        self.assertNotIn(PRIVATE_SENTINEL, serialized)
        self.assertNotIn(raw_directory, serialized)

    def test_malformed_dataset_fails_as_safe_planning_issue(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            workspace = _agent_workspace(root, planning=_complete_planning())
            dataset = root / "tasks.json"
            dataset.write_text('{"tasks":"private malformed value"}', encoding="utf-8")
            manifest_path = workspace / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset"]["sha256"] = _digest(dataset)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = plan_experiment(workspace)
        self.assertEqual(report["state"], "blocked")
        self.assertEqual(
            report["issues"][0]["code"], "runeconomy.plan.pairing_undefined"
        )
        self.assertNotIn("private malformed", json.dumps(report))

    def test_unresolved_workspace_is_blocked_by_reused_validator(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = _agent_workspace(Path(raw_directory))
            descriptor_path = workspace / "workspace.json"
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            descriptor["generation_status"] = "unresolved"
            descriptor["unresolved_fields"] = [
                {"path": "evaluator", "reason": PRIVATE_SENTINEL}
            ]
            descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
            report = plan_experiment(workspace)
        self.assertEqual(report["state"], "blocked")
        self.assertEqual(report["execution_readiness"]["status"], "unresolved")
        self.assertNotIn(PRIVATE_SENTINEL, json.dumps(report))

    def test_overlap_multiple_changes_and_outcome_stopping_are_blocking(self) -> None:
        planning = _complete_planning()
        planning["heldout"] = {
            "artifact_sha256": "sha256:" + "a" * 64,
            "operator_declared_unseen": True,
        }
        planning["intervention_variables"] = ["prompt_order", "model_identity"]
        planning["stopping_rule"] = {
            "kind": "predeclared_adaptive",
            "favorable_result_dependent": True,
        }
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(Path(raw_directory), planning=planning)
            )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["state"], "blocked")
        self.assertIn("runeconomy.plan.calibration_heldout_overlap", codes)
        self.assertIn("runeconomy.plan.multiple_intervention_variables", codes)
        self.assertIn("runeconomy.plan.outcome_dependent_stopping", codes)

    def test_undeclared_identity_change_and_consumed_heldout_are_blocking(self) -> None:
        planning = _complete_planning()
        planning["identity_change"] = True
        planning["heldout"] = {
            "artifact_sha256": "sha256:" + "b" * 64,
            "operator_declared_unseen": False,
        }
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(Path(raw_directory), planning=planning)
            )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["state"], "blocked")
        self.assertIn("runeconomy.plan.identity_change_undeclared", codes)
        self.assertIn("runeconomy.plan.heldout_consumed", codes)

    def test_engine_change_target_requires_declared_backend_identity_intervention(self) -> None:
        planning = _complete_planning()
        with TemporaryDirectory() as raw_directory:
            workspace = _agent_workspace(Path(raw_directory), planning=planning)
            manifest_path = workspace / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["acceptance"]["optimization_target"] = "engine_change_latency"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            blocked = plan_experiment(workspace)

            manifest["design"]["planning"]["identity_change"] = True
            manifest["design"]["planning"]["intervention_variables"] = [
                "backend_identity"
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            declared = plan_experiment(workspace)

        self.assertIn(
            "runeconomy.plan.engine_change_not_declared",
            {issue["code"] for issue in blocked["issues"]},
        )
        self.assertNotIn(
            "runeconomy.plan.engine_change_not_declared",
            {issue["code"] for issue in declared["issues"]},
        )

    def test_operator_heldout_declaration_is_not_independent_verification(self) -> None:
        planning = _complete_planning()
        planning["heldout"] = {"artifact_sha256": "sha256:" + "b" * 64}
        with TemporaryDirectory() as raw_directory:
            report = plan_experiment(
                _agent_workspace(Path(raw_directory), planning=planning)
            )
        self.assertEqual(report["state"], "ready_for_calibration")
        self.assertEqual(report["design_declarations"]["heldout_status"], "unknown")
        self.assertFalse(
            report["design_declarations"]["heldout_status_is_operator_declared"]
        )

    def test_static_mechanism_metric_cannot_support_workflow_claim(self) -> None:
        source = Path(__file__).parents[1] / "examples" / "experiments" / "timestamp-placement.json"
        with TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "manifest.json"
            manifest = json.loads(source.read_text(encoding="utf-8"))
            planning = _complete_planning()
            planning["claim"] = {
                "level": "workflow",
                "primary_metric": "server_ttft_ms",
            }
            planning["measurement_scope"]["concurrency"] = True
            manifest["design"] = {"planning": planning}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            report = plan_experiment(path)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["proposed_claim"]["metric_role"], "mechanism_metric")
        self.assertIn("runeconomy.plan.mechanism_metric_product_claim", codes)
        self.assertIn("runeconomy.plan.static_replay_not_live_workflow", codes)
        self.assertIn("runeconomy.plan.unsupported_concurrency_claim", codes)

    def test_human_and_json_outputs_have_the_same_state_and_codes(self) -> None:
        with TemporaryDirectory() as raw_directory:
            workspace = _agent_workspace(Path(raw_directory))
            human = StringIO()
            machine = StringIO()
            with redirect_stdout(human):
                human_exit = main(["experiment", "plan", str(workspace)])
            with redirect_stdout(machine):
                json_exit = main(
                    ["experiment", "plan", str(workspace), "--json"]
                )
            payload = json.loads(machine.getvalue())
        self.assertEqual(human_exit, json_exit)
        self.assertEqual(payload["state"], "needs_design_input")
        self.assertIn("needs_design_input", human.getvalue())
        for issue in payload["issues"]:
            self.assertIn(issue["code"], human.getvalue())

    def test_planning_is_network_free_read_only_and_privacy_safe(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            workspace = _agent_workspace(root, planning=_complete_planning())
            before = {
                path.relative_to(workspace): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in workspace.rglob("*")
                if path.is_file()
            }
            with patch.object(
                socket, "socket", side_effect=AssertionError("network attempted")
            ):
                report = plan_experiment(workspace)
                repeated = plan_experiment(workspace)
            after = {
                path.relative_to(workspace): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in workspace.rglob("*")
                if path.is_file()
            }
            serialized = json.dumps(report)
        self.assertEqual(before, after)
        self.assertEqual(report, repeated)
        self.assertRegex(
            report["design_declarations"]["sha256"], r"^sha256:[0-9a-f]{64}$"
        )
        self.assertNotIn(PRIVATE_SENTINEL, serialized)
        self.assertNotIn(raw_directory, serialized)
        self.assertNotIn("private-baseline-agent", serialized)


if __name__ == "__main__":
    unittest.main()
