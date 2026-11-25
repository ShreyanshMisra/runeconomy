"""Planning schema 0.2: power, sample size, failure-model tolerance, and preregistration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy import paired_inference as pi
from runeconomy.experiments import _paired_success_statistics
from runeconomy.planning import plan_experiment, recommend_pairs, worked_examples
from runeconomy.workspaces import create_agent_workspace


CALIBRATION = "sha256:" + "a" * 64
HELDOUT = "sha256:" + "b" * 64
HYPOTHESIS_TEXT = "PRIVATE_OPERATOR_HYPOTHESIS_TEXT"
TASK_OUTCOME_POLICY = {
    "schema_version": "0.1",
    "common_mode_paired_failures": "task_outcome",
}


def _planning(**overrides: object) -> dict:
    planning: dict = {
        "schema_version": "0.2",
        "hypothesis": HYPOTHESIS_TEXT,
        "intervention": "Serve the pinned model with FP8 weights instead of BF16.",
        "predicted_direction": "decrease",
        "claim": {"level": "workflow", "primary_metric": "workflow_latency_ms"},
        "calibration": {"artifact_sha256": CALIBRATION},
        "heldout": {"artifact_sha256": HELDOUT, "operator_declared_unseen": True},
        "evaluator_frozen": True,
        "gates_frozen": True,
        "stopping_rule": {"kind": "fixed_sample", "favorable_result_dependent": False},
        "claim_boundary": "One pinned agent workload on the declared environment.",
        "intervention_variables": ["backend_identity"],
        "identity_change": True,
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
        "target_power": 0.8,
        "success_assumptions": {
            "source": "calibration",
            "artifact_sha256": CALIBRATION,
            "expected_success_rate": 0.90,
            "expected_discordance_rate": 0.06,
            "expected_difference": 0.0,
        },
        "completion_model": {
            "source": "calibration",
            "artifact_sha256": CALIBRATION,
            "infrastructure_failure_rate": 0.005,
            "application_failure_rate": 0.01,
            "pass_probability": 0.95,
        },
        "cost_assumptions": {"source": "calibration", "usd_per_attempt": 0.002},
    }
    planning.update(overrides)
    return planning


def _proof_3_workspace(
    root: Path,
    *,
    tasks: int = 200,
    method: str = "paired_score_exact",
    planning: dict | None = None,
    min_completed_rate: float | None = None,
    cost_ceiling_usd: float = 5.0,
) -> Path:
    dataset = root / "tasks.json"
    evaluator = root / "evaluator.py"
    dataset.write_text(
        json.dumps({"tasks": [{"task_id": f"task-{index}"} for index in range(tasks)]}),
        encoding="utf-8",
    )
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    if min_completed_rate is None:
        tolerance = pi.completion_tolerance(tasks, 0.015, 0.95)
        min_completed_rate = (tasks - tolerance) / tasks
    workspace = root / "experiment"
    create_agent_workspace(
        output=workspace,
        baseline_command="python agent.py --profile bf16",
        candidate_command="python agent.py --profile fp8",
        dataset_path=dataset,
        evaluator_path=evaluator,
        max_task_success_regression=0.05,
        min_task_success_rate=0.80,
        min_completed_rate=min_completed_rate,
        optimization_target="engine_change_latency",
        cost_ceiling_usd=cost_ceiling_usd,
        task_success_method=method,
        completion_policy=TASK_OUTCOME_POLICY,
    )
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["design"] = {"planning": planning if planning is not None else _planning()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return workspace


def _codes(plan: dict) -> set[str]:
    return {issue["code"] for issue in plan["issues"]}


class PoweredProof3DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with TemporaryDirectory() as raw_directory:
            cls.plan = plan_experiment(_proof_3_workspace(Path(raw_directory)))
            cls.raw_directory = raw_directory

    def test_realistic_proof_3_reaches_target_power_at_an_affordable_n(self) -> None:
        sample = self.plan["sample_size"]
        self.assertEqual(sample["status"], "scenario_estimate")
        self.assertEqual(sample["method_id"], "paired_score_exact")
        self.assertEqual(sample["gate"], "quality_gates")
        self.assertGreaterEqual(sample["planned_power"]["quality_gates"], 0.80)
        self.assertLessEqual(sample["minimum_pairs_for_target_power"], 200)
        self.assertTrue(sample["planned_pairs_meet_target"])
        # The legacy method is underpowered for the same frozen sample.
        legacy = sample["method_comparison"]["paired_exact"]
        self.assertLess(legacy["planned_power"]["quality_gates"], 0.60)
        self.assertGreater(legacy["minimum_pairs"], sample["minimum_pairs_for_target_power"])
        components = sample["planned_power"]["components"]
        self.assertEqual(
            set(components), {"task_success_gates", "completion_both_variants", "latency"}
        )
        self.assertLessEqual(
            sample["planned_power"]["all_gates_lower_bound"],
            sample["planned_power"]["all_gates_independence_approximation"],
        )
        cost = self.plan["cost_estimate"]
        self.assertAlmostEqual(cost["estimated_usd"], 0.8)
        self.assertTrue(cost["within_ceiling"])

    def test_complete_powered_design_is_ready_to_freeze(self) -> None:
        self.assertEqual(self.plan["state"], "ready_to_freeze", self.plan["issues"])
        checks = self.plan["design_declarations"]["freeze_checks"]
        for name in (
            "task_success_power_meets_target",
            "success_assumptions_from_calibration",
            "completion_threshold_from_failure_model",
            "cost_estimate_within_ceiling",
        ):
            self.assertTrue(checks[name], name)

    def test_planner_prints_the_model_derived_integer_tolerance(self) -> None:
        model = self.plan["completion_model"]["expected_failure_model"]
        tolerance = pi.completion_tolerance(200, 0.015, 0.95)
        self.assertEqual(model["status"], "derived")
        self.assertEqual(model["derived_max_noncompleted_attempts_per_variant"], tolerance)
        self.assertTrue(model["declared_matches_model"])
        self.assertEqual(self.plan["completion_model"]["common_mode_paired_failures"], "task_outcome")
        message = next(
            issue["why"]
            for issue in self.plan["issues"]
            if issue["code"] == "runeconomy.plan.completion_model_tolerance"
        )
        self.assertIn(f"integer tolerance of {tolerance} failed or incomplete", message)

    def test_score_method_feasibility_is_reported_for_the_declared_method(self) -> None:
        paired = self.plan["task_success"]["paired_noninferiority"]
        self.assertEqual(paired["method_id"], "paired_score_exact")
        self.assertTrue(paired["feasible_in_best_case"])
        legacy = _paired_success_statistics([(True, True)] * 200)["one_sided_lower_bound"]
        self.assertGreater(paired["best_case_one_sided_lower_bound"], legacy)
        self.assertGreaterEqual(paired["max_baseline_only_with_zero_candidate_only"], 3)
        self.assertEqual(
            self.plan["task_success"]["method_properties"]["method"], "paired_score_exact"
        )

    def test_preregistration_draft_is_filled_without_copying_free_text(self) -> None:
        draft = self.plan["preregistration"]
        self.assertEqual(draft["artifact_type"], "runeconomy.preregistration_draft")
        self.assertEqual(draft["status"], "draft_not_frozen")
        self.assertRegex(draft["template_sha256"], r"^sha256:[0-9a-f]{64}$")
        for field in (
            "claim",
            "population",
            "gates",
            "method",
            "n",
            "tolerance",
            "cost_estimate",
            "stopping_rule",
        ):
            self.assertIn(field, draft)
        self.assertEqual(draft["unknown_fields"], [])
        self.assertEqual(draft["n"]["planned_pairs"], 200)
        self.assertEqual(draft["method"]["task_success_method"], "paired_score_exact")
        self.assertEqual(
            draft["tolerance"]["max_noncompleted_attempts_per_variant"],
            pi.completion_tolerance(200, 0.015, 0.95),
        )
        self.assertTrue(draft["tolerance"]["derived_from_failure_model"])
        self.assertEqual(draft["claim"]["statement_fields"]["hypothesis"], "design.planning.hypothesis")
        serialized = json.dumps(self.plan)
        self.assertNotIn(HYPOTHESIS_TEXT, serialized)
        self.assertNotIn("--profile", serialized)


class PoweredDesignIssueTests(unittest.TestCase):
    def _plan(self, **options: object) -> dict:
        with TemporaryDirectory() as raw_directory:
            return plan_experiment(_proof_3_workspace(Path(raw_directory), **options))

    def test_underpowered_sample_blocks_with_the_required_pairs(self) -> None:
        plan = self._plan(tasks=120)
        self.assertEqual(plan["state"], "blocked")
        issue = next(
            issue
            for issue in plan["issues"]
            if issue["code"] == "runeconomy.plan.success_sample_below_target_power"
        )
        self.assertIn(
            str(plan["sample_size"]["minimum_pairs_for_target_power"]), issue["next_action"]
        )

    def test_zero_tolerance_completion_gate_is_flagged_against_the_model(self) -> None:
        plan = self._plan(min_completed_rate=1.0)
        self.assertIn(
            "runeconomy.plan.completion_threshold_stricter_than_failure_model", _codes(plan)
        )
        model = plan["completion_model"]["expected_failure_model"]
        self.assertLess(model["declared_pass_probability_both_variants"], 0.1)
        self.assertFalse(
            plan["design_declarations"]["freeze_checks"][
                "completion_threshold_from_failure_model"
            ]
        )
        self.assertNotEqual(plan["state"], "ready_to_freeze")

    def test_rates_from_a_heldout_result_are_rejected(self) -> None:
        heldout_model = dict(_planning()["completion_model"])
        heldout_model["source"] = "heldout"
        plan = self._plan(planning=_planning(completion_model=heldout_model))
        self.assertIn("runeconomy.plan.assumption_not_from_calibration", _codes(plan))
        self.assertEqual(plan["state"], "blocked")
        reused = dict(_planning()["success_assumptions"])
        reused["artifact_sha256"] = HELDOUT
        plan = self._plan(planning=_planning(success_assumptions=reused))
        self.assertIn("runeconomy.plan.assumption_not_from_calibration", _codes(plan))

    def test_scenario_assumptions_can_plan_but_not_freeze(self) -> None:
        scenario = dict(_planning()["success_assumptions"])
        scenario["source"] = "scenario"
        scenario.pop("artifact_sha256")
        plan = self._plan(planning=_planning(success_assumptions=scenario))
        self.assertEqual(plan["sample_size"]["assumptions"]["source"], "scenario")
        self.assertFalse(
            plan["design_declarations"]["freeze_checks"][
                "success_assumptions_from_calibration"
            ]
        )
        self.assertEqual(plan["state"], "ready_for_calibration")

    def test_cost_estimate_above_ceiling_blocks(self) -> None:
        plan = self._plan(cost_ceiling_usd=0.5)
        self.assertIn("runeconomy.plan.cost_estimate_exceeds_ceiling", _codes(plan))
        self.assertFalse(plan["preregistration"]["cost_estimate"]["within_ceiling"])

    def test_powered_keys_require_planning_schema_0_2(self) -> None:
        plan = self._plan(planning=_planning(schema_version="0.1"))
        unsupported = [
            issue["field_path"]
            for issue in plan["issues"]
            if issue["code"] == "runeconomy.plan.design_schema_unsupported"
        ]
        self.assertIn("design.planning.success_assumptions", unsupported)
        self.assertIn("design.planning.completion_model", unsupported)
        self.assertEqual(plan["sample_size"]["status"], "unknown")

    def test_inconsistent_success_assumptions_are_rejected(self) -> None:
        inconsistent = dict(_planning()["success_assumptions"])
        inconsistent["expected_difference"] = 0.2
        plan = self._plan(planning=_planning(success_assumptions=inconsistent))
        self.assertIn("runeconomy.plan.success_assumptions_inconsistent", _codes(plan))


class SampleSizeRecommendationTests(unittest.TestCase):
    def test_recommended_pairs_is_the_first_count_meeting_target(self) -> None:
        recommendation = recommend_pairs(
            margin=0.05,
            expected_success_rate=0.95,
            expected_discordance_rate=0.04,
            method="paired_score_exact",
        )
        minimum = recommendation["minimum_pairs"]
        self.assertIsNotNone(minimum)
        self.assertGreaterEqual(
            recommendation["power_at_minimum"]["noninferiority"], 0.8
        )
        cells = recommendation["assumptions"]["cell_probabilities"]

        def power(pairs: int) -> float:
            return pi.gate_power(
                pairs=pairs, method="paired_score_exact", margin=0.05, cells=cells
            )["noninferiority"]

        # Stable minimum: every count up to the scanned top meets the target,
        # and the count just below does not.
        for pairs in range(minimum, recommendation["stable_through_pairs"] + 1):
            self.assertGreaterEqual(power(pairs), 0.8, pairs)
        self.assertLess(power(minimum - 1), 0.8)
        for pairs in recommendation["isolated_smaller_pairs_meeting_target"]:
            self.assertLess(pairs, minimum - 1)
            self.assertGreaterEqual(power(pairs), 0.8)

    def test_target_beyond_the_margin_cannot_be_powered(self) -> None:
        recommendation = recommend_pairs(
            margin=0.05,
            expected_success_rate=0.95,
            expected_discordance_rate=0.08,
            expected_difference=-0.06,
            method="paired_score_exact",
        )
        self.assertIsNone(recommendation["minimum_pairs"])


class WorkedExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = worked_examples()

    def test_n40_case_shows_the_minus_8_8_point_ceiling(self) -> None:
        case = self.examples["n40"]["best_case_all_pairs_agree"]
        self.assertAlmostEqual(
            case["margin_0.05"]["paired_exact"]["one_sided_lower_bound"], -0.0881, places=4
        )
        self.assertAlmostEqual(
            case["margin_0.05"]["paired_score_exact"]["one_sided_lower_bound"], -0.0722, places=3
        )
        self.assertFalse(case["margin_0.05"]["paired_exact"]["passes"])
        self.assertFalse(case["margin_0.05"]["paired_score_exact"]["passes"])
        self.assertTrue(case["margin_0.10"]["paired_score_exact"]["passes"])
        minimum = self.examples["n40"]["minimum_pairs_for_all_agreement_to_pass_margin_0.05"]
        self.assertEqual(minimum["paired_exact"], 72)
        self.assertLess(minimum["paired_score_exact"], minimum["paired_exact"])
        self.assertEqual(self.examples["n40"]["completion_tolerance_per_variant"], 0)

    def test_n180_case_matches_the_frozen_proof_2_record(self) -> None:
        case = self.examples["n180"]
        self.assertAlmostEqual(
            case["observed_lower_bound"]["paired_exact"], -0.020285204570113013, places=12
        )
        self.assertGreater(
            case["observed_lower_bound"]["paired_score_exact"],
            case["observed_lower_bound"]["paired_exact"],
        )
        self.assertEqual(case["completion"]["declared_tolerance_per_variant"], 0)
        self.assertLess(case["completion"]["zero_tolerance_pass_probability_both_variants"], 0.1)
        self.assertEqual(case["completion"]["model_derived_tolerance_per_variant"], 3)
        self.assertIn("remains invalid", case["completion"]["common_mode_policy_note"])

    def test_proof_3_scenario_is_powered_and_affordable(self) -> None:
        case = self.examples["proof_3_scenario"]
        score = case["minimum_pairs"]["paired_score_exact"]
        legacy = case["minimum_pairs"]["paired_exact"]
        self.assertLessEqual(score, 250)
        self.assertLess(score, legacy)
        self.assertGreaterEqual(case["power_at_minimum"]["paired_score_exact"]["quality_gates"], 0.8)
        self.assertLess(case["cost_estimate"]["estimated_usd"], 5.0)

    def test_shared_binomial_refactor_preserves_planner_outputs(self) -> None:
        """Pin user-visible planning numbers from before the consolidation."""

        n40 = self.examples["n40"]
        self.assertAlmostEqual(
            n40["best_case_all_pairs_agree"]["margin_0.05"]["paired_exact"][
                "one_sided_lower_bound"
            ],
            -0.08809730287880221,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            n40["scenario_power"]["margin_0.10"]["paired_exact"]["quality_gates"],
            0.296568056029391,
            delta=1e-12,
        )
        n180 = self.examples["n180"]
        self.assertAlmostEqual(
            n180["observed_lower_bound"]["paired_exact"],
            -0.020285204570112604,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            n180["completion"]["zero_tolerance_pass_probability_both_variants"],
            0.06652510315599597,
            delta=1e-12,
        )
        proof_3 = self.examples["proof_3_scenario"]
        self.assertEqual(
            proof_3["minimum_pairs"],
            {"paired_exact": 345, "paired_score_exact": 197},
        )
        self.assertAlmostEqual(
            proof_3["power_at_minimum"]["paired_exact"]["quality_gates"],
            0.8080195818368676,
            delta=1e-12,
        )


if __name__ == "__main__":
    unittest.main()


class RegressionDetectionPowerTests(unittest.TestCase):
    def test_power_matches_the_decision_engine_absolute_gate(self) -> None:
        from runeconomy.experiments import _clopper_pearson_interval
        from runeconomy.planning import PLANNING_ALPHA, regression_detection_power
        from runeconomy.binomial import binomial_lower_bound

        for successes in (0, 30, 85, 100):
            self.assertAlmostEqual(
                binomial_lower_bound(successes, 100, PLANNING_ALPHA / 2),
                _clopper_pearson_interval(successes, 100)[0],
            )
        strong = regression_detection_power(100, baseline_rate=0.99, candidate_rate=0.55, floor=0.8)
        self.assertGreater(strong, 0.99)
        # No harm, no detection: a candidate as good as the baseline almost never fails.
        null = regression_detection_power(100, baseline_rate=0.99, candidate_rate=0.99, floor=0.8)
        self.assertLess(null, 0.01)
        # Near the floor the design is underpowered at small n.
        self.assertLess(regression_detection_power(20, baseline_rate=0.9, candidate_rate=0.78, floor=0.8), 0.8)
