from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_JSON = ROOT / "docs" / "proof-2-preregistration-proposal.json"
PROPOSAL_MD = ROOT / "docs" / "proof-2-preregistration-proposal.md"


class Proof2PreregistrationProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proposal = json.loads(PROPOSAL_JSON.read_text(encoding="utf-8"))
        cls.narrative = PROPOSAL_MD.read_text(encoding="utf-8")

    def test_proposal_is_explicitly_unfrozen_and_unexecuted(self) -> None:
        self.assertEqual(self.proposal["status"], "proposed_unfrozen")
        self.assertEqual(self.proposal["review_state"], "freeze_ready")
        self.assertFalse(self.proposal["contract_frozen"])
        self.assertFalse(self.proposal["heldout_executed"])
        self.assertFalse(self.proposal["freeze_state"]["evaluator_frozen"])
        self.assertFalse(self.proposal["freeze_state"]["gates_frozen"])
        self.assertFalse(self.proposal["freeze_state"]["heldout_runner_enabled"])
        self.assertFalse(self.proposal["execution_readiness"]["ready"])
        self.assertFalse(self.proposal["provenance"]["heldout_compute_used"])
        self.assertFalse(self.proposal["provenance"]["heldout_compute_authorized"])
        self.assertTrue(
            self.proposal["approval_boundary"]["required_before_contract_freeze"]
        )
        self.assertTrue(
            self.proposal["approval_boundary"]["required_before_heldout_execution"]
        )

    def test_fixed_sample_matches_quality_planning(self) -> None:
        sample = self.proposal["proposed_sample"]
        plan = self.proposal["sample_size_plan"]
        self.assertEqual(sample["task_count"], 180)
        self.assertEqual(sample["attempt_count"], 360)
        self.assertEqual(
            sample["family_counts"],
            {"call_expression": 90, "string_literal": 90},
        )
        self.assertEqual(
            sample["blocks"],
            {
                "block_a": {
                    "task_count": 90,
                    "family_counts": {
                        "call_expression": 45,
                        "string_literal": 45,
                    },
                },
                "block_b": {
                    "task_count": 90,
                    "family_counts": {
                        "call_expression": 45,
                        "string_literal": 45,
                    },
                },
            },
        )
        self.assertEqual(plan["calibration_pair_count"], 40)
        self.assertEqual(plan["calibration_baseline_successes"], 40)
        self.assertEqual(plan["calibration_candidate_successes"], 40)
        self.assertEqual(plan["approximate_required_latency_pairs"], 6)
        self.assertEqual(plan["selected_pairs"], 180)
        self.assertEqual(
            plan["minimum_pairs_by_baseline_only_with_zero_candidate_only"],
            {"0": 72, "1": 110, "2": 142},
        )

    def test_crossover_swaps_devices_and_keeps_pooled_analysis_primary(self) -> None:
        crossover = self.proposal["intervention"]["crossover"]
        self.assertEqual(
            crossover["block_a_device_assignment"],
            {"baseline_bf16": 0, "candidate_fp8": 1},
        )
        self.assertEqual(
            crossover["block_b_device_assignment"],
            {"baseline_bf16": 1, "candidate_fp8": 0},
        )
        window = self.proposal["measurement_window"]
        self.assertTrue(window["applied_identically_before_each_block"])
        self.assertEqual(
            window["per_profile_per_block"]["structured_warmup_requests"], 2
        )
        self.assertEqual(
            window["per_profile_per_block"]["prefix_cache_resets_after_warmup"],
            1,
        )
        primary = self.proposal["estimands"]["primary_latency"]
        self.assertIn("Pool all eligible pairs", primary["pooling"])
        sensitivity = self.proposal["estimands"]["descriptive_sensitivity"]
        self.assertIn("descriptive only", sensitivity["decision_role"])
        self.assertEqual(len(sensitivity["outputs"]), 3)

    def test_heldout_spending_ceiling_and_abort_rules_are_explicit(self) -> None:
        spending = self.proposal["heldout_spending"]
        self.assertEqual(spending["hard_ceiling"], 8.0)
        self.assertEqual(spending["phase_cumulative_ceiling"], 40.0)
        self.assertEqual(
            spending["maximum_resource_projection_before_egress"], 2.267712
        )
        self.assertGreaterEqual(len(spending["abort_rules"]), 4)
        self.assertTrue(any("15 minutes" in rule for rule in spending["abort_rules"]))
        self.assertTrue(any("invalid" in rule for rule in spending["abort_rules"]))

    def test_stop_and_reporting_rules_are_outcome_independent(self) -> None:
        rules = self.proposal["stopping_and_rerun_rules"]
        self.assertEqual(rules["heldout_runs"], 1)
        self.assertFalse(rules["favorable_result_dependent"])
        self.assertFalse(rules["rerun_on_negative_result"])
        self.assertFalse(rules["rerun_on_invalid_result"])
        self.assertFalse(rules["partial_rerun"])
        self.assertFalse(rules["replacement"])
        self.assertTrue(self.proposal["reporting_commitment"]["prepare_all_outcomes"])
        self.assertTrue(
            self.proposal["reporting_commitment"][
                "external_publication_requires_separate_approval"
            ]
        )

    def test_narrative_exposes_no_private_runtime_credentials_or_tasks(self) -> None:
        lowered = self.narrative.lower()
        for private_term in (
            "base_url",
            "server_id",
            "api_key",
            "modal-token",
            "model_output",
            "bearer ",
        ):
            self.assertNotIn(private_term, lowered)
        self.assertIn("freeze-ready, proposed, and unfrozen", lowered)
        self.assertIn("not executed and not authorized", lowered)
        self.assertIn("$2.25252396", self.narrative)
        self.assertIn("180 task pairs", lowered)
        self.assertIn("pooled test is the sole confirmatory", lowered)

    def test_calibration_runner_cannot_execute_the_proposed_sample(self) -> None:
        runner = (ROOT / "deploy" / "modal_proof_2.py").read_text(encoding="utf-8")
        builder = (ROOT / "scripts" / "build_proof_2_proposal.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("expected_task_count not in {5, 40}", runner)
        self.assertIn("CONSUMED_CALIBRATION_TASKS_SHA256", runner)
        self.assertIn("CONSUMED_CALIBRATION_EVALUATOR_SHA256", runner)
        self.assertNotIn("def heldout(", runner)
        self.assertNotIn("expected_task_count == 180", runner)
        self.assertIn('"evaluator_frozen": False', builder)
        self.assertIn('"gates_frozen": False', builder)
        self.assertNotIn("run_comparison", builder)
        self.assertNotIn("modal.", builder)


if __name__ == "__main__":
    unittest.main()
