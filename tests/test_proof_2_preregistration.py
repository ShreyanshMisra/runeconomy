from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_JSON = ROOT / "docs" / "proof-2-preregistration-proposal.json"
PROPOSAL_MD = ROOT / "docs" / "proof-2-preregistration-proposal.md"
FREEZE_RECORD = ROOT / "docs" / "proof-2-freeze-record.json"


class Proof2PreregistrationProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.proposal = json.loads(PROPOSAL_JSON.read_text(encoding="utf-8"))
        cls.narrative = PROPOSAL_MD.read_text(encoding="utf-8")

    def test_contract_is_frozen_but_unexecuted_and_unauthorized(self) -> None:
        self.assertEqual(self.proposal["status"], "frozen")
        self.assertEqual(self.proposal["review_state"], "approved_frozen")
        self.assertTrue(self.proposal["contract_frozen"])
        self.assertFalse(self.proposal["heldout_executed"])
        self.assertTrue(self.proposal["freeze_state"]["evaluator_frozen"])
        self.assertTrue(self.proposal["freeze_state"]["gates_frozen"])
        self.assertTrue(self.proposal["freeze_state"]["heldout_runner_enabled"])
        self.assertFalse(self.proposal["execution_readiness"]["ready"])
        self.assertFalse(self.proposal["provenance"]["heldout_compute_used"])
        self.assertFalse(self.proposal["provenance"]["heldout_compute_authorized"])
        self.assertFalse(
            self.proposal["approval_boundary"]["required_before_contract_freeze"]
        )
        self.assertTrue(
            self.proposal["approval_boundary"]["required_before_heldout_execution"]
        )

    def test_frozen_contract_digest_matches_runner_lock(self) -> None:
        digest = "sha256:" + hashlib.sha256(PROPOSAL_JSON.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "sha256:b349073a6f93a9391d628f9abe38d49120cbd0f4b5cc4202f190f924da6a9816",
        )
        runner = (ROOT / "deploy" / "modal_proof_2.py").read_text(encoding="utf-8")
        self.assertIn(digest, runner)

    def test_freeze_record_digests_match_public_artifacts(self) -> None:
        record = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
        paths = {
            "contract_sha256": PROPOSAL_JSON,
            "human_rendering_sha256": PROPOSAL_MD,
            "runner_sha256": ROOT / "deploy" / "modal_proof_2.py",
            "agent_sha256": ROOT / "examples" / "proof_2" / "agent.py",
            "crossover_allocator_sha256": ROOT / "src" / "runeconomy" / "crossover.py",
        }
        for name, path in paths.items():
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(record["artifacts"][name], digest)
        self.assertFalse(record["heldout_executed"])
        self.assertTrue(record["separate_execution_authorization_required"])
        self.assertTrue(record["execution_safeguards"]["atomic_one_shot_ledger"])
        self.assertFalse(record["privacy"]["raw_tasks_committed"])

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
        self.assertIn("approved and frozen", lowered)
        self.assertIn("not executed and not authorized", lowered)
        self.assertIn("$2.25252396", self.narrative)
        self.assertIn("180 task pairs", lowered)
        self.assertIn("pooled test is the sole confirmatory", lowered)

    def test_heldout_runner_is_digest_locked_and_separately_authorized(self) -> None:
        runner = (ROOT / "deploy" / "modal_proof_2.py").read_text(encoding="utf-8")
        builder = (ROOT / "scripts" / "build_proof_2_proposal.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("expected_task_count not in {5, 40}", runner)
        self.assertIn("CONSUMED_CALIBRATION_TASKS_SHA256", runner)
        self.assertIn("CONSUMED_CALIBRATION_EVALUATOR_SHA256", runner)
        self.assertIn("def heldout(", runner)
        self.assertIn("def run_heldout(", runner)
        self.assertIn("FROZEN_HELDOUT_TASKS_SHA256", runner)
        self.assertIn("FROZEN_HELDOUT_EVALUATOR_SHA256", runner)
        self.assertIn("FROZEN_HELDOUT_AUTHORIZATION", runner)
        self.assertIn("skip_if_exists=True", runner)
        self.assertIn("max_task_success_regression=0.05", runner)
        self.assertIn("min_task_success_rate=0.90", runner)
        self.assertIn("block A consumed the frozen half-window", runner)
        self.assertIn('"evaluator_frozen": False', builder)
        self.assertIn('"gates_frozen": False', builder)
        self.assertNotIn("run_comparison", builder)
        self.assertNotIn("modal.", builder)


if __name__ == "__main__":
    unittest.main()
