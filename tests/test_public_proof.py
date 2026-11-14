from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from runeconomy.reports import load_demo_result, render_demo_html
from scripts.build_public_proof_artifact import (
    DEFAULT_CASE_STUDY,
    DEFAULT_DATASET,
    DEFAULT_METADATA,
    DEFAULT_RESULT,
    build,
    render_case_study,
    validate_public_artifact,
)


ROOT = Path(__file__).parents[1]


class PublicProofTests(unittest.TestCase):
    def test_bundled_artifact_is_deterministically_rebuilt_from_private_evidence(self) -> None:
        if not all(path.exists() for path in (DEFAULT_RESULT, DEFAULT_METADATA, DEFAULT_DATASET)):
            self.skipTest("private frozen evidence is intentionally not distributed")
        self.assertEqual(
            load_demo_result("rejection"),
            build(DEFAULT_RESULT, DEFAULT_METADATA, DEFAULT_DATASET),
        )
        self.assertEqual(
            DEFAULT_CASE_STUDY.read_text(encoding="utf-8"),
            render_case_study(load_demo_result("rejection")),
        )

    def test_public_artifact_retains_frozen_invalid_verdict_and_gates(self) -> None:
        artifact = load_demo_result("rejection")
        self.assertEqual(artifact["artifact_version"], "0.2")
        self.assertEqual(artifact["decision"]["status"], "invalid")
        self.assertEqual(
            artifact["decision"]["failed_quality_gates"],
            ["baseline_minimum_task_success", "task_success_noninferiority"],
        )
        self.assertTrue(artifact["experiment"]["consumed_held_out_workload"])
        self.assertEqual(artifact["outcomes"]["baseline"]["task_success_count"], 37)
        self.assertEqual(artifact["outcomes"]["candidate"]["task_success_count"], 38)
        self.assertEqual(
            artifact["conditional_effects"]["workflow_latency_ms"]["paired_count"],
            36,
        )
        self.assertEqual(
            artifact["intervention"]["changed_property"],
            "system-message block order",
        )
        self.assertEqual(
            artifact["decision"]["decisive_stage"],
            "baseline_completion_and_absolute_task_success",
        )
        self.assertEqual(len(artifact["decision"]["gating_order"]), 6)

    def test_every_published_result_number_matches_private_aggregate_evidence(self) -> None:
        if not DEFAULT_RESULT.exists():
            self.skipTest("private frozen evidence is intentionally not distributed")
        private = json.loads(DEFAULT_RESULT.read_text(encoding="utf-8"))
        public = load_demo_result("rejection")
        for variant in ("baseline", "candidate"):
            source = private["summary"]["variants"][variant]
            published = public["outcomes"][variant]
            self.assertEqual(published["attempt_count"], source["attempt_count"])
            self.assertEqual(
                published["task_success_count"], source["task_success_count"]
            )
            self.assertEqual(
                published["task_success_rate"], source["task_success_rate"]
            )
            self.assertEqual(
                published["task_success_95_ci"], source["task_success_95_ci"]
            )
        for metric, published in public["conditional_effects"].items():
            source = private["summary"]["effects"][metric]
            self.assertEqual(
                published["candidate_minus_baseline_mean"],
                source["candidate_minus_baseline_mean"],
            )
            self.assertEqual(
                published["confidence_interval"], source["confidence_interval"]
            )
            self.assertEqual(published["paired_count"], source["paired_count"])
        self.assertEqual(
            public["paired_success"]["one_sided_lower_bound"],
            private["summary"]["effects"]["task_success"][
                "one_sided_lower_bound"
            ],
        )

    def test_public_artifact_contains_no_forbidden_material(self) -> None:
        artifact = load_demo_result("rejection")
        validate_public_artifact(artifact)
        rendered = json.dumps(artifact, sort_keys=True)
        forbidden = (
            "https://",
            "http://",
            "modal.direct",
            "/" + "Users/",
            "/" + "root/",
            "Authorization:",
            "Bearer ",
            '"attempts"',
            '"task_id"',
            '"question"',
            '"stderr"',
            '"stdout"',
            '"trace"',
        )
        for marker in forbidden:
            self.assertNotIn(marker, rendered)

    def test_private_task_and_generated_content_do_not_appear_in_public_artifact(self) -> None:
        if not DEFAULT_DATASET.exists() or not DEFAULT_RESULT.exists():
            self.skipTest("private frozen evidence is intentionally not distributed")
        artifact_text = json.dumps(load_demo_result("rejection"), sort_keys=True)
        case_study_text = DEFAULT_CASE_STUDY.read_text(encoding="utf-8")
        dataset = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
        for task in dataset["tasks"]:
            self.assertNotIn(task["task_id"], artifact_text)
            self.assertNotIn(task["question"], artifact_text)
            self.assertNotIn(task["task_id"], case_study_text)
            self.assertNotIn(task["question"], case_study_text)
        result = json.loads(DEFAULT_RESULT.read_text(encoding="utf-8"))
        for attempt in result["attempts"]:
            self.assertNotIn(attempt["attempt_id"], artifact_text)
            self.assertNotIn(attempt["output_digest"], artifact_text)

    def test_validator_fails_closed_for_new_or_sensitive_fields(self) -> None:
        artifact = load_demo_result("rejection")
        unexpected = copy.deepcopy(artifact)
        unexpected["notes"] = "looks safe"
        with self.assertRaisesRegex(ValueError, "top-level fields"):
            validate_public_artifact(unexpected)

        leaked = copy.deepcopy(artifact)
        leaked["sanitization"]["stderr"] = "private output"
        with self.assertRaisesRegex(ValueError, "forbidden public-artifact field"):
            validate_public_artifact(leaked)

    def test_flagship_report_makes_rejection_and_claim_boundary_dominant(self) -> None:
        rendered = render_demo_html(load_demo_result("rejection"))
        verdict = rendered.index("INVALID — quality gates failed")
        conditional = rendered.index("Conditional observations only")
        self.assertLess(verdict, conditional)
        self.assertIn("RunEconomy declined to verify this candidate", rendered)
        self.assertIn("Failures did not enter the latency sample", rendered)
        self.assertIn("Intervention tested", rendered)
        self.assertIn("system-message block order", rendered)
        self.assertIn("Decision gates, in order", rendered)
        self.assertIn("Experiment limitations", rendered)
        self.assertIn("This report is not a positive benchmark claim", rendered)


if __name__ == "__main__":
    unittest.main()
