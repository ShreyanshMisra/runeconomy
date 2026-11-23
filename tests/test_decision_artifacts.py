from __future__ import annotations

import json
from importlib.resources import files

import jsonschema
import pytest

from runeconomy.decision_artifacts import (
    DECISION_SCHEMA_VERSION,
    enrich_decision_artifact,
)


def _summary(
    *,
    both_success: int,
    both_failure: int = 0,
    baseline_only: int = 0,
    candidate_only: int = 0,
) -> dict[str, object]:
    paired = both_success + both_failure + baseline_only + candidate_only
    baseline_success = both_success + baseline_only
    candidate_success = both_success + candidate_only
    completed_count = (
        paired if both_failure == baseline_only == candidate_only == 0 else paired - 1
    )
    status_counts = (
        {"completed": paired}
        if both_failure == baseline_only == candidate_only == 0
        else {"completed": paired - 1, "application_failure": 1}
    )
    return {
        "variants": {
            "baseline": {
                "attempt_count": paired,
                "evaluated_count": paired,
                "completed_rate": completed_count / paired,
                "task_success_count": baseline_success,
                "task_success_rate": baseline_success / paired,
                "status_counts": status_counts,
            },
            "candidate": {
                "attempt_count": paired,
                "evaluated_count": paired,
                "completed_rate": completed_count / paired,
                "task_success_count": candidate_success,
                "task_success_rate": candidate_success / paired,
                "status_counts": status_counts,
            },
        },
        "effects": {
            "task_success": {
                "paired_count": paired,
                "outcome_counts": {
                    "both_success": both_success,
                    "both_failure": both_failure,
                    "baseline_only": baseline_only,
                    "candidate_only": candidate_only,
                },
            }
        },
    }


def _invalid_completion_decision() -> dict[str, object]:
    return {
        "status": "invalid",
        "reason": "the baseline did not establish a valid workload",
        "checks": [
            {
                "name": "trace_and_environment_identity",
                "passed": True,
                "observed": {"complete": True},
                "threshold": "complete and stable",
            },
            {
                "name": "baseline_completion",
                "passed": False,
                "observed": 179 / 180,
                "threshold": 1.0,
            },
            {
                "name": "candidate_completion",
                "passed": False,
                "observed": 179 / 180,
                "threshold": 1.0,
            },
            {
                "name": "successful_task_workflow_latency_improvement",
                "passed": True,
                "observed": {"mean_ms": -391.0},
                "threshold": "upper interval bound below zero",
            },
        ],
    }


def test_common_mode_failure_keeps_terminality_separate_from_validity() -> None:
    artifact = enrich_decision_artifact(
        _invalid_completion_decision(), _summary(both_success=179, both_failure=1)
    )

    assert artifact["schema_version"] == DECISION_SCHEMA_VERSION
    assert artifact["status"] == "invalid"
    assert artifact["evidence_validity"]["status"] == "valid"
    assert artifact["execution_terminality"]["status"] == "terminal"
    assert artifact["task_outcome"]["status"] == "failures_observed"
    assert artifact["failure_origin"] == {
        "category": "common_mode_paired_failure",
        "attribution": "causal_origin_unresolved",
        "paired_outcome_basis": {
            "both_success": 179,
            "both_failure": 1,
            "baseline_only": 0,
            "candidate_only": 0,
        },
    }
    assert artifact["common_mode_paired_failures"]["present"] is True
    assert artifact["common_mode_paired_failures"]["both_failure_rate"] == 1 / 180
    assert artifact["performance_eligibility"]["eligible"] is False
    assert artifact["decisive_gate"]["name"] == "baseline_completion"
    assert artifact["decisive_gate"]["category"] == "completion"
    assert "observed 0.9944444444444445; threshold 1.0" in artifact[
        "decisive_gate"
    ]["explanation"]


def test_candidate_specific_failure_is_not_called_common_mode() -> None:
    decision = {
        "status": "regression",
        "reason": "candidate task outcome failed",
        "checks": [
            {
                "name": "candidate_minimum_task_success",
                "passed": False,
                "observed": 0.9,
                "threshold": 0.95,
            }
        ],
    }
    artifact = enrich_decision_artifact(
        decision, _summary(both_success=19, baseline_only=1)
    )

    assert artifact["failure_origin"]["category"] == "candidate_specific_task_failure"
    assert artifact["common_mode_paired_failures"]["present"] is False
    assert artifact["performance_eligibility"]["status"] == "ineligible"


def test_schema_0_2_is_additive_idempotent_and_validated() -> None:
    decision = {
        "status": "verified",
        "reason": "all required gates passed",
        "checks": [
            {
                "name": "trace_and_environment_identity",
                "passed": True,
                "observed": "complete and stable",
                "threshold": "complete and stable",
            },
            {
                "name": "completion_rate",
                "passed": True,
                "observed": {"baseline": 1.0, "candidate": 1.0},
                "threshold": 1.0,
            },
            {
                "name": "workflow_latency_ms_improvement",
                "passed": True,
                "observed": -10.0,
                "threshold": 0.0,
            },
        ],
    }
    summary = _summary(both_success=20)
    artifact = enrich_decision_artifact(decision, summary)
    schema = json.loads(
        files("runeconomy")
        .joinpath("data/decision-0.2.schema.json")
        .read_text(encoding="utf-8")
    )

    jsonschema.validate(artifact, schema)
    assert enrich_decision_artifact(artifact, summary) == artifact
    assert artifact["performance_eligibility"]["eligible"] is True
    assert artifact["failure_origin"]["category"] == "none"
    assert artifact["status"] == decision["status"]
    assert artifact["reason"] == decision["reason"]
    assert artifact["checks"] == decision["checks"]


def test_absent_required_evidence_checks_are_unknown_and_ineligible() -> None:
    decision = {
        "status": "verified",
        "reason": "legacy gates passed",
        "checks": [
            {
                "name": "workflow_latency_ms_improvement",
                "passed": True,
                "observed": -10.0,
                "threshold": 0.0,
            }
        ],
    }

    artifact = enrich_decision_artifact(decision, _summary(both_success=20))

    assert artifact["evidence_validity"] == {
        "status": "unknown",
        "required_check_count": 0,
        "passed_required_checks": [],
        "failed_required_checks": [],
        "unresolved_required_checks": [],
    }
    assert artifact["performance_eligibility"] == {
        "status": "ineligible",
        "eligible": False,
        "reason": "Evidence validity was not established.",
    }


def test_required_evidence_checks_must_explicitly_pass_to_be_valid() -> None:
    decision = {
        "status": "verified",
        "reason": "all required gates passed",
        "checks": [
            {
                "name": "evaluator_integrity",
                "passed": True,
                "observed": 0,
                "threshold": 0,
            }
        ],
    }

    artifact = enrich_decision_artifact(decision, _summary(both_success=20))

    assert artifact["evidence_validity"] == {
        "status": "valid",
        "required_check_count": 1,
        "passed_required_checks": ["evaluator_integrity"],
        "failed_required_checks": [],
        "unresolved_required_checks": [],
    }
    assert artifact["performance_eligibility"]["eligible"] is True


def test_failed_required_evidence_check_is_invalid_and_ineligible() -> None:
    decision = {
        "status": "invalid",
        "reason": "evaluator evidence was incomplete",
        "checks": [
            {
                "name": "evaluator_integrity",
                "passed": False,
                "observed": 1,
                "threshold": 0,
            }
        ],
    }

    artifact = enrich_decision_artifact(decision, _summary(both_success=20))

    assert artifact["evidence_validity"] == {
        "status": "invalid",
        "required_check_count": 1,
        "passed_required_checks": [],
        "failed_required_checks": ["evaluator_integrity"],
        "unresolved_required_checks": [],
    }
    assert artifact["failure_origin"]["category"] == "evidence_or_environment"
    assert artifact["performance_eligibility"]["eligible"] is False


def test_legacy_schema_is_accepted_but_unknown_future_schema_is_rejected() -> None:
    legacy = {
        "schema_version": "0.1",
        "status": "inconclusive",
        "reason": "more evidence needed",
        "checks": [],
    }

    assert enrich_decision_artifact(legacy)["schema_version"] == "0.2"
    with pytest.raises(ValueError, match="unsupported decision schema version"):
        enrich_decision_artifact({**legacy, "schema_version": "9.9"})
