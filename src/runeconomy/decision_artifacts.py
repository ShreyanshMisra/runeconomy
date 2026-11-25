"""Additive, compatibility-safe structure for future experiment decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


DECISION_SCHEMA_VERSION = "0.2"
LEGACY_DECISION_SCHEMA_VERSION = "0.1"
SUPPORTED_DECISION_SCHEMA_VERSIONS = frozenset(
    {LEGACY_DECISION_SCHEMA_VERSION, DECISION_SCHEMA_VERSION}
)

_DERIVED_FIELDS = {
    "common_mode_paired_failures",
    "decisive_gate",
    "evidence_validity",
    "execution_terminality",
    "failure_origin",
    "identity_assurance",
    "performance_eligibility",
    "schema_version",
    "task_outcome",
}


def _required(check: Mapping[str, Any]) -> bool:
    return check.get("required", True) is not False


def _gate_category(name: str) -> str:
    lowered = name.lower()
    if any(
        marker in lowered
        for marker in (
            "identity",
            "trace",
            "evaluator",
            "evidence",
            "integrity",
        )
    ):
        return "evidence"
    if "completion" in lowered:
        return "completion"
    if any(marker in lowered for marker in ("task_success", "task_quality")):
        return "task_outcome"
    return "performance"


def _variant_outcomes(
    summary: Mapping[str, Any], variant: str
) -> Mapping[str, Any]:
    variants = summary.get("variants")
    if not isinstance(variants, Mapping):
        return {}
    value = variants.get(variant)
    if not isinstance(value, Mapping):
        return {}
    outcomes = value.get("outcomes")
    return outcomes if isinstance(outcomes, Mapping) else value


def _status_counts(outcomes: Mapping[str, Any]) -> dict[str, int]:
    value = outcomes.get("status_counts")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): int(count)
        for name, count in value.items()
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0
    }


def _variant_terminality(outcomes: Mapping[str, Any]) -> dict[str, Any]:
    counts = _status_counts(outcomes)
    recorded = sum(counts.values())
    explicit_attempts = outcomes.get("attempt_count")
    evaluated = outcomes.get("evaluated_count")
    expected = (
        explicit_attempts
        if isinstance(explicit_attempts, int) and not isinstance(explicit_attempts, bool)
        else evaluated
        if isinstance(evaluated, int) and not isinstance(evaluated, bool)
        else recorded
        if recorded
        else None
    )
    if expected is None:
        status = "unknown"
    elif expected == 0:
        status = "terminal"
    elif recorded >= expected:
        status = "terminal"
    else:
        status = "unterminated"
    return {
        "status": status,
        "expected_attempt_count": expected,
        "terminal_record_count": recorded if counts else None,
        "terminal_status_counts": counts,
    }


def _paired_outcome_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    effects = summary.get("effects")
    if not isinstance(effects, Mapping):
        return {}
    task_success = effects.get("task_success")
    if not isinstance(task_success, Mapping):
        return {}
    counts = task_success.get("outcome_counts")
    if not isinstance(counts, Mapping):
        return {}
    names = ("both_success", "both_failure", "baseline_only", "candidate_only")
    return {
        name: int(counts.get(name, 0))
        for name in names
        if isinstance(counts.get(name, 0), int)
        and not isinstance(counts.get(name, 0), bool)
        and counts.get(name, 0) >= 0
    }


def _task_outcome(
    summary: Mapping[str, Any], paired_counts: Mapping[str, int]
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name in ("baseline", "candidate"):
        outcomes = _variant_outcomes(summary, name)
        variants[name] = {
            "evaluated_count": outcomes.get("evaluated_count"),
            "task_success_count": outcomes.get("task_success_count"),
            "task_success_rate": outcomes.get("task_success_rate"),
        }
    paired_count = sum(paired_counts.values()) if paired_counts else 0
    failure_count = (
        paired_counts.get("both_failure", 0)
        + paired_counts.get("baseline_only", 0)
        + paired_counts.get("candidate_only", 0)
    )
    status = (
        "unknown"
        if not paired_count
        else "failures_observed"
        if failure_count
        else "all_success"
    )
    return {
        "status": status,
        "variants": variants,
        "paired_outcome_counts": dict(paired_counts),
    }


def _common_mode_diagnostics(
    counts: Mapping[str, int], policy: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    paired_count = sum(counts.values()) if counts else 0
    both_failure = counts.get("both_failure", 0)
    diagnostics: dict[str, Any] = {
        "paired_count": paired_count,
        "both_failure_count": both_failure,
        "both_failure_rate": both_failure / paired_count if paired_count else None,
        "present": both_failure > 0,
        "interpretation": (
            "Concordant paired failures are consistent with a common-mode or shared-task failure, "
            "but aggregate outcomes alone do not identify the causal origin."
            if both_failure
            else "No concordant paired task failures were observed."
        ),
    }
    if isinstance(policy, Mapping):
        # Present only when the frozen contract declared a completion policy,
        # so legacy artifacts keep their exact shape.
        same_failure = policy.get("common_mode_pair_count")
        reclassified = policy.get("reclassified_as_task_outcome")
        diagnostics["policy"] = {
            "schema_version": policy.get("schema_version"),
            "common_mode_paired_failures": policy.get("common_mode_paired_failures"),
        }
        diagnostics["same_failure_pair_count"] = (
            same_failure if isinstance(same_failure, int) else None
        )
        diagnostics["classification"] = (
            "task_outcome"
            if policy.get("common_mode_paired_failures") == "task_outcome"
            else "variant_non_completion"
        )
        diagnostics["reclassified_pair_count"] = (
            reclassified if isinstance(reclassified, int) else 0
        )
    return diagnostics


def _failure_origin(
    *,
    evidence_status: str,
    paired_counts: Mapping[str, int],
    task_status: str,
) -> dict[str, Any]:
    both_failure = paired_counts.get("both_failure", 0)
    baseline_only = paired_counts.get("baseline_only", 0)
    candidate_only = paired_counts.get("candidate_only", 0)
    if evidence_status == "invalid":
        category = "evidence_or_environment"
        attribution = "identified_at_gate_category"
    elif both_failure and not baseline_only and not candidate_only:
        category = "common_mode_paired_failure"
        attribution = "causal_origin_unresolved"
    elif baseline_only and not candidate_only and not both_failure:
        category = "candidate_specific_task_failure"
        attribution = "localized_by_paired_outcome"
    elif candidate_only and not baseline_only and not both_failure:
        category = "baseline_specific_task_failure"
        attribution = "localized_by_paired_outcome"
    elif task_status == "all_success":
        category = "none"
        attribution = "no_task_failure_observed"
    elif task_status == "unknown":
        category = "unknown"
        attribution = "insufficient_aggregate_evidence"
    else:
        category = "mixed_or_unresolved_task_failure"
        attribution = "causal_origin_unresolved"
    return {
        "category": category,
        "attribution": attribution,
        "paired_outcome_basis": dict(paired_counts),
    }


def _brief(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple)) and len(value) <= 4 and all(
        item is None or isinstance(item, (bool, int, float, str)) for item in value
    ):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, Mapping) and len(value) <= 4 and all(
        item is None or isinstance(item, (bool, int, float, str))
        for item in value.values()
    ):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "a structured value recorded on the decisive check"


def _decisive_gate(checks: list[dict[str, Any]], status: str, reason: str) -> dict[str, Any]:
    decisive_index = next(
        (
            index
            for index, check in enumerate(checks)
            if _required(check) and check.get("passed") is False
        ),
        None,
    )
    if decisive_index is None:
        return {
            "name": "all_required_gates",
            "category": "final_verdict",
            "check_index": None,
            "passed": status == "verified",
            "explanation": reason,
        }
    check = checks[decisive_index]
    name = str(check.get("name", "unnamed_gate"))
    category = _gate_category(name)
    consequence = (
        "Performance evidence is ineligible for a claim."
        if category in {"evidence", "completion", "task_outcome"}
        else "The declared performance criterion was not established."
    )
    return {
        "name": name,
        "category": category,
        "check_index": decisive_index,
        "passed": False,
        "observed_reference": f"checks[{decisive_index}].observed",
        "threshold_reference": f"checks[{decisive_index}].threshold",
        "explanation": (
            f"Required {category} gate {name!r} failed: observed "
            f"{_brief(check.get('observed'))}; threshold "
            f"{_brief(check.get('threshold'))}. {consequence}"
        ),
    }


def _identity_assurance(
    record: Mapping[str, Any], status: str
) -> dict[str, Any]:
    """State which claim a verdict supports under graded identity.

    Present only for contracts that preregistered an identity policy.
    """

    scope = record.get("claim_scope")
    if status == "invalid":
        supported_claim = "No claim: the verdict is invalid."
    elif scope == "pinned_configuration":
        supported_claim = (
            "The verdict applies to the pinned, operator-verified serving "
            "configurations recorded in identity."
        )
    else:
        supported_claim = (
            "The verdict applies only to the declared workload and to the "
            "provider-declared service as observed during this run. It is not "
            "a claim about a pinned model or serving configuration."
        )
    variants = record.get("variants")
    variants = variants if isinstance(variants, Mapping) else {}
    return {
        "schema_version": record.get("schema_version"),
        "policy": record.get("policy"),
        "grades": record.get("grades"),
        "claim_scope": scope,
        "supported_claim": supported_claim,
        "claim_limits": list(record.get("claim_limits") or []),
        "variants": {
            name: {
                key: value.get(key)
                for key in (
                    "grade",
                    "valid",
                    "meets_minimum_grade",
                    "missing_fields",
                    "placeholder_fields",
                    "undisclosed_fields",
                    "provider_fingerprint",
                )
            }
            for name, value in sorted(variants.items())
            if isinstance(value, Mapping)
        },
    }


def enrich_decision_artifact(
    decision: Mapping[str, Any], summary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a schema-0.2 decision while preserving the legacy verdict fields.

    Decisions without a schema version are treated as legacy schema 0.1. The
    transformation is additive and idempotent; unsupported future versions are
    rejected instead of guessed.
    """

    incoming_version = str(
        decision.get("schema_version", LEGACY_DECISION_SCHEMA_VERSION)
    )
    if incoming_version not in SUPPORTED_DECISION_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported decision schema version {incoming_version!r}")
    source_summary = summary if isinstance(summary, Mapping) else {}
    base = {key: value for key, value in decision.items() if key not in _DERIVED_FIELDS}
    status = str(base.get("status", "unknown"))
    reason = str(base.get("reason", "No decision reason was recorded."))
    raw_checks = base.get("checks")
    checks = [dict(check) for check in raw_checks if isinstance(check, Mapping)] if isinstance(raw_checks, list) else []
    base["checks"] = checks

    required_evidence_checks = [
        check
        for check in checks
        if _required(check)
        and _gate_category(str(check.get("name", ""))) == "evidence"
    ]
    failed_evidence = [
        str(check.get("name", "unnamed_gate"))
        for check in required_evidence_checks
        if check.get("passed") is False
    ]
    passed_evidence = [
        str(check.get("name", "unnamed_gate"))
        for check in required_evidence_checks
        if check.get("passed") is True
    ]
    unresolved_evidence = [
        str(check.get("name", "unnamed_gate"))
        for check in required_evidence_checks
        if check.get("passed") is not True and check.get("passed") is not False
    ]
    evidence_status = (
        "invalid"
        if failed_evidence
        else "unknown"
        if not required_evidence_checks or unresolved_evidence
        else "valid"
    )
    terminality_variants = {
        name: _variant_terminality(_variant_outcomes(source_summary, name))
        for name in ("baseline", "candidate")
    }
    terminality_statuses = {
        value["status"] for value in terminality_variants.values()
    }
    terminality_status = (
        "terminal"
        if terminality_statuses == {"terminal"}
        else "unterminated"
        if "unterminated" in terminality_statuses
        else "unknown"
    )
    paired_counts = _paired_outcome_counts(source_summary)
    task_outcome = _task_outcome(source_summary, paired_counts)
    required_preperformance_failures = [
        str(check.get("name", "unnamed_gate"))
        for check in checks
        if _required(check)
        and check.get("passed") is False
        and _gate_category(str(check.get("name", "")))
        in {"evidence", "completion", "task_outcome"}
    ]
    eligible = (
        evidence_status == "valid"
        and terminality_status == "terminal"
        and not required_preperformance_failures
        and status != "invalid"
    )
    if required_preperformance_failures:
        eligibility_reason = (
            "Required pre-performance gate(s) failed: "
            + ", ".join(required_preperformance_failures)
            + "."
        )
    elif evidence_status != "valid":
        eligibility_reason = "Evidence validity was not established."
    elif terminality_status != "terminal":
        eligibility_reason = "Execution terminality was not established."
    elif status == "invalid":
        eligibility_reason = "The legacy verdict is invalid."
    else:
        eligibility_reason = "Evidence, terminality, and pre-performance gates permit performance interpretation."

    return {
        **base,
        "schema_version": DECISION_SCHEMA_VERSION,
        "evidence_validity": {
            "status": evidence_status,
            "required_check_count": len(required_evidence_checks),
            "passed_required_checks": passed_evidence,
            "failed_required_checks": failed_evidence,
            "unresolved_required_checks": unresolved_evidence,
        },
        "execution_terminality": {
            "status": terminality_status,
            "variants": terminality_variants,
        },
        "task_outcome": task_outcome,
        "failure_origin": _failure_origin(
            evidence_status=evidence_status,
            paired_counts=paired_counts,
            task_status=str(task_outcome["status"]),
        ),
        "performance_eligibility": {
            "status": "eligible" if eligible else "ineligible",
            "eligible": eligible,
            "reason": eligibility_reason,
        },
        "common_mode_paired_failures": _common_mode_diagnostics(
            paired_counts,
            source_summary.get("completion_policy")
            if isinstance(source_summary.get("completion_policy"), Mapping)
            else None,
        ),
        "decisive_gate": _decisive_gate(checks, status, reason),
        **(
            {
                "identity_assurance": _identity_assurance(
                    source_summary["identity_assurance"], status
                )
            }
            if isinstance(source_summary.get("identity_assurance"), Mapping)
            else {}
        ),
    }
