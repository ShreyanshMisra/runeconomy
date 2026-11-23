"""Read-only experiment-design assessment without execution or network access."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    _clopper_pearson_interval,
    _paired_success_statistics,
)
from .workspaces import load_workspace, validate_experiment_artifact


PLAN_SCHEMA_VERSION = "0.1"
PLAN_STATES = frozenset(
    {"blocked", "needs_design_input", "ready_for_calibration", "ready_to_freeze"}
)
_DIRECTIONS = frozenset({"decrease", "increase", "no_change"})
_CLAIM_LEVELS = frozenset({"workflow", "request", "mechanism", "unknown"})
_MEASUREMENT_KEYS = frozenset(
    {"tool_time", "concurrency", "routing", "production_behavior", "live_agent"}
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_LABEL = re.compile(r"[a-z][a-z0-9_.-]{0,63}")


def _issue(
    code: str,
    severity: str,
    field_path: str,
    why: str,
    next_action: str,
) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "field_path": field_path,
        "why": why,
        "next_action": next_action,
    }


def _digest(value: Mapping[str, Any]) -> str | None:
    if not value:
        return None
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _minimum_successes_for_lower_bound(
    units: int, threshold: float, *, confidence: float = 0.95
) -> int | None:
    """Return the first success count whose existing exact lower bound passes."""

    for successes in range(units + 1):
        interval = _clopper_pearson_interval(
            successes, units, confidence=confidence
        )
        assert interval is not None
        if interval[0] >= threshold:
            return successes
    return None


def _paired_feasibility(
    pairs: int, margin: float, *, confidence: float = 0.95
) -> dict[str, Any]:
    best = _paired_success_statistics(
        [(True, True)] * pairs, confidence=confidence
    )
    best_lower = best["one_sided_lower_bound"]
    max_baseline_only = None
    scenarios = []
    for baseline_only in range(pairs + 1):
        outcomes = [(True, False)] * baseline_only + [
            (True, True)
        ] * (pairs - baseline_only)
        statistics = _paired_success_statistics(outcomes, confidence=confidence)
        lower = statistics["one_sided_lower_bound"]
        passes = bool(lower is not None and lower >= -margin)
        if passes:
            max_baseline_only = baseline_only
        if baseline_only in {0, 1, 2, pairs}:
            scenarios.append(
                {
                    "baseline_only": baseline_only,
                    "candidate_only": 0,
                    "one_sided_lower_bound": lower,
                    "passes": passes,
                }
            )
    return {
        "method": best["method"],
        "confidence_level": confidence,
        "margin": margin,
        "best_case_one_sided_lower_bound": best_lower,
        "feasible_in_best_case": bool(
            best_lower is not None and best_lower >= -margin
        ),
        "max_baseline_only_with_zero_candidate_only": max_baseline_only,
        "scenario_table": scenarios,
        "discordance_explanation": (
            "Candidate-only outcomes improve the paired difference. Baseline-only "
            "outcomes weaken it. Agreement still leaves finite-sample uncertainty."
        ),
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentValidationError("planning input could not be read") from exc


def _agent_tasks(manifest: Mapping[str, Any], workspace: Path) -> tuple[int, bool]:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("path"), str):
        raise ExperimentValidationError("agent dataset reference is unavailable")
    path = Path(dataset["path"])
    if not path.is_absolute():
        path = workspace / path
    payload = _read_json(path)
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else payload
    if not isinstance(tasks, list) or not tasks:
        raise ExperimentValidationError("agent dataset has no tasks")
    identifiers = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ExperimentValidationError("agent dataset contains an invalid task")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ExperimentValidationError("agent dataset task ID is invalid")
        identifiers.append(task_id)
    return len(identifiers), len(identifiers) == len(set(identifiers))


def _planning_block(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    design = manifest.get("design")
    if not isinstance(design, Mapping):
        return {}, False
    planning = design.get("planning")
    if planning is None:
        return {}, False
    if not isinstance(planning, Mapping):
        return {}, True
    return dict(planning), False


def _presence(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _safe_direction(value: Any) -> str:
    return value if value in _DIRECTIONS else "unknown"


def _safe_claim_level(value: Any) -> str:
    return value if value in _CLAIM_LEVELS else "unknown"


def _safe_measurement_scope(value: Any) -> dict[str, bool | None]:
    mapping = value if isinstance(value, Mapping) else {}
    return {
        key: mapping.get(key) if isinstance(mapping.get(key), bool) else None
        for key in sorted(_MEASUREMENT_KEYS)
    }


def _static_units(manifest: ExperimentManifest) -> tuple[int, int]:
    case_count = len(manifest.cases)
    if not manifest.execution.cases_form_workflow:
        return case_count, case_count
    workflow_ids = [case.workflow_id for case in manifest.cases]
    if all(value is not None for value in workflow_ids):
        return case_count, len(set(workflow_ids))
    return case_count, 1


def _latency_power(
    planning: Mapping[str, Any], planned_pairs: int
) -> dict[str, Any]:
    assumptions = planning.get("latency_assumptions")
    if not isinstance(assumptions, Mapping):
        return {
            "status": "unknown",
            "reason": "expected paired effect and variability were not declared",
            "calibration_needed": [
                "paired candidate-minus-baseline mean",
                "paired-difference standard deviation",
            ],
        }
    effect = assumptions.get("expected_difference_ms")
    deviation = assumptions.get("paired_standard_deviation_ms")
    if (
        isinstance(effect, bool)
        or not isinstance(effect, (int, float))
        or isinstance(deviation, bool)
        or not isinstance(deviation, (int, float))
        or float(effect) == 0
        or float(deviation) <= 0
    ):
        return {
            "status": "unknown",
            "reason": "latency assumptions are incomplete or invalid",
            "calibration_needed": [
                "non-zero expected paired effect",
                "positive paired-difference standard deviation",
            ],
        }
    # Two-sided 95% normal approximation with 80% target power. The executed
    # experiment still uses its paired Student-t interval, not this estimate.
    required = math.ceil(
        ((1.959963984540054 + 0.8416212335729143) * float(deviation) / abs(float(effect)))
        ** 2
    )
    return {
        "status": "scenario_estimate",
        "method": "paired normal approximation, two-sided alpha 0.05, 80% power",
        "expected_difference_ms": float(effect),
        "paired_standard_deviation_ms": float(deviation),
        "estimated_required_pairs": required,
        "planned_pairs": planned_pairs,
        "planned_pairs_meet_estimate": planned_pairs >= required,
        "limitations": (
            "This is a planning approximation from declared calibration assumptions. "
            "The experiment uses a paired Student-t interval and may need more data."
        ),
    }


def _blocked_plan(
    validation_status: str, issue: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "artifact_type": "runeconomy.experiment_plan",
        "schema_version": PLAN_SCHEMA_VERSION,
        "state": "blocked",
        "execution_readiness": {
            "status": validation_status,
            "runnable": validation_status == "runnable",
        },
        "proposed_claim": {"level": "unknown", "primary_metric": "unknown"},
        "experimental_unit": None,
        "sample": None,
        "task_success": None,
        "performance": None,
        "design_declarations": None,
        "issues": [dict(issue)],
        "next_action_code": "runeconomy.action.resolve_execution_validation",
    }


def plan_experiment(path: Path) -> dict[str, Any]:
    """Assess one design without executing, networking, or mutating its workspace."""

    validation = validate_experiment_artifact(path)
    issues: list[dict[str, str]] = []
    if not validation.get("runnable"):
        return _blocked_plan(
            str(validation.get("status", "invalid")),
            _issue(
                "runeconomy.plan.execution_not_ready",
                "error",
                "workspace",
                "The execution validator did not find a runnable workspace.",
                "Resolve the validator issues before using design planning.",
            ),
        )

    descriptor: Mapping[str, Any] | None = None
    if path.is_dir():
        descriptor, raw_manifest = load_workspace(path)
        mode = str(descriptor.get("mode"))
        workspace = path
    else:
        raw_manifest = _read_json(path)
        if not isinstance(raw_manifest, Mapping):
            raise ExperimentValidationError("planning manifest must be an object")
        mode = "static_request"
        workspace = path.parent

    planning, malformed_planning = _planning_block(raw_manifest)
    if malformed_planning or (
        planning and planning.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        issues.append(
            _issue(
                "runeconomy.plan.design_schema_unsupported",
                "error",
                "design.planning.schema_version",
                "The optional planning declaration is malformed or unsupported.",
                "Use planning schema 0.1 or remove the optional block.",
            )
        )

    design = raw_manifest.get("design")
    design = design if isinstance(design, Mapping) else {}
    derived_hypothesis = design.get("hypothesis")
    hypothesis_present = _presence(planning.get("hypothesis")) or isinstance(
        derived_hypothesis, Mapping
    )
    derived_prediction = (
        derived_hypothesis.get("prediction")
        if isinstance(derived_hypothesis, Mapping)
        else None
    )
    predicted_direction = _safe_direction(planning.get("predicted_direction"))
    if predicted_direction == "unknown" and isinstance(derived_prediction, Mapping):
        predicted_direction = _safe_direction(derived_prediction.get("direction"))
    intervention_present = _presence(planning.get("intervention")) or bool(
        isinstance(descriptor, Mapping)
        and isinstance(descriptor.get("intervention"), Mapping)
    )

    if mode == "agent_process":
        try:
            unique_tasks, ids_unique = _agent_tasks(raw_manifest, workspace)
        except ExperimentValidationError:
            return _blocked_plan(
                str(validation.get("status", "invalid")),
                _issue(
                    "runeconomy.plan.pairing_undefined",
                    "error",
                    "dataset.tasks",
                    "The dataset does not define usable task-level pairing.",
                    "Correct the dataset structure and stable task IDs, then validate and plan again.",
                ),
            )
        source_cases = unique_tasks
        execution = raw_manifest.get("execution")
        acceptance = raw_manifest.get("acceptance")
        assert isinstance(execution, Mapping)
        assert isinstance(acceptance, Mapping)
        repetitions = int(execution.get("repetitions", 1))
        primary_metric = "workflow_latency_ms"
        min_success = float(acceptance.get("min_task_success_rate", 0.8))
        margin = float(acceptance.get("max_task_success_regression", 0.05))
        unit_name = "task attempt identified by (task_id, repetition)"
        workflow_kind = "complete_agent_process"
    else:
        parsed = ExperimentManifest.from_dict(raw_manifest)
        source_cases, unique_tasks = _static_units(parsed)
        ids_unique = True
        repetitions = parsed.execution.repetitions
        primary_metric = parsed.acceptance.primary_metric
        min_success = parsed.acceptance.min_task_success_rate
        margin = parsed.acceptance.max_task_success_regression
        unit_name = "paired case repetition"
        workflow_kind = (
            "sequential_fixture_workflow"
            if parsed.execution.cases_form_workflow
            else "isolated_request_replay"
        )

    planned_pairs = unique_tasks * repetitions
    total_attempts = planned_pairs * 2
    if not ids_unique:
        issues.append(
            _issue(
                "runeconomy.plan.duplicate_task_id",
                "error",
                "dataset.tasks[].task_id",
                "Duplicate task IDs make task-level pairing ambiguous.",
                "Assign one unique stable task ID to every dataset task.",
            )
        )
    if repetitions > 1:
        issues.append(
            _issue(
                "runeconomy.plan.repetitions_not_independent_tasks",
                "warning",
                "execution.repetitions",
                "Repeated attempts add paired units but are not additional unique tasks.",
                "Describe the evidence as repeated task attempts and account for within-task dependence.",
            )
        )

    minimum_successes = _minimum_successes_for_lower_bound(
        planned_pairs, min_success
    )
    all_success_interval = _clopper_pearson_interval(planned_pairs, planned_pairs)
    if minimum_successes is None:
        issues.append(
            _issue(
                "runeconomy.plan.absolute_gate_infeasible",
                "error",
                "acceptance.min_task_success_rate",
                "Even all planned units succeeding cannot clear the configured exact lower bound.",
                "Increase the frozen sample or revise the gate before observing held-out outcomes.",
            )
        )
    paired = _paired_feasibility(planned_pairs, margin)
    if not paired["feasible_in_best_case"]:
        issues.append(
            _issue(
                "runeconomy.plan.paired_gate_infeasible",
                "error",
                "acceptance.max_task_success_regression",
                "The planned pairs cannot clear the paired exact non-inferiority gate even with complete agreement.",
                "Increase the frozen sample or predeclare a scientifically justified margin.",
            )
        )

    claim = planning.get("claim")
    claim = claim if isinstance(claim, Mapping) else {}
    claim_level = _safe_claim_level(claim.get("level"))
    declared_metric = claim.get("primary_metric")
    metric_matches = declared_metric == primary_metric if isinstance(declared_metric, str) else None
    metric_role = (
        "workflow_outcome"
        if primary_metric in {"workflow_latency_ms", "total_latency_ms"}
        else "mechanism_metric"
    )
    if metric_matches is False:
        issues.append(
            _issue(
                "runeconomy.plan.claim_metric_mismatch",
                "error",
                "design.planning.claim.primary_metric",
                "The proposed claim names a different primary metric from the execution gate.",
                "Align the claim and acceptance metric before calibration.",
            )
        )
    if claim_level == "workflow" and metric_role == "mechanism_metric":
        issues.append(
            _issue(
                "runeconomy.plan.mechanism_metric_product_claim",
                "error",
                "design.planning.claim.level",
                "A mechanism metric cannot by itself establish a workflow-level performance claim.",
                "Use end-to-end workflow latency or narrow the claim to the measured mechanism.",
            )
        )
    if mode == "static_request" and claim_level == "workflow":
        issues.append(
            _issue(
                "runeconomy.plan.static_replay_not_live_workflow",
                "warning",
                "design.planning.claim.level",
                "Sequential fixture replay is not a complete stateful live-agent workflow.",
                "Narrow the claim or use agent-process mode for a complete workflow claim.",
            )
        )

    scope = _safe_measurement_scope(planning.get("measurement_scope"))
    if claim_level == "workflow" and any(value is None for value in scope.values()):
        issues.append(
            _issue(
                "runeconomy.plan.measurement_scope_unknown",
                "warning",
                "design.planning.measurement_scope",
                "The design does not declare every relevant workflow measurement boundary.",
                "Declare whether tool time, concurrency, routing, production behavior, and a live agent are measured.",
            )
        )
    for key in ("concurrency", "routing", "production_behavior"):
        if scope[key] is True and (
            mode == "static_request" or key in {"concurrency", "production_behavior"}
        ):
            issues.append(
                _issue(
                    f"runeconomy.plan.unsupported_{key}_claim",
                    "error",
                    f"design.planning.measurement_scope.{key}",
                    f"The declared design does not measure {key.replace('_', ' ')}.",
                    "Remove that claim or use a design that directly measures it.",
                )
            )

    variables = planning.get("intervention_variables")
    variables_valid = bool(
        isinstance(variables, list)
        and all(
            isinstance(value, str) and _SAFE_LABEL.fullmatch(value)
            for value in variables
        )
    )
    variables = variables if variables_valid else []
    if planning.get("intervention_variables") is not None and not variables_valid:
        issues.append(
            _issue(
                "runeconomy.plan.design_schema_unsupported",
                "error",
                "design.planning.intervention_variables",
                "Intervention variables must be a list of stable content-free labels.",
                "Replace runtime or free-form values with fixed labels.",
            )
        )
    variable_count = len(variables)
    if variable_count > 1:
        issues.append(
            _issue(
                "runeconomy.plan.multiple_intervention_variables",
                "error",
                "design.planning.intervention_variables",
                "Multiple declared changes prevent a clean attribution to one intervention.",
                "Isolate one change or explicitly narrow the causal claim.",
            )
        )
    identity_change = planning.get("identity_change")
    if identity_change is True and not any(
        value in {"model_identity", "backend_identity"} for value in variables
    ):
        issues.append(
            _issue(
                "runeconomy.plan.identity_change_undeclared",
                "error",
                "design.planning.identity_change",
                "A model or backend identity change is not named as the intervention.",
                "Declare the identity change as the intervention or keep identity fixed.",
            )
        )

    calibration = planning.get("calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    heldout = planning.get("heldout")
    heldout = heldout if isinstance(heldout, Mapping) else {}
    calibration_digest = calibration.get("artifact_sha256")
    heldout_digest = heldout.get("artifact_sha256")
    calibration_present = bool(
        isinstance(calibration_digest, str) and _SHA256.fullmatch(calibration_digest)
    )
    heldout_present = bool(
        isinstance(heldout_digest, str) and _SHA256.fullmatch(heldout_digest)
    )
    overlap = bool(
        calibration_present
        and heldout_present
        and calibration_digest == heldout_digest
    )
    if overlap:
        issues.append(
            _issue(
                "runeconomy.plan.calibration_heldout_overlap",
                "error",
                "design.planning.heldout.artifact_sha256",
                "Calibration and held-out declarations identify the same artifact.",
                "Freeze a separate unseen workload before held-out execution.",
            )
        )
    operator_unseen = heldout.get("operator_declared_unseen")
    heldout_status = (
        "operator_declared_unseen"
        if operator_unseen is True
        else "operator_declared_consumed"
        if operator_unseen is False
        else "unknown"
    )
    if heldout_status == "unknown":
        issues.append(
            _issue(
                "runeconomy.plan.heldout_status_unknown",
                "warning",
                "design.planning.heldout.operator_declared_unseen",
                "RunEconomy cannot determine whether the workload was previously inspected.",
                "Record the operator declaration without treating it as independently verified.",
            )
        )
    elif heldout_status == "operator_declared_consumed":
        issues.append(
            _issue(
                "runeconomy.plan.heldout_consumed",
                "error",
                "design.planning.heldout.operator_declared_unseen",
                "The operator declaration says this workload has already been consumed.",
                "Use a new unseen workload for any fresh held-out claim.",
            )
        )

    stopping = planning.get("stopping_rule")
    stopping = stopping if isinstance(stopping, Mapping) else {}
    stopping_kind = stopping.get("kind")
    stopping_declared = stopping_kind in {"fixed_sample", "predeclared_adaptive"}
    favorable_dependent = stopping.get("favorable_result_dependent") is True
    if favorable_dependent:
        issues.append(
            _issue(
                "runeconomy.plan.outcome_dependent_stopping",
                "error",
                "design.planning.stopping_rule",
                "The stopping rule depends on observing a favorable result.",
                "Use a fixed sample or a predeclared outcome-independent rule.",
            )
        )

    required_design = {
        "hypothesis": hypothesis_present,
        "intervention": intervention_present,
        "predicted_direction": predicted_direction != "unknown",
        "claim_level": claim_level != "unknown",
        "claim_metric": metric_matches is True,
    }
    for field, present in required_design.items():
        if not present:
            issues.append(
                _issue(
                    "runeconomy.plan.design_input_missing",
                    "warning",
                    f"design.planning.{field}",
                    f"The design does not declare {field.replace('_', ' ')}.",
                    "Add the declaration before treating the design as calibration-ready.",
                )
            )

    power = _latency_power(planning, planned_pairs)
    if power["status"] == "unknown":
        issues.append(
            _issue(
                "runeconomy.plan.latency_power_unknown",
                "warning",
                "design.planning.latency_assumptions",
                "Latency power is unknown without a paired effect and variability assumption.",
                "Estimate both from calibration before freezing the held-out sample.",
            )
        )
    elif (
        predicted_direction == "decrease"
        and power["expected_difference_ms"] > 0
    ) or (
        predicted_direction == "increase"
        and power["expected_difference_ms"] < 0
    ):
        issues.append(
            _issue(
                "runeconomy.plan.assumption_direction_mismatch",
                "error",
                "design.planning.latency_assumptions.expected_difference_ms",
                "The latency assumption points opposite to the predicted direction.",
                "Correct the predeclared sign or the predicted direction before calibration.",
            )
        )
    elif power["planned_pairs_meet_estimate"] is False:
        issues.append(
            _issue(
                "runeconomy.plan.latency_sample_below_assumption",
                "error",
                "design.planning.latency_assumptions",
                "The planned pairs are below the declared latency scenario estimate.",
                "Increase the frozen sample or recalibrate the effect and variability assumptions.",
            )
        )

    freeze_checks = {
        "calibration_evidence": calibration_present,
        "separate_heldout_workload": heldout_present and not overlap,
        "operator_declared_unseen": operator_unseen is True,
        "evaluator_frozen": planning.get("evaluator_frozen") is True,
        "gates_frozen": planning.get("gates_frozen") is True,
        "stopping_rule_frozen": stopping_declared and not favorable_dependent,
        "claim_boundary": _presence(planning.get("claim_boundary")),
        "single_intervention_variable": variable_count == 1,
    }
    if calibration_present or heldout_present:
        for field, present in freeze_checks.items():
            if not present and field not in {
                "separate_heldout_workload",
                "operator_declared_unseen",
                "single_intervention_variable",
            }:
                issues.append(
                    _issue(
                        "runeconomy.plan.freeze_input_missing",
                        "warning",
                        f"design.planning.{field}",
                        f"The design has not established {field.replace('_', ' ')}.",
                        "Complete and review the freeze declaration before held-out execution.",
                    )
                )
    has_errors = any(issue["severity"] == "error" for issue in issues)
    if has_errors:
        state = "blocked"
        next_action = "runeconomy.action.repair_experiment_design"
    elif not all(required_design.values()):
        state = "needs_design_input"
        next_action = "runeconomy.action.complete_design_declarations"
    elif all(freeze_checks.values()):
        state = "ready_to_freeze"
        next_action = "runeconomy.action.review_and_freeze_design"
    else:
        state = "ready_for_calibration"
        next_action = "runeconomy.action.run_free_calibration_first"

    return {
        "artifact_type": "runeconomy.experiment_plan",
        "schema_version": PLAN_SCHEMA_VERSION,
        "state": state,
        "execution_readiness": {
            "status": validation["status"],
            "runnable": True,
        },
        "proposed_claim": {
            "level": claim_level,
            "primary_metric": primary_metric,
            "metric_role": metric_role,
            "matches_declared_metric": metric_matches,
        },
        "experimental_unit": {
            "name": unit_name,
            "pairing_key": ["task_id", "repetition"],
            "workflow_kind": workflow_kind,
            "repetitions_are_independent_tasks": False,
            "absolute_success_unit": "task_attempt",
            "paired_success_unit": "baseline_candidate_task_attempt_pair",
            "latency_unit": "both_successful_baseline_candidate_pair",
        },
        "sample": {
            "source_case_count": source_cases,
            "unique_task_count": unique_tasks,
            "task_ids_unique": ids_unique,
            "baseline_candidate_use_same_tasks": True,
            "repetitions": repetitions,
            "planned_pairs": planned_pairs,
            "total_attempts": total_attempts,
            "maximum_analyzable_latency_pairs": planned_pairs,
        },
        "task_success": {
            "confidence_level": 0.95,
            "absolute_method": "Clopper-Pearson exact binomial",
            "minimum_lower_bound": min_success,
            "minimum_successful_units": minimum_successes,
            "planned_units": planned_pairs,
            "all_success_lower_bound": (
                all_success_interval[0] if all_success_interval else None
            ),
            "all_success_is_not_true_100_percent": True,
            "paired_noninferiority": paired,
        },
        "performance": {
            "primary_metric": primary_metric,
            "metric_role": metric_role,
            "conditional_on_both_variants_succeeding": True,
            "task_failures_reduce_analyzable_pairs": True,
            "maximum_analyzable_pairs": planned_pairs,
            "power": power,
        },
        "design_declarations": {
            "sha256": _digest(planning),
            "planning_schema_present": bool(planning),
            "hypothesis": hypothesis_present,
            "intervention": intervention_present,
            "predicted_direction": predicted_direction,
            "calibration_evidence": calibration_present,
            "heldout_workload": heldout_present,
            "heldout_status": heldout_status,
            "heldout_status_is_operator_declared": operator_unseen is not None,
            "evaluator_frozen": planning.get("evaluator_frozen") is True,
            "gates_frozen": planning.get("gates_frozen") is True,
            "stopping_rule": stopping_kind if stopping_declared else "unknown",
            "claim_boundary": _presence(planning.get("claim_boundary")),
            "intervention_variable_count": variable_count,
            "measurement_scope": scope,
            "freeze_checks": freeze_checks,
        },
        "issues": issues,
        "next_action_code": next_action,
    }
