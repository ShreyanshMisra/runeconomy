"""Read-only experiment-design assessment without execution or network access."""

from __future__ import annotations

import hashlib
import json
import math
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .binomial import binomial_lower_bound
from .completion_policy import (
    CompletionPolicy,
    CompletionPolicyError,
    VARIANT_NON_COMPLETION,
)
from .comparison import agent_arrival_schedule
from .execution import CLOSED_LOOP, DRAIN, SUSTAIN, LoadSpec
from .experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    _clopper_pearson_interval,
    _paired_success_statistics,
    static_arrival_schedule,
)
from .metric_gates import (
    MetricGate,
    MetricGateError,
    minimum_sample,
    parse_metric_gates,
)
from .paired_inference import (
    PAIRED_EXACT,
    PAIRED_METHODS,
    PAIRED_SCORE_EXACT,
    PAIRED_SCORE_EXACT_LABEL,
    PairedInferenceError,
    asymptotic_score_pairs,
    binomial_cdf,
    completion_tolerance,
    gate_power,
    method_properties,
    normal_cdf,
    paired_cell_probabilities,
    score_exact_lower_bound,
    score_exact_rejects,
)
from .workspaces import agent_load_spec, load_workspace, validate_experiment_artifact


PLAN_SCHEMA_VERSION = "0.1"
# Planning input 0.2 adds success, completion-failure, and cost assumptions.
# Their checks apply only to designs that declare 0.2, so plans for existing
# 0.1 designs keep their states and issues.
PLANNING_INPUT_VERSIONS = frozenset({"0.1", "0.2"})
POWERED_PLANNING_VERSION = "0.2"
_POWERED_PLANNING_KEYS = (
    "success_assumptions",
    "completion_model",
    "cost_assumptions",
    "target_power",
)
_ASSUMPTION_SOURCES = frozenset({"calibration", "scenario"})
DEFAULT_TARGET_POWER = 0.8
DEFAULT_COMPLETION_PASS_PROBABILITY = 0.95
MAX_RECOMMENDED_PAIRS = 2000
PLANNING_ALPHA = 0.05
PREREGISTRATION_TEMPLATE_VERSION = "0.1"
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
    """Return the first success count whose existing exact lower bound passes.

    The lower bound is monotone in the success count, so a binary search finds
    the same count as a linear scan.
    """

    def lower(successes: int) -> float:
        interval = _clopper_pearson_interval(successes, units, confidence=confidence)
        assert interval is not None
        return interval[0]

    if lower(units) < threshold:
        return None
    low, high = 0, units
    while low < high:
        middle = (low + high) // 2
        if lower(middle) >= threshold:
            high = middle
        else:
            low = middle + 1
    return low


def _completion_tolerance(units: int, threshold: float) -> dict[str, Any]:
    """Describe the exact integer consequence of a completion-rate gate."""

    minimum_completed = min(units, max(0, math.ceil(threshold * units - 1e-12)))
    maximum_failures = units - minimum_completed
    return {
        "threshold": threshold,
        "planned_attempts_per_variant": units,
        "minimum_completed_attempts_per_variant": minimum_completed,
        "maximum_failed_or_incomplete_attempts_per_variant": maximum_failures,
        "warning": (
            f"A completion threshold of {threshold:g} requires at least "
            f"{minimum_completed} of {units} completed attempts per variant and "
            f"tolerates exactly {maximum_failures} failed or incomplete "
            f"attempt{'s' if maximum_failures != 1 else ''} per variant."
        ),
    }


def _paired_feasibility(
    pairs: int,
    margin: float,
    *,
    confidence: float = 0.95,
    method: str = PAIRED_EXACT,
) -> dict[str, Any]:
    if method == PAIRED_SCORE_EXACT:
        return _score_feasibility(pairs, margin, confidence=confidence)
    best = _paired_success_statistics(
        [(True, True)] * pairs, confidence=confidence
    )
    best_lower = best["one_sided_lower_bound"]
    max_baseline_only = None
    scenarios = []

    def minimum_pairs(baseline_only: int) -> int | None:
        for candidate_pairs in range(max(1, baseline_only), 5001):
            outcomes = [(True, False)] * baseline_only + [
                (True, True)
            ] * (candidate_pairs - baseline_only)
            candidate_statistics = _paired_success_statistics(
                outcomes, confidence=confidence
            )
            candidate_lower = candidate_statistics["one_sided_lower_bound"]
            if candidate_lower is not None and candidate_lower >= -margin:
                return candidate_pairs
        return None

    minimum_pairs_by_baseline_only = {
        str(baseline_only): minimum_pairs(baseline_only)
        for baseline_only in (0, 1, 2)
    }
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
                    "minimum_pairs_with_zero_candidate_only": (
                        minimum_pairs_by_baseline_only.get(str(baseline_only))
                    ),
                }
            )
    return {
        "method": best["method"],
        "method_id": PAIRED_EXACT,
        "confidence_level": confidence,
        "margin": margin,
        "best_case_one_sided_lower_bound": best_lower,
        "feasible_in_best_case": bool(
            best_lower is not None and best_lower >= -margin
        ),
        "max_baseline_only_with_zero_candidate_only": max_baseline_only,
        "minimum_pairs_by_baseline_only_with_zero_candidate_only": (
            minimum_pairs_by_baseline_only
        ),
        "scenario_table": scenarios,
        "discordance_explanation": (
            "Candidate-only outcomes improve the paired difference. Baseline-only "
            "outcomes weaken it. Agreement still leaves finite-sample uncertainty."
        ),
    }


def _score_minimum_pairs(
    baseline_only: int, margin: float, *, alpha: float, limit: int = 5000
) -> int | None:
    """Smallest pairs at which ``baseline_only`` losses and no wins still pass.

    Found by doubling and bisection, then a short downward check for the
    sawtooth that exact tests show in ``n``.
    """

    def passes(pairs: int) -> bool:
        return score_exact_rejects(0, baseline_only, pairs, margin, alpha=alpha)

    low = max(1, baseline_only)
    if passes(low):
        return low
    high = low
    while True:
        next_high = min(limit, high * 2)
        if passes(next_high):
            high = next_high
            break
        if next_high >= limit:
            return None
        low = high = next_high
    while high - low > 1:
        middle = (low + high) // 2
        if passes(middle):
            high = middle
        else:
            low = middle
    for candidate in range(high - 1, max(low, high - 4), -1):
        if passes(candidate):
            high = candidate
    return high


def _score_feasibility(
    pairs: int, margin: float, *, confidence: float = 0.95
) -> dict[str, Any]:
    alpha = 1.0 - confidence

    def lower(baseline_only: int) -> float:
        return score_exact_lower_bound(
            0, baseline_only, pairs, alpha=alpha, anchor=-margin, tolerance=1e-5
        )

    def passes(baseline_only: int) -> bool:
        return score_exact_rejects(0, baseline_only, pairs, margin, alpha=alpha)

    best_lower = lower(0)
    max_baseline_only: int | None = None
    for baseline_only in range(pairs + 1):
        if not passes(baseline_only):
            break
        max_baseline_only = baseline_only
    minimum_pairs = {
        str(baseline_only): _score_minimum_pairs(baseline_only, margin, alpha=alpha)
        for baseline_only in (0, 1, 2)
    }
    scenarios = []
    for baseline_only in sorted({0, 1, 2, pairs}):
        if baseline_only > pairs:
            continue
        scenarios.append(
            {
                "baseline_only": baseline_only,
                "candidate_only": 0,
                "one_sided_lower_bound": lower(baseline_only),
                "passes": bool(
                    max_baseline_only is not None and baseline_only <= max_baseline_only
                ),
                "minimum_pairs_with_zero_candidate_only": minimum_pairs.get(
                    str(baseline_only)
                ),
            }
        )
    return {
        "method": PAIRED_SCORE_EXACT_LABEL,
        "method_id": PAIRED_SCORE_EXACT,
        "confidence_level": confidence,
        "margin": margin,
        "best_case_one_sided_lower_bound": best_lower,
        "feasible_in_best_case": passes(0),
        "max_baseline_only_with_zero_candidate_only": max_baseline_only,
        "minimum_pairs_by_baseline_only_with_zero_candidate_only": minimum_pairs,
        "scenario_table": scenarios,
        "discordance_explanation": (
            "Candidate-only outcomes improve the paired difference. Baseline-only "
            "outcomes weaken it. Agreement still leaves finite-sample uncertainty."
        ),
    }


def recommend_pairs(
    *,
    margin: float,
    expected_success_rate: float,
    expected_discordance_rate: float,
    method: str = PAIRED_SCORE_EXACT,
    target_power: float = DEFAULT_TARGET_POWER,
    alpha: float = PLANNING_ALPHA,
    expected_difference: float = 0.0,
    min_task_success_rate: float = 0.0,
    max_pairs: int = MAX_RECOMMENDED_PAIRS,
) -> dict[str, Any]:
    """Minimum paired units for a target power under one preregistered method.

    Power is exact under the declared cell probabilities (see
    ``paired_inference.gate_power``). With an absolute success floor, the
    target applies to the joint probability that both absolute gates and the
    paired gate pass.

    Exact tests have a sawtooth power curve in ``n``: one more pair can lower
    power. ``minimum_pairs`` is therefore the stable minimum, the smallest
    count from which every count up to the top of the scanned window meets the
    target (the window extends at least 5% above the bracketed crossing).
    Isolated smaller counts within five pairs below it that also meet the
    target are listed separately, because losing a single pair to attrition
    there would drop power below the target.
    """

    if method not in PAIRED_METHODS:
        raise PairedInferenceError(f"sample size is defined only for {sorted(PAIRED_METHODS)}")
    if not 0.0 < target_power < 1.0:
        raise PairedInferenceError("target_power must be in (0, 1)")
    cells = paired_cell_probabilities(
        expected_success_rate=expected_success_rate,
        expected_discordance_rate=expected_discordance_rate,
        expected_difference=expected_difference,
    )
    gate = "quality_gates" if min_task_success_rate > 0 else "noninferiority"
    evaluated: dict[int, dict[str, Any]] = {}

    def power(pairs: int) -> dict[str, Any]:
        if pairs not in evaluated:
            evaluated[pairs] = gate_power(
                pairs=pairs,
                method=method,
                margin=margin,
                cells=cells,
                min_task_success_rate=min_task_success_rate,
                alpha=alpha,
            )
        return evaluated[pairs]

    def meets(pairs: int) -> bool:
        return power(pairs)[gate] >= target_power

    base = {
        "method": method,
        "gate": gate,
        "target_power": target_power,
        "alpha": alpha,
        "margin": margin,
        "min_task_success_rate": min_task_success_rate,
        "assumptions": {
            "expected_success_rate": expected_success_rate,
            "expected_discordance_rate": expected_discordance_rate,
            "expected_difference": expected_difference,
            "cell_probabilities": cells,
        },
        "max_pairs_searched": max_pairs,
    }
    start = asymptotic_score_pairs(
        margin=margin, cells=cells, alpha=alpha, power=target_power
    )
    if start is None:
        return {
            **base,
            "minimum_pairs": None,
            "reason": "the expected difference is at or below the margin, so no sample can power the gate",
        }
    start = min(max(start, 2), max_pairs)

    def unreachable() -> dict[str, Any]:
        return {
            **base,
            "minimum_pairs": None,
            "asymptotic_starting_point": start,
            "power_at_max_pairs": power(max_pairs),
            "reason": f"the target power is not reached within {max_pairs} pairs",
        }

    # Bracket the envelope crossing: ``failing`` misses the target, ``meeting``
    # reaches it.
    meeting = start
    while not meets(meeting):
        if meeting >= max_pairs:
            return unreachable()
        meeting = min(max_pairs, max(meeting + 1, math.ceil(meeting * 1.2)))
    failing = 1
    while meeting > 2:
        smaller = max(2, int(meeting / 1.2))
        if not meets(smaller):
            failing = smaller
            break
        meeting = smaller
    while meeting - failing > 1:
        middle = (meeting + failing) // 2
        if meets(middle):
            meeting = middle
        else:
            failing = middle
    window = max(3, math.ceil(0.05 * meeting))
    top = min(max_pairs, meeting + window)
    while not meets(top):
        if top >= max_pairs:
            return unreachable()
        top = min(max_pairs, top + window)
    stable = top
    while stable > 2 and meets(stable - 1):
        stable -= 1
    isolated = [
        count
        for count in range(stable - 2, max(1, stable - 7), -1)
        if meets(count)
    ]
    return {
        **base,
        "minimum_pairs": stable,
        "power_at_minimum": power(stable),
        "stable_through_pairs": top,
        "isolated_smaller_pairs_meeting_target": sorted(isolated),
        "asymptotic_starting_point": start,
        "search": (
            "normal-approximation start, geometric bracketing and bisection of the "
            "crossing, then a downward scan from at least 5% above it so that every "
            "count from minimum_pairs to stable_through_pairs meets the target"
        ),
    }


def _probability_value(
    block: Mapping[str, Any],
    name: str,
    *,
    field_path: str,
    issues: list[dict[str, str]],
    default: float | None = None,
    open_low: bool = False,
    open_high: bool = False,
) -> float | None:
    value = block.get(name, default)
    if value is None:
        return None
    valid = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (0.0 < float(value) if open_low else 0.0 <= float(value))
        and (float(value) < 1.0 if open_high else float(value) <= 1.0)
    )
    if not valid:
        issues.append(
            _issue(
                "runeconomy.plan.design_schema_unsupported",
                "error",
                f"{field_path}.{name}",
                "A declared planning rate is not a probability in the supported range.",
                "Declare the value as a number between 0 and 1.",
            )
        )
        return None
    return float(value)


def _assumption_source(
    block: Mapping[str, Any],
    *,
    field_path: str,
    heldout_digest: Any,
    issues: list[dict[str, str]],
) -> str:
    """Return the declared source, rejecting anything derived from held-out data."""

    source = block.get("source")
    digest = block.get("artifact_sha256")
    from_heldout = source not in _ASSUMPTION_SOURCES or (
        isinstance(digest, str)
        and isinstance(heldout_digest, str)
        and digest == heldout_digest
    )
    if from_heldout:
        issues.append(
            _issue(
                "runeconomy.plan.assumption_not_from_calibration",
                "error",
                f"{field_path}.source",
                "Planning rates must come from calibration or a declared scenario, never from a held-out result.",
                "Re-estimate the rates on calibration data, or declare them as an explicit scenario.",
            )
        )
        return "invalid"
    if source == "calibration" and digest is not None and not (
        isinstance(digest, str) and _SHA256.fullmatch(digest)
    ):
        issues.append(
            _issue(
                "runeconomy.plan.design_schema_unsupported",
                "error",
                f"{field_path}.artifact_sha256",
                "The calibration artifact digest is malformed.",
                "Use a sha256:<64 hex> digest of the calibration artifact.",
            )
        )
    return str(source)


def _powered_inputs(
    planning: Mapping[str, Any],
    *,
    version: Any,
    heldout_digest: Any,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    """Parse planning-0.2 assumptions; reject them under planning 0.1."""

    declared = [key for key in _POWERED_PLANNING_KEYS if key in planning]
    if version != POWERED_PLANNING_VERSION:
        for key in declared:
            issues.append(
                _issue(
                    "runeconomy.plan.design_schema_unsupported",
                    "error",
                    f"design.planning.{key}",
                    "Power, completion-failure, and cost assumptions require planning schema 0.2.",
                    "Set design.planning.schema_version to 0.2 before declaring them.",
                )
            )
        return {"powered": False}
    result: dict[str, Any] = {"powered": True}
    target = planning.get("target_power", DEFAULT_TARGET_POWER)
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not 0.5 <= float(target) < 1.0
    ):
        issues.append(
            _issue(
                "runeconomy.plan.design_schema_unsupported",
                "error",
                "design.planning.target_power",
                "Target power must be at least 0.5 and below 1.",
                "Declare a target power such as 0.8 or 0.9.",
            )
        )
        target = DEFAULT_TARGET_POWER
    result["target_power"] = float(target)

    for key in ("success_assumptions", "completion_model", "cost_assumptions"):
        value = planning.get(key)
        if value is not None and not isinstance(value, Mapping):
            issues.append(
                _issue(
                    "runeconomy.plan.design_schema_unsupported",
                    "error",
                    f"design.planning.{key}",
                    "The planning assumption block must be an object.",
                    "Declare the block as an object or remove it.",
                )
            )

    success = planning.get("success_assumptions")
    if isinstance(success, Mapping):
        path = "design.planning.success_assumptions"
        source = _assumption_source(
            success, field_path=path, heldout_digest=heldout_digest, issues=issues
        )
        rate = _probability_value(
            success, "expected_success_rate", field_path=path, issues=issues, open_low=True
        )
        discordance = _probability_value(
            success, "expected_discordance_rate", field_path=path, issues=issues
        )
        difference = success.get("expected_difference", 0.0)
        if (
            isinstance(difference, bool)
            or not isinstance(difference, (int, float))
            or not -1.0 < float(difference) < 1.0
        ):
            issues.append(
                _issue(
                    "runeconomy.plan.design_schema_unsupported",
                    "error",
                    f"{path}.expected_difference",
                    "The expected candidate-minus-baseline difference must be in (-1, 1).",
                    "Declare 0 for no expected change, or a calibrated difference.",
                )
            )
            difference = None
        if rate is not None and discordance is not None and difference is not None:
            try:
                cells = paired_cell_probabilities(
                    expected_success_rate=rate,
                    expected_discordance_rate=discordance,
                    expected_difference=float(difference),
                )
            except PairedInferenceError:
                issues.append(
                    _issue(
                        "runeconomy.plan.success_assumptions_inconsistent",
                        "error",
                        path,
                        "The success rate, discordance rate, and expected difference cannot all hold at once.",
                        "Check that discordance covers the difference and fits within the success and failure rates.",
                    )
                )
            else:
                result["success"] = {
                    "source": source,
                    "expected_success_rate": rate,
                    "expected_discordance_rate": discordance,
                    "expected_difference": float(difference),
                    "cell_probabilities": cells,
                }

    completion_model = planning.get("completion_model")
    if isinstance(completion_model, Mapping):
        path = "design.planning.completion_model"
        source = _assumption_source(
            completion_model,
            field_path=path,
            heldout_digest=heldout_digest,
            issues=issues,
        )
        infrastructure = _probability_value(
            completion_model, "infrastructure_failure_rate", field_path=path, issues=issues
        )
        application = _probability_value(
            completion_model, "application_failure_rate", field_path=path, issues=issues
        )
        pass_probability = _probability_value(
            completion_model,
            "pass_probability",
            field_path=path,
            issues=issues,
            default=DEFAULT_COMPLETION_PASS_PROBABILITY,
            open_low=True,
            open_high=True,
        )
        if None not in (infrastructure, application, pass_probability):
            result["completion_model"] = {
                "source": source,
                "infrastructure_failure_rate": infrastructure,
                "application_failure_rate": application,
                "pass_probability": pass_probability,
            }

    cost = planning.get("cost_assumptions")
    if isinstance(cost, Mapping):
        path = "design.planning.cost_assumptions"
        source = _assumption_source(
            cost, field_path=path, heldout_digest=heldout_digest, issues=issues
        )
        per_attempt = cost.get("usd_per_attempt")
        fixed = cost.get("fixed_usd", 0.0)
        valid = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in (per_attempt, fixed)
        )
        if not valid:
            issues.append(
                _issue(
                    "runeconomy.plan.design_schema_unsupported",
                    "error",
                    path,
                    "Cost assumptions must be non-negative US-dollar amounts.",
                    "Declare usd_per_attempt and optional fixed_usd as numbers.",
                )
            )
        else:
            result["cost"] = {
                "source": source,
                "usd_per_attempt": float(per_attempt),
                "fixed_usd": float(fixed),
            }
    return result


def _completion_model_section(
    model: Mapping[str, Any] | None,
    *,
    attempts_per_variant: int,
    declared_threshold: float,
    policy: CompletionPolicy | None,
    powered: bool,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    """Derive the integer completion tolerance from the expected-failure model."""

    declared = _completion_tolerance(attempts_per_variant, declared_threshold)
    declared_tolerance = declared["maximum_failed_or_incomplete_attempts_per_variant"]
    classification = (
        policy.common_mode_paired_failures if policy is not None else VARIANT_NON_COMPLETION
    )
    if model is None:
        if powered:
            issues.append(
                _issue(
                    "runeconomy.plan.completion_failure_model_unknown",
                    "warning",
                    "design.planning.completion_model",
                    "The completion threshold is not derived from an expected-failure model.",
                    "Declare calibration infrastructure and application failure rates so the planner can derive the integer tolerance.",
                )
            )
        return {
            "status": "unknown",
            "reason": "no expected-failure model was declared",
            "common_mode_paired_failures": classification,
        }
    rate = min(
        1.0, model["infrastructure_failure_rate"] + model["application_failure_rate"]
    )
    target = model["pass_probability"]
    derived = completion_tolerance(attempts_per_variant, rate, target)
    derived_rate = (attempts_per_variant - derived) / attempts_per_variant
    declared_pass = binomial_cdf(declared_tolerance, attempts_per_variant, rate)
    derived_pass = binomial_cdf(derived, attempts_per_variant, rate)
    issues.append(
        _issue(
            "runeconomy.plan.completion_model_tolerance",
            "warning",
            "acceptance.min_completed_rate",
            (
                f"An expected non-completion rate of {rate:.4%} per attempt gives an "
                f"integer tolerance of {derived} failed or incomplete attempt"
                f"{'s' if derived != 1 else ''} per variant out of "
                f"{attempts_per_variant} (min_completed_rate "
                f"{derived_rate:.6g}) at {target:.0%} pass probability per variant."
            ),
            "Freeze the model-derived tolerance; never derive it from a held-out result.",
        )
    )
    if declared_tolerance < derived:
        issues.append(
            _issue(
                "runeconomy.plan.completion_threshold_stricter_than_failure_model",
                "warning",
                "acceptance.min_completed_rate",
                (
                    f"The declared threshold tolerates {declared_tolerance} per variant; "
                    "under the expected-failure model both variants pass it with "
                    f"probability {declared_pass * declared_pass:.3f}, so an otherwise "
                    "clean result is likely to be invalid."
                ),
                f"Set min_completed_rate to {derived_rate:.6g} before freezing.",
            )
        )
    elif declared_tolerance > derived:
        issues.append(
            _issue(
                "runeconomy.plan.completion_threshold_looser_than_failure_model",
                "warning",
                "acceptance.min_completed_rate",
                (
                    f"The declared threshold tolerates {declared_tolerance} per variant, "
                    f"more than the {derived} the expected-failure model needs."
                ),
                f"Set min_completed_rate to {derived_rate:.6g}, or document why more failures are acceptable.",
            )
        )
    return {
        "status": "derived",
        "source": model["source"],
        "infrastructure_failure_rate": model["infrastructure_failure_rate"],
        "application_failure_rate": model["application_failure_rate"],
        "per_attempt_noncompletion_rate": rate,
        "pass_probability_target_per_variant": target,
        "expected_noncompleted_attempts_per_variant": rate * attempts_per_variant,
        "derived_max_noncompleted_attempts_per_variant": derived,
        "derived_min_completed_rate": derived_rate,
        "derived_pass_probability_per_variant": derived_pass,
        "declared_max_noncompleted_attempts_per_variant": declared_tolerance,
        "declared_pass_probability_per_variant": declared_pass,
        "declared_pass_probability_both_variants": declared_pass * declared_pass,
        "declared_matches_model": declared_tolerance == derived,
        "common_mode_paired_failures": classification,
        "assumptions": [
            "Attempts fail independently at the declared per-attempt rate.",
            "Both-variant pass probabilities multiply the per-variant probabilities, which is a lower bound under positively associated failures.",
            "Application failures are counted in full even when a task_outcome policy will reclassify common-mode pairs, so the tolerance is not tightened by an unverified common-mode share.",
        ],
        "rule": "Completion thresholds come from calibration failure rates, never from a held-out result.",
    }


REGRESSION_DETECTION = "regression"
DETECTION_SEARCH_LIMIT = 2000


def _binomial_pmf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0 if successes == 0 else 0.0
    if probability >= 1.0:
        return 1.0 if successes == trials else 0.0
    log = (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        + successes * math.log(probability)
        + (trials - successes) * math.log1p(-probability)
    )
    return math.exp(log)


def regression_detection_power(
    pairs: int, *, baseline_rate: float, candidate_rate: float, floor: float
) -> float:
    """Exact probability that the baseline clears the absolute floor and the
    candidate does not.

    Both use the lower end of the two-sided 95% Clopper-Pearson interval, the
    absolute-success gate the decision applies. The decision also returns
    ``regression`` on a point-estimate drop beyond the margin, so this is a
    lower bound on the probability of a ``regression`` verdict. The arms are
    treated as independent, which is conservative for positively correlated
    pairs.
    """

    def clears(rate: float) -> float:
        return sum(
            _binomial_pmf(x, pairs, rate)
            for x in range(pairs + 1)
            if binomial_lower_bound(x, pairs, PLANNING_ALPHA / 2) >= floor
        )

    return clears(baseline_rate) * (1.0 - clears(candidate_rate))


def _detection_section(
    success: Mapping[str, Any], *, planned_pairs: int, target: float, floor: float
) -> dict[str, Any]:
    baseline = float(success["expected_success_rate"])
    candidate = max(0.0, min(1.0, baseline + float(success["expected_difference"])))

    def power(pairs: int) -> float:
        return regression_detection_power(pairs, baseline_rate=baseline, candidate_rate=candidate, floor=floor)

    minimum = next(
        (pairs for pairs in range(5, DETECTION_SEARCH_LIMIT + 1) if power(pairs) >= target), None
    )
    return {
        "detection_target": REGRESSION_DETECTION,
        "expected_baseline_success_rate": baseline,
        "expected_candidate_success_rate": candidate,
        "absolute_floor": floor,
        "planned_detection_power": power(planned_pairs),
        "minimum_pairs_for_target_power": minimum,
        "rule": (
            "P(baseline two-sided 95% Clopper-Pearson lower bound >= floor) x "
            "P(candidate lower bound < floor); a lower bound on P(regression verdict)"
        ),
    }


def _sample_size_section(
    powered: Mapping[str, Any],
    *,
    method: str,
    margin: float,
    min_success: float,
    planned_pairs: int,
    completion_section: Mapping[str, Any],
    latency_assumptions: Any,
    issues: list[dict[str, str]],
    detection_target: Any = None,
) -> dict[str, Any]:
    if detection_target not in (None, REGRESSION_DETECTION):
        issues.append(
            _issue(
                "runeconomy.plan.detection_target_unsupported",
                "error",
                "design.planning.detection_target",
                "Only a regression detection target is supported.",
                "Remove detection_target or set it to regression.",
            )
        )
        detection_target = None
    if not powered.get("powered"):
        return {
            "status": "unknown",
            "reason": "task-success power requires planning schema 0.2 success assumptions",
        }
    success = powered.get("success")
    if success is None:
        issues.append(
            _issue(
                "runeconomy.plan.success_power_unknown",
                "warning",
                "design.planning.success_assumptions",
                "Task-success power is unknown without an expected success rate and paired discordance rate.",
                "Estimate both from calibration before freezing the held-out sample.",
            )
        )
        return {
            "status": "unknown",
            "reason": "expected success and discordance rates were not declared",
            "calibration_needed": [
                "expected baseline task-success rate",
                "expected paired discordance rate",
            ],
        }
    if method not in PAIRED_METHODS:
        return {
            "status": "not_applicable",
            "reason": f"{method} has no sampling-error control, so power is undefined",
        }
    target = float(powered["target_power"])
    cells = success["cell_probabilities"]
    alpha = PLANNING_ALPHA

    def recommendation(name: str) -> dict[str, Any]:
        return recommend_pairs(
            margin=margin,
            expected_success_rate=success["expected_success_rate"],
            expected_discordance_rate=success["expected_discordance_rate"],
            expected_difference=success["expected_difference"],
            method=name,
            target_power=target,
            alpha=alpha,
            min_task_success_rate=min_success,
        )

    comparison: dict[str, Any] = {}
    for name in (PAIRED_EXACT, PAIRED_SCORE_EXACT):
        planned = gate_power(
            pairs=planned_pairs,
            method=name,
            margin=margin,
            cells=cells,
            min_task_success_rate=min_success,
            alpha=alpha,
        )
        recommended = recommendation(name)
        comparison[name] = {
            "planned_power": planned,
            "minimum_pairs": recommended.get("minimum_pairs"),
            "power_at_minimum": recommended.get("power_at_minimum"),
        }
    declared = comparison[method]
    gate = "quality_gates" if min_success > 0 else "noninferiority"
    components: dict[str, float] = {"task_success_gates": declared["planned_power"][gate]}
    if completion_section.get("status") == "derived":
        components["completion_both_variants"] = completion_section[
            "declared_pass_probability_both_variants"
        ]
    latency = _latency_gate_power(
        latency_assumptions, planned_pairs * cells["both_success"]
    )
    if latency is not None:
        components["latency"] = latency
    product = 1.0
    for value in components.values():
        product *= value
    lower = max(0.0, 1.0 - sum(1.0 - value for value in components.values()))
    detection = None
    if detection_target == REGRESSION_DETECTION:
        # A caught-regression design is powered to detect the declared harm,
        # not to pass quality gates it is predicted to fail.
        detection = _detection_section(success, planned_pairs=planned_pairs, target=target, floor=min_success)
        if detection["planned_detection_power"] < target:
            issues.append(
                _issue(
                    "runeconomy.plan.regression_detection_below_target_power",
                    "error",
                    "execution.repetitions",
                    (
                        f"The planned {planned_pairs} pairs detect the declared regression with probability "
                        f"{detection['planned_detection_power']:.3f}, below the {target:.0%} target."
                    ),
                    "Increase the frozen sample before observing held-out outcomes.",
                )
            )
    elif declared["minimum_pairs"] is None or planned_pairs < declared["minimum_pairs"]:
        issues.append(
            _issue(
                "runeconomy.plan.success_sample_below_target_power",
                "error",
                "execution.repetitions",
                (
                    f"The planned {planned_pairs} pairs give {declared['planned_power'][gate]:.3f} "
                    f"task-success gate power under {method}, below the {target:.0%} target."
                ),
                (
                    "Increase the frozen sample"
                    + (
                        f" to at least {declared['minimum_pairs']} pairs"
                        if declared["minimum_pairs"] is not None
                        else ""
                    )
                    + ", or revise the margin before observing held-out outcomes."
                ),
            )
        )
    return {
        "status": "scenario_estimate",
        "method_id": method,
        "alpha": alpha,
        "gate": gate,
        "target_power": target,
        "assumptions": {
            key: success[key]
            for key in (
                "source",
                "expected_success_rate",
                "expected_discordance_rate",
                "expected_difference",
                "cell_probabilities",
            )
        },
        "planned_pairs": planned_pairs,
        "planned_power": {
            **declared["planned_power"],
            "components": components,
            "all_gates_independence_approximation": product,
            "all_gates_lower_bound": lower,
        },
        "minimum_pairs_for_target_power": (
            detection["minimum_pairs_for_target_power"] if detection else declared["minimum_pairs"]
        ),
        "power_at_minimum_pairs": declared["power_at_minimum"],
        "planned_pairs_meet_target": bool(
            detection["planned_detection_power"] >= target
            if detection
            else declared["minimum_pairs"] is not None and planned_pairs >= declared["minimum_pairs"]
        ),
        **({"regression_detection": detection} if detection else {}),
        "method_comparison": comparison,
        "limitations": [
            "Power is exact under the declared cell probabilities, which are themselves estimates.",
            "Repeated attempts on one task are treated as independent pairs; within-task dependence reduces effective power.",
            "The all-gates lower bound needs no independence assumption; the product assumes independent gates.",
        ],
    }


def _latency_gate_power(assumptions: Any, analyzable_pairs: float) -> float | None:
    if not isinstance(assumptions, Mapping):
        return None
    effect = assumptions.get("expected_difference_ms")
    deviation = assumptions.get("paired_standard_deviation_ms")
    if (
        isinstance(effect, bool)
        or not isinstance(effect, (int, float))
        or isinstance(deviation, bool)
        or not isinstance(deviation, (int, float))
        or float(effect) == 0
        or float(deviation) <= 0
        or analyzable_pairs < 2
    ):
        return None
    # Same paired normal approximation as the latency scenario estimate.
    standard_error = float(deviation) / math.sqrt(analyzable_pairs)
    return normal_cdf(abs(float(effect)) / standard_error - 1.959963984540054)


def _cost_section(
    powered: Mapping[str, Any],
    *,
    total_attempts: int,
    ceiling: Any,
    recommended_pairs: int | None,
    issues: list[dict[str, str]],
    attempts_per_pair: int = 2,
) -> dict[str, Any]:
    cost = powered.get("cost")
    ceiling_value = (
        float(ceiling)
        if isinstance(ceiling, (int, float)) and not isinstance(ceiling, bool)
        else None
    )
    if cost is None:
        return {
            "status": "unknown",
            "reason": "no per-attempt cost was declared",
            "ceiling_usd": ceiling_value,
        }
    estimate = cost["fixed_usd"] + cost["usd_per_attempt"] * total_attempts
    within = None if ceiling_value is None else estimate <= ceiling_value
    if within is False:
        issues.append(
            _issue(
                "runeconomy.plan.cost_estimate_exceeds_ceiling",
                "error",
                "cost_ceiling_usd",
                f"The estimated cost of ${estimate:.2f} exceeds the declared ${ceiling_value:.2f} ceiling.",
                "Raise the ceiling with owner approval or reduce the planned sample.",
            )
        )
    return {
        "status": "estimate",
        "source": cost["source"],
        "usd_per_attempt": cost["usd_per_attempt"],
        "fixed_usd": cost["fixed_usd"],
        "planned_attempts": total_attempts,
        "estimated_usd": estimate,
        "estimated_usd_at_minimum_pairs": (
            None
            if recommended_pairs is None
            else cost["fixed_usd"]
            + cost["usd_per_attempt"] * recommended_pairs * attempts_per_pair
        ),
        "ceiling_usd": ceiling_value,
        "within_ceiling": within,
        "excludes": "warm-up requests, retries, and idle accelerator time unless included in usd_per_attempt or fixed_usd",
    }


def _preregistration_template() -> tuple[dict[str, Any], str]:
    resource = files("runeconomy").joinpath(
        "data", f"preregistration-{PREREGISTRATION_TEMPLATE_VERSION}.template.json"
    )
    raw = resource.read_bytes()
    return json.loads(raw), "sha256:" + hashlib.sha256(raw).hexdigest()


def _identity_gate_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Preregistered identity requirements, as they will be enforced at run time."""

    acceptance = manifest.get("acceptance")
    acceptance = acceptance if isinstance(acceptance, Mapping) else {}
    rows: list[dict[str, Any]] = []
    policy = acceptance.get("identity_policy")
    if isinstance(policy, Mapping):
        rows.append(
            {
                "name": "serving_identity_grade",
                "rule": "each variant's graded serving identity must meet the minimum grade, else invalid",
                "schema_version": policy.get("schema_version"),
                "minimum_grade": policy.get("minimum_grade"),
            }
        )
    if acceptance.get("require_environment_identity") is True:
        rows.append(
            {
                "name": "environment_identity",
                "rule": "recorded serving identity must be complete and stable across the run, else invalid",
            }
        )
    return rows


def _preregistration_draft(context: Mapping[str, Any]) -> dict[str, Any]:
    """Fill the preregistration template from the assessed design."""

    template, template_digest = _preregistration_template()
    draft = {
        "claim": context["claim"],
        "population": context["population"],
        "gates": context["gates"],
        "method": context["method"],
        "n": context["n"],
        "tolerance": context["tolerance"],
        "cost_estimate": context["cost_estimate"],
        "stopping_rule": context["stopping_rule"],
    }
    unknown: list[str] = []

    def collect(prefix: str, value: Any) -> None:
        if value is None or value == "unknown":
            unknown.append(prefix)
        elif isinstance(value, Mapping):
            if value.get("status") == "unknown":
                unknown.append(prefix)
            for key, item in value.items():
                if key in {"status", "reason"}:
                    continue
                collect(f"{prefix}.{key}", item)

    for field in template["fields"]:
        collect(field, draft.get(field))
    return {
        "artifact_type": "runeconomy.preregistration_draft",
        "template_version": template["template_version"],
        "template_sha256": template_digest,
        "status": "draft_not_frozen",
        **draft,
        "unknown_fields": sorted(set(unknown)),
        "freeze_note": (
            "This draft is generated from the design and is not a freeze record. "
            "Review it, resolve every unknown field, and freeze it before held-out execution."
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


def _agent_task_list(manifest: Mapping[str, Any], workspace: Path) -> list[dict[str, Any]]:
    dataset = manifest.get("dataset")
    assert isinstance(dataset, Mapping)
    path = Path(dataset["path"])
    if not path.is_absolute():
        path = workspace / path
    payload = _read_json(path)
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else payload
    return [dict(task) for task in tasks]


def _load_section(
    *,
    load: LoadSpec | None,
    schedule_digest: str | None,
    planned_pairs: int,
    unique_tasks: int,
    repetitions: int,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    """Assess the declared arrival pattern and whether it measures concurrency."""

    if load is None:
        return {
            "arrival_pattern": "sequential",
            "measures_concurrency": False,
            "stages": 1,
            "reason": "one trial at a time; no concurrency or offered load is measured",
        }
    measures = load.open_loop or load.gate_level >= 2
    stage_units = unique_tasks * repetitions
    # Interleaved designs run a whole stage as one block per variant pool.
    effective_block = (
        min(load.block_units or unique_tasks, stage_units)
        if load.pairing == "variant_blocked"
        else stage_units
    )
    pinned = load.schedule_sha256 is not None
    pin_matches = None if not pinned else load.schedule_sha256 == schedule_digest
    if pin_matches is False:
        issues.append(
            _issue(
                "runeconomy.plan.arrival_schedule_pin_mismatch",
                "error",
                "execution.load.schedule_sha256",
                "The pinned arrival-schedule digest does not match the schedule this manifest builds.",
                "Re-pin the digest the planner reports, or restore the frozen load, seed, order, and units.",
            )
        )
    elif not pinned:
        issues.append(
            _issue(
                "runeconomy.plan.arrival_schedule_not_pinned",
                "warning",
                "execution.load.schedule_sha256",
                "The arrival schedule is deterministic but not pinned in the manifest.",
                f"Set execution.load.schedule_sha256 to {schedule_digest} before held-out execution.",
            )
        )
    tail_fraction = None
    if load.arrival_pattern == CLOSED_LOOP:
        tail_fraction = min(1.0, load.gate_level / effective_block)
        if load.tail == DRAIN and effective_block < 4 * load.gate_level:
            issues.append(
                _issue(
                    "runeconomy.plan.load_tail_fraction_high",
                    "warning",
                    "execution.load.block_units",
                    "More than a quarter of each block runs while in-flight work drains below the declared concurrency.",
                    "Use tail 'sustain', or blocks of at least four times the gate concurrency.",
                )
            )
    if load.tail == SUSTAIN:
        issues.append(
            _issue(
                "runeconomy.plan.load_filler_compute",
                "warning",
                "execution.load.tail",
                "Sustain-tail load fillers add data-dependent compute that the cost estimate excludes.",
                "Budget extra attempts for fillers; they are never part of the analyzed population.",
            )
        )
    stages = len(load.levels)
    return {
        "arrival_pattern": load.arrival_pattern,
        "measures_concurrency": measures,
        "claim_boundary": load.claim_boundary(),
        "level_name": load.level_name,
        "levels": list(load.levels),
        "gate_level": load.gate_level,
        "gate_stage": load.gate_stage,
        "stages": stages,
        "sweep_stages_are_descriptive": stages > 1,
        "pairing": load.pairing,
        "serving_topology": load.serving_topology,
        "tail": load.tail,
        "block_units": effective_block if load.pairing == "variant_blocked" else None,
        "closed_loop_drain_fraction_upper_bound": tail_fraction,
        "arrival_schedule_sha256": schedule_digest,
        "schedule_pinned": pinned,
        "schedule_pin_matches": pin_matches,
        "population_attempts_per_stage": planned_pairs * 2,
        "population_attempts_all_stages": planned_pairs * 2 * stages,
        "filler_attempts": "data_dependent" if load.tail == SUSTAIN else 0,
        "open_loop_max_in_flight": load.max_in_flight,
        "max_dispatch_lag_ms": load.max_dispatch_lag_ms,
    }


def _metric_gate_section(
    gates: tuple[MetricGate, ...],
    *,
    mode: str,
    planned_pairs: int,
    unique_tasks: int,
    issues: list[dict[str, str]],
    cost_model: Any = None,
    requests_per_pair: int | None = None,
) -> list[dict[str, Any]]:
    """Check each preregistered metric gate against its documented minimum n.

    ``requests_per_pair`` is known only when every paired unit issues a fixed
    number of requests (isolated static replay issues exactly one). Agent
    workflows decide their own request count, so request-level gates stay a
    calibration question there.
    """

    rows: list[dict[str, Any]] = []
    for index, gate in enumerate(gates):
        minimum = minimum_sample(gate)
        level = gate.definition.level
        if level == "workflow":
            observations_bound: int | None = planned_pairs
        elif level == "request" and requests_per_pair is not None:
            observations_bound = planned_pairs * requests_per_pair
        else:
            observations_bound = None
        meets_observations = (
            None
            if minimum["observations"] is None or observations_bound is None
            else observations_bound >= minimum["observations"]
        )
        meets_clusters = (
            None if minimum["clusters"] is None else unique_tasks >= minimum["clusters"]
        )
        field = f"acceptance.metric_gates[{index}]"
        if meets_observations is False or meets_clusters is False:
            issues.append(
                _issue(
                    "runeconomy.plan.metric_gate_sample_below_minimum",
                    "error",
                    field,
                    f"Gate {gate.name} cannot pass: the planned gate-stage sample is below the documented minimum for {gate.method}.",
                    "Increase unique tasks or repetitions, use a less extreme statistic, or remove the gate before freezing.",
                )
            )
        if level == "request" and observations_bound is None:
            issues.append(
                _issue(
                    "runeconomy.plan.metric_gate_request_count_data_dependent",
                    "warning",
                    field,
                    f"Gate {gate.name} counts requests, which depend on what the workflow does.",
                    "Confirm from calibration that each workflow yields enough requests to clear the minimum.",
                )
            )
        if level == "ratio" and not (
            isinstance(cost_model, Mapping)
            and (cost_model.get("kind") == "gpu_hourly" or gate.metric == "usd_per_1k_successful_workflows")
        ):
            # A declared gpu_hourly model prices every attempt from wall time, so
            # both ratio gates are computable. A per_token model prices cost but
            # records no GPU time, so the goodput gate still needs economics.
            issues.append(
                _issue(
                    "runeconomy.plan.metric_gate_requires_economics",
                    "warning",
                    field,
                    (
                        "Static replay records no cost or GPU time; this gate stays unknown unless a cost model attaches economics to every attempt."
                        if mode == "static_request"
                        else "Every population attempt must record cost or GPU time, including failures; unknown values keep the gate unknown."
                    ),
                    "Attach a versioned cost model or record runeconomy.cost_usd and runeconomy.gpu_time_ms on every request.",
                )
            )
        rows.append(
            {
                "name": gate.name,
                **gate.to_dict(),
                "unit_level": level,
                "minimum_sample": minimum,
                "planned_observations_upper_bound": observations_bound,
                "planned_clusters": unique_tasks,
                "meets_minimum_observations": meets_observations,
                "meets_minimum_clusters": meets_clusters,
                "population_note": (
                    "pairs where both variants succeed; failures shrink the analyzable sample"
                    if gate.population == "successful_pairs"
                    else "all population attempts; failures count against the candidate"
                ),
            }
        )
    return rows


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
        planning and planning.get("schema_version") not in PLANNING_INPUT_VERSIONS
    ):
        issues.append(
            _issue(
                "runeconomy.plan.design_schema_unsupported",
                "error",
                "design.planning.schema_version",
                "The optional planning declaration is malformed or unsupported.",
                "Use planning schema 0.1 or 0.2, or remove the optional block.",
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
        optimization_target = str(
            acceptance.get("optimization_target", "successful_workflow_latency")
        )
        latency_semantics = str(acceptance.get("latency_semantics", "superiority"))
        max_latency_regression_ms = float(
            acceptance.get("max_latency_regression_ms", 0.0)
        )
        min_success = float(acceptance.get("min_task_success_rate", 0.8))
        min_completed = float(acceptance.get("min_completed_rate", 1.0))
        margin = float(acceptance.get("max_task_success_regression", 0.05))
        unit_name = "task attempt identified by (task_id, repetition)"
        workflow_kind = "complete_agent_process"
        method = str(acceptance.get("task_success_method", PAIRED_EXACT))
        try:
            completion_policy = (
                CompletionPolicy.from_value(acceptance["completion_policy"])
                if "completion_policy" in acceptance
                else None
            )
        except CompletionPolicyError:
            completion_policy = None
        load_spec = agent_load_spec(raw_manifest)
        try:
            metric_gates = parse_metric_gates(acceptance.get("metric_gates"))
        except MetricGateError as exc:
            raise ExperimentValidationError(str(exc)) from exc
        schedule_digest = (
            None
            if load_spec is None
            else agent_arrival_schedule(
                load_spec,
                _agent_task_list(raw_manifest, workspace),
                repetitions=repetitions,
                seed=int(execution.get("seed", 0)),
                order=str(execution.get("order", "randomized")),
            ).digest()
        )
        cost_ceiling = raw_manifest.get("cost_ceiling_usd")
        dataset = raw_manifest.get("dataset")
        population_digest = (
            dataset.get("sha256") if isinstance(dataset, Mapping) else None
        )
    else:
        parsed = ExperimentManifest.from_dict(raw_manifest)
        source_cases, unique_tasks = _static_units(parsed)
        ids_unique = True
        repetitions = parsed.execution.repetitions
        primary_metric = parsed.acceptance.primary_metric
        optimization_target = parsed.acceptance.optimization_target
        latency_semantics = "superiority"
        max_latency_regression_ms = 0.0
        min_success = parsed.acceptance.min_task_success_rate
        min_completed = parsed.acceptance.min_completed_rate
        margin = parsed.acceptance.max_task_success_regression
        unit_name = "paired case repetition"
        workflow_kind = (
            "sequential_fixture_workflow"
            if parsed.execution.cases_form_workflow
            else "isolated_request_replay"
        )
        method = parsed.acceptance.task_success_method
        completion_policy = None
        load_spec = parsed.execution.load
        metric_gates = parsed.metric_gates
        schedule_digest = (
            None if load_spec is None else static_arrival_schedule(parsed).digest()
        )
        cost_ceiling = None
        population_digest = "sha256:" + parsed.reproducibility_fingerprint()

    planned_pairs = unique_tasks * repetitions
    stage_count = len(load_spec.levels) if load_spec is not None else 1
    total_attempts = planned_pairs * 2 * stage_count
    load_section = _load_section(
        load=load_spec,
        schedule_digest=schedule_digest,
        planned_pairs=planned_pairs,
        unique_tasks=unique_tasks,
        repetitions=repetitions,
        issues=issues,
    )
    metric_gate_rows = _metric_gate_section(
        metric_gates,
        mode=mode,
        planned_pairs=planned_pairs,
        unique_tasks=unique_tasks,
        issues=issues,
        cost_model=raw_manifest.get("cost_model"),
        requests_per_pair=1 if workflow_kind == "isolated_request_replay" else None,
    )
    completion = _completion_tolerance(planned_pairs, min_completed)
    issues.append(
        _issue(
            "runeconomy.plan.completion_failure_tolerance",
            "warning",
            "acceptance.min_completed_rate",
            completion["warning"],
            "Freeze this exact integer failure tolerance before held-out execution.",
        )
    )
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
    analysis_method = method if method in PAIRED_METHODS else PAIRED_EXACT
    paired = _paired_feasibility(planned_pairs, margin, method=analysis_method)
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
    elif paired["max_baseline_only_with_zero_candidate_only"] == 0:
        required_for_one = paired[
            "minimum_pairs_by_baseline_only_with_zero_candidate_only"
        ]["1"]
        issues.append(
            _issue(
                "runeconomy.plan.paired_gate_fragile",
                "warning",
                "acceptance.max_task_success_regression",
                "The planned quality gate passes only with zero baseline-only outcomes; one candidate quality loss would be inconclusive under the exact paired bound.",
                f"Plan at least {required_for_one} pairs to tolerate one baseline-only outcome with zero offsetting candidate-only outcomes, or freeze the current limitation explicitly.",
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
    # A concurrency claim is legal exactly when the frozen design measures it:
    # a concurrent arrival pattern whose gated level puts more than one unit
    # in flight per variant (or any open-loop offered load).
    concurrency_measured = bool(load_section["measures_concurrency"])
    for key in ("concurrency", "routing", "production_behavior"):
        if key == "concurrency" and concurrency_measured:
            continue
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
    if optimization_target == "engine_change_latency" and (
        identity_change is not True or "backend_identity" not in variables
    ):
        issues.append(
            _issue(
                "runeconomy.plan.engine_change_not_declared",
                "error",
                "design.planning.intervention_variables",
                "The engine-change latency target requires backend identity to be the declared intervention.",
                "Set identity_change=true and declare exactly backend_identity before calibration.",
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

    powered = _powered_inputs(
        planning,
        version=planning.get("schema_version"),
        heldout_digest=heldout_digest,
        issues=issues,
    )
    failure_model = _completion_model_section(
        powered.get("completion_model"),
        attempts_per_variant=planned_pairs,
        declared_threshold=min_completed,
        policy=completion_policy,
        powered=bool(powered.get("powered")),
        issues=issues,
    )
    sample_size = _sample_size_section(
        powered,
        method=method,
        margin=margin,
        min_success=min_success,
        planned_pairs=planned_pairs,
        completion_section=failure_model,
        latency_assumptions=planning.get("latency_assumptions"),
        issues=issues,
        detection_target=planning.get("detection_target"),
    )
    cost_estimate = _cost_section(
        powered,
        total_attempts=total_attempts,
        ceiling=cost_ceiling,
        recommended_pairs=sample_size.get("minimum_pairs_for_target_power"),
        issues=issues,
        attempts_per_pair=2 * stage_count,
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
    if load_spec is not None:
        freeze_checks["arrival_schedule_pinned"] = (
            load_section["schedule_pin_matches"] is True
        )
    if powered.get("powered"):
        success_assumptions = powered.get("success") or {}
        freeze_checks.update(
            {
                "task_success_power_meets_target": sample_size.get(
                    "planned_pairs_meet_target"
                )
                is True,
                "success_assumptions_from_calibration": success_assumptions.get(
                    "source"
                )
                == "calibration",
                "completion_threshold_from_failure_model": bool(
                    failure_model.get("declared_matches_model") is True
                    and failure_model.get("source") == "calibration"
                ),
                "cost_estimate_within_ceiling": bool(
                    cost_estimate.get("status") == "estimate"
                    and cost_estimate.get("within_ceiling") is not False
                ),
            }
        )
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

    classification = (
        completion_policy.common_mode_paired_failures
        if completion_policy is not None
        else VARIANT_NON_COMPLETION
    )
    completion_model = {
        "common_mode_paired_failures": classification,
        "completion_policy": (
            completion_policy.to_dict() if completion_policy is not None else None
        ),
        "declared_max_noncompleted_attempts_per_variant": completion[
            "maximum_failed_or_incomplete_attempts_per_variant"
        ],
        "expected_failure_model": failure_model,
    }
    gate_name = "quality_gates" if min_success > 0 else "noninferiority"
    planned_power = sample_size.get("planned_power")
    preregistration = _preregistration_draft(
        {
            "claim": {
                "level": claim_level,
                "primary_metric": primary_metric,
                "predicted_direction": predicted_direction,
                "intervention_variables": list(variables),
                "noninferiority_margin": margin,
                "statement_fields": {
                    name: f"design.planning.{name}"
                    for name in ("hypothesis", "intervention", "claim_boundary")
                    if _presence(planning.get(name))
                },
                "design_declarations_sha256": _digest(planning),
            },
            "population": {
                "experimental_unit": unit_name,
                "pairing_key": ["task_id", "repetition"],
                "unique_task_count": unique_tasks,
                "repetitions": repetitions,
                "planned_pairs": planned_pairs,
                "total_attempts": total_attempts,
                "workload_sha256": population_digest,
                "heldout_artifact_sha256": heldout_digest if heldout_present else None,
                "heldout_status": heldout_status,
                "arrival_schedule": {
                    "arrival_pattern": load_section["arrival_pattern"],
                    **(
                        {}
                        if load_spec is None
                        else {
                            "load": load_spec.to_dict(include_pin=False),
                            "gate_level": load_spec.gate_level,
                            "claim_boundary": load_spec.claim_boundary(),
                            "sha256": schedule_digest,
                            "pinned": load_section["schedule_pinned"],
                        }
                    ),
                },
            },
            "gates": [
                {
                    "name": "minimum_task_success_rate",
                    "applies_to": ["baseline", "candidate"],
                    "rule": "lower end of the two-sided 95% Clopper-Pearson interval >= threshold",
                    "threshold": min_success,
                    "minimum_successful_units": minimum_successes,
                },
                {
                    "name": "task_success_noninferiority",
                    "rule": "reject H0: candidate minus baseline <= -margin",
                    "method": method,
                    "margin": margin,
                    "alpha": PLANNING_ALPHA,
                },
                {
                    "name": "completion_rate",
                    "applies_to": ["baseline", "candidate"],
                    "threshold": min_completed,
                    "max_noncompleted_attempts_per_variant": completion[
                        "maximum_failed_or_incomplete_attempts_per_variant"
                    ],
                    "common_mode_paired_failures": classification,
                },
                {
                    "name": primary_metric,
                    "semantics": latency_semantics,
                    "population": "pairs where both variants succeeded",
                    "max_regression_ms": max_latency_regression_ms,
                },
                *(
                    []
                    if load_spec is None or not load_spec.open_loop
                    else [
                        {
                            "name": "baseline_arrival_schedule_integrity",
                            "rule": "baseline p95 dispatch lag on the gate stage <= threshold, else invalid",
                            "threshold_ms": load_spec.max_dispatch_lag_ms,
                        },
                        {
                            "name": "candidate_offered_load_completion",
                            "rule": "candidate p95 dispatch lag on the gate stage <= threshold, else regression",
                            "threshold_ms": load_spec.max_dispatch_lag_ms,
                        },
                    ]
                ),
                *[
                    {
                        "name": row["name"],
                        "rule": "two-sided 95% interval for the candidate-worse difference; pass iff upper < limit",
                        **{
                            key: row[key]
                            for key in (
                                "metric",
                                "statistic",
                                "semantics",
                                "method",
                                "population",
                                "margin",
                                "relative_margin",
                                "minimum_sample",
                            )
                            if key in row
                        },
                    }
                    for row in metric_gate_rows
                ],
                *_identity_gate_rows(raw_manifest),
            ],
            "method": {
                "task_success_method": method,
                "alpha": PLANNING_ALPHA,
                "properties": method_properties(method),
            },
            "n": {
                "planned_pairs": planned_pairs,
                "target_power": sample_size.get("target_power"),
                "task_success_gate_power": (
                    planned_power.get(gate_name)
                    if isinstance(planned_power, Mapping)
                    else None
                ),
                "all_gates_lower_bound": (
                    planned_power.get("all_gates_lower_bound")
                    if isinstance(planned_power, Mapping)
                    else None
                ),
                "minimum_pairs_for_target_power": sample_size.get(
                    "minimum_pairs_for_target_power"
                ),
            },
            "tolerance": {
                "min_completed_rate": min_completed,
                "max_noncompleted_attempts_per_variant": completion[
                    "maximum_failed_or_incomplete_attempts_per_variant"
                ],
                "derived_from_failure_model": failure_model.get(
                    "declared_matches_model"
                )
                is True,
                "failure_model_source": failure_model.get("source", "unknown"),
                "common_mode_paired_failures": classification,
            },
            "cost_estimate": {
                "status": cost_estimate["status"],
                "estimated_usd": cost_estimate.get("estimated_usd"),
                "ceiling_usd": cost_estimate.get("ceiling_usd"),
                "within_ceiling": cost_estimate.get("within_ceiling"),
                "source": cost_estimate.get("source", "unknown"),
            },
            "stopping_rule": stopping_kind if stopping_declared else "unknown",
        }
    )
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
        "load": load_section,
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
            "method_id": method,
            "method_properties": method_properties(method),
            "paired_noninferiority": paired,
        },
        "completion": completion,
        "completion_model": completion_model,
        "sample_size": sample_size,
        "cost_estimate": cost_estimate,
        "preregistration": preregistration,
        "performance": {
            "primary_metric": primary_metric,
            "optimization_target": optimization_target,
            "latency_semantics": latency_semantics,
            "max_latency_regression_ms": max_latency_regression_ms,
            "latency_population": "task pairs where both variants succeeded",
            "metric_role": metric_role,
            "conditional_on_both_variants_succeeding": True,
            "task_failures_reduce_analyzable_pairs": True,
            "maximum_analyzable_pairs": planned_pairs,
            "power": power,
            "metric_gates": metric_gate_rows,
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


# Measured Proof 2 held-out spend divided by its 360 attempts. Used only as a
# labelled reference scenario; a real multi-step agent may cost more.
PROOF_2_HELDOUT_USD_PER_ATTEMPT = 0.69699111 / 360


def worked_examples() -> dict[str, Any]:
    """Reference planning cases for the two historical sizes and a Proof 3 design.

    The n=40 and n=180 cases reproduce the frozen Proof 1 and Proof 2 gate
    choices and show what each method could and could not establish. The
    Proof 3 case is a forward-looking scenario. Every rate in it is a declared
    assumption that calibration must replace before a freeze. Nothing here
    reinterprets a recorded verdict; both proofs remain invalid.
    """

    alpha = PLANNING_ALPHA

    def best_case(pairs: int, margin: float) -> dict[str, Any]:
        legacy = _paired_success_statistics([(True, True)] * pairs)
        return {
            "paired_exact": {
                "one_sided_lower_bound": legacy["one_sided_lower_bound"],
                "passes": legacy["one_sided_lower_bound"] >= -margin,
            },
            "paired_score_exact": {
                "one_sided_lower_bound": score_exact_lower_bound(
                    0, 0, pairs, alpha=alpha, anchor=-margin, tolerance=1e-5
                ),
                "passes": score_exact_rejects(0, 0, pairs, margin, alpha=alpha),
            },
        }

    def powers(pairs: int, margin: float, cells: Mapping[str, float], floor: float) -> dict[str, Any]:
        return {
            name: gate_power(
                pairs=pairs,
                method=name,
                margin=margin,
                cells=cells,
                min_task_success_rate=floor,
                alpha=alpha,
            )
            for name in (PAIRED_EXACT, PAIRED_SCORE_EXACT)
        }

    reference_cells = paired_cell_probabilities(
        expected_success_rate=0.95, expected_discordance_rate=0.04
    )
    legacy_minimum_40 = _paired_feasibility(40, 0.05)[
        "minimum_pairs_by_baseline_only_with_zero_candidate_only"
    ]["0"]
    n40 = {
        "contract": {
            "pairs": 40,
            "max_task_success_regression": 0.10,
            "min_task_success_rate": 0.80,
            "min_completed_rate": 1.0,
        },
        "best_case_all_pairs_agree": {
            "margin_0.10": best_case(40, 0.10),
            "margin_0.05": best_case(40, 0.05),
        },
        "minimum_pairs_for_all_agreement_to_pass_margin_0.05": {
            "paired_exact": legacy_minimum_40,
            "paired_score_exact": _score_minimum_pairs(0, 0.05, alpha=alpha),
        },
        "scenario_power": {
            "assumptions": {
                "source": "scenario",
                "expected_success_rate": 0.95,
                "expected_discordance_rate": 0.04,
            },
            "margin_0.10": powers(40, 0.10, reference_cells, 0.80),
        },
        "completion_tolerance_per_variant": _completion_tolerance(40, 1.0)[
            "maximum_failed_or_incomplete_attempts_per_variant"
        ],
        "reading": (
            "At n=40 no valid method can verify a 5 pp margin even when every "
            "pair agrees. The score method narrows the best case from -8.8 pp to "
            "about -7.2 pp, but quality-gate power stays far below 80%."
        ),
    }

    proof_2_pairs = [(True, True)] * 179 + [(False, False)]
    legacy_180 = _paired_success_statistics(proof_2_pairs)
    scenario_rate = 0.0025 + 0.005
    derived_180 = completion_tolerance(180, scenario_rate, DEFAULT_COMPLETION_PASS_PROBABILITY)
    full_completion_pass = binomial_cdf(0, 180, scenario_rate)
    n180 = {
        "contract": {
            "pairs": 180,
            "max_task_success_regression": 0.05,
            "min_task_success_rate": 0.90,
            "min_completed_rate": 1.0,
        },
        "observed_paired_outcomes": legacy_180["outcome_counts"],
        "observed_lower_bound": {
            "paired_exact": legacy_180["one_sided_lower_bound"],
            "paired_score_exact": score_exact_lower_bound(
                0, 0, 180, alpha=alpha, anchor=-0.05, tolerance=1e-5
            ),
        },
        "completion": {
            "declared_tolerance_per_variant": _completion_tolerance(180, 1.0)[
                "maximum_failed_or_incomplete_attempts_per_variant"
            ],
            "scenario_failure_model": {
                "source": "scenario",
                "infrastructure_failure_rate": 0.0025,
                "application_failure_rate": 0.005,
            },
            "zero_tolerance_pass_probability_both_variants": full_completion_pass**2,
            "model_derived_tolerance_per_variant": derived_180,
            "model_derived_min_completed_rate": (180 - derived_180) / 180,
            "common_mode_policy_note": (
                "Under a prospectively declared task_outcome policy, a shared "
                "failure with identical status and provenance in both arms would be "
                "a both-failure task outcome, not non-completion. Proof 2 did not "
                "declare that policy and remains invalid under its frozen contract."
            ),
        },
        "scenario_power": {
            "assumptions": {
                "source": "scenario",
                "expected_success_rate": 0.95,
                "expected_discordance_rate": 0.04,
            },
            "margin_0.05": powers(180, 0.05, reference_cells, 0.90),
        },
        "reading": (
            "At n=180 a 100% completion gate tolerates zero failures, so even a "
            "small expected failure rate makes an invalid verdict likely. The "
            "score method raises paired-gate power at this size."
        ),
    }

    proof_3_assumptions = {
        "source": "scenario",
        "expected_success_rate": 0.90,
        "expected_discordance_rate": 0.06,
        "expected_difference": 0.0,
    }
    recommendations = {
        name: recommend_pairs(
            margin=0.05,
            expected_success_rate=0.90,
            expected_discordance_rate=0.06,
            method=name,
            target_power=DEFAULT_TARGET_POWER,
            alpha=alpha,
            min_task_success_rate=0.80,
        )
        for name in (PAIRED_EXACT, PAIRED_SCORE_EXACT)
    }
    recommended = recommendations[PAIRED_SCORE_EXACT]["minimum_pairs"]
    proof_3_failure_rate = 0.005 + 0.01
    proof_3_tolerance = (
        None
        if recommended is None
        else completion_tolerance(
            recommended, proof_3_failure_rate, DEFAULT_COMPLETION_PASS_PROBABILITY
        )
    )
    proof_3 = {
        "design": "FP8 versus BF16 serving of one pinned model on a real multi-step agent",
        "contract": {
            "max_task_success_regression": 0.05,
            "min_task_success_rate": 0.80,
            "task_success_method": PAIRED_SCORE_EXACT,
            "completion_policy": {
                "schema_version": "0.1",
                "common_mode_paired_failures": "task_outcome",
            },
        },
        "assumptions": proof_3_assumptions,
        "target_power": DEFAULT_TARGET_POWER,
        "minimum_pairs": {
            name: value.get("minimum_pairs") for name, value in recommendations.items()
        },
        "power_at_minimum": {
            name: value.get("power_at_minimum")
            for name, value in recommendations.items()
        },
        "completion": {
            "scenario_failure_model": {
                "source": "scenario",
                "infrastructure_failure_rate": 0.005,
                "application_failure_rate": 0.01,
            },
            "model_derived_tolerance_per_variant": proof_3_tolerance,
            "model_derived_min_completed_rate": (
                None
                if recommended is None or proof_3_tolerance is None
                else (recommended - proof_3_tolerance) / recommended
            ),
        },
        "cost_estimate": {
            "source": "scenario: Proof 2 measured held-out cost per attempt",
            "usd_per_attempt": PROOF_2_HELDOUT_USD_PER_ATTEMPT,
            "estimated_usd": (
                None
                if recommended is None
                else PROOF_2_HELDOUT_USD_PER_ATTEMPT * recommended * 2
            ),
        },
        "reading": (
            "Under these scenario rates the score method reaches 80% power for "
            "the joint quality gates with substantially fewer pairs than "
            "paired_exact needs. Calibration must replace every assumed rate "
            "before a freeze."
        ),
    }
    return {
        "artifact_type": "runeconomy.planning_worked_examples",
        "alpha": alpha,
        "n40": n40,
        "n180": n180,
        "proof_3_scenario": proof_3,
    }
