"""Static, shareable reports for controlled experiment results."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class ReportCard:
    label: str
    value: str
    description: str


@dataclass(frozen=True)
class QualityRequirement:
    question: str
    rows: tuple[tuple[str, str], ...]
    explanation: str
    passed: bool


@dataclass(frozen=True)
class TechnicalSection:
    label: str
    body_html: str
    body_markdown: str


@dataclass(frozen=True)
class ReportView:
    experiment_id: str
    status: str
    status_label: str
    primary_message: str
    verdict_explanation: str
    invalid_explanation: str | None
    recommendation: str
    badges: tuple[str, ...]
    intervention_intro: str
    intervention_baseline: str
    intervention_candidate: str
    intervention_why: str
    intervention_control: str
    intervention_kind: str
    result_cards: tuple[ReportCard, ...]
    result_qualification: str
    outcome_items: tuple[tuple[str, str], ...]
    cache_position_items: tuple[tuple[str, str], ...]
    quality_requirements: tuple[QualityRequirement, ...]
    decision_path: tuple[tuple[str, str], ...]
    later_checks: tuple[str, ...]
    experiment_intro: str
    experiment_facts: tuple[tuple[str, str], ...]
    controls: str
    technical_sections: tuple[TechnicalSection, ...]


def _number(value: Any, digits: int = 1) -> str:
    return "Unknown" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 1) -> str:
    return "Unknown" if value is None else f"{float(value) * 100:.{digits}f}%"


def _prose(value: Any) -> str:
    """Normalize recorded prose to the report's punctuation contract."""

    text = str(value).replace(" — ", ". ").replace("—", "-")
    parts = [part.strip() for part in text.split(";")]
    normalized = []
    for part in parts:
        if not part:
            continue
        normalized.append(part[0].upper() + part[1:] if normalized else part)
    return ". ".join(normalized)


def _escape(value: Any) -> str:
    return html.escape(_prose(value))


def _interval(effect: Mapping[str, Any], digits: int = 1) -> str:
    interval = effect.get("confidence_interval", effect.get("bootstrap_95_ci"))
    if not isinstance(interval, list) or len(interval) != 2:
        return "Unknown"
    if interval[0] is None or interval[1] is None:
        return "Unknown"
    return f"{float(interval[0]):.{digits}f} to {float(interval[1]):.{digits}f}"


def _percentage_points(value: Any, digits: int = 2) -> str:
    if value is None:
        return "Unknown"
    return f"{float(value) * 100:.{digits}f} percentage points"


def _status_copy(status: str, reason: str) -> tuple[str, str, str | None, str]:
    normalized = status.lower()
    if normalized == "verified":
        return (
            "Improvement verified",
            "The declared task-quality and performance requirements passed. The candidate is eligible for deployment review.",
            None,
            "Review operational risk and production safeguards before deploying the candidate.",
        )
    if normalized == "regression":
        return (
            "Candidate regressed",
            "A declared candidate guardrail failed. The recorded evidence does not support shipping this change.",
            None,
            "Do not ship the candidate. Revise or reject the change before a new experiment.",
        )
    if normalized == "inconclusive":
        return (
            "Result inconclusive",
            "The evidence was usable, but it was not sufficient to establish the declared performance claim.",
            None,
            "Gather more paired evidence under a new frozen design before making a performance claim.",
        )
    if normalized == "invalid":
        return (
            "Result not verified",
            "The candidate may have favorable measurements, but this experiment did not establish the evidence required for a verified claim.",
            "Invalid does not mean the candidate was slower. It means the attractive measurements were not eligible to become a verified improvement or regression claim.",
            "Repair the decisive evidence or baseline-quality problem, then use a new frozen held-out workload.",
        )
    return (
        "No verdict available",
        _prose(reason or "The result did not contain a recognized verdict."),
        None,
        "Inspect the result and complete the missing experiment evidence.",
    )


def _preregistration_rows(result: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Freeze status and economics provenance. Empty for results without them."""

    rows: list[tuple[str, str]] = []
    record = result.get("preregistration")
    if isinstance(record, Mapping):
        rows.extend(
            [
                ("Preregistration record", str(record.get("record_sha256", "unknown"))),
                ("Frozen at", str(record.get("frozen_at", "unknown"))),
                ("Locks at run start", str(record.get("status", "unknown"))),
                ("Design completeness", str(record.get("design_completeness", "unknown"))),
            ]
        )
    provenance = result.get("provenance")
    design = result.get("design")
    cost = None
    for source in (provenance, design):
        if isinstance(source, Mapping) and isinstance(source.get("cost_model"), Mapping):
            cost = source["cost_model"]
    if cost is not None:
        model = cost.get("model") if isinstance(cost.get("model"), Mapping) else {}
        rows.append(
            (
                "Cost model",
                f"{model.get('kind', 'unknown')} ({cost.get('sha256', 'unknown')})"
                + (f"; source: {model['source']}" if model.get("source") else ""),
            )
        )
    return tuple(rows)


def _identity_assurance_rows(
    decision: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Graded identity, shown separately from gates. Empty for legacy results."""

    record = decision.get("identity_assurance")
    if not isinstance(record, Mapping):
        return ()
    policy = record.get("policy") if isinstance(record.get("policy"), Mapping) else {}
    rows: list[tuple[str, str]] = [
        (
            "Identity policy",
            f"schema {policy.get('schema_version', 'unknown')}; minimum grade "
            f"{policy.get('minimum_grade', 'unknown')}",
        ),
        ("Claim scope", str(record.get("claim_scope", "unknown"))),
        ("Supported claim", str(record.get("supported_claim", "Not recorded."))),
    ]
    variants = record.get("variants") if isinstance(record.get("variants"), Mapping) else {}
    for name in ("baseline", "candidate"):
        item = variants.get(name)
        if not isinstance(item, Mapping):
            continue
        undisclosed = ", ".join(item.get("undisclosed_fields") or []) or "none"
        missing = ", ".join(
            [*(item.get("missing_fields") or []), *(item.get("placeholder_fields") or [])]
        ) or "none"
        fingerprint = item.get("provider_fingerprint")
        fingerprint = fingerprint if isinstance(fingerprint, Mapping) else {}
        rows.append(
            (
                f"{name.title()} identity",
                f"Grade: {item.get('grade') or 'unknown'}. Meets minimum: "
                f"{'yes' if item.get('meets_minimum_grade') else 'no'}. "
                f"Undisclosed: {undisclosed}. Missing or placeholder: {missing}.",
            )
        )
        rows.append(
            (
                f"{name.title()} provider fingerprint",
                f"{fingerprint.get('distinct_count', 0)} distinct value(s) across "
                f"{fingerprint.get('reported_count', 0)} of "
                f"{fingerprint.get('request_count', 0)} requests. Drift indicator "
                "only; not part of identity and never affects validity.",
            )
        )
    for limit in record.get("claim_limits") or []:
        if isinstance(limit, Mapping):
            rows.append(
                (
                    f"Claim limit · {limit.get('variant', 'unknown')}",
                    f"{limit.get('code', 'unknown')}: {limit.get('statement', '')}",
                )
            )
    return tuple(rows)


def _serving_capability_rows(
    result: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Adapter capability downgrades. Empty for results that record none."""

    record = result.get("serving_capabilities")
    if not isinstance(record, Mapping):
        return ()
    rows: list[tuple[str, str]] = [
        ("Backend kind", str(record.get("backend_kind", "unknown"))),
        (
            "Capability table",
            f"version {record.get('capability_table_version', 'unknown')}",
        ),
    ]
    for item in record.get("downgrades") or []:
        if isinstance(item, Mapping):
            rows.append(
                (
                    f"Downgraded · {item.get('capability', 'unknown')}",
                    f"{item.get('status', 'unknown')}: {item.get('effect', '')}",
                )
            )
    if len(rows) == 2:
        rows.append(("Downgrades", "None recorded."))
    return tuple(rows)


def _generic_recommendation(status: str, decision: Mapping[str, Any]) -> str:
    reason = str(decision.get("reason", "")).lower()
    decisive = str(decision.get("decisive_stage", "")).lower()
    assurance = decision.get("identity_assurance")
    if (
        status == "verified"
        and isinstance(assurance, Mapping)
        and assurance.get("claim_scope") == "observed_service_window"
    ):
        return (
            "The candidate is eligible for deployment review for this workload only. "
            "Identity is provider-declared, so the result describes the provider "
            "service as observed during the run, not a pinned configuration. "
            "Re-verify before relying on it later."
        )
    if status == "verified":
        return "The candidate is eligible for deployment review. Review operational risk and production safeguards first."
    if status == "regression":
        return "Do not ship the candidate. Reject or revise the change before a new experiment."
    if status == "inconclusive":
        return "Gather more paired evidence under a new frozen design before making a performance claim."
    if status == "invalid" and any(
        marker in reason or marker in decisive
        for marker in ("baseline", "task success", "task_success", "quality")
    ):
        return "Stabilize the workload, evaluator or baseline, then test under a new frozen design."
    if status == "invalid":
        return "Repair the instrumentation or environment-identity evidence, then run a new frozen experiment."
    return "Inspect the result and complete the missing experiment evidence."


def _metric_effect(
    effects: Mapping[str, Any], *names: str
) -> tuple[str | None, Mapping[str, Any]]:
    for name in names:
        value = effects.get(name)
        if isinstance(value, Mapping):
            return name, value
    return None, {}


def _card(
    label: str,
    effect: Mapping[str, Any],
    *,
    positive_is_more: bool = False,
    unit: str = "",
) -> ReportCard:
    delta = effect.get("candidate_minus_baseline_mean")
    if delta is None:
        return ReportCard(label, "Unknown", "This metric was not available.")
    number = float(delta)
    magnitude = abs(number)
    rounded = f"{magnitude:.0f}"
    if positive_is_more:
        value = f"+{rounded}" if number > 0 else f"-{rounded}" if number < 0 else "0"
        description = (
            "More prompt tokens were served from cache."
            if number > 0
            else "Fewer prompt tokens were served from cache."
            if number < 0
            else "Mean cached prompt tokens did not change."
        )
    else:
        value = f"{rounded} {unit} lower" if number < 0 else f"{rounded} {unit} higher"
        description = (
            "The first token arrived sooner at the server."
            if label == "Server time to first token" and number < 0
            else "The first token arrived later at the server."
            if label == "Server time to first token"
            else "Successful task pairs completed sooner."
            if number < 0
            else "Successful task pairs completed later."
        )
    return ReportCard(label, value.strip(), description)


def _html_table(rows: Sequence[tuple[str, str]], headings: tuple[str, str]) -> str:
    body = "".join(
        f'<tr><th scope="row">{_escape(label)}</th><td>{_escape(value)}</td></tr>'
        for label, value in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"<th>{_escape(headings[0])}</th><th>{_escape(headings[1])}</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _markdown_table(rows: Sequence[tuple[str, str]], headings: tuple[str, str]) -> str:
    lines = [f"| {headings[0]} | {headings[1]} |", "|---|---|"]
    lines.extend(f"| {_prose(label)} | {_prose(value)} |" for label, value in rows)
    return "\n".join(lines)


def _recorded_label(key: Any) -> str:
    words = str(key).replace("_", " ").split()
    return " ".join("DAG" if word == "dag" else word for word in words)


def _recorded_value(value: Any, *, nested: bool = False) -> str:
    """Describe a recorded gate value in words without changing its numbers.

    Gate thresholds and observations are stored as numbers, intervals, and
    nested objects. Numbers keep their full recorded precision so the report
    remains auditable against the result file.
    """

    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "unknown"
    if isinstance(value, Mapping):
        if not value:
            return "none recorded"
        if not nested and all(isinstance(item, Mapping) for item in value.values()):
            return ". ".join(
                f"{_recorded_label(key).capitalize()}: "
                f"{_recorded_value(item, nested=True)}"
                for key, item in value.items()
            )
        parts = []
        for key, item in value.items():
            text = _recorded_value(item, nested=True)
            if isinstance(item, Mapping):
                text = f"({text})"
            parts.append(f"{_recorded_label(key)} {text}")
        return ", ".join(parts)
    if isinstance(value, (list, tuple)):
        scalars = all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
        if len(value) == 2 and scalars:
            return f"{value[0]} to {value[1]}"
        return ", ".join(_recorded_value(item, nested=True) for item in value)
    return _prose(value)


def _technical_sections_for_public(
    result: Mapping[str, Any],
) -> tuple[TechnicalSection, ...]:
    checks = result["gates"]["checks"]
    check_rows = tuple(
        (
            str(item.get("name", "unknown")),
            f"{'Passed' if item.get('passed') else 'Failed'}. "
            f"Threshold: {_recorded_value(item.get('threshold'))}. "
            f"Observed: {_recorded_value(item.get('observed'))}",
        )
        for item in checks
    )
    effects = result["conditional_effects"]
    effect_rows = tuple(
        (
            name.replace("_", " ").title(),
            f"Mean candidate minus baseline: {_number(item.get('candidate_minus_baseline_mean'), 3)}. 95% interval: {_interval(item, 3)}. Pairs: {item.get('paired_count', 0)}",
        )
        for name, item in sorted(effects.items())
    )
    paired = result["paired_success"]
    paired_interval = paired.get("confidence_interval") or [None, None]
    effect_rows += (
        (
            "Paired task success",
            f"Mean candidate minus baseline: {_percentage_points(paired.get('candidate_minus_baseline_mean'))}. 95% interval: {_percentage_points(paired_interval[0])} to {_percentage_points(paired_interval[1])}. One-sided lower bound: {_percentage_points(paired.get('one_sided_lower_bound'))}. Method: {_prose(paired.get('method', 'unknown'))}",
        ),
    )
    position_rows = tuple(
        (
            item.get("label", "Unknown call"),
            f"Baseline mean: {_number(item.get('baseline_mean'), 2)}. Candidate mean: {_number(item.get('candidate_mean'), 2)}. Difference: {_number(item.get('candidate_minus_baseline_mean'), 2)} cached tokens. Pairs: {item.get('paired_count', 0)}",
        )
        for item in result.get("request_position_effects", [])
    )
    if position_rows:
        effect_rows += position_rows
    provenance = result["provenance"]
    environment_rows = tuple(
        (
            key.replace("_", " ").title(),
            json.dumps(value, sort_keys=True)
            if isinstance(value, Mapping)
            else str(value),
        )
        for key, value in sorted(provenance.items())
    ) + tuple(
        (f"{name.replace('_', ' ').title()} hash", str(value))
        for name, value in sorted(result["frozen_hashes"].items())
    )
    evidence = result["evidence_integrity"]
    failure = result["failure_categories"]
    evidence_rows = tuple(
        (key.replace("_", " ").title(), str(value))
        for key, value in sorted(evidence.items())
    ) + tuple(
        (key.replace("_", " ").title(), str(value))
        for key, value in sorted(failure.items())
    )
    claim = result["claim_boundary"]
    limitation_rows = (
        ("Supported claim", str(claim.get("supported", "unknown"))),
        (
            "Claims not supported",
            ", ".join(str(item) for item in claim.get("not_supported", [])),
        ),
        ("Limitations", " ".join(str(item) for item in result.get("limitations", []))),
        ("Artifact source", "src/runeconomy/data/public-proof-rejection.json"),
    )
    return tuple(
        TechnicalSection(
            label,
            _html_table(rows, ("Field", "Recorded value")),
            _markdown_table(rows, ("Field", "Recorded value")),
        )
        for label, rows in (
            ("Acceptance checks", check_rows),
            ("Statistical details", effect_rows),
            ("Environment and provenance", environment_rows),
            ("Evidence integrity", evidence_rows),
            ("Limitations", limitation_rows),
        )
    )


def _generic_summary(
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    raw = result.get("summary")
    if not isinstance(raw, Mapping):
        raise ValueError("result must contain a summary object")
    summary = raw.get("workflow", raw)
    if not isinstance(summary, Mapping):
        raise ValueError("result summary must be an object")
    variants = summary.get("variants")
    effects = summary.get("effects")
    if not isinstance(variants, Mapping) or not isinstance(effects, Mapping):
        raise ValueError("result summary must contain variants and effects objects")
    if not all(
        isinstance(variants.get(name), Mapping) for name in ("baseline", "candidate")
    ):
        raise ValueError("result summary must contain baseline and candidate variants")
    return raw, variants, effects


def _generic_outcomes(variants: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    for name in ("baseline", "candidate"):
        outcomes = variants[name].get("outcomes", variants[name])
        if not isinstance(outcomes, Mapping):
            outcomes = {}
        count = outcomes.get("task_success_count")
        attempts = outcomes.get("attempt_count", outcomes.get("evaluated_count"))
        if attempts == 0:
            items.append((name.title(), "Task success was not evaluated."))
            continue
        if count is None and outcomes.get("task_success_rate") is not None and attempts:
            count = round(float(outcomes["task_success_rate"]) * int(attempts))
        if count is None or attempts is None:
            value = f"Success rate: {_percent(outcomes.get('task_success_rate'))}"
        else:
            value = f"{count} of {attempts} tasks succeeded."
        items.append((name.title(), value))
    return tuple(items)


def _generic_technical_sections(
    result: Mapping[str, Any], effects: Mapping[str, Any]
) -> tuple[TechnicalSection, ...]:
    decision = result.get("decision", {})
    checks = decision.get("checks", []) if isinstance(decision, Mapping) else []
    check_rows = (
        (
            "Recorded decision reason",
            str(decision.get("reason", "No reason recorded.")),
        ),
    ) + tuple(
        (
            str(item.get("name", "unknown")),
            f"{'Passed' if item.get('passed') else 'Failed'}. Required: {'yes' if item.get('required', True) else 'no'}",
        )
        for item in checks
        if isinstance(item, Mapping)
    ) or (("Checks", "No acceptance checks were recorded."),)
    effect_rows = tuple(
        (
            str(name).replace("_", " ").title(),
            f"Mean candidate minus baseline: {_number(item.get('candidate_minus_baseline_mean'), 3)}. 95% interval: {_interval(item, 3)}. Pairs: {item.get('paired_count', 'unknown')}",
        )
        for name, item in sorted(effects.items())
        if isinstance(item, Mapping)
    ) or (("Effects", "No paired effects were recorded."),)
    raw_summary = result.get("summary", {})
    predeclared = (
        raw_summary.get("predeclared_prediction")
        if isinstance(raw_summary, Mapping)
        else None
    )
    if isinstance(predeclared, Mapping):
        effect_rows += (
            (
                "Predeclared hypothesis",
                f"Mechanism: {predeclared.get('causal_mechanism', 'unknown')}. Metric: {predeclared.get('metric', 'unknown')}. Expected direction: {predeclared.get('direction', 'unknown')}. Assessment: {predeclared.get('status', 'unknown')}",
            ),
        )
    provenance = result.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    environment_rows = tuple(
        (
            str(key).replace("_", " ").title(),
            json.dumps(value, sort_keys=True)
            if isinstance(value, Mapping)
            else str(value),
        )
        for key, value in sorted(provenance.items())
    ) or (("Environment", "No environment identity was recorded."),)
    integrity_rows = (
        (
            "Evaluator infrastructure errors",
            str(len(result.get("evaluator_errors", []))),
        ),
        ("Generated content", "Generated output is not embedded in this report."),
    )
    evidence_quality = result.get("evidence_quality")
    evidence_rows: tuple[tuple[str, str], ...]
    if isinstance(evidence_quality, Mapping):
        raw_fields = evidence_quality.get("fields", {})
        field_rows = tuple(
            (
                str(name).replace("_", " ").title(),
                f"{item.get('grade', 'unresolved')}: {item.get('source', 'unknown')}",
            )
            for name, item in sorted(raw_fields.items())
            if isinstance(item, Mapping)
        ) if isinstance(raw_fields, Mapping) else ()
        evidence_rows = (
            ("Overall grade", str(evidence_quality.get("overall_grade", "unresolved"))),
            *field_rows,
        )
    else:
        evidence_rows = (("Overall grade", "Not recorded by this result schema."),)
    limitations = result.get("limitations", [])
    limitation_rows = (
        (
            "Claim boundary",
            "The verdict applies only to the recorded workload, backend, hardware and acceptance rules.",
        ),
        (
            "Limitations",
            " ".join(str(item) for item in limitations)
            or "No additional limitations were recorded.",
        ),
    )
    return tuple(
        TechnicalSection(
            label,
            _html_table(rows, ("Field", "Recorded value")),
            _markdown_table(rows, ("Field", "Recorded value")),
        )
        for label, rows in (
            ("Acceptance checks", check_rows),
            ("Statistical details", effect_rows),
            ("Environment and provenance", environment_rows),
            ("Evidence quality", evidence_rows),
            ("Preregistration and economics", _preregistration_rows(result)),
            ("Identity assurance", _identity_assurance_rows(decision)),
            ("Serving capabilities", _serving_capability_rows(result)),
            ("Evidence integrity", integrity_rows),
            ("Limitations", limitation_rows),
        )
        if rows
    )


def _quality_question(name: str) -> str:
    labels = {
        "baseline_minimum_task_success": "Was the baseline workload reliable enough?",
        "candidate_minimum_task_success": "Did the candidate meet absolute task quality?",
        "minimum_task_success_rate": "Did the variants meet the task-quality floor?",
        "task_success_noninferior": "Did the candidate preserve task quality?",
        "task_success_noninferiority": "Did the candidate preserve task quality?",
    }
    return labels.get(name, name.replace("_", " ").capitalize())


def _build_public_proof_view(result: Mapping[str, Any]) -> ReportView:
    if result.get("decision", {}).get("status") != "invalid":
        raise ValueError("the flagship public proof must retain an invalid verdict")
    experiment = result["experiment"]
    outcomes = result["outcomes"]
    effects = result["conditional_effects"]
    paired = result["paired_success"]
    provenance = result["provenance"]
    acceptance = result["gates"]["predeclared"]
    primary, _, invalid_explanation, _ = _status_copy("invalid", "")
    outcome_counts = paired["outcome_counts"]
    baseline_lower = outcomes["baseline"]["task_success_95_ci"][0]
    paired_lower = paired["one_sided_lower_bound"]
    min_success = acceptance["min_task_success_rate"]
    max_regression = acceptance["max_task_success_regression"]
    baseline_shortfall = min_success - baseline_lower
    paired_shortfall = (-max_regression) - paired_lower
    task_count = experiment["task_count"]
    attempt_count = result["evidence_integrity"]["attempt_count"]
    model = str(provenance["model"]).split("/")[-1]
    return ReportView(
        experiment_id="frozen-public-proof",
        status="invalid",
        status_label="Invalid",
        primary_message=primary,
        verdict_explanation=(
            "The candidate showed better cache reuse and lower measured latency, but this experiment did not establish the task-quality evidence required for a verified performance claim."
        ),
        invalid_explanation=invalid_explanation,
        recommendation=(
            "Do not present this candidate as a verified latency improvement. Establish a more reliable baseline and test the intervention on a new, sufficiently powered held-out workload. The current 40-task workload has been consumed and cannot be reused as fresh evidence."
        ),
        badges=(
            model,
            f"vLLM {provenance['backend_version']}",
            str(provenance["hardware"]),
            f"{task_count} paired tasks",
            f"{attempt_count} attempts",
        ),
        intervention_intro=(
            "vLLM can reuse cached prompt tokens while a new prompt matches a previous prompt from the beginning. In the baseline, changing metadata appeared first, so the shared prefix ended immediately. The candidate moved that metadata after the stable instructions."
        ),
        intervention_baseline="Changing run metadata appeared before stable system instructions.",
        intervention_candidate="The same changing metadata moved after stable system instructions.",
        intervention_why="A longer stable prefix may allow more prompt tokens to be reused from cache.",
        intervention_control="The prompt content stayed the same. Only the block order changed.",
        intervention_kind="prefix-order",
        result_cards=(
            _card(
                "Cached input tokens",
                effects["cached_input_tokens"],
                positive_is_more=True,
            ),
            _card("Server time to first token", effects["server_ttft_ms"], unit="ms"),
            _card("Workflow latency", effects["workflow_latency_ms"], unit="ms"),
        ),
        result_qualification=(
            "Measured across the 36 task pairs where both variants succeeded. These measurements are not a verified improvement until the quality requirements pass."
        ),
        outcome_items=(
            ("Baseline", "37 of 40 tasks succeeded."),
            ("Candidate", "38 of 40 tasks succeeded."),
            ("Both succeeded", f"{outcome_counts['both_success']} pairs"),
            ("Baseline only succeeded", f"{outcome_counts['baseline_only']} pair"),
            ("Candidate only succeeded", f"{outcome_counts['candidate_only']} pairs"),
            ("Both failed", f"{outcome_counts['both_failure']} pair"),
        ),
        cache_position_items=tuple(
            (
                item["label"],
                f"Baseline {_number(item['baseline_mean'], 2)}, candidate {_number(item['candidate_mean'], 2)}, difference {_number(item['candidate_minus_baseline_mean'], 2)} cached tokens",
            )
            for item in result.get("request_position_effects", [])
        ),
        quality_requirements=(
            QualityRequirement(
                "Was the baseline workload reliable enough?",
                (
                    ("Required lower confidence bound", _percent(min_success, 2)),
                    ("Observed baseline lower bound", _percent(baseline_lower, 2)),
                    (
                        "Result",
                        f"Failed by {baseline_shortfall * 100:.2f} percentage points",
                    ),
                ),
                "The baseline succeeded on 37 of 40 tasks. With only 40 tasks, the lower confidence bound remained slightly below the required threshold.",
                False,
            ),
            QualityRequirement(
                "Did the candidate preserve task quality?",
                (
                    (
                        "Maximum allowed regression",
                        f"{max_regression * 100:.2f} percentage points",
                    ),
                    (
                        "Observed lower-bound difference",
                        f"negative {abs(paired_lower) * 100:.2f} percentage points",
                    ),
                    (
                        "Result",
                        f"Failed by {paired_shortfall * 100:.2f} percentage points",
                    ),
                ),
                "The candidate's observed success rate was higher, but the uncertainty was too wide to rule out a regression larger than the allowed margin.",
                False,
            ),
        ),
        decision_path=(
            ("Evidence usable?", "Passed"),
            ("Baseline reliable enough?", "Failed"),
            ("Performance claim", "Not eligible for verification"),
        ),
        later_checks=(
            "Candidate absolute quality: Passed",
            "Paired quality preservation: Failed",
            "Conditional workflow latency: Favorable but not eligible",
        ),
        experiment_intro=(
            "Each task gave the agent a code clue and a bounded set of repository files. The agent planned a search, used a repository-search tool and identified the enclosing Python function or class."
        ),
        experiment_facts=(
            (
                "Task families",
                "13 direct-call lookup tasks, 13 docstring lookup tasks and 14 unique-string lookup tasks",
            ),
            ("Paired workload", "40 baseline and candidate pairs, 80 total attempts"),
            (
                "Workflow",
                "2 model calls and 1 repository-search tool call per attempt",
            ),
            (
                "Environment",
                f"{model}, vLLM {provenance['backend_version']}, Modal {provenance['hardware']}",
            ),
            (
                "Execution",
                "Chat Completions, colocated driver and sequential execution",
            ),
        ),
        controls=(
            "The model, backend, hardware, agent, tasks, evaluator and acceptance rules were chosen and locked before the held-out workload was run."
        ),
        technical_sections=_technical_sections_for_public(result),
    )


def _build_generic_view(result: Mapping[str, Any]) -> ReportView:
    _, variants, effects = _generic_summary(result)
    decision = result.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("result must contain a decision object")
    status = str(decision.get("status", "unknown")).lower()
    reason = str(decision.get("reason", "No reason recorded."))
    primary, explanation, invalid_explanation, _ = _status_copy(status, reason)
    recommendation = _generic_recommendation(status, decision)
    provenance = result.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    context = result.get("measurement_context", {})
    if not isinstance(context, Mapping):
        context = {}
    _, cached = _metric_effect(effects, "cached_input_tokens", "cached_tokens")
    _, ttft = _metric_effect(effects, "server_ttft_ms")
    _, latency = _metric_effect(effects, "workflow_latency_ms", "total_latency_ms")
    pair_count = latency.get("paired_count", cached.get("paired_count", "the recorded"))
    evaluated = sum(
        int(
            variants[name]
            .get("outcomes", variants[name])
            .get("evaluated_count", 0)
            or 0
        )
        for name in ("baseline", "candidate")
    )
    badges = tuple(
        str(item)
        for item in (
            provenance.get("model"),
            f"vLLM {provenance['backend_version']}"
            if provenance.get("backend_version")
            else None,
            provenance.get("hardware"),
            f"{evaluated} evaluated attempts" if evaluated else None,
        )
        if item
    )
    baseline_tools = context.get("baseline_tool_count")
    candidate_tools = context.get("candidate_tool_count")
    if baseline_tools is not None and candidate_tools is not None:
        intervention_intro = "The experiment compared the number of tool definitions attached to each request."
        baseline_copy = f"Each request exposed {baseline_tools} tool definitions."
        candidate_copy = f"Each request exposed {candidate_tools} tool definitions."
        why = "A smaller tool schema may reduce prompt processing work."
        control = "The recorded manifest identifies the request fields that were held constant."
    else:
        intervention = result.get("intervention", {})
        if not isinstance(intervention, Mapping):
            intervention = {}
        intervention_intro = _prose(
            intervention.get(
                "summary",
                "The experiment compared a recorded baseline with a recorded candidate.",
            )
        )
        baseline_copy = _prose(
            intervention.get("baseline", "The recorded baseline request or workflow.")
        )
        candidate_copy = _prose(
            intervention.get("candidate", "The recorded candidate request or workflow.")
        )
        why = _prose(
            intervention.get(
                "expected_mechanism",
                "The candidate was expected to improve the recorded optimization target.",
            )
        )
        control = _prose(
            intervention.get(
                "held_constant",
                "Only the changes declared in the experiment manifest were intended to vary.",
            )
        )
    qualification = (
        f"Measured across {pair_count} paired observations. The declared quality and performance requirements passed."
        if status == "verified"
        else f"Measured across {pair_count} paired observations. These measurements are not a verified improvement until the quality requirements pass."
    )
    checks = [item for item in decision.get("checks", []) if isinstance(item, Mapping)]
    quality_requirements = tuple(
        QualityRequirement(
            _quality_question(str(item.get("name", "Recorded acceptance requirement"))),
            (
                ("Required", "Yes" if item.get("required", True) else "No"),
                ("Result", "Passed" if item.get("passed") else "Failed"),
            ),
            "See the exact threshold and observed value in Acceptance checks.",
            bool(item.get("passed")),
        )
        for item in checks
        if "success" in str(item.get("name", ""))
        or "quality" in str(item.get("name", ""))
    )
    facts: list[tuple[str, str]] = []
    if context.get("measured_workflow_pair_count") is not None:
        facts.append(
            (
                "Paired workload",
                f"{context['measured_workflow_pair_count']} measured workflow pairs",
            )
        )
    if context.get("workflow_step_count") is not None:
        facts.append(
            (
                "Workflow",
                f"{context['workflow_step_count']} sequential inference calls per workflow",
            )
        )
    if provenance.get("model"):
        facts.append(("Model", str(provenance["model"])))
    if provenance.get("backend_version"):
        facts.append(("Backend", f"vLLM {provenance['backend_version']}"))
    if provenance.get("hardware"):
        facts.append(("Hardware", str(provenance["hardware"])))
    if not facts:
        facts.append(("Experiment", str(result.get("experiment_id", "unknown"))))
    return ReportView(
        experiment_id=str(result.get("experiment_id", "unknown-experiment")),
        status=status,
        status_label=status.title(),
        primary_message=primary,
        verdict_explanation=explanation,
        invalid_explanation=invalid_explanation,
        recommendation=recommendation,
        badges=badges,
        intervention_intro=intervention_intro,
        intervention_baseline=baseline_copy,
        intervention_candidate=candidate_copy,
        intervention_why=why,
        intervention_control=control,
        intervention_kind="generic",
        result_cards=(
            _card("Cached input tokens", cached, positive_is_more=True),
            _card("Server time to first token", ttft, unit="ms"),
            _card("Workflow latency", latency, unit="ms"),
        ),
        result_qualification=qualification,
        outcome_items=_generic_outcomes(variants),
        cache_position_items=(),
        quality_requirements=quality_requirements,
        decision_path=(("Recorded verdict", status.title()),),
        later_checks=(),
        experiment_intro=(
            "This report describes the recorded baseline and candidate comparison without adding evidence that was not present in the result."
        ),
        experiment_facts=tuple(facts),
        controls=(
            "The report uses only the workload, environment and acceptance rules recorded in the experiment result."
        ),
        technical_sections=_generic_technical_sections(result, effects),
    )


def _build_report_view_model(result: Mapping[str, Any]) -> ReportView:
    """Convert a recorded artifact into a deterministic presentation model."""

    if result.get("artifact_kind") == "runeconomy.public_proof":
        return _build_public_proof_view(result)
    return _build_generic_view(result)


def _render_header(view: ReportView) -> str:
    badges = "".join(f"<li>{_escape(item)}</li>" for item in view.badges)
    return (
        '<header class="report-header">'
        "<h1>RunEconomy Experiment Report</h1>"
        f'<ul class="badges" aria-label="Experiment facts">{badges}</ul>'
        "</header>"
    )


def _render_intervention(view: ReportView) -> str:
    if view.intervention_kind == "prefix-order":
        visual = """
<figure class="prefix-visual" aria-labelledby="prefix-caption">
  <div class="variant-row">
    <strong>Baseline</strong>
    <div class="attempt"><span class="metadata">Run metadata A</span><span class="stable">Stable instructions</span></div>
    <div class="attempt"><span class="metadata">Run metadata B</span><span class="stable">Stable instructions</span></div>
    <p>First tokens differ, so the shared prefix stops early.</p>
  </div>
  <div class="variant-row">
    <strong>Candidate</strong>
    <div class="attempt"><span class="stable">Stable instructions</span><span class="metadata">Run metadata A</span></div>
    <div class="attempt"><span class="stable">Stable instructions</span><span class="metadata">Run metadata B</span></div>
    <p>Stable instructions form a reusable shared prefix.</p>
  </div>
  <figcaption id="prefix-caption">Synthetic labels show ordering only. No private prompt content is included.</figcaption>
</figure>"""
    else:
        visual = (
            '<div class="comparison" role="group" aria-label="Baseline and candidate">'
            f"<article><h3>Baseline</h3><p>{_escape(view.intervention_baseline)}</p></article>"
            f"<article><h3>Candidate</h3><p>{_escape(view.intervention_candidate)}</p></article>"
            "</div>"
        )
    return f"""
<section aria-labelledby="intervention-heading">
  <h2 id="intervention-heading">Intervention Tested</h2>
  <p>{_escape(view.intervention_intro)}</p>
  {visual}
  <p><strong>Why this might help:</strong> {_escape(view.intervention_why)}</p>
  <p><strong>Control:</strong> {_escape(view.intervention_control)}</p>
</section>"""


def _render_results(view: ReportView) -> str:
    cards = "".join(
        '<article class="result-card">'
        f"<h3>{_escape(card.label)}</h3>"
        f'<p class="result-value">{_escape(card.value)}</p>'
        f"<p>{_escape(card.description)}</p>"
        "</article>"
        for card in view.result_cards
    )
    outcomes = "".join(
        f"<li><strong>{_escape(label)}:</strong> {_escape(value)}</li>"
        for label, value in view.outcome_items
    )
    position_breakdown = (
        '<div class="position-breakdown"><h3>Cache reuse by agent call</h3>'
        + _html_table(
            view.cache_position_items,
            ("Agent call", "Mean cached input tokens across both-successful pairs"),
        )
        + "</div>"
        if view.cache_position_items
        else ""
    )
    return f"""
<section aria-labelledby="results-heading">
  <h2 id="results-heading">Results</h2>
  <div class="result-grid">{cards}</div>
  <p class="qualification"><strong>Important:</strong> {_escape(view.result_qualification)}</p>
  <ul class="outcome-grid" aria-label="Task outcomes">{outcomes}</ul>
  {position_breakdown}
</section>"""


def _render_verdict(view: ReportView) -> str:
    requirements = "".join(
        '<article class="quality-card">'
        f"<h3>{_escape(item.question)}</h3>"
        f"{_html_table(item.rows, ('Measure', 'Recorded result'))}"
        f"<p>{_escape(item.explanation)}</p>"
        "</article>"
        for item in view.quality_requirements
    )
    decision_path = "".join(
        f"<li><span>{_escape(label)}</span><strong>{_escape(value)}</strong></li>"
        for label, value in view.decision_path
    )
    later = "".join(f"<li>{_escape(item)}</li>" for item in view.later_checks)
    invalid = (
        f'<p class="invalid-explanation">{_escape(view.invalid_explanation)}</p>'
        if view.invalid_explanation
        else ""
    )
    requirements_html = (
        f'<div class="quality-grid">{requirements}</div>' if requirements else ""
    )
    later_html = f'<ul class="later-checks">{later}</ul>' if later else ""
    return f"""
<section aria-labelledby="verdict-heading">
  <h2 id="verdict-heading">Verdict</h2>
  <div class="verdict verdict-{_escape(view.status)}">
    <p class="verdict-message">{_escape(view.primary_message)}</p>
    <p>{_escape(view.verdict_explanation)}</p>
    <p class="canonical-status">Verdict: {_escape(view.status_label)}</p>
    {invalid}
  </div>
  {requirements_html}
  <ol class="decision-path" aria-label="Decision path">{decision_path}</ol>
  {later_html}
  <p class="recommendation"><strong>Recommended next step:</strong> {_escape(view.recommendation)}</p>
</section>"""


def _render_experiment_details(view: ReportView) -> str:
    facts = _html_table(view.experiment_facts, ("Experiment fact", "Recorded value"))
    return f"""
<section aria-labelledby="details-heading">
  <h2 id="details-heading">Experiment Details</h2>
  <p>{_escape(view.experiment_intro)}</p>
  {facts}
  <p><strong>Controls:</strong> {_escape(view.controls)}</p>
</section>"""


def _render_technical_details(view: ReportView) -> str:
    details = "".join(
        f"<details><summary>{_escape(section.label)}</summary>{section.body_html}</details>"
        for section in view.technical_sections
    )
    return f"""
<section class="technical" aria-label="Technical details">
  {details}
</section>"""


_REPORT_CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #fff; color: #151515; font: 15px/1.55 Arial, Helvetica, sans-serif; }
main { width: min(1040px, calc(100% - 32px)); margin: 28px auto 64px; }
.report-header { border-bottom: 2px solid #1f2933; padding-bottom: 16px; margin-bottom: 28px; }
h1 { margin: 0; font: 700 28px/1.2 Georgia, "Times New Roman", serif; }
h2 { margin: 34px 0 12px; font: 700 21px/1.25 Georgia, "Times New Roman", serif; }
h3 { margin: 0 0 8px; font-size: 15px; }
p { max-width: 880px; margin: 8px 0; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0 0; padding: 0; list-style: none; }
.badges li { border: 1px solid #8b949e; padding: 3px 8px; font-size: 13px; }
.comparison, .result-grid, .quality-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 14px 0; }
.comparison article, .result-card, .quality-card { border: 1px solid #9aa1a8; padding: 14px; }
.result-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.result-value { margin: 5px 0; font: 700 27px/1.15 Georgia, "Times New Roman", serif; font-variant-numeric: tabular-nums; }
.qualification { border-left: 4px solid #4f5963; padding: 10px 12px; background: #f4f5f6; }
.outcome-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; padding: 0; list-style: none; }
.outcome-grid li { border: 1px solid #b5bac0; padding: 9px; }
.verdict { border: 2px solid #31383f; padding: 16px; }
.verdict-message { margin: 0 0 8px; font: 700 24px/1.2 Georgia, "Times New Roman", serif; }
.canonical-status { font-size: 13px; color: #3f464d; }
.invalid-explanation { font-weight: 700; }
.decision-path { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; padding: 0; list-style: none; counter-reset: path; }
.decision-path li { border-top: 3px solid #343a40; padding: 10px; }
.decision-path span, .decision-path strong { display: block; }
.decision-path span { font-size: 13px; color: #4b535b; }
.later-checks { margin: 10px 0; }
.recommendation { border: 1px solid #5d646b; padding: 12px; }
.prefix-visual { margin: 16px 0; padding: 14px; border: 1px solid #8d959d; }
.variant-row + .variant-row { border-top: 1px solid #c0c5ca; margin-top: 16px; padding-top: 16px; }
.attempt { display: flex; min-width: 0; margin: 7px 0; font: 12px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace; }
.attempt span { border: 1px solid #4c545c; padding: 7px; overflow-wrap: anywhere; }
.attempt .stable { flex: 1; background: #eef1f3; }
.attempt .metadata { background: #fff; }
figcaption { color: #4b535b; font-size: 12px; }
.table-wrap { overflow-x: auto; margin: 10px 0 16px; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #a5abb1; padding: 7px 8px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
thead th { background: #eef0f2; }
code { font: 12px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
details { border-top: 1px solid #aeb4ba; padding: 9px 0; }
summary { cursor: pointer; font-weight: 700; }
summary:focus-visible { outline: 3px solid #1d4ed8; outline-offset: 3px; }
@media (max-width: 700px) {
  main { width: min(100% - 20px, 1040px); margin-top: 18px; }
  .result-grid, .comparison, .quality-grid, .outcome-grid, .decision-path { grid-template-columns: 1fr; }
  h1 { font-size: 24px; }
  .attempt { font-size: 11px; }
}
"""


def render_experiment_html(result: Mapping[str, Any]) -> str:
    """Render a self-contained experiment report from recorded evidence."""

    view = _build_report_view_model(result)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RunEconomy Experiment Report</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<main>
{_render_header(view)}
{_render_intervention(view)}
{_render_results(view)}
{_render_verdict(view)}
{_render_experiment_details(view)}
{_render_technical_details(view)}
</main>
</body>
</html>"""


def _render_markdown(view: ReportView) -> str:
    lines = ["# RunEconomy Experiment Report", ""]
    if view.badges:
        lines.extend([" · ".join(view.badges), ""])
    lines.extend(
        [
            "## Intervention Tested",
            "",
            view.intervention_intro,
            "",
            f"- **Baseline:** {view.intervention_baseline}",
            f"- **Candidate:** {view.intervention_candidate}",
            f"- **Why this might help:** {view.intervention_why}",
            f"- **Control:** {view.intervention_control}",
            "",
            "## Results",
            "",
        ]
    )
    for card in view.result_cards:
        lines.extend(
            [
                f"### {card.label}",
                "",
                f"**{card.value}**",
                "",
                card.description,
                "",
            ]
        )
    lines.extend([f"**Important:** {view.result_qualification}", ""])
    lines.extend(f"- **{label}:** {value}" for label, value in view.outcome_items)
    if view.cache_position_items:
        lines.extend(
            [
                "",
                "### Cache reuse by agent call",
                "",
                _markdown_table(
                    view.cache_position_items,
                    (
                        "Agent call",
                        "Mean cached input tokens across both-successful pairs",
                    ),
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"### {view.primary_message}",
            "",
            view.verdict_explanation,
            "",
            f"Verdict: {view.status_label}",
            "",
        ]
    )
    if view.invalid_explanation:
        lines.extend([view.invalid_explanation, ""])
    for requirement in view.quality_requirements:
        lines.extend(
            [
                f"### {requirement.question}",
                "",
                _markdown_table(requirement.rows, ("Measure", "Recorded result")),
                "",
                requirement.explanation,
                "",
            ]
        )
    if view.decision_path:
        lines.extend(["### Decision path", ""])
        lines.extend(
            f"1. **{label}{'' if label.endswith('?') else ':'}** {value}"
            for label, value in view.decision_path
        )
        lines.append("")
    lines.extend(f"- {item}" for item in view.later_checks)
    if view.later_checks:
        lines.append("")
    lines.extend(
        [
            f"**Recommended next step:** {view.recommendation}",
            "",
            "## Experiment Details",
            "",
            view.experiment_intro,
            "",
            _markdown_table(
                view.experiment_facts, ("Experiment fact", "Recorded value")
            ),
            "",
            f"**Controls:** {view.controls}",
            "",
            "## Technical Details",
            "",
        ]
    )
    for section in view.technical_sections:
        lines.extend(
            [
                "<details>",
                f"<summary>{section.label}</summary>",
                "",
                section.body_markdown,
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(_prose(line) for line in lines).rstrip() + "\n"


def render_experiment_markdown(result: Mapping[str, Any]) -> str:
    """Render the same report view model as accessible Markdown."""

    return _render_markdown(_build_report_view_model(result))


def render_public_proof_markdown(result: Mapping[str, Any]) -> str:
    """Render the public proof case study from its sanitized artifact."""

    if result.get("artifact_kind") != "runeconomy.public_proof":
        raise ValueError("result is not a RunEconomy public-proof artifact")
    return _render_markdown(_build_public_proof_view(result))


def load_demo_result(case: str = "rejection") -> dict[str, Any]:
    """Load one bundled, sanitized result used by the CPU-only demo."""

    resources = {
        "rejection": "public-proof-rejection.json",
        "historical": "repository-agent-tool-pruning.json",
    }
    try:
        name = resources[case]
    except KeyError as exc:
        raise ValueError(f"unknown demo case: {case}") from exc
    resource = files("runeconomy").joinpath(f"data/{name}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bundled demo result must be a JSON object")
    return payload


def render_demo_html(result: Mapping[str, Any]) -> str:
    """Render either the flagship invalid proof or historical experiment result."""

    return render_experiment_html(result)
