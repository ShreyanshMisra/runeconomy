"""Static, shareable reports for controlled experiment results."""

from __future__ import annotations

import html
import json
from importlib.resources import files
from typing import Any, Mapping


def _number(value: Any, digits: int = 1) -> str:
    return "unknown" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 1) -> str:
    return "unknown" if value is None else f"{float(value) * 100:.{digits}f}%"


def _percentage_points(value: Any, digits: int = 1) -> str:
    return (
        "unknown"
        if value is None
        else f"{float(value) * 100:.{digits}f} percentage points"
    )


def _interval(effect: Mapping[str, Any], digits: int = 1) -> str:
    interval = effect.get("bootstrap_95_ci")
    if not isinstance(interval, list) or len(interval) != 2:
        return "unknown"
    return f"[{_number(interval[0], digits)}, {_number(interval[1], digits)}]"


def _percent_interval(effect: Mapping[str, Any]) -> str:
    interval = effect.get("confidence_interval", effect.get("bootstrap_95_ci"))
    if not isinstance(interval, list) or len(interval) != 2:
        return "unknown"
    if interval[0] is None or interval[1] is None:
        return "unknown"
    return f"[{float(interval[0]) * 100:.1f}, {float(interval[1]) * 100:.1f}] pp"


def render_experiment_markdown(result: Mapping[str, Any]) -> str:
    if not isinstance(result.get("summary"), dict) or not isinstance(
        result.get("decision"), dict
    ):
        raise ValueError("result must contain summary and decision objects")
    experiment_id = result.get("experiment_id", "unknown-experiment")
    decision = result["decision"]
    raw_summary = result["summary"]
    summary = raw_summary.get("workflow", raw_summary)
    variants = summary["variants"]
    effects = summary["effects"]
    provenance = result.get("provenance", {})

    metrics = (
        ("input_tokens", "Input tokens", 1),
        ("cached_tokens", "Cached tokens", 1),
        ("cache_coverage", "Cache coverage", 3),
        ("server_ttft_ms", "Server TTFT (ms)", 1),
        ("client_ttft_ms", "Client TTFT (ms)", 1),
        ("client_server_ttft_gap_ms", "Client − server TTFT gap (ms)", 1),
        ("total_latency_ms", "Total latency (ms)", 1),
        ("output_tokens", "Output tokens", 1),
        ("task_success", "Task success", 3),
    )
    lines = [
        f"# NegativeSplit report: {experiment_id}",
        "",
        f"**Decision: {str(decision.get('status', 'unknown')).upper()}**",
        "",
        str(decision.get("reason", "No decision reason recorded.")),
        "",
        "## Environment",
        "",
        f"- Model: `{provenance.get('model', 'unknown')}`",
        f"- Model revision: `{provenance.get('model_revision', 'unknown')}`",
        f"- Backend: vLLM `{provenance.get('backend_version', 'unknown')}`",
        f"- API dialect: `{provenance.get('api_dialect', 'unknown')}`",
        f"- Hardware: {provenance.get('hardware', 'unknown')}",
        f"- Cache treatment: `{provenance.get('cache_treatment', 'unknown')}`",
        f"- Cache isolation: `{provenance.get('cache_isolation', 'unknown')}`",
        f"- Trial order: `{provenance.get('order', 'unknown')}`",
        f"- Optimization target: `{provenance.get('acceptance', {}).get('optimization_target', 'cache_reuse')}`",
        f"- Reproducibility manifest SHA-256: `{provenance.get('reproducibility_manifest_sha256', provenance.get('manifest_sha256', 'unknown'))}`",
        "",
        "## Outcomes",
        "",
        "| Variant | Completion rate | Task-success rate | Exact 95% CI | Evaluated trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("baseline", "candidate"):
        outcomes = variants[variant]["outcomes"]
        lines.append(
            f"| {variant.title()} | {_percent(outcomes.get('completed_rate'))} | "
            f"{_percent(outcomes.get('task_success_rate'))} | "
            f"{_percent_interval({'confidence_interval': outcomes.get('task_success_95_ci')})} | "
            f"{outcomes.get('evaluated_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Inference results",
            "",
            "| Metric | Baseline mean | Candidate mean | Candidate − baseline | 95% interval |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label, digits in metrics:
        if (
            key not in variants["baseline"]
            or key not in variants["candidate"]
            or key not in effects
        ):
            continue
        baseline = variants["baseline"][key]["mean"]
        candidate = variants["candidate"][key]["mean"]
        effect = effects[key]
        if key in {"cache_coverage", "task_success"}:
            baseline_text = _percent(baseline)
            candidate_text = _percent(candidate)
            delta = effect.get("candidate_minus_baseline_mean")
            delta_text = "unknown" if delta is None else f"{float(delta) * 100:.1f} pp"
            interval_text = _percent_interval(effect)
        else:
            baseline_text = _number(baseline, digits)
            candidate_text = _number(candidate, digits)
            delta_text = _number(effect.get("candidate_minus_baseline_mean"), digits)
            interval_text = _interval(effect, digits)
        lines.append(
            f"| {label} | {baseline_text} | {candidate_text} | "
            f"{delta_text} | {interval_text} |"
        )

    prediction = summary.get("prediction_vs_observation", {})
    predeclared = raw_summary.get("predeclared_prediction")
    lines.extend(
        [
            "",
            "## Prediction versus observation",
            "",
            f"- Predicted cached-token delta: {_number(prediction.get('predicted_candidate_minus_baseline_cached_tokens'))}",
            f"- Observed cached-token delta: {_number(prediction.get('observed_candidate_minus_baseline_cached_tokens'))}",
            f"- Realization ratio: {_number(prediction.get('realization_ratio'), 3)}",
        ]
    )
    if isinstance(predeclared, Mapping):
        lines.extend(
            [
                "",
                "## Predeclared hypothesis",
                "",
                f"- Mechanism: {predeclared.get('causal_mechanism', 'unknown')}",
                f"- Prediction: `{predeclared.get('metric', 'unknown')}` should "
                f"`{predeclared.get('direction', 'unknown')}`",
                f"- Assessment: **{str(predeclared.get('status', 'unknown')).upper()}**",
                f"- Observed candidate − baseline: "
                f"{_number(predeclared.get('candidate_minus_baseline_mean'))}",
                f"- Paired 95% interval: "
                f"{predeclared.get('confidence_interval', 'unknown')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Verification checks",
            "",
            "| Check | Required | Passed |",
            "|---|---:|---:|",
        ]
    )
    for check in decision.get("checks", []):
        required = check.get("required", True)
        lines.append(
            f"| `{check.get('name', 'unknown')}` | "
            f"{'yes' if required else 'no'} | "
            f"{'yes' if check.get('passed') else 'no'} |"
        )
    if summary is not raw_summary:
        lines.extend(
            [
                "",
                "## Workflow aggregation",
                "",
                f"- Sequential inference steps: {summary.get('step_count', 'unknown')}",
                f"- Claim: {summary.get('claim', 'unknown')}",
            ]
        )
    evaluated_count = sum(
        int(variants[name].get("outcomes", {}).get("evaluated_count", 0) or 0)
        for name in ("baseline", "candidate")
    )
    evaluation_text = (
        "Generated output text was evaluated locally; generated text was not persisted in "
        "raw trial records."
        if evaluated_count
        else "Task success was not evaluated for this experiment."
    )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"{evaluation_text} An exact output-digest match is not treated as semantic "
            "correctness. The decision above applies only to the pinned workload, backend, "
            "hardware, and acceptance thresholds recorded here.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _signed_percent(value: Any) -> str:
    if value is None:
        return "unknown"
    return f"{float(value) * 100:+.1f}%"


def render_experiment_html(result: Mapping[str, Any]) -> str:
    """Render a generic report using only fields recorded in an experiment result."""

    if not isinstance(result.get("summary"), dict) or not isinstance(
        result.get("decision"), dict
    ):
        raise ValueError("result must contain summary and decision objects")
    raw_summary = result["summary"]
    summary = raw_summary.get("workflow", raw_summary)
    variants = summary.get("variants")
    effects = summary.get("effects")
    if not isinstance(variants, dict) or not isinstance(effects, dict):
        raise ValueError("result summary must contain variants and effects objects")
    if not all(
        isinstance(variants.get(name), dict) for name in ("baseline", "candidate")
    ):
        raise ValueError("result summary must contain baseline and candidate variants")

    metric_specs = (
        ("input_tokens", "Input tokens", 1, False),
        ("cached_tokens", "Cached tokens", 1, False),
        ("cache_coverage", "Cache coverage", 1, True),
        ("server_ttft_ms", "Server TTFT (ms)", 1, False),
        ("client_ttft_ms", "Client TTFT (ms)", 1, False),
        ("client_server_ttft_gap_ms", "Client − server TTFT gap (ms)", 1, False),
        ("total_latency_ms", "Total latency (ms)", 1, False),
        ("output_tokens", "Output tokens", 1, False),
        ("task_success", "Task success", 1, True),
    )
    metric_rows: list[str] = []
    for key, label, digits, percentage in metric_specs:
        baseline_metric = variants["baseline"].get(key)
        candidate_metric = variants["candidate"].get(key)
        effect = effects.get(key)
        if not all(
            isinstance(item, dict)
            for item in (baseline_metric, candidate_metric, effect)
        ):
            continue
        baseline_value = baseline_metric.get("mean")
        candidate_value = candidate_metric.get("mean")
        delta = effect.get("candidate_minus_baseline_mean")
        if percentage:
            baseline_text = _percent(baseline_value)
            candidate_text = _percent(candidate_value)
            delta_text = "unknown" if delta is None else f"{float(delta) * 100:.1f} pp"
            interval_text = _percent_interval(effect)
        else:
            baseline_text = _number(baseline_value, digits)
            candidate_text = _number(candidate_value, digits)
            delta_text = _number(delta, digits)
            interval_text = _interval(effect, digits)
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f'<td class="number">{html.escape(baseline_text)}</td>'
            f'<td class="number">{html.escape(candidate_text)}</td>'
            f'<td class="number">{html.escape(delta_text)}</td>'
            f'<td class="number">{html.escape(interval_text)}</td>'
            "</tr>"
        )

    outcome_rows: list[str] = []
    evaluated_total = 0
    for name in ("baseline", "candidate"):
        outcomes = variants[name].get("outcomes", {})
        if not isinstance(outcomes, dict):
            outcomes = {}
        evaluated = int(outcomes.get("evaluated_count", 0) or 0)
        evaluated_total += evaluated
        outcome_rows.append(
            "<tr>"
            f"<td>{name.title()}</td>"
            f'<td class="number">{html.escape(_percent(outcomes.get("completed_rate")))}</td>'
            f'<td class="number">{html.escape(_percent(outcomes.get("task_success_rate")))}</td>'
            f'<td class="number">{html.escape(_percent_interval({"confidence_interval": outcomes.get("task_success_95_ci")}))}</td>'
            f'<td class="number">{evaluated}</td>'
            "</tr>"
        )

    decision = result["decision"]
    check_rows = []
    for check in decision.get("checks", []):
        if not isinstance(check, dict):
            continue
        check_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(check.get('name', 'unknown')))}</code></td>"
            f"<td>{'yes' if check.get('required', True) else 'no'}</td>"
            f"<td>{'yes' if check.get('passed') else 'no'}</td>"
            "</tr>"
        )
    provenance = result.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    workflow_note = ""
    if summary is not raw_summary:
        workflow_note = (
            "<p><strong>Workflow aggregation:</strong> "
            f"{html.escape(str(summary.get('step_count', 'unknown')))} sequential "
            "inference steps per workflow unit.</p>"
        )
    evaluation_note = (
        "Generated output was evaluated locally. Generated text is not stored in the "
        "trial result."
        if evaluated_total
        else "Task success was not evaluated for this experiment."
    )
    evaluator_errors = result.get("evaluator_errors", [])
    error_count = len(evaluator_errors) if isinstance(evaluator_errors, list) else 0
    error_note = (
        f"<p><strong>Evaluator infrastructure errors:</strong> {error_count}. "
        "Affected trials have unknown task success and cannot support a verified decision.</p>"
        if error_count
        else ""
    )
    experiment_id = html.escape(str(result.get("experiment_id", "unknown")))
    status = html.escape(str(decision.get("status", "unknown")))
    reason = html.escape(str(decision.get("reason", "No reason recorded.")))
    predeclared = raw_summary.get("predeclared_prediction")
    prediction_note = ""
    if isinstance(predeclared, Mapping):
        prediction_note = (
            "<h2>Predeclared hypothesis</h2><table><tbody>"
            f"<tr><th>Causal mechanism</th><td>{html.escape(str(predeclared.get('causal_mechanism', 'unknown')))}</td></tr>"
            f"<tr><th>Prediction</th><td><code>{html.escape(str(predeclared.get('metric', 'unknown')))}</code> should {html.escape(str(predeclared.get('direction', 'unknown')))}</td></tr>"
            f"<tr><th>Assessment</th><td>{html.escape(str(predeclared.get('status', 'unknown')))}</td></tr>"
            f"<tr><th>Candidate − baseline</th><td>{html.escape(_number(predeclared.get('candidate_minus_baseline_mean')))}</td></tr>"
            f"<tr><th>Paired 95% interval</th><td>{html.escape(str(predeclared.get('confidence_interval', 'unknown')))}</td></tr>"
            "</tbody></table>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RunEconomy NegativeSplit report — {experiment_id}</title>
<style>
body {{ margin: 0; background: #fff; color: #111; font: 14px/1.45 Arial, sans-serif; }}
main {{ width: min(980px, calc(100% - 32px)); margin: 32px auto 64px; }}
header {{ border-bottom: 2px solid #111; padding-bottom: 12px; }}
h1 {{ margin: 0 0 4px; font: 700 25px/1.2 Georgia, serif; }}
h2 {{ margin: 26px 0 8px; font: 700 18px/1.2 Georgia, serif; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0 18px; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid #aaa; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eee; }} .number {{ text-align: right; white-space: nowrap; }}
code {{ font: 12px ui-monospace, monospace; overflow-wrap: anywhere; }}
.note {{ color: #444; }} footer {{ border-top: 1px solid #999; margin-top: 28px; padding-top: 10px; color: #555; }}
</style></head><body><main>
<header><h1>RunEconomy NegativeSplit report</h1><div>Experiment: <code>{experiment_id}</code></div></header>
<h2>Decision</h2><p><strong>{status}</strong> — {reason}</p>{workflow_note}{error_note}
<h2>Outcomes</h2><table><thead><tr><th>Variant</th><th class="number">Completion rate</th><th class="number">Task-success rate</th><th class="number">Exact 95% CI</th><th class="number">Evaluated trials</th></tr></thead><tbody>{"".join(outcome_rows)}</tbody></table>
<h2>Inference results</h2><table><thead><tr><th>Metric</th><th class="number">Baseline mean</th><th class="number">Candidate mean</th><th class="number">Candidate − baseline</th><th class="number">95% interval</th></tr></thead><tbody>{"".join(metric_rows)}</tbody></table>
{prediction_note}
<h2>Verification checks</h2><table><thead><tr><th>Check</th><th>Required</th><th>Passed</th></tr></thead><tbody>{"".join(check_rows)}</tbody></table>
<h2>Recorded environment</h2><table><tbody>
<tr><th>Model</th><td><code>{html.escape(str(provenance.get("model", "unknown")))}</code></td></tr>
<tr><th>Model revision</th><td><code>{html.escape(str(provenance.get("model_revision", "unknown")))}</code></td></tr>
<tr><th>Backend</th><td><code>{html.escape(str(provenance.get("backend_kind", "vllm")))} {html.escape(str(provenance.get("backend_version", "unknown")))}</code></td></tr>
<tr><th>API dialect</th><td><code>{html.escape(str(provenance.get("api_dialect", "unknown")))}</code></td></tr>
<tr><th>Hardware</th><td>{html.escape(str(provenance.get("hardware", "unknown")))}</td></tr>
<tr><th>Cache treatment</th><td><code>{html.escape(str(provenance.get("cache_treatment", "unknown")))}</code></td></tr>
<tr><th>Cache isolation</th><td><code>{html.escape(str(provenance.get("cache_isolation", "unknown")))}</code></td></tr>
<tr><th>Reproducibility manifest SHA-256</th><td><code>{html.escape(str(provenance.get("reproducibility_manifest_sha256", provenance.get("manifest_sha256", "unknown"))))}</code></td></tr>
</tbody></table>
<h2>Claim boundary</h2><p>{html.escape(evaluation_note)} Exact output-digest agreement is not treated as semantic correctness. The decision applies only to the recorded workload, backend, hardware, and acceptance thresholds.</p>
<footer>Generated from recorded RunEconomy aggregate result fields. No remote scripts, fonts, or network resources are used.</footer>
</main></body></html>"""


def _render_public_proof_html(result: Mapping[str, Any]) -> str:
    """Render the aggregate-only, quality-gated public proof."""

    if result.get("artifact_kind") != "runeconomy.public_proof":
        raise ValueError("result is not a RunEconomy public-proof artifact")
    decision = result.get("decision", {})
    experiment = result.get("experiment", {})
    outcomes = result.get("outcomes", {})
    effects = result.get("conditional_effects", {})
    gates = result.get("gates", {})
    provenance = result.get("provenance", {})
    hashes = result.get("frozen_hashes", {})
    evidence = result.get("evidence_integrity", {})
    failures = result.get("failure_categories", {})
    claim = result.get("claim_boundary", {})
    paired_success = result.get("paired_success", {})
    intervention = result.get("intervention", {})
    limitations = result.get("limitations", [])
    if decision.get("status") != "invalid":
        raise ValueError("the flagship public proof must retain an invalid verdict")

    def effect_row(key: str, label: str) -> str:
        item = effects.get(key, {})
        interval = item.get("confidence_interval")
        interval_text = (
            f"[{_number(interval[0], 2)}, {_number(interval[1], 2)}]"
            if isinstance(interval, list) and len(interval) == 2
            else "unavailable"
        )
        return (
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f'<td class="number">{html.escape(_number(item.get("candidate_minus_baseline_mean"), 2))}</td>'
            f'<td class="number">{html.escape(interval_text)}</td>'
            f'<td class="number">{html.escape(str(item.get("paired_count", 0)))}</td>'
            "</tr>"
        )

    outcome_rows = []
    for variant in ("baseline", "candidate"):
        item = outcomes.get(variant, {})
        interval = item.get("task_success_95_ci") or [None, None]
        outcome_rows.append(
            "<tr>"
            f"<td>{variant.title()}</td>"
            f'<td class="number">{item.get("task_success_count", "unknown")} / {item.get("attempt_count", "unknown")}</td>'
            f'<td class="number">{html.escape(_percent(item.get("task_success_rate")))}</td>'
            f'<td class="number">{html.escape(_percent(interval[0], 2))}–{html.escape(_percent(interval[1], 2))}</td>'
            f'<td class="number">{html.escape(_percent(item.get("completed_rate")))}</td>'
            "</tr>"
        )

    check_rows = []
    for item in gates.get("checks", []):
        passed = bool(item.get("passed"))
        check_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(item.get('name', 'unknown')))}</code></td>"
            f"<td>{html.escape(str(item.get('threshold', 'unknown')))}</td>"
            f'<td class="{"met" if passed else "failed"}">{"MET" if passed else "FAILED"}</td>'
            "</tr>"
        )

    family_text = ", ".join(
        f"{html.escape(str(name))}: {count}"
        for name, count in experiment.get("task_family_counts", {}).items()
    )
    failed_names = ", ".join(
        f"<code>{html.escape(str(name))}</code>"
        for name in decision.get("failed_quality_gates", [])
    )
    not_supported = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in claim.get("not_supported", [])
    )
    gate_order = "".join(
        "<li>"
        f"<code>{html.escape(str(item.get('stage', 'unknown')))}</code> — "
        f"{html.escape(str(item.get('observed_result', 'unknown')))}"
        f"{' <strong>(decisive)</strong>' if item.get('decisive') else ''}"
        "</li>"
        for item in decision.get("gating_order", [])
    )
    limitation_items = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in limitations
    )
    hash_rows = "".join(
        "<tr>"
        f"<th>{html.escape(str(name).replace('_', ' ').title())}</th>"
        f"<td><code>{html.escape(str(value))}</code></td>"
        "</tr>"
        for name, value in hashes.items()
    )
    paired_interval = paired_success.get("confidence_interval") or [None, None]
    lower_bound = paired_success.get("one_sided_lower_bound")
    acceptance = gates.get("predeclared", {})
    noninferiority_margin = acceptance.get("max_task_success_regression")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RunEconomy NegativeSplit — rejected candidate</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; color: #111; font: 14px/1.45 Arial, Helvetica, sans-serif; }}
main {{ width: min(980px, calc(100% - 32px)); margin: 32px auto 64px; }}
header {{ border-bottom: 3px solid #111; padding-bottom: 14px; }}
h1 {{ margin: 0 0 5px; font: 700 26px/1.2 Georgia, "Times New Roman", serif; }}
h2 {{ margin: 28px 0 8px; font: 700 18px/1.25 Georgia, "Times New Roman", serif; }}
.verdict {{ border: 3px double #111; margin: 22px 0; padding: 14px; }}
.verdict h2 {{ margin: 0 0 8px; font-size: 22px; }}
.failed {{ font-weight: 700; text-decoration: underline; }}
.met {{ font-weight: 700; }}
.conditional {{ border-left: 5px solid #777; padding-left: 12px; }}
.note {{ color: #333; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0 18px; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid #999; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eee; }} td.number, th.number {{ text-align: right; white-space: nowrap; }}
code {{ font: 12px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }}
footer {{ border-top: 1px solid #999; margin-top: 30px; padding-top: 10px; color: #555; font-size: 12px; }}
@media(max-width:700px) {{ body {{ font-size: 13px; }} th,td {{ padding: 5px; }} main {{ width:calc(100% - 20px); }} }}
</style></head><body><main>
<header><h1>RunEconomy NegativeSplit</h1><div>Frozen real-agent held-out proof · aggregate-only public artifact</div></header>

<section class="verdict">
<h2>INVALID — quality gates failed</h2>
<p><strong>RunEconomy declined to verify this candidate.</strong> {html.escape(str(decision.get("reason", "No reason recorded.")))}.</p>
<p>Failed predeclared gates: {failed_names or "unknown"}.</p>
</section>

<h2>Intervention tested</h2>
<table><tbody>
<tr><th>Baseline</th><td>{html.escape(str(intervention.get('baseline', 'unknown')))}.</td></tr>
<tr><th>Candidate</th><td>{html.escape(str(intervention.get('candidate', 'unknown')))}.</td></tr>
<tr><th>Changed property</th><td>{html.escape(str(intervention.get('changed_property', 'unknown')))}.</td></tr>
<tr><th>Held constant</th><td>{html.escape(str(intervention.get('held_constant', 'unknown')))}.</td></tr>
<tr><th>Expected mechanism</th><td>{html.escape(str(intervention.get('expected_mechanism', 'unknown')))}.</td></tr>
</tbody></table>

<h2>Task-quality result</h2>
<table><thead><tr><th>Variant</th><th class="number">Successes</th><th class="number">Success rate</th><th class="number">Exact 95% interval</th><th class="number">Process completion</th></tr></thead><tbody>{''.join(outcome_rows)}</tbody></table>
<p>The paired candidate-minus-baseline success estimate was {_percentage_points(paired_success.get("candidate_minus_baseline_mean"))}; its 95% interval was {_percentage_points(paired_interval[0], 2)} to {_percentage_points(paired_interval[1], 2)}, and the one-sided lower bound was {_percentage_points(lower_bound, 2)}. The predeclared non-inferiority margin was −{_percentage_points(noninferiority_margin, 2)}.</p>

<h2>Predeclared acceptance checks</h2>
<table><thead><tr><th>Check</th><th>Threshold</th><th>Result</th></tr></thead><tbody>{''.join(check_rows)}</tbody></table>

<h2>Decision gates, in order</h2>
<ol>{gate_order}</ol>
<p><strong>Decisive stage:</strong> <code>{html.escape(str(decision.get('decisive_stage', 'unknown')))}</code>. The baseline validity failure made the verdict <code>invalid</code> before successful-pair latency could verify the candidate. Paired non-inferiority also failed independently.</p>

<section class="conditional">
<h2>Conditional observations only — not verified improvements</h2>
<p>The candidate increased cache reuse and reduced TTFT and measured successful-task latency in this run. These paired effects use only the {effects.get('workflow_latency_ms', {}).get('paired_count', 'unknown')} task pairs where both variants succeeded. Failures did not enter the latency sample, and the failed quality gates prevent a verified performance claim.</p>
<table><thead><tr><th>Candidate − baseline metric</th><th class="number">Mean effect</th><th class="number">Paired 95% interval</th><th class="number">Pairs</th></tr></thead><tbody>
{effect_row('cached_input_tokens', 'Cached input tokens')}
{effect_row('server_ttft_ms', 'Server TTFT (ms)')}
{effect_row('client_ttft_ms', 'Client TTFT (ms)')}
{effect_row('workflow_latency_ms', 'End-to-end workflow latency (ms)')}
{effect_row('critical_path_ms', 'Critical-path latency (ms)')}
{effect_row('inference_span_ms', 'Inference-span latency (ms)')}
</tbody></table>
</section>

<h2>Frozen experiment design</h2>
<table><tbody>
<tr><th>Workload</th><td>{experiment.get('task_count', 'unknown')} frozen held-out tasks; {family_text or 'family counts unavailable'}.</td></tr>
<tr><th>Order</th><td><code>{html.escape(str(experiment.get('order', 'unknown')))}</code>, seed <code>{html.escape(str(experiment.get('seed', 'unknown')))}</code>.</td></tr>
<tr><th>Primary outcome</th><td>{html.escape(str(experiment.get('primary_metric', 'unknown')))}.</td></tr>
<tr><th>Latency population</th><td>{html.escape(str(experiment.get('latency_population', 'unknown')))}.</td></tr>
<tr><th>Reuse restriction</th><td>This {experiment.get('task_count', 'unknown')}-task held-out workload has been consumed. It cannot be rerun or retuned and presented as fresh held-out evidence.</td></tr>
</tbody></table>

<h2>Recorded environment</h2>
<table><tbody>
<tr><th>Model</th><td><code>{html.escape(str(provenance.get('model', 'unknown')))}</code></td></tr>
<tr><th>Model revision</th><td><code>{html.escape(str(provenance.get('model_revision', 'unknown')))}</code></td></tr>
<tr><th>Backend</th><td><code>{html.escape(str(provenance.get('backend', 'unknown')))} {html.escape(str(provenance.get('backend_version', 'unknown')))}</code></td></tr>
<tr><th>API dialect</th><td><code>{html.escape(str(provenance.get('api_dialect', 'unknown')))}</code></td></tr>
<tr><th>Hardware</th><td>{html.escape(str(provenance.get('hardware', 'unknown')))}</td></tr>
<tr><th>Chat-template digest</th><td><code>{html.escape(str(provenance.get('chat_template_digest', 'unknown')))}</code></td></tr>
</tbody></table>

<h2>Evidence integrity</h2>
<p>{evidence.get('attempt_count', 'unknown')} attempts retained status evidence; {evidence.get('valid_trace_count', 'unknown')} valid traces and {evidence.get('complete_dag_count', 'unknown')} complete explicit DAGs were recorded. Environment identity was complete and stable: {str(bool(evidence.get('environment_identity_complete') and evidence.get('environment_identity_stable'))).lower()}. Raw attempts and generated content are not included.</p>
<p>High-level failures: {failures.get('task_quality_failure', 'unknown')} task-quality failures; {failures.get('timeout', 'unknown')} timeouts; {failures.get('process_or_infrastructure_failure', 'unknown')} process/infrastructure failures; {failures.get('evaluator_failure', 'unknown')} evaluator failures; {failures.get('invalid_trace', 'unknown')} invalid traces.</p>

<h2>Frozen hashes</h2><table><tbody>{hash_rows}</tbody></table>

<h2>Claim boundary</h2>
<p><strong>Supported:</strong> {html.escape(str(claim.get('supported', 'unknown')))}</p>
<p><strong>Not supported:</strong></p><ul>{not_supported}</ul>
<h2>Experiment limitations</h2><ul>{limitation_items}</ul>
<p class="note">The favorable mechanism and latency measurements are conditional observations. This report is not a positive benchmark claim.</p>

<footer>Generated from a checked-in aggregate-only artifact. No prompts, task answers, model output, stderr, credentials, endpoints, local paths, raw traces, or private workload records are embedded. No remote scripts, fonts, or network resources are used.</footer>
</main></body></html>"""


def render_demo_html(result: Mapping[str, Any]) -> str:
    """Render either the flagship rejection or the historical static demo."""

    if result.get("artifact_kind") == "runeconomy.public_proof":
        return _render_public_proof_html(result)
    return _render_historical_demo_html(result)


def _render_historical_demo_html(result: Mapping[str, Any]) -> str:
    """Render the repository-agent demonstration from its bundled evidence."""

    if not isinstance(result.get("summary"), dict) or not isinstance(
        result.get("decision"), dict
    ):
        raise ValueError("result must contain summary and decision objects")
    raw_summary = result["summary"]
    summary = raw_summary.get("workflow", raw_summary)
    variants = summary["variants"]
    effects = summary["effects"]
    decision = result["decision"]
    provenance = result.get("provenance", {})
    sanitization = result.get("sanitization", {})
    context = result.get("measurement_context", {})

    def mean(variant: str, metric: str) -> Any:
        return variants[variant][metric]["mean"]

    baseline_input = mean("baseline", "input_tokens")
    candidate_input = mean("candidate", "input_tokens")
    baseline_latency = mean("baseline", "total_latency_ms")
    candidate_latency = mean("candidate", "total_latency_ms")
    baseline_success = variants["baseline"]["outcomes"]["task_success_rate"]
    candidate_success = variants["candidate"]["outcomes"]["task_success_rate"]
    input_effect = effects["input_tokens"]
    latency_effect = effects["total_latency_ms"]
    latency_ci = latency_effect.get("bootstrap_95_ci") or [None, None]
    status = html.escape(str(decision.get("status", "unknown")))
    experiment_id = html.escape(str(result.get("experiment_id", "unknown")))
    model = html.escape(str(provenance.get("model", "unknown")))
    hardware = html.escape(str(provenance.get("hardware", "unknown")))
    backend = html.escape(str(provenance.get("backend_version", "unknown")))
    measured_at = html.escape(str(result.get("created_at", "unknown")))
    raw_trials = html.escape(str(sanitization.get("raw_trial_count", "unknown")))
    manifest_sha = html.escape(
        str(provenance.get("reproducibility_manifest_sha256", "unknown"))
    )
    legacy_manifest_sha = html.escape(
        str(provenance.get("legacy_manifest_sha256", "unknown"))
    )
    baseline_tools = html.escape(str(context.get("baseline_tool_count", "unknown")))
    candidate_tools = html.escape(str(context.get("candidate_tool_count", "unknown")))
    measured_pairs = html.escape(
        str(
            context.get(
                "measured_workflow_pair_count",
                input_effect.get("paired_count", "unknown"),
            )
        )
    )
    warmup_pairs = html.escape(
        str(context.get("warmup_workflow_pair_count", "unknown"))
    )
    measured_calls = html.escape(
        str(context.get("measured_inference_call_count", "unknown"))
    )
    warmup_calls = html.escape(
        str(context.get("warmup_inference_call_count", "unknown"))
    )
    evaluator_kind = html.escape(str(context.get("evaluator_kind", "unknown")))
    candidate_selection = html.escape(
        str(context.get("candidate_selection", "unknown"))
    )
    workflow_steps = html.escape(str(context.get("workflow_step_count", "unknown")))
    candidate_tool_names = context.get("candidate_tool_names", [])
    candidate_tools_text = (
        ", ".join(
            f"<code>{html.escape(str(name))}</code>" for name in candidate_tool_names
        )
        or "not recorded"
    )
    changed_fields = context.get("changed_request_fields", [])
    changed_fields_text = (
        ", ".join(f"<code>{html.escape(str(name))}</code>" for name in changed_fields)
        or "not recorded"
    )
    raw_result_sha = html.escape(str(sanitization.get("raw_result_sha256", "unknown")))

    server_baseline = mean("baseline", "server_ttft_ms")
    server_candidate = mean("candidate", "server_ttft_ms")
    server_effect = effects["server_ttft_ms"]
    server_ci = server_effect.get("bootstrap_95_ci") or [None, None]
    client_baseline = mean("baseline", "client_ttft_ms")
    client_candidate = mean("candidate", "client_ttft_ms")
    client_effect = effects["client_ttft_ms"]
    client_ci = client_effect.get("bootstrap_95_ci") or [None, None]
    success_effect = effects["task_success"]
    success_ci = success_effect.get("confidence_interval") or [None, None]
    baseline_success_ci = variants["baseline"]["outcomes"].get(
        "task_success_95_ci"
    ) or [None, None]
    candidate_success_ci = variants["candidate"]["outcomes"].get(
        "task_success_95_ci"
    ) or [None, None]
    baseline_gap = mean("baseline", "client_server_ttft_gap_ms")
    candidate_gap = mean("candidate", "client_server_ttft_gap_ms")

    check_rows = []
    for check in decision.get("checks", []):
        required = check.get("required", True)
        passed = bool(check.get("passed"))
        check_rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(check.get('name', 'unknown')))}</code></td>"
            f"<td>{'required' if required else 'informational'}</td>"
            f'<td class="{"pass" if passed else ""}">'
            f"{'PASS' if passed else 'NOT MET'}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RunEconomy NegativeSplit report — {experiment_id}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; color: #111; font: 14px/1.45 Arial, Helvetica, sans-serif; }}
main {{ width: min(980px, calc(100% - 32px)); margin: 32px auto 64px; }}
header {{ border-bottom: 2px solid #111; padding-bottom: 14px; margin-bottom: 24px; }}
h1 {{ margin: 0 0 4px; font: 700 26px/1.2 Georgia, "Times New Roman", serif; }}
h2 {{ margin: 28px 0 8px; font: 700 18px/1.25 Georgia, "Times New Roman", serif; }}
p {{ max-width: 880px; margin: 8px 0; }}
.subtle {{ color: #555; }}
.note {{ color: #333; font-size: 13px; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0 18px; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid #aaa; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #eee; font-weight: 700; }}
td.number, th.number {{ text-align: right; white-space: nowrap; }}
code {{ font: 12px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }}
.pass {{ font-weight: 700; }}
ul {{ margin-top: 6px; padding-left: 22px; }}
footer {{ border-top: 1px solid #999; margin-top: 30px; padding-top: 10px; color: #555; font-size: 12px; }}
@media(max-width: 700px) {{ body {{ font-size: 13px; }} th, td {{ padding: 5px; }} main {{ width: calc(100% - 20px); }} }}
</style>
</head>
<body><main>
<header>
  <h1>RunEconomy NegativeSplit report</h1>
  <div>Experiment: <code>{experiment_id}</code></div>
  <div class="subtle">Recorded at {measured_at}</div>
</header>

<h2>Result</h2>
<p><strong>Runner decision:</strong> {status}.</p>
<p>The runner compared a baseline containing {baseline_tools} tool definitions per request with a candidate containing {candidate_tools}. In this pinned replay, the candidate used fewer input tokens and had lower summed inference latency. Both variants had a {_percent(candidate_success)} workflow evaluator pass rate.</p>
<p class="note"><strong>Why this is inconclusive:</strong> the measured latency effect is positive evidence, but the candidate allowlist was not selected on a separate calibration workload and fifteen agreeing outcome pairs do not establish a zero-regression task-success margin.</p>

<h2>Measured workflow results</h2>
<table>
<thead><tr><th>Metric</th><th class="number">Baseline mean</th><th class="number">Candidate mean</th><th class="number">Candidate − baseline</th><th class="number">Relative change</th><th class="number">95% interval for difference</th></tr></thead>
<tbody>
<tr><td>Input tokens, summed over two inference calls</td><td class="number">{_number(baseline_input, 0)}</td><td class="number">{_number(candidate_input, 0)}</td><td class="number">{_number(input_effect.get("candidate_minus_baseline_mean"), 0)}</td><td class="number">{_signed_percent(input_effect.get("relative_change"))}</td><td class="number">[{_number(input_effect.get("bootstrap_95_ci")[0], 0)}, {_number(input_effect.get("bootstrap_95_ci")[1], 0)}]</td></tr>
<tr><td>Server TTFT, summed (ms)</td><td class="number">{_number(server_baseline, 1)}</td><td class="number">{_number(server_candidate, 1)}</td><td class="number">{_number(server_effect.get("candidate_minus_baseline_mean"), 1)}</td><td class="number">{_signed_percent(server_effect.get("relative_change"))}</td><td class="number">[{_number(server_ci[0], 1)}, {_number(server_ci[1], 1)}]</td></tr>
<tr><td>Client TTFT, summed (ms)</td><td class="number">{_number(client_baseline, 1)}</td><td class="number">{_number(client_candidate, 1)}</td><td class="number">{_number(client_effect.get("candidate_minus_baseline_mean"), 1)}</td><td class="number">{_signed_percent(client_effect.get("relative_change"))}</td><td class="number">[{_number(client_ci[0], 1)}, {_number(client_ci[1], 1)}]</td></tr>
<tr><td>Client − server TTFT gap, summed (ms)</td><td class="number">{_number(baseline_gap, 1)}</td><td class="number">{_number(candidate_gap, 1)}</td><td class="number">{_number(effects["client_server_ttft_gap_ms"].get("candidate_minus_baseline_mean"), 1)}</td><td class="number">{_signed_percent(effects["client_server_ttft_gap_ms"].get("relative_change"))}</td><td class="number">{_interval(effects["client_server_ttft_gap_ms"], 1)}</td></tr>
<tr><td>Inference latency, summed (ms)</td><td class="number">{_number(baseline_latency, 1)}</td><td class="number">{_number(candidate_latency, 1)}</td><td class="number">{_number(latency_effect.get("candidate_minus_baseline_mean"), 1)}</td><td class="number">{_signed_percent(latency_effect.get("relative_change"))}</td><td class="number">[{_number(latency_ci[0], 1)}, {_number(latency_ci[1], 1)}]</td></tr>
<tr><td>Workflow evaluator pass rate</td><td class="number">{_percent(baseline_success)} (exact CI {_percent(baseline_success_ci[0])}–{_percent(baseline_success_ci[1])})</td><td class="number">{_percent(candidate_success)} (exact CI {_percent(candidate_success_ci[0])}–{_percent(candidate_success_ci[1])})</td><td class="number">{float(success_effect.get("candidate_minus_baseline_mean", 0)) * 100:.1f} percentage points</td><td class="number">{_signed_percent(success_effect.get("relative_change"))}</td><td class="number">[{float(success_ci[0]) * 100:.1f}, {float(success_ci[1]) * 100:.1f}] percentage points</td></tr>
</tbody></table>

<h2>Experiment design</h2>
<table><tbody>
<tr><th>Workflow unit</th><td>{workflow_steps} sequential inference calls treated as critical-path steps. Reported workflow latency is their sum and excludes fixture access and tool runtime.</td></tr>
<tr><th>Baseline</th><td>{baseline_tools} tool definitions on each request.</td></tr>
<tr><th>Candidate</th><td>{candidate_tools} tool definitions on each request: {candidate_tools_text}.</td></tr>
<tr><th>Changed request field</th><td>{changed_fields_text}. Within each case, the baseline and candidate request objects otherwise matched.</td></tr>
<tr><th>Evaluation</th><td>{evaluator_kind}.</td></tr>
<tr><th>Candidate selection</th><td><code>{candidate_selection}</code>; no held-out evaluation workload was recorded.</td></tr>
<tr><th>Measured sample</th><td>{measured_pairs} paired workflows = {measured_calls} measured inference calls.</td></tr>
<tr><th>Warmup sample</th><td>{warmup_pairs} paired workflows = {warmup_calls} warmup inference calls.</td></tr>
<tr><th>Total retained before sanitization</th><td>{raw_trials} inference trial records.</td></tr>
</tbody></table>

<h2>Acceptance checks</h2>
<table><thead><tr><th>Check</th><th>Role in decision</th><th>Recorded result</th></tr></thead>
<tbody>{"".join(check_rows)}</tbody></table>

<h2>Recorded environment</h2>
<table><tbody>
<tr><th>Model</th><td><code>{model}</code></td></tr>
<tr><th>Model revision</th><td><code>{html.escape(str(provenance.get("model_revision", "unknown")))}</code></td></tr>
<tr><th>Backend</th><td>vLLM <code>{backend}</code></td></tr>
<tr><th>Hardware</th><td>{hardware}</td></tr>
<tr><th>Cache treatment</th><td><code>{html.escape(str(provenance.get("cache_treatment", "unknown")))}</code></td></tr>
<tr><th>Trial order</th><td><code>{html.escape(str(provenance.get("order", "unknown")))}</code></td></tr>
<tr><th>Reproducibility manifest fingerprint</th><td><code>{manifest_sha}</code></td></tr>
<tr><th>Legacy private-manifest fingerprint</th><td><code>{legacy_manifest_sha}</code></td></tr>
</tbody></table>

<h2>Provenance note</h2>
<p>The checked-in recorded manifest preserves the measured prompt and fixture content while replacing the private endpoint and credential-variable name. The reproducibility fingerprint excludes only those transport locations. The legacy fingerprint remains recorded for the original private resolved manifest.</p>

<h2>Limitations and claim boundary</h2>
<ul>
  <li>The workload was a fixture-backed repository-analysis replay, not a live autonomous agent executing arbitrary tools.</li>
  <li>The workflow evaluator used two deterministic regex checks. “100% pass rate” refers to those checks, not an independent assessment of general answer quality.</li>
  <li>The exact 95% interval for each 15/15 pass rate extends below 100%, and the conservative paired interval includes a material regression; agreement is not proof of zero failure probability.</li>
  <li>The three-tool candidate allowlist was not shown to be predeclared or selected on a separate calibration workload.</li>
  <li>The measurement used one pinned small model, one A10 GPU, sequential traffic, and cold-cache treatment.</li>
  <li>Client TTFT exceeded server TTFT by {_number(baseline_gap, 1)} ms for the baseline and {_number(candidate_gap, 1)} ms for the candidate on average, so proxy and network overhead remain visible.</li>
  <li>The latency metric sums two inference calls and excludes fixture access and tool runtime.</li>
  <li>The experiment does not establish behavior under high concurrency, multiple replicas, larger models, SGLang, or other workloads.</li>
</ul>

<h2>Sanitization</h2>
<p>This HTML was generated from a checked-in aggregate derived from the local raw result. Performance measurements are recomputed from the raw trials; the task-success intervals, evidence review, and verdict use the repaired post-measurement methodology. Per-trial records, prompt token IDs, output digests, generated output, and request content are not embedded in this report.</p>
<p>Local raw-result SHA-256 recorded during sanitization: <code>{raw_result_sha}</code>.</p>

<footer>No remote scripts, fonts, or network resources are used.</footer>
</main></body></html>"""
