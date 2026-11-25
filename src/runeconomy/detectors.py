"""Workflow waste detectors.

Each detector is deterministic, states its evidence requirements, and stays
silent when the evidence it needs is unknown. A finding is a hypothesis with a
named experiment template, never a verdict: the template's paired experiment
confirms or refutes the predicted effect.

Trace detectors (called from :func:`runeconomy.analysis.analyze_trace`):

``context.tool_output_carryover``
    A large tool result is re-sent, token for token, in two or more later
    requests of the same workflow. Requires exact prompt tokens and token
    boundaries for ``tool_results`` segments.
``workflow.serialized_independent_steps``
    Steps with no declared dependency between them ran one after another, so
    the observed workflow span exceeds the dependency-DAG critical path.
    Requires ``complete_explicit`` dependency evidence.
``models.step_overprovisioning``
    Short-output steps (routing, extraction, formatting) run on a large model.
    Requires the model's parameter count in its name and per-step token counts.
``output.structured_step_inflation``
    Tool-call steps generate far more output tokens than a tool call needs,
    which indicates verbose or reasoning-token inflation on structured steps.
    Requires output token counts and invoked-tool evidence.

Result detector (called from :func:`diagnose_result` on a load-run result):

``serving.kv_pressure``
    Server telemetry shows preemptions or a saturated KV cache with a waiting
    queue during measured load blocks. Requires sampled server metrics.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping

from .critical_path import calculate_critical_path
from .models import ExecutionSpan, InferenceRequest, Trace


CARRYOVER_MIN_SEGMENT_TOKENS = 256
CARRYOVER_MIN_CARRIES = 2
CARRYOVER_MIN_TOTAL_TOKENS = 1024
SERIALIZATION_MIN_GAP_MS = 50.0
SERIALIZATION_MIN_GAP_SHARE = 0.10
OVERPROVISION_MAX_OUTPUT_TOKENS = 32
OVERPROVISION_MIN_PARAMETERS_B = 7.0
INFLATION_MIN_OUTPUT_TOKENS = 256
KV_SATURATION_RATIO = 0.95

_PARAMETERS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")

# template id -> (mode, intervention, predicted metric, direction, quality risk)
DETECTOR_TEMPLATES: dict[str, dict[str, str]] = {
    "context.compact_tool_output": {
        "mode": "agent-process",
        "intervention": "Carry earlier tool results forward as a compact summary or reference instead of verbatim.",
        "predicted_metric": "input_tokens",
        "predicted_direction": "decrease",
        "quality_risk": "Later steps may need detail that the compaction removed.",
    },
    "workflow.parallelize_independent_steps": {
        "mode": "agent-process",
        "intervention": "Run steps that have no declared dependency between them concurrently.",
        "predicted_metric": "workflow_latency_ms",
        "predicted_direction": "decrease",
        "quality_risk": "An undeclared dependency would surface as task failure or nondeterminism.",
    },
    "models.route_step_to_smaller_model": {
        "mode": "agent-process",
        "intervention": "Route the identified short-output steps to a smaller model; keep other steps unchanged.",
        "predicted_metric": "workflow_latency_ms",
        "predicted_direction": "decrease",
        "quality_risk": "The smaller model may misroute or misformat, lowering task success.",
    },
    "output.constrain_structured_step": {
        "mode": "agent-process",
        "intervention": "Constrain tool-call steps (structured output, a tight max_tokens, or reasoning disabled).",
        "predicted_metric": "output_tokens",
        "predicted_direction": "decrease",
        "quality_risk": "Constraining generation can truncate or degrade the tool call.",
    },
    "serving.reduce_kv_pressure": {
        "mode": "agent-process",
        "intervention": (
            "Reduce KV pressure on the candidate server (lower max_num_seqs, a larger KV budget, "
            "an FP8 KV cache, or a shorter max_model_len) under the same load."
        ),
        "predicted_metric": "workflow_latency_ms",
        "predicted_direction": "decrease",
        "quality_risk": "Engine changes such as an FP8 KV cache can change outputs.",
    },
}


def _finding(
    *,
    trace_id: str,
    rule_id: str,
    affected: tuple[str, ...],
    title: str,
    confidence: str,
    evidence_level: str,
    summary: str,
    template_id: str,
    supporting_metrics: dict[str, Any],
    parameters: dict[str, Any],
    required_capabilities: tuple[str, ...],
    on_critical_path: bool | None = None,
    reducible_tokens: int | None = None,
    reducible_ms: float | None = None,
    optimization_target: str,
    mechanism: str,
) -> dict[str, Any]:
    template = DETECTOR_TEMPLATES[template_id]
    identity = "\0".join((trace_id, rule_id, *affected))
    return {
        "finding_id": "finding_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "rule_id": rule_id,
        "request_id": affected[0] if affected else "",
        "affected_request_ids": list(affected),
        "title": title,
        "evidence_level": evidence_level,
        "confidence": confidence,
        "summary": summary,
        "suggested_experiment": (
            f"{template['intervention']} Run a paired {template['mode']} experiment; require non-inferior "
            f"task success and measure {template['predicted_metric']}. Risk: {template['quality_risk']}"
        ),
        "supporting_metrics": supporting_metrics,
        "on_critical_path": on_critical_path,
        "estimated_reducible_tokens": reducible_tokens,
        "estimated_reducible_ms": reducible_ms,
        "optimization_target": optimization_target,
        "experiment_template_id": template_id,
        "experiment_parameters": parameters,
        "required_capabilities": list(required_capabilities),
        "unsupported_reason": (
            "this template changes application or serving behavior that RunEconomy cannot generate; "
            "implement the candidate and run it as an agent-process experiment"
        ),
        "causal_mechanism": mechanism,
        "predicted_metric": template["predicted_metric"],
        "predicted_direction": template["predicted_direction"],
    }


def _bounded(request: InferenceRequest, kind: str) -> list[tuple[int, int, str]]:
    return [
        (segment.start_token, segment.end_token, segment.label)
        for segment in request.segments
        if segment.kind == kind
        and segment.start_token is not None
        and segment.end_token is not None
        and segment.attribution_quality in {"exact_token_boundary", "block_level"}
    ]


def detect_tool_output_carryover(trace: Trace) -> list[dict[str, Any]]:
    first_seen: dict[str, dict[str, Any]] = {}
    for request in trace.requests:
        tokens = request.prompt_token_ids
        if tokens is None:
            continue
        cached = request.metrics.cached_input_tokens
        for start, end, label in _bounded(request, "tool_results"):
            length = end - start
            if length < CARRYOVER_MIN_SEGMENT_TOKENS or end > len(tokens):
                continue
            digest = hashlib.sha256(repr(tokens[start:end]).encode()).hexdigest()
            entry = first_seen.get(digest)
            if entry is None:
                first_seen[digest] = {
                    "origin": request.request_id,
                    "label": label,
                    "tokens": length,
                    "carried_into": [],
                    "uncached_carried_tokens": 0,
                    "cache_known": True,
                }
                continue
            entry["carried_into"].append(request.request_id)
            if cached is None:
                entry["cache_known"] = False
            else:
                entry["uncached_carried_tokens"] += max(0, end - max(start, cached))
    carried = [
        entry
        for entry in first_seen.values()
        if len(entry["carried_into"]) >= CARRYOVER_MIN_CARRIES
    ]
    total = sum(entry["tokens"] * len(entry["carried_into"]) for entry in carried)
    if not carried or total < CARRYOVER_MIN_TOTAL_TOKENS:
        return []
    affected = tuple(dict.fromkeys(rid for entry in carried for rid in (entry["origin"], *entry["carried_into"])))
    uncached = (
        sum(entry["uncached_carried_tokens"] for entry in carried)
        if all(entry["cache_known"] for entry in carried)
        else None
    )
    return [
        _finding(
            trace_id=trace.trace_id,
            rule_id="context.tool_output_carryover",
            affected=affected,
            title="Large tool results are re-sent verbatim in later steps",
            confidence="high",
            evidence_level="observed",
            summary=(
                f"{len(carried)} tool result(s) of at least {CARRYOVER_MIN_SEGMENT_TOKENS} tokens were re-sent "
                f"token-for-token in later requests, {total} carried prompt tokens in total"
                + (
                    f", of which {uncached} were not covered by the reported prefix cache."
                    if uncached is not None
                    else "; cache coverage of the carried tokens is unknown."
                )
            ),
            template_id="context.compact_tool_output",
            supporting_metrics={
                "carried_tool_results": [
                    {
                        "origin_request_id": entry["origin"],
                        "segment_label": entry["label"],
                        "tokens": entry["tokens"],
                        "carried_into_request_ids": entry["carried_into"],
                    }
                    for entry in carried
                ],
                "total_carried_tokens": total,
                "uncached_carried_tokens": uncached,
            },
            parameters={"carried_segment_kind": "tool_results", "minimum_segment_tokens": CARRYOVER_MIN_SEGMENT_TOKENS},
            required_capabilities=("exact_prompt_tokens", "task_evaluator", "agent_process"),
            reducible_tokens=total,
            optimization_target="input_tokens",
            mechanism=(
                "Every later request re-prefills the carried tool output. Prefix caching only helps while "
                "the carried text stays inside an unchanged prefix."
            ),
        )
    ]


def _ancestors(spans: tuple[ExecutionSpan, ...]) -> dict[str, set[str]]:
    by_id = {span.span_id: span for span in spans}
    memo: dict[str, set[str]] = {}

    def visit(span_id: str) -> set[str]:
        if span_id not in memo:
            found: set[str] = set()
            for parent in by_id[span_id].depends_on:
                found.add(parent)
                found |= visit(parent)
            memo[span_id] = found
        return memo[span_id]

    for span in spans:
        visit(span.span_id)
    return memo


def detect_serialized_independent_steps(
    trace: Trace, dependency_evidence: str | None
) -> list[dict[str, Any]]:
    spans = trace.spans
    if dependency_evidence != "complete_explicit" or len(spans) < 2:
        return []
    path = calculate_critical_path(spans)
    if path is None:
        return []
    wall = max(span.end_ms for span in spans) - min(span.start_ms for span in spans)
    gap = wall - path.duration_ms
    if gap < SERIALIZATION_MIN_GAP_MS or wall <= 0 or gap / wall < SERIALIZATION_MIN_GAP_SHARE:
        return []
    ancestors = _ancestors(spans)
    serialized: list[tuple[str, str]] = []
    ordered = sorted(spans, key=lambda span: span.start_ms)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            independent = first.span_id not in ancestors[second.span_id] and second.span_id not in ancestors[first.span_id]
            if independent and first.end_ms <= second.start_ms:
                serialized.append((first.span_id, second.span_id))
    if not serialized:
        return []
    involved = tuple(dict.fromkeys(span_id for pair in serialized for span_id in pair))
    requests = tuple(
        span.request_id for span in spans if span.span_id in involved and span.request_id is not None
    )
    return [
        _finding(
            trace_id=trace.trace_id,
            rule_id="workflow.serialized_independent_steps",
            affected=requests or involved,
            title="Independent steps ran one after another",
            confidence="high",
            evidence_level="observed",
            summary=(
                f"The workflow took {wall:.1f} ms, but its declared dependency DAG has a {path.duration_ms:.1f} ms "
                f"critical path. {len(serialized)} pair(s) of steps without a declared dependency ran sequentially. "
                f"Up to {gap:.1f} ms is attributable to serialization if the declared edges are complete."
            ),
            template_id="workflow.parallelize_independent_steps",
            supporting_metrics={
                "observed_workflow_span_ms": wall,
                "critical_path_ms": path.duration_ms,
                "serialization_gap_ms": gap,
                "serialized_independent_pairs": [list(pair) for pair in serialized[:20]],
                "serialized_pair_count": len(serialized),
            },
            parameters={"independent_span_ids": list(involved)},
            required_capabilities=("explicit_dependencies", "task_evaluator", "agent_process"),
            reducible_ms=gap,
            on_critical_path=None,
            optimization_target="workflow_latency_ms",
            mechanism=(
                "Steps with no data dependency wait for each other; running them concurrently shortens the "
                "workflow toward its critical path."
            ),
        )
    ]


def model_parameters_billions(model: str) -> float | None:
    matches = [float(value) for value in _PARAMETERS.findall(model or "")]
    return max(matches) if matches else None


def detect_step_overprovisioning(trace: Trace) -> list[dict[str, Any]]:
    candidates: list[InferenceRequest] = []
    for request in trace.requests:
        parameters = model_parameters_billions(request.model)
        output = request.metrics.output_tokens
        if (
            parameters is None
            or parameters < OVERPROVISION_MIN_PARAMETERS_B
            or output is None
            or output > OVERPROVISION_MAX_OUTPUT_TOKENS
        ):
            continue
        candidates.append(request)
    others = [
        request
        for request in trace.requests
        if request not in candidates and request.metrics.output_tokens is not None
    ]
    if not candidates or not others:
        # A workflow made only of short steps gives no contrast; stay silent.
        return []
    latency = [request.metrics.total_latency_ms for request in candidates]
    step_ms = sum(latency) if all(value is not None for value in latency) else None
    return [
        _finding(
            trace_id=trace.trace_id,
            rule_id="models.step_overprovisioning",
            affected=tuple(request.request_id for request in candidates),
            title="Short-output steps run on a large model",
            confidence="low",
            evidence_level="heuristic",
            summary=(
                f"{len(candidates)} step(s) produced at most {OVERPROVISION_MAX_OUTPUT_TOKENS} output tokens on a "
                f"model of at least {OVERPROVISION_MIN_PARAMETERS_B:g}B parameters while other steps produced longer "
                "output. Short output suggests routing, extraction or formatting, which a smaller model may handle. "
                "Output length alone does not establish that the step is simple."
            ),
            template_id="models.route_step_to_smaller_model",
            supporting_metrics={
                "steps": [
                    {
                        "request_id": request.request_id,
                        "model": request.model,
                        "parameters_billions": model_parameters_billions(request.model),
                        "input_tokens": request.metrics.input_tokens,
                        "output_tokens": request.metrics.output_tokens,
                        "total_latency_ms": request.metrics.total_latency_ms,
                    }
                    for request in candidates
                ],
                "candidate_step_latency_ms": step_ms,
            },
            parameters={"step_request_ids": [request.request_id for request in candidates]},
            required_capabilities=("task_evaluator", "agent_process"),
            reducible_ms=None,
            optimization_target="workflow_latency_ms",
            mechanism="Prefill and decode cost scale with model size; a short structured step may not need it.",
        )
    ]


def detect_structured_step_inflation(trace: Trace) -> list[dict[str, Any]]:
    inflated = [
        request
        for request in trace.requests
        if request.invoked_tools
        and request.metrics.output_tokens is not None
        and request.metrics.output_tokens >= INFLATION_MIN_OUTPUT_TOKENS
    ]
    if not inflated:
        return []
    output = sum(request.metrics.output_tokens or 0 for request in inflated)
    decode = [request.metrics.decode_ms for request in inflated]
    return [
        _finding(
            trace_id=trace.trace_id,
            rule_id="output.structured_step_inflation",
            affected=tuple(request.request_id for request in inflated),
            title="Tool-call steps generate far more output than a tool call needs",
            confidence="medium",
            evidence_level="observed",
            summary=(
                f"{len(inflated)} step(s) that ended in a tool call generated at least {INFLATION_MIN_OUTPUT_TOKENS} "
                f"output tokens ({output} in total). Verbose preambles or reasoning tokens before a structured call "
                "add decode time on every such step."
            ),
            template_id="output.constrain_structured_step",
            supporting_metrics={
                "steps": [
                    {
                        "request_id": request.request_id,
                        "invoked_tools": list(request.invoked_tools),
                        "output_tokens": request.metrics.output_tokens,
                        "decode_ms": request.metrics.decode_ms,
                    }
                    for request in inflated
                ],
                "total_output_tokens": output,
                "total_decode_ms": sum(decode) if all(value is not None for value in decode) else None,
            },
            parameters={"step_request_ids": [request.request_id for request in inflated]},
            required_capabilities=("task_evaluator", "agent_process"),
            reducible_tokens=None,
            optimization_target="output_tokens",
            mechanism="Decode time grows with every generated token; a constrained tool call needs few.",
        )
    ]


def run_trace_detectors(trace: Trace, *, dependency_evidence: str | None = None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    findings += detect_tool_output_carryover(trace)
    findings += detect_serialized_independent_steps(trace, dependency_evidence)
    findings += detect_step_overprovisioning(trace)
    findings += detect_structured_step_inflation(trace)
    return findings


def _gauge(target: Mapping[str, Any], name: str, statistic: str) -> float | None:
    gauges = target.get("gauges")
    value = gauges.get(name) if isinstance(gauges, Mapping) else None
    result = value.get(statistic) if isinstance(value, Mapping) else None
    return float(result) if isinstance(result, (int, float)) and math.isfinite(result) else None


def detect_kv_pressure(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocks = result.get("load_blocks")
    if not isinstance(blocks, list):
        return []
    observations: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        telemetry = block.get("telemetry")
        metrics = telemetry.get("server_metrics") if isinstance(telemetry, Mapping) else None
        targets = metrics.get("targets") if isinstance(metrics, Mapping) else None
        if not isinstance(targets, Mapping):
            continue
        for name, target in targets.items():
            if not isinstance(target, Mapping) or target.get("status") != "measured":
                continue
            counters = target.get("counters") if isinstance(target.get("counters"), Mapping) else {}
            preemptions = counters.get("preemptions_total_delta")
            usage = _gauge(target, "kv_cache_usage_ratio", "max")
            waiting = _gauge(target, "waiting_requests", "p95")
            saturated = usage is not None and usage >= KV_SATURATION_RATIO and (waiting or 0) > 0
            preempted = isinstance(preemptions, (int, float)) and preemptions > 0
            if saturated or preempted:
                observations.append(
                    {
                        "block": block.get("block", block.get("index")),
                        "stage": block.get("stage"),
                        "target": name,
                        "preemptions": preemptions,
                        "kv_cache_usage_max": usage,
                        "waiting_requests_p95": waiting,
                    }
                )
    if not observations:
        return []
    experiment = str(result.get("experiment_id", "result"))
    preempting = sorted({str(item["target"]) for item in observations if (item["preemptions"] or 0) > 0})
    return [
        _finding(
            trace_id=experiment,
            rule_id="serving.kv_pressure",
            affected=tuple(f"{item['target']}:{item['block']}" for item in observations),
            title="KV-cache pressure or preemption under load",
            confidence="high" if preempting else "medium",
            evidence_level="observed",
            summary=(
                f"{len(observations)} measured block/server observation(s) showed "
                + ("preemptions" if preempting else "a saturated KV cache with a waiting queue")
                + ". Preempted requests are recomputed, which inflates tail latency under load."
            ),
            template_id="serving.reduce_kv_pressure",
            supporting_metrics={"observations": observations, "preempting_targets": preempting},
            parameters={"targets": sorted({str(item["target"]) for item in observations})},
            required_capabilities=("server_metrics", "task_evaluator", "agent_process"),
            reducible_ms=None,
            optimization_target="workflow_latency_ms",
            mechanism=(
                "When the KV cache is full, the scheduler preempts and later recomputes running requests, "
                "and new requests queue."
            ),
        )
    ]


def diagnose_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Run result-level detectors and return an analyze-shaped report."""

    findings = detect_kv_pressure(result)
    return {
        "report_version": "0.2",
        "trace_id": str(result.get("experiment_id", "result")),
        "workload": "experiment_result",
        "success": None,
        "summary": {"request_count": 0, "finding_count": len(findings), "source": "experiment_result"},
        "critical_path": None,
        "requests": [],
        "findings": findings,
    }
