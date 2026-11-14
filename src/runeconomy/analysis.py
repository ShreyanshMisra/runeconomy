"""Deterministic prefix and cache-gap analysis."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .cache import eligible_prefix_tokens, longest_common_prefix
from .critical_path import calculate_critical_path
from .models import InferenceRequest, PromptSegment, Trace


@dataclass(frozen=True)
class RequestAnalysis:
    request_id: str
    token_evidence_quality: str
    attribution_qualities: tuple[str, ...]
    prefix_analysis_supported: bool
    baseline_request_id: str | None
    shared_prefix_tokens: int
    eligible_prefix_tokens: int
    actual_cached_tokens: int | None
    unrealized_eligible_tokens: int | None
    divergence_token: int | None
    divergence_segment_kind: str | None
    divergence_segment_label: str | None
    eligibility_model: str
    on_critical_path: bool | None


@dataclass(frozen=True)
class Finding:
    finding_id: str
    rule_id: str
    request_id: str
    affected_request_ids: tuple[str, ...]
    title: str
    evidence_level: str
    confidence: str
    summary: str
    suggested_experiment: str
    supporting_metrics: dict[str, Any]
    on_critical_path: bool | None
    estimated_reducible_tokens: int | None
    estimated_reducible_ms: float | None
    optimization_target: str
    experiment_template_id: str
    experiment_parameters: dict[str, Any]
    required_capabilities: tuple[str, ...]
    unsupported_reason: str | None
    causal_mechanism: str
    predicted_metric: str
    predicted_direction: str


def _finding_id(
    trace_id: str, rule_id: str, affected_request_ids: tuple[str, ...]
) -> str:
    identity = "\0".join((trace_id, rule_id, *affected_request_ids))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"finding-{digest}"


def _segment_at(request: InferenceRequest, token_index: int) -> PromptSegment | None:
    for segment in request.segments:
        if (
            segment.start_token is not None
            and segment.end_token is not None
            and segment.start_token <= token_index < segment.end_token
        ):
            return segment
    return None


def _best_baseline(
    request: InferenceRequest, prior: list[InferenceRequest]
) -> tuple[InferenceRequest | None, int]:
    best: InferenceRequest | None = None
    best_lcp = 0
    if request.prompt_token_ids is None:
        return None, 0
    for candidate in prior:
        if candidate.cache_identity != request.cache_identity:
            continue
        if candidate.prompt_token_ids is None:
            continue
        lcp = longest_common_prefix(candidate.prompt_token_ids, request.prompt_token_ids)
        if best is None or lcp > best_lcp:
            best = candidate
            best_lcp = lcp
    return best, best_lcp


def _finding_for_divergence(
    trace_id: str,
    analysis: RequestAnalysis,
    segment: PromptSegment,
    request: InferenceRequest,
) -> Finding | None:
    templates = {
        "dynamic_metadata": (
            "prompt.move_dynamic_metadata",
            "Move changing metadata after the stable shared prefix and replay the same "
            "workload with prompt semantics held constant.",
            "Changing metadata currently invalidates otherwise stable prefix blocks.",
            "cached_tokens",
            "increase",
        ),
        "tools": (
            "prompt.canonicalize_tools",
            "Canonicalize tool definitions and ordering, then compare cache reuse and "
            "task success on paired requests.",
            "Unstable tool serialization changes prompt tokens before reusable content.",
            "cached_tokens",
            "increase",
        ),
        "retrieved_context": (
            "retrieval.stabilize_order",
            "Stabilize document ordering or separate stable reference context from "
            "per-step retrieval, then run a paired replay.",
            "Unstable retrieval ordering changes otherwise reusable prompt blocks.",
            "cached_tokens",
            "increase",
        ),
        "history": (
            "history.preserve_prefix",
            "Preserve earlier turns byte-for-byte and append new turns; compare against "
            "the current history reconstruction.",
            "Reconstructed history changes tokens that should remain prefix-stable.",
            "cached_tokens",
            "increase",
        ),
    }
    template = templates.get(segment.kind)
    if template is None:
        return None
    template_id, experiment, mechanism, predicted_metric, direction = template
    eligible = analysis.eligible_prefix_tokens
    affected = tuple(
        request_id
        for request_id in (analysis.baseline_request_id, analysis.request_id)
        if request_id is not None
    )
    assert request.prompt_token_ids is not None
    maximum_eligible = eligible_prefix_tokens(request, len(request.prompt_token_ids))
    return Finding(
        finding_id=_finding_id(trace_id, f"prefix_divergence.{segment.kind}", affected),
        rule_id=f"prefix_divergence.{segment.kind}",
        request_id=analysis.request_id,
        affected_request_ids=affected,
        title=f"Reusable prefix diverges in {segment.label}",
        evidence_level="observed",
        confidence=(
            "high" if segment.attribution_quality == "exact_token_boundary" else "medium"
        ),
        summary=(
            f"The best comparable prior request diverges at token "
            f"{analysis.divergence_token} inside segment {segment.label!r}. "
            f"Only {eligible} leading tokens are statically eligible for reuse."
        ),
        suggested_experiment=experiment,
        supporting_metrics={
            "baseline_request_id": analysis.baseline_request_id,
            "shared_prefix_tokens": analysis.shared_prefix_tokens,
            "eligible_prefix_tokens": eligible,
            "divergence_token": analysis.divergence_token,
            "maximum_full_block_eligible_tokens": maximum_eligible,
            "attribution_quality": segment.attribution_quality,
            "attribution_source": segment.source,
        },
        on_critical_path=analysis.on_critical_path,
        estimated_reducible_tokens=max(maximum_eligible - eligible, 0),
        estimated_reducible_ms=None,
        optimization_target="cache_reuse",
        experiment_template_id=template_id,
        experiment_parameters={
            "segment_kind": segment.kind,
            "segment_label": segment.label,
            "divergence_token": analysis.divergence_token,
            "baseline_request_id": analysis.baseline_request_id,
        },
        required_capabilities=(
            "exact_prompt_tokens",
            "cache_telemetry",
            "task_evaluator",
        ),
        unsupported_reason=(
            "the normalized trace does not contain replayable request bodies or a "
            "safe application-level candidate transformation"
        ),
        causal_mechanism=mechanism,
        predicted_metric=predicted_metric,
        predicted_direction=direction,
    )


def _finding_for_runtime_gap(
    trace_id: str, analysis: RequestAnalysis
) -> Finding | None:
    gap = analysis.unrealized_eligible_tokens
    if gap is None or gap <= 0:
        return None
    affected = tuple(
        request_id
        for request_id in (analysis.baseline_request_id, analysis.request_id)
        if request_id is not None
    )
    rule_id = "cache.unrealized_eligibility"
    return Finding(
        finding_id=_finding_id(trace_id, rule_id, affected),
        rule_id=rule_id,
        request_id=analysis.request_id,
        affected_request_ids=affected,
        title="Eligible prefix reuse was not realized",
        evidence_level="inferred",
        confidence="medium",
        summary=(
            f"Static analysis found {analysis.eligible_prefix_tokens} eligible tokens, "
            f"but the runtime reported {analysis.actual_cached_tokens} cached tokens. "
            f"The {gap}-token gap can result from eviction, routing, cache state, or "
            "backend-specific eligibility not present in this trace."
        ),
        suggested_experiment=(
            "Replay on one isolated backend with a documented warm-cache sequence. If "
            "the gap disappears, test routing and eviction separately."
        ),
        supporting_metrics={
            "baseline_request_id": analysis.baseline_request_id,
            "eligible_prefix_tokens": analysis.eligible_prefix_tokens,
            "actual_cached_tokens": analysis.actual_cached_tokens,
            "unrealized_eligible_tokens": gap,
        },
        on_critical_path=analysis.on_critical_path,
        estimated_reducible_tokens=gap,
        estimated_reducible_ms=None,
        optimization_target="cache_reuse",
        experiment_template_id="cache.isolated_warm_replay",
        experiment_parameters={
            "baseline_request_id": analysis.baseline_request_id,
            "warmup_sequence_required": True,
        },
        required_capabilities=(
            "exact_prompt_tokens",
            "cache_telemetry",
            "warm_cache_replay",
        ),
        unsupported_reason=(
            "the trace cannot distinguish eviction, routing, cache state, and "
            "backend-specific eligibility without a replayable workload"
        ),
        causal_mechanism=(
            "Eligible prefix blocks were not available on the serving instance that "
            "processed this request."
        ),
        predicted_metric="cached_tokens",
        predicted_direction="increase",
    )


def _finding_for_unused_tools(
    trace_id: str, request: InferenceRequest, on_critical_path: bool | None
) -> Finding | None:
    available = request.available_tools
    if len(available) < 4:
        return None
    invoked = set(request.invoked_tools)
    unused = tuple(tool for tool in available if tool not in invoked)
    if len(unused) < 3 or len(unused) / len(available) < 0.5:
        return None
    tool_tokens = sum(
        segment.end_token - segment.start_token
        for segment in request.segments
        if segment.kind == "tools"
        and segment.start_token is not None
        and segment.end_token is not None
    )
    rule_id = "tools.unused_exposure"
    affected = (request.request_id,)
    return Finding(
        finding_id=_finding_id(trace_id, rule_id, affected),
        rule_id=rule_id,
        request_id=request.request_id,
        affected_request_ids=affected,
        title="Most exposed tools were unused on this step",
        evidence_level="observed",
        confidence="medium",
        summary=(
            f"The request exposed {len(available)} tools but invoked "
            f"{len(invoked)}; {len(unused)} were unused in this trace. "
            f"The tool segment occupied {tool_tokens} prompt tokens. Observed "
            "non-use does not prove a tool is unnecessary on every task."
        ),
        suggested_experiment=(
            "Replay representative tasks with the full catalog versus a "
            "task-scoped allowlist. Require non-inferior workflow success and "
            "measure input tokens plus total critical-path latency."
        ),
        supporting_metrics={
            "available_tool_count": len(available),
            "invoked_tool_count": len(invoked),
            "unused_tool_count": len(unused),
            "unused_tool_names": list(unused),
            "tool_segment_tokens": tool_tokens,
            "candidate_tool_names": list(request.invoked_tools),
            "optimization_target": "input_tokens",
            "primary_metric": "total_latency_ms",
            "reducible_tokens_upper_bound": tool_tokens,
        },
        on_critical_path=on_critical_path,
        estimated_reducible_tokens=None,
        estimated_reducible_ms=None,
        optimization_target="input_tokens",
        experiment_template_id="tools.scope_allowlist",
        experiment_parameters={
            "candidate_tool_names": list(request.invoked_tools),
            "observed_unused_tool_names": list(unused),
        },
        required_capabilities=(
            "exact_prompt_tokens",
            "task_evaluator",
        ),
        unsupported_reason=(
            "the normalized trace records tool names but not complete tool definitions, "
            "a replayable request, representative cases, or an evaluator"
        ),
        causal_mechanism=(
            "Removing irrelevant tool definitions reduces prompt prefill work on this "
            "inference step."
        ),
        predicted_metric="input_tokens",
        predicted_direction="decrease",
    )


def analyze_trace(trace: Trace) -> dict[str, Any]:
    """Analyze a normalized trace and return a JSON-serializable report."""
    analyses: list[RequestAnalysis] = []
    findings: list[Finding] = []
    prior: list[InferenceRequest] = []
    critical_path = calculate_critical_path(trace.spans)
    critical_request_ids = (
        set(critical_path.inference_request_ids) if critical_path is not None else None
    )

    for request in trace.requests:
        baseline, shared = _best_baseline(request, prior)
        eligible = eligible_prefix_tokens(request, shared) if baseline else 0
        actual = request.metrics.cached_input_tokens
        gap = None if actual is None else max(eligible - actual, 0)
        divergence = None
        segment = None
        if (
            baseline is not None
            and baseline.prompt_token_ids is not None
            and request.prompt_token_ids is not None
            and shared
            < min(len(baseline.prompt_token_ids), len(request.prompt_token_ids))
        ):
            divergence = shared
            segment = _segment_at(request, divergence)

        result = RequestAnalysis(
            request_id=request.request_id,
            token_evidence_quality=request.token_evidence_quality,
            attribution_qualities=tuple(
                dict.fromkeys(
                    segment.attribution_quality for segment in request.segments
                )
            ),
            prefix_analysis_supported=request.prompt_token_ids is not None,
            baseline_request_id=baseline.request_id if baseline else None,
            shared_prefix_tokens=shared,
            eligible_prefix_tokens=eligible,
            actual_cached_tokens=actual,
            unrealized_eligible_tokens=gap,
            divergence_token=divergence,
            divergence_segment_kind=segment.kind if segment else None,
            divergence_segment_label=segment.label if segment else None,
            eligibility_model="vllm_full_blocks" if request.backend == "vllm" else "generic_tokens",
            on_critical_path=(
                request.request_id in critical_request_ids
                if critical_request_ids is not None
                else None
            ),
        )
        analyses.append(result)
        if segment is not None:
            finding = _finding_for_divergence(
                trace.trace_id, result, segment, request
            )
            if finding is not None:
                findings.append(finding)
        runtime_finding = _finding_for_runtime_gap(trace.trace_id, result)
        if runtime_finding is not None:
            findings.append(runtime_finding)
        unused_tools_finding = _finding_for_unused_tools(
            trace.trace_id, request, result.on_critical_path
        )
        if unused_tools_finding is not None:
            findings.append(unused_tools_finding)
        prior.append(request)

    total_input = sum(
        request.metrics.input_tokens
        if request.metrics.input_tokens is not None
        else request.prompt_token_count or 0
        for request in trace.requests
    )
    cache_telemetry_complete = all(
        request.metrics.cached_input_tokens is not None for request in trace.requests
    )
    total_actual_cached = (
        sum(request.metrics.cached_input_tokens or 0 for request in trace.requests)
        if cache_telemetry_complete
        else None
    )
    total_eligible = sum(item.eligible_prefix_tokens for item in analyses)
    observed_cache_coverage = (
        total_actual_cached / total_input
        if total_input and total_actual_cached is not None
        else None
    )

    return {
        "report_version": "0.2",
        "trace_id": trace.trace_id,
        "workload": trace.workload,
        "success": trace.success,
        "summary": {
            "request_count": len(trace.requests),
            "exact_token_request_count": sum(
                request.token_evidence_quality == "exact"
                for request in trace.requests
            ),
            "count_only_request_count": sum(
                request.token_evidence_quality == "count_only"
                for request in trace.requests
            ),
            "unknown_token_request_count": sum(
                request.token_evidence_quality == "unknown"
                for request in trace.requests
            ),
            "total_input_tokens": total_input,
            "total_eligible_prefix_tokens": total_eligible,
            "total_actual_cached_tokens": total_actual_cached,
            "cache_telemetry_complete": cache_telemetry_complete,
            "observed_cache_coverage": observed_cache_coverage,
            "finding_count": len(findings),
            "critical_path_duration_ms": (
                critical_path.duration_ms if critical_path is not None else None
            ),
        },
        "critical_path": asdict(critical_path) if critical_path is not None else None,
        "requests": [asdict(item) for item in analyses],
        "findings": [asdict(item) for item in findings],
    }
