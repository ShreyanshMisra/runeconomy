"""Deterministic prefix and cache-gap analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .cache import eligible_prefix_tokens, longest_common_prefix
from .critical_path import calculate_critical_path
from .models import InferenceRequest, PromptSegment, Trace


@dataclass(frozen=True)
class RequestAnalysis:
    request_id: str
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
    rule_id: str
    request_id: str
    title: str
    evidence_level: str
    confidence: str
    summary: str
    suggested_experiment: str
    supporting_metrics: dict[str, Any]
    on_critical_path: bool | None


def _segment_at(request: InferenceRequest, token_index: int) -> PromptSegment | None:
    for segment in request.segments:
        if segment.start_token <= token_index < segment.end_token:
            return segment
    return None


def _best_baseline(
    request: InferenceRequest, prior: list[InferenceRequest]
) -> tuple[InferenceRequest | None, int]:
    best: InferenceRequest | None = None
    best_lcp = 0
    for candidate in prior:
        if candidate.cache_identity != request.cache_identity:
            continue
        lcp = longest_common_prefix(candidate.prompt_token_ids, request.prompt_token_ids)
        if best is None or lcp > best_lcp:
            best = candidate
            best_lcp = lcp
    return best, best_lcp


def _finding_for_divergence(
    analysis: RequestAnalysis, segment: PromptSegment
) -> Finding | None:
    experiments = {
        "dynamic_metadata": (
            "Move changing metadata after the stable shared prefix and replay the same "
            "workload with prompt semantics held constant."
        ),
        "tools": (
            "Canonicalize tool definitions and ordering, then compare cache reuse and "
            "task success on paired requests."
        ),
        "retrieved_context": (
            "Stabilize document ordering or separate stable reference context from "
            "per-step retrieval, then run a paired replay."
        ),
        "history": (
            "Preserve earlier turns byte-for-byte and append new turns; compare against "
            "the current history reconstruction."
        ),
    }
    experiment = experiments.get(segment.kind)
    if experiment is None:
        return None
    eligible = analysis.eligible_prefix_tokens
    return Finding(
        rule_id=f"prefix_divergence.{segment.kind}",
        request_id=analysis.request_id,
        title=f"Reusable prefix diverges in {segment.label}",
        evidence_level="observed",
        confidence="high",
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
        },
        on_critical_path=analysis.on_critical_path,
    )


def _finding_for_runtime_gap(analysis: RequestAnalysis) -> Finding | None:
    gap = analysis.unrealized_eligible_tokens
    if gap is None or gap <= 0:
        return None
    return Finding(
        rule_id="cache.unrealized_eligibility",
        request_id=analysis.request_id,
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
    )


def _finding_for_unused_tools(
    request: InferenceRequest, on_critical_path: bool | None
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
    )
    return Finding(
        rule_id="tools.unused_exposure",
        request_id=request.request_id,
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
        },
        on_critical_path=on_critical_path,
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
        if baseline is not None and shared < min(
            len(baseline.prompt_token_ids), len(request.prompt_token_ids)
        ):
            divergence = shared
            segment = _segment_at(request, divergence)

        result = RequestAnalysis(
            request_id=request.request_id,
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
            finding = _finding_for_divergence(result, segment)
            if finding is not None:
                findings.append(finding)
        runtime_finding = _finding_for_runtime_gap(result)
        if runtime_finding is not None:
            findings.append(runtime_finding)
        unused_tools_finding = _finding_for_unused_tools(
            request, result.on_critical_path
        )
        if unused_tools_finding is not None:
            findings.append(unused_tools_finding)
        prior.append(request)

    total_input = sum(
        request.metrics.input_tokens
        if request.metrics.input_tokens is not None
        else len(request.prompt_token_ids)
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
        "report_version": "0.1",
        "trace_id": trace.trace_id,
        "workload": trace.workload,
        "success": trace.success,
        "summary": {
            "request_count": len(trace.requests),
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
