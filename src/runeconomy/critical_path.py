"""Critical-path calculation for explicit agent workflow dependency DAGs."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ExecutionSpan, TraceValidationError


@dataclass(frozen=True)
class CriticalPath:
    duration_ms: float
    span_ids: tuple[str, ...]
    inference_request_ids: tuple[str, ...]
    inference_ms: float
    tool_ms: float


def calculate_critical_path(spans: tuple[ExecutionSpan, ...]) -> CriticalPath | None:
    """Return the longest duration-weighted path through an execution DAG."""
    if not spans:
        return None

    by_id = {span.span_id: span for span in spans}
    indegree = {span.span_id: len(span.depends_on) for span in spans}
    successors: dict[str, list[str]] = {span.span_id: [] for span in spans}
    for span in spans:
        for dependency in span.depends_on:
            successors[dependency].append(span.span_id)

    ready = sorted(span_id for span_id, degree in indegree.items() if degree == 0)
    topological: list[str] = []
    while ready:
        span_id = ready.pop(0)
        topological.append(span_id)
        for successor in sorted(successors[span_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(topological) != len(spans):
        raise TraceValidationError("trace.spans: dependency graph contains a cycle")

    longest: dict[str, float] = {}
    predecessor: dict[str, str | None] = {}
    for span_id in topological:
        span = by_id[span_id]
        if not span.depends_on:
            longest[span_id] = span.duration_ms
            predecessor[span_id] = None
            continue
        best_dependency = max(
            span.depends_on, key=lambda item: (longest[item], item)
        )
        longest[span_id] = longest[best_dependency] + span.duration_ms
        predecessor[span_id] = best_dependency

    final_id = max(topological, key=lambda item: (longest[item], item))
    reversed_path: list[str] = []
    cursor: str | None = final_id
    while cursor is not None:
        reversed_path.append(cursor)
        cursor = predecessor[cursor]
    path_ids = tuple(reversed(reversed_path))
    path_spans = [by_id[span_id] for span_id in path_ids]
    return CriticalPath(
        duration_ms=longest[final_id],
        span_ids=path_ids,
        inference_request_ids=tuple(
            span.request_id
            for span in path_spans
            if span.kind == "inference" and span.request_id is not None
        ),
        inference_ms=sum(
            span.duration_ms for span in path_spans if span.kind == "inference"
        ),
        tool_ms=sum(span.duration_ms for span in path_spans if span.kind == "tool"),
    )
