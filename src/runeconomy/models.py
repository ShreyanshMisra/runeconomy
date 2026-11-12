"""Versioned, dependency-free models for normalized agent traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TraceValidationError(ValueError):
    """Raised when a normalized trace is invalid or internally inconsistent."""


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise TraceValidationError(f"{path}.{key}: required field is missing")
    return mapping[key]


def _optional_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceValidationError(f"{path}: expected a number or null")
    if value < 0:
        raise TraceValidationError(f"{path}: must be non-negative")
    return float(value)


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TraceValidationError(f"{path}: expected a list of non-empty strings")
    if len(value) != len(set(value)):
        raise TraceValidationError(f"{path}: values must be unique")
    return tuple(value)


@dataclass(frozen=True)
class PromptSegment:
    kind: str
    label: str
    start_token: int
    end_token: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> "PromptSegment":
        try:
            kind = str(_required(data, "kind", path))
            label = str(data.get("label", kind))
            start = int(_required(data, "start_token", path))
            end = int(_required(data, "end_token", path))
        except (TypeError, ValueError) as exc:
            raise TraceValidationError(f"{path}: invalid segment field: {exc}") from exc
        if start < 0 or end <= start:
            raise TraceValidationError(
                f"{path}: expected 0 <= start_token < end_token"
            )
        return cls(kind=kind, label=label, start_token=start, end_token=end)


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    block_size: int = 16
    cache_salt_digest: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> "CacheConfig":
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TraceValidationError(f"{path}.enabled: expected a boolean")
        try:
            block_size = int(data.get("block_size", 16))
        except (TypeError, ValueError) as exc:
            raise TraceValidationError(f"{path}.block_size: expected an integer") from exc
        if block_size <= 0:
            raise TraceValidationError(f"{path}.block_size: must be positive")
        salt = data.get("cache_salt_digest")
        if salt is not None and not isinstance(salt, str):
            raise TraceValidationError(
                f"{path}.cache_salt_digest: expected a string or null"
            )
        return cls(enabled=enabled, block_size=block_size, cache_salt_digest=salt)


@dataclass(frozen=True)
class RequestMetrics:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    ttft_ms: float | None = None
    queue_ms: float | None = None
    total_latency_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> "RequestMetrics":
        integer_fields: dict[str, int | None] = {}
        for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
            value = data.get(name)
            if value is None:
                integer_fields[name] = None
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TraceValidationError(
                    f"{path}.{name}: expected a non-negative integer or null"
                )
            integer_fields[name] = value
        return cls(
            **integer_fields,
            ttft_ms=_optional_number(data.get("ttft_ms"), f"{path}.ttft_ms"),
            queue_ms=_optional_number(data.get("queue_ms"), f"{path}.queue_ms"),
            total_latency_ms=_optional_number(
                data.get("total_latency_ms"), f"{path}.total_latency_ms"
            ),
        )


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    sequence: int
    model: str
    backend: str
    prompt_token_ids: tuple[int, ...]
    model_revision: str | None = None
    backend_version: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    chat_template_digest: str | None = None
    segments: tuple[PromptSegment, ...] = field(default_factory=tuple)
    available_tools: tuple[str, ...] = field(default_factory=tuple)
    invoked_tools: tuple[str, ...] = field(default_factory=tuple)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    @property
    def cache_identity(self) -> tuple[Any, ...]:
        return (
            self.backend,
            self.backend_version,
            self.model,
            self.model_revision,
            self.tokenizer,
            self.tokenizer_revision,
            self.chat_template_digest,
            self.cache.block_size,
            self.cache.cache_salt_digest,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> "InferenceRequest":
        request_id = str(_required(data, "request_id", path))
        try:
            sequence = int(_required(data, "sequence", path))
        except (TypeError, ValueError) as exc:
            raise TraceValidationError(f"{path}.sequence: expected an integer") from exc
        if sequence < 0:
            raise TraceValidationError(f"{path}.sequence: must be non-negative")

        raw_tokens = _required(data, "prompt_token_ids", path)
        if not isinstance(raw_tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in raw_tokens
        ):
            raise TraceValidationError(
                f"{path}.prompt_token_ids: expected a list of non-negative integers"
            )

        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            raise TraceValidationError(f"{path}.segments: expected a list")
        segments = tuple(
            PromptSegment.from_dict(item, f"{path}.segments[{index}]")
            for index, item in enumerate(raw_segments)
        )
        previous_end = 0
        for index, segment in enumerate(segments):
            if segment.end_token > len(raw_tokens):
                raise TraceValidationError(
                    f"{path}.segments[{index}]: range exceeds prompt token length"
                )
            if index and segment.start_token < previous_end:
                raise TraceValidationError(
                    f"{path}.segments[{index}]: segments must be ordered and non-overlapping"
                )
            previous_end = segment.end_token

        metrics = RequestMetrics.from_dict(data.get("metrics", {}), f"{path}.metrics")
        if metrics.input_tokens is not None and metrics.input_tokens != len(raw_tokens):
            raise TraceValidationError(
                f"{path}.metrics.input_tokens: does not match prompt_token_ids length"
            )
        available_tools = _string_tuple(
            data.get("available_tools"), f"{path}.available_tools"
        )
        invoked_tools = _string_tuple(
            data.get("invoked_tools"), f"{path}.invoked_tools"
        )
        unknown_invocations = set(invoked_tools) - set(available_tools)
        if unknown_invocations:
            raise TraceValidationError(
                f"{path}.invoked_tools: tools were not present in available_tools: "
                f"{sorted(unknown_invocations)}"
            )

        return cls(
            request_id=request_id,
            sequence=sequence,
            model=str(_required(data, "model", path)),
            backend=str(_required(data, "backend", path)).lower(),
            prompt_token_ids=tuple(raw_tokens),
            model_revision=data.get("model_revision"),
            backend_version=data.get("backend_version"),
            tokenizer=data.get("tokenizer"),
            tokenizer_revision=data.get("tokenizer_revision"),
            chat_template_digest=data.get("chat_template_digest"),
            segments=segments,
            available_tools=available_tools,
            invoked_tools=invoked_tools,
            cache=CacheConfig.from_dict(data.get("cache", {}), f"{path}.cache"),
            metrics=metrics,
        )


@dataclass(frozen=True)
class ExecutionSpan:
    span_id: str
    kind: str
    start_ms: float
    end_ms: float
    request_id: str | None = None
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> "ExecutionSpan":
        span_id = str(_required(data, "span_id", path))
        kind = str(_required(data, "kind", path))
        start = _optional_number(_required(data, "start_ms", path), f"{path}.start_ms")
        end = _optional_number(_required(data, "end_ms", path), f"{path}.end_ms")
        assert start is not None and end is not None
        if end < start:
            raise TraceValidationError(f"{path}: end_ms must be >= start_ms")
        request_id = data.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise TraceValidationError(f"{path}.request_id: expected a string or null")
        dependencies = data.get("depends_on", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise TraceValidationError(f"{path}.depends_on: expected a list of strings")
        if span_id in dependencies:
            raise TraceValidationError(f"{path}.depends_on: a span cannot depend on itself")
        return cls(
            span_id=span_id,
            kind=kind,
            start_ms=start,
            end_ms=end,
            request_id=request_id,
            depends_on=tuple(dependencies),
        )


@dataclass(frozen=True)
class Trace:
    schema_version: str
    trace_id: str
    workload: str
    success: bool | None
    spans: tuple[ExecutionSpan, ...]
    requests: tuple[InferenceRequest, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        if not isinstance(data, dict):
            raise TraceValidationError("trace: expected a JSON object")
        schema_version = str(_required(data, "schema_version", "trace"))
        if schema_version != "0.1":
            raise TraceValidationError(
                f"trace.schema_version: unsupported version {schema_version!r}"
            )
        success = data.get("success")
        if success is not None and not isinstance(success, bool):
            raise TraceValidationError("trace.success: expected a boolean or null")
        raw_requests = _required(data, "requests", "trace")
        if not isinstance(raw_requests, list):
            raise TraceValidationError("trace.requests: expected a list")
        requests = tuple(
            InferenceRequest.from_dict(item, f"trace.requests[{index}]")
            for index, item in enumerate(raw_requests)
        )
        request_ids = [request.request_id for request in requests]
        if len(request_ids) != len(set(request_ids)):
            raise TraceValidationError("trace.requests: request_id values must be unique")
        sequences = [request.sequence for request in requests]
        if len(sequences) != len(set(sequences)):
            raise TraceValidationError("trace.requests: sequence values must be unique")

        raw_spans = data.get("spans", [])
        if not isinstance(raw_spans, list):
            raise TraceValidationError("trace.spans: expected a list")
        spans = tuple(
            ExecutionSpan.from_dict(item, f"trace.spans[{index}]")
            for index, item in enumerate(raw_spans)
        )
        span_ids = [span.span_id for span in spans]
        if len(span_ids) != len(set(span_ids)):
            raise TraceValidationError("trace.spans: span_id values must be unique")
        span_id_set = set(span_ids)
        request_id_set = set(request_ids)
        for index, span in enumerate(spans):
            missing = set(span.depends_on) - span_id_set
            if missing:
                raise TraceValidationError(
                    f"trace.spans[{index}].depends_on: unknown span IDs {sorted(missing)}"
                )
            if span.request_id is not None and span.request_id not in request_id_set:
                raise TraceValidationError(
                    f"trace.spans[{index}].request_id: unknown request ID "
                    f"{span.request_id!r}"
                )
        span_by_id = {span.span_id: span for span in spans}
        indegree = {span.span_id: len(span.depends_on) for span in spans}
        successors: dict[str, list[str]] = {span.span_id: [] for span in spans}
        for span in spans:
            for dependency_id in span.depends_on:
                successors[dependency_id].append(span.span_id)
        ready = [span_id for span_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            span_id = ready.pop()
            visited += 1
            for successor in successors[span_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if visited != len(spans):
            raise TraceValidationError("trace.spans: dependency graph contains a cycle")
        for index, span in enumerate(spans):
            for dependency_id in span.depends_on:
                dependency = span_by_id[dependency_id]
                if span.start_ms < dependency.end_ms:
                    raise TraceValidationError(
                        f"trace.spans[{index}]: starts before dependency "
                        f"{dependency_id!r} ends"
                    )
        return cls(
            schema_version=schema_version,
            trace_id=str(_required(data, "trace_id", "trace")),
            workload=str(data.get("workload", "unknown")),
            success=success,
            spans=spans,
            requests=tuple(sorted(requests, key=lambda request: request.sequence)),
        )
