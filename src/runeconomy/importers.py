"""Import external inference evidence without manufacturing missing detail."""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .evidence_quality import build_evidence_quality, evidence_field
from .identity import (
    PROVIDER_DECLARED,
    TRACE_FINGERPRINT_ATTRIBUTE,
    TRACE_GRADE_ATTRIBUTE,
    TRACE_UNDISCLOSED_ATTRIBUTE,
    IdentityPolicyError,
    fingerprint_digest,
    parse_grade,
    parse_undisclosed,
)
from .models import Trace, TraceValidationError


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    if isinstance(tool.get("name"), str):
        return tool["name"]
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TraceValidationError(f"cannot read {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(
                f"{path}:{line_number}:{exc.colno}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise TraceValidationError(
                f"{path}:{line_number}: capture record must be a JSON object"
            )
        records.append(value)
    if not records:
        raise TraceValidationError(f"{path}: no capture records found")
    return records


def _structural_segments(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe request blocks while making no claim about rendered token offsets."""

    segments: list[dict[str, Any]] = []
    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        segments.append(
            {
                "kind": "tools",
                "label": "tool definitions",
                "start_token": None,
                "end_token": None,
                "attribution_quality": "structural",
                "source": "request.tools",
            }
        )
    messages = request.get("messages", request.get("input"))
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if isinstance(messages, list):
        message_objects = [item for item in messages if isinstance(item, dict)]
        latest_user = max(
            (
                index
                for index, item in enumerate(message_objects)
                if item.get("role") == "user"
            ),
            default=None,
        )
        for index, message in enumerate(message_objects):
            role = str(message.get("role", "unknown"))
            if role in {"system", "developer"}:
                kind = "system_instructions"
            elif role == "tool":
                kind = "tool_results"
            elif role == "user" and index == latest_user:
                kind = "latest_user_input"
            else:
                kind = "history"
            segments.append(
                {
                    "kind": kind,
                    "label": f"{role} message {index}",
                    "start_token": None,
                    "end_token": None,
                    "attribution_quality": "structural",
                    "source": f"request.messages[{index}]",
                }
            )
    for key, kind in (
        ("retrieved_context", "retrieved_context"),
        ("documents", "retrieved_context"),
        ("metadata", "dynamic_metadata"),
    ):
        if key in request and request[key] not in (None, [], {}):
            segments.append(
                {
                    "kind": kind,
                    "label": key.replace("_", " "),
                    "start_token": None,
                    "end_token": None,
                    "attribution_quality": "structural",
                    "source": f"request.{key}",
                }
            )
    if not segments:
        segments.append(
            {
                "kind": "unknown",
                "label": "unattributed prompt",
                "start_token": None,
                "end_token": None,
                "attribution_quality": "unknown",
                "source": "capture",
            }
        )
    return segments


def _capture_segments(
    record: dict[str, Any], request: dict[str, Any], token_ids: list[int]
) -> list[dict[str, Any]]:
    explicit = record.get("segments")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise TraceValidationError("capture segments must be an array")
        return explicit
    rendered = record.get("rendered_segments")
    if rendered is not None:
        if not isinstance(rendered, list):
            raise TraceValidationError("rendered_segments must be an array")
        flattened: list[int] = []
        result: list[dict[str, Any]] = []
        for index, item in enumerate(rendered):
            if not isinstance(item, dict) or not isinstance(item.get("token_ids"), list):
                raise TraceValidationError(
                    f"rendered_segments[{index}].token_ids must be an array"
                )
            ids = item["token_ids"]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
                raise TraceValidationError(
                    f"rendered_segments[{index}].token_ids must contain integers"
                )
            start = len(flattened)
            flattened.extend(ids)
            if ids:
                result.append(
                    {
                        "kind": str(item.get("kind", "unknown")),
                        "label": str(item.get("label", item.get("kind", "unknown"))),
                        "start_token": start,
                        "end_token": len(flattened),
                        "attribution_quality": "exact_token_boundary",
                        "source": "capture.rendered_segments",
                    }
                )
        if flattened != token_ids:
            raise TraceValidationError(
                "rendered_segments do not concatenate to prompt_token_ids"
            )
        return result
    return _structural_segments(request)


def import_vllm_jsonl(
    path: Path,
    *,
    trace_id: str,
    workload: str,
    model: str,
    model_revision: str | None = None,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    chat_template_digest: str | None = None,
    backend_version: str | None = None,
    block_size: int = 16,
    success: bool | None = None,
) -> dict[str, Any]:
    """Normalize one JSON object per captured inference request.

    Each record may be a raw ``capture-vllm`` result or an envelope containing
    ``capture`` and the original ``request``. Exact ``prompt_token_ids`` are
    required. Exact semantic boundaries are accepted only from explicit segment
    evidence or verified rendered components; otherwise request blocks are marked
    as structural attribution with null token offsets.
    """

    requests: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    elapsed_ms = 0.0
    for sequence, record in enumerate(_records(path)):
        capture = record.get("capture", record)
        if not isinstance(capture, dict):
            raise TraceValidationError(
                f"capture record {sequence}: capture must be an object"
            )
        token_ids = capture.get("prompt_token_ids")
        if not isinstance(token_ids, list):
            raise TraceValidationError(
                f"capture record {sequence}: prompt_token_ids are required"
            )
        original_request = record.get("request", {})
        if not isinstance(original_request, dict):
            raise TraceValidationError(
                f"capture record {sequence}: request must be an object"
            )
        request_id = str(record.get("request_id", f"request-{sequence + 1}"))
        tools = original_request.get("tools", [])
        if not isinstance(tools, list):
            tools = []
        available_tools = list(
            dict.fromkeys(name for tool in tools if (name := _tool_name(tool)))
        )
        invoked_tools = record.get("invoked_tools", [])
        if not isinstance(invoked_tools, list):
            raise TraceValidationError(
                f"capture record {sequence}: invoked_tools must be an array"
            )
        total_latency = capture.get("total_latency_ms")
        duration = (
            float(total_latency)
            if isinstance(total_latency, (int, float))
            and not isinstance(total_latency, bool)
            else 0.0
        )
        start_ms = record.get("start_ms", elapsed_ms)
        end_ms = record.get("end_ms", float(start_ms) + duration)
        span_id = str(record.get("span_id", f"inference-{sequence + 1}"))
        depends_on = record.get(
            "depends_on", [] if sequence == 0 else [spans[-1]["span_id"]]
        )
        segments = _capture_segments(record, original_request, token_ids)
        request_model = original_request.get("model", model)
        request_payload: dict[str, Any] = {
            "request_id": request_id,
            "sequence": sequence,
            "model": request_model,
            "backend": "vllm",
            "prompt_token_ids": token_ids,
            "prompt_token_count": len(token_ids),
            "token_evidence_quality": "exact",
            "segments": segments,
            "available_tools": available_tools,
            "invoked_tools": invoked_tools,
            "cache": {"enabled": True, "block_size": block_size},
            "metrics": {
                "input_tokens": len(token_ids),
                "output_tokens": capture.get("output_tokens"),
                "cached_input_tokens": capture.get("cached_tokens"),
                "ttft_ms": capture.get("server_ttft_ms"),
                "client_ttft_ms": capture.get("client_ttft_ms"),
                "queue_ms": capture.get("server_queue_ms"),
                "total_latency_ms": total_latency,
            },
        }
        optional_identity = {
            "model_revision": model_revision,
            "backend_version": backend_version,
            "tokenizer": tokenizer,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_digest": chat_template_digest,
        }
        request_payload.update(
            {
                key: value
                for key, value in optional_identity.items()
                if value is not None
            }
        )
        requests.append(request_payload)
        spans.append(
            {
                "span_id": span_id,
                "kind": "inference",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "request_id": request_id,
                "depends_on": depends_on,
            }
        )
        elapsed_ms = max(elapsed_ms, float(end_ms))

    payload = {
        "schema_version": "0.2",
        "trace_id": trace_id,
        "workload": workload,
        "success": success,
        "dependency_evidence": "derived_sequential_capture_order",
        "spans": spans,
        "requests": requests,
    }
    identity_complete = all(
        all(
            request.get(field) not in {None, "", "unknown"}
            for field in (
                "model",
                "model_revision",
                "backend",
                "backend_version",
                "tokenizer",
                "tokenizer_revision",
                "chat_template_digest",
            )
        )
        for request in requests
    )
    payload["evidence_quality"] = build_evidence_quality(
        "vllm_capture_jsonl",
        {
            "prompt_tokens": evidence_field(
                "directly_observed", "vLLM render or tokenize response"
            ),
            "cached_input_tokens": evidence_field(
                "directly_observed"
                if all(
                    request["metrics"].get("cached_input_tokens") is not None
                    for request in requests
                )
                else "unresolved",
                "vLLM completed-response usage details"
                if all(
                    request["metrics"].get("cached_input_tokens") is not None
                    for request in requests
                )
                else "cached-token field absent from one or more responses",
            ),
            "server_ttft": evidence_field(
                "directly_observed"
                if all(request["metrics"].get("ttft_ms") is not None for request in requests)
                else "unresolved",
                "vLLM per-request metrics"
                if all(request["metrics"].get("ttft_ms") is not None for request in requests)
                else "server TTFT absent from one or more responses",
            ),
            "workflow_dependencies": evidence_field(
                "reliably_derived",
                "sequential capture order",
                detail="capture order is not an explicitly observed workflow DAG",
            ),
            "backend_identity": evidence_field(
                "weakly_inferred" if identity_complete else "unresolved",
                "operator-supplied import arguments and request model"
                if identity_complete
                else "complete pinned identity was not supplied",
            ),
            "task_identity": evidence_field(
                "unresolved", "vLLM capture records do not establish task pairing"
            ),
            "request_latency": evidence_field(
                "directly_observed", "capture client monotonic timer"
            ),
        },
    )
    Trace.from_dict(payload)
    return payload


def _any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in (
        "stringValue",
        "boolValue",
        "intValue",
        "doubleValue",
        "bytesValue",
    ):
        if key in value:
            raw = value[key]
            if key == "intValue":
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return raw
            return raw
    if "arrayValue" in value:
        values = value["arrayValue"].get("values", [])
        return [_any_value(item) for item in values]
    if "kvlistValue" in value:
        return _attributes(value["kvlistValue"].get("values", []))
    return value


def _attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, list):
        return {}
    result: dict[str, Any] = {}
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("key"), str):
            result[item["key"]] = _any_value(item.get("value"))
    return result


def _otel_spans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("spans"), list):
        return [item for item in payload["spans"] if isinstance(item, dict)]
    spans: list[dict[str, Any]] = []
    resources = payload.get("resourceSpans", payload.get("resource_spans", []))
    if not isinstance(resources, list):
        return spans
    for resource_span in resources:
        if not isinstance(resource_span, dict):
            continue
        resource = resource_span.get("resource", {})
        resource_attrs = _attributes(
            resource.get("attributes", []) if isinstance(resource, dict) else []
        )
        scopes = resource_span.get("scopeSpans", resource_span.get("scope_spans", []))
        for scope_span in scopes if isinstance(scopes, list) else []:
            if not isinstance(scope_span, dict):
                continue
            for span in scope_span.get("spans", []):
                if isinstance(span, dict):
                    normalized = dict(span)
                    normalized["_resource_attributes"] = resource_attrs
                    spans.append(normalized)
    return spans


def _span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    attributes = _attributes(span.get("attributes", {}))
    resource = _attributes(span.get("resource", {}))
    resource.update(_attributes(span.get("_resource_attributes", {})))
    return {**resource, **attributes}


def _first(attributes: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in attributes and attributes[key] is not None:
            return attributes[key]
    return None


def _consistent_first(
    attributes: dict[str, Any], *keys: str, field: str
) -> tuple[Any, str | None]:
    """Return the first alias only when all present aliases agree."""

    present = [(key, attributes[key]) for key in keys if attributes.get(key) is not None]
    if not present:
        return None, None
    canonical = {
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        for _, value in present
    }
    if len(canonical) > 1:
        names = ", ".join(key for key, _ in present)
        raise TraceValidationError(
            f"conflicting {field} evidence across attributes: {names}"
        )
    return present[0][1], present[0][0]


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


_MESSAGE_ROLE = re.compile(r"^llm\.input_messages\.(\d+)\.message\.role$")
_TOOL_NAME = re.compile(r"^llm\.tools\.(\d+)\.tool\.name$")
_TOOL_CALL = re.compile(
    r"^llm\.output_messages\.\d+\.message\.tool_calls\.\d+\.tool_call\.function\.name$"
)


def _openinference_segments(
    attributes: dict[str, Any], exact_token_ids: list[int] | None
) -> list[dict[str, Any]]:
    explicit = attributes.get("runeconomy.prompt_segments")
    if isinstance(explicit, str):
        try:
            explicit = json.loads(explicit)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(
                "runeconomy.prompt_segments must be valid JSON"
            ) from exc
    if explicit is not None:
        if not isinstance(explicit, list) or exact_token_ids is None:
            raise TraceValidationError(
                "runeconomy.prompt_segments requires exact prompt token IDs and an array value"
            )
        flattened: list[int] = []
        result: list[dict[str, Any]] = []
        for index, item in enumerate(explicit):
            if not isinstance(item, dict) or not isinstance(item.get("token_ids"), list):
                raise TraceValidationError(
                    f"runeconomy.prompt_segments[{index}].token_ids must be an array"
                )
            ids = item["token_ids"]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
                raise TraceValidationError(
                    f"runeconomy.prompt_segments[{index}].token_ids must contain integers"
                )
            start = len(flattened)
            flattened.extend(ids)
            if ids:
                result.append(
                    {
                        "kind": str(item.get("kind", "unknown")),
                        "label": str(item.get("label", item.get("kind", "unknown"))),
                        "start_token": start,
                        "end_token": len(flattened),
                        "attribution_quality": "exact_token_boundary",
                        "source": "runeconomy.prompt_segments",
                    }
                )
        if flattened != exact_token_ids:
            raise TraceValidationError(
                "runeconomy.prompt_segments do not concatenate to prompt token IDs"
            )
        return result
    roles = sorted(
        (
            (int(match.group(1)), str(value))
            for key, value in attributes.items()
            if (match := _MESSAGE_ROLE.match(key))
        ),
        key=lambda item: item[0],
    )
    request: dict[str, Any] = {
        "messages": [{"role": role} for _, role in roles],
        "tools": [
            {"name": value}
            for key, value in attributes.items()
            if _TOOL_NAME.match(key)
        ],
    }
    return _structural_segments(request)


def _exact_token_ids(attributes: dict[str, Any]) -> list[int] | None:
    value = attributes.get("runeconomy.prompt_token_ids")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, list) and all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0
        for token in value
    ):
        return value
    return None


def _explicit_dependencies(attributes: dict[str, Any]) -> list[str]:
    value = attributes.get("runeconomy.depends_on", [])
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in value.split(",") if item.strip()]
        value = parsed
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TraceValidationError(
            "runeconomy.depends_on must be an array of span IDs or a JSON array string"
        )
    return list(dict.fromkeys(value))


def import_openinference(
    path: Path,
    *,
    trace_id: str | None = None,
    workload: str = "unknown",
    backend: str = "unknown",
    block_size: int = 16,
) -> dict[str, Any]:
    """Normalize one OpenInference/OpenTelemetry trace from OTLP JSON.

    OpenInference message structure is retained as structural attribution. Exact
    prompt tokens are imported only from the explicit RunEconomy extension
    ``runeconomy.prompt_token_ids``.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceValidationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise TraceValidationError(f"{path}: expected an OTLP JSON object")
    raw_spans = _otel_spans(payload)
    trace_ids = {
        str(span.get("traceId", span.get("trace_id")))
        for span in raw_spans
        if span.get("traceId", span.get("trace_id")) is not None
    }
    if trace_id is None:
        if len(trace_ids) != 1:
            raise TraceValidationError(
                "OpenTelemetry input must contain exactly one trace; pass --trace-id"
            )
        trace_id = next(iter(trace_ids))
    selected = [
        span
        for span in raw_spans
        if str(span.get("traceId", span.get("trace_id"))) == trace_id
    ]
    if not selected:
        raise TraceValidationError(f"trace_id {trace_id!r} was not found")

    starts = [
        int(span.get("startTimeUnixNano", span.get("start_time_unix_nano", 0)))
        for span in selected
    ]
    origin = min(starts)
    requests: list[dict[str, Any]] = []
    normalized_spans: list[dict[str, Any]] = []
    any_error = False
    dependency_declarations = 0
    task_outcomes: set[bool] = set()
    task_ids: set[str] = set()
    attempt_ids: set[str] = set()
    cached_token_grades: list[str] = []
    server_ttft_grades: list[str] = []
    for span in selected:
        attributes = _span_attributes(span)
        explicit_success = attributes.get("runeconomy.task_success")
        if isinstance(explicit_success, bool):
            task_outcomes.add(explicit_success)
        if isinstance(attributes.get("runeconomy.task_id"), str):
            task_ids.add(attributes["runeconomy.task_id"])
        if isinstance(attributes.get("runeconomy.attempt_id"), str):
            attempt_ids.add(attributes["runeconomy.attempt_id"])
        span_kind = str(attributes.get("openinference.span.kind", "")).upper()
        operation = str(attributes.get("gen_ai.operation.name", "")).lower()
        is_llm = span_kind == "LLM" or operation in {"chat", "text_completion"}
        is_work = is_llm or span_kind in {"TOOL", "RETRIEVER"}
        if not is_work:
            continue
        if "runeconomy.depends_on" in attributes:
            dependency_declarations += 1
        span_id = str(span.get("spanId", span.get("span_id", f"span-{len(normalized_spans)}")))
        start_ns = int(span.get("startTimeUnixNano", span.get("start_time_unix_nano", origin)))
        end_ns = int(span.get("endTimeUnixNano", span.get("end_time_unix_nano", start_ns)))
        status = span.get("status", {})
        if isinstance(status, dict) and str(
            status.get("code", status.get("statusCode", ""))
        ).upper() in {"ERROR", "STATUS_CODE_ERROR", "2"}:
            any_error = True
        request_id: str | None = None
        if is_llm:
            request_id = span_id
            exact_ids = _exact_token_ids(attributes)
            prompt_value, _ = _consistent_first(
                attributes,
                "llm.token_count.prompt",
                "gen_ai.usage.input_tokens",
                field="prompt-token count",
            )
            prompt_count = _integer(prompt_value)
            if exact_ids is not None:
                prompt_count = len(exact_ids)
                token_quality = "exact"
            elif prompt_count is not None:
                token_quality = "count_only"
            else:
                token_quality = "unknown"
            available = [
                str(value)
                for key, value in attributes.items()
                if _TOOL_NAME.match(key) and isinstance(value, str)
            ]
            invoked = [
                str(value)
                for key, value in attributes.items()
                if _TOOL_CALL.match(key) and isinstance(value, str)
            ]
            available = list(dict.fromkeys([*available, *invoked]))
            model_value, _ = _consistent_first(
                attributes,
                "llm.model_name",
                "llm.request.model_name",
                "gen_ai.request.model",
                field="model identity",
            )
            backend_value, _ = _consistent_first(
                attributes,
                "llm.provider",
                "llm.system",
                "gen_ai.provider.name",
                field="backend identity",
            )
            output_value, _ = _consistent_first(
                attributes,
                "llm.token_count.completion",
                "gen_ai.usage.output_tokens",
                field="output-token count",
            )
            cached_value, cached_source = _consistent_first(
                attributes,
                "runeconomy.cached_input_tokens",
                "llm.token_count.prompt_details.cache_read",
                "gen_ai.usage.cache_read.input_tokens",
                field="cached-token count",
            )
            server_ttft_value, server_ttft_source = _consistent_first(
                attributes,
                "runeconomy.server_ttft_ms",
                "llm.time_to_first_token_ms",
                "gen_ai.ttft_ms",
                field="server TTFT",
            )
            cached_token_grades.append(
                "directly_observed" if cached_source is not None else "unresolved"
            )
            server_ttft_grades.append(
                "directly_observed"
                if server_ttft_source == "runeconomy.server_ttft_ms"
                else "weakly_inferred"
                if server_ttft_source is not None
                else "unresolved"
            )
            request_payload: dict[str, Any] = {
                "request_id": request_id,
                "sequence": len(requests),
                "model": str(model_value or "unknown"),
                "backend": str(backend_value or backend).lower(),
                "prompt_token_ids": exact_ids,
                "prompt_token_count": prompt_count,
                "token_evidence_quality": token_quality,
                "segments": _openinference_segments(attributes, exact_ids),
                "available_tools": available,
                "invoked_tools": list(dict.fromkeys(invoked)),
                "cache": {"enabled": True, "block_size": block_size},
                "metrics": {
                    "input_tokens": prompt_count,
                    "output_tokens": _integer(output_value),
                    "cached_input_tokens": _integer(cached_value),
                    "ttft_ms": _float(server_ttft_value),
                    "client_ttft_ms": _float(
                        attributes.get("runeconomy.client_ttft_ms")
                    ),
                    "queue_ms": _float(
                        _first(attributes, "llm.queue_time_ms", "gen_ai.queue_time_ms")
                    ),
                    "prefill_ms": _float(
                        _first(attributes, "llm.prefill_time_ms", "gen_ai.prefill_time_ms")
                    ),
                    "decode_ms": _float(
                        _first(attributes, "llm.decode_time_ms", "gen_ai.decode_time_ms")
                    ),
                    "cost_usd": _float(
                        _first(
                            attributes,
                            "llm.cost.total",
                            "gen_ai.usage.cost",
                            "runeconomy.cost_usd",
                        )
                    ),
                    "gpu_time_ms": _float(attributes.get("runeconomy.gpu_time_ms")),
                    "total_latency_ms": (end_ns - start_ns) / 1_000_000,
                },
            }
            step_role = attributes.get("runeconomy.step_role")
            if isinstance(step_role, str) and step_role.strip():
                # Declared step role: identity stability is checked per role,
                # so deliberate step-level model routing is not drift.
                request_payload["step_role"] = step_role.strip()
            model_revision, _ = _consistent_first(
                attributes,
                "llm.model_revision",
                "runeconomy.model_revision",
                field="model revision",
            )
            backend_version, _ = _consistent_first(
                attributes,
                "llm.backend_version",
                "runeconomy.backend_version",
                field="backend version",
            )
            identity_fields = {
                "model_revision": model_revision,
                "backend_version": backend_version,
                "tokenizer": attributes.get("runeconomy.tokenizer"),
                "tokenizer_revision": attributes.get(
                    "runeconomy.tokenizer_revision"
                ),
                "chat_template_digest": attributes.get(
                    "runeconomy.chat_template_digest"
                ),
            }
            request_payload.update(
                {
                    key: str(value)
                    for key, value in identity_fields.items()
                    if value is not None
                }
            )
            # Graded identity declarations (identity policy 0.1). Keys are added
            # only when a trace declares them, so legacy traces are unchanged.
            try:
                grade = parse_grade(attributes.get(TRACE_GRADE_ATTRIBUTE))
                undisclosed = parse_undisclosed(
                    attributes.get(TRACE_UNDISCLOSED_ATTRIBUTE)
                )
            except IdentityPolicyError as exc:
                raise TraceValidationError(f"span {span_id}: {exc}") from exc
            if grade is not None:
                request_payload["identity_grade"] = grade
            if undisclosed is not None:
                request_payload["identity_undisclosed_fields"] = undisclosed
            fingerprint = fingerprint_digest(
                attributes.get(TRACE_FINGERPRINT_ATTRIBUTE)
            )
            if fingerprint is not None:
                request_payload["provider_fingerprint_sha256"] = fingerprint
            requests.append(request_payload)
        normalized_spans.append(
            {
                "span_id": span_id,
                "kind": "inference" if is_llm else span_kind.lower(),
                "start_ms": (start_ns - origin) / 1_000_000,
                "end_ms": (end_ns - origin) / 1_000_000,
                "request_id": request_id,
                "depends_on": _explicit_dependencies(attributes),
                "parent_span_id": span.get("parentSpanId", span.get("parent_span_id")),
            }
        )
    if not requests:
        raise TraceValidationError("selected trace contains no OpenInference LLM spans")
    if len(task_outcomes) > 1:
        raise TraceValidationError("conflicting runeconomy.task_success values in one trace")
    if len(task_ids) > 1:
        raise TraceValidationError("conflicting runeconomy.task_id values in one trace")
    if len(attempt_ids) > 1:
        raise TraceValidationError("conflicting runeconomy.attempt_id values in one trace")
    dependency_evidence = (
        "complete_explicit"
        if dependency_declarations == len(normalized_spans)
        else "partial_explicit"
        if dependency_declarations
        else "absent"
    )
    identity_complete = all(
        all(
            request.get(field) not in {None, "", "unknown"}
            for field in (
                "model",
                "model_revision",
                "backend",
                "backend_version",
                "tokenizer",
                "tokenizer_revision",
                "chat_template_digest",
            )
        )
        for request in requests
    )
    normalized = {
        "schema_version": "0.2",
        "trace_id": trace_id,
        "workload": workload,
        "success": False if any_error else next(iter(task_outcomes)) if task_outcomes else None,
        "task_id": next(iter(task_ids)) if len(task_ids) == 1 else None,
        "attempt_id": next(iter(attempt_ids)) if len(attempt_ids) == 1 else None,
        "dependency_evidence": dependency_evidence,
        "spans": normalized_spans,
        "requests": requests,
    }
    normalized["evidence_quality"] = build_evidence_quality(
        "openinference_otlp",
        {
            "prompt_tokens": evidence_field(
                "directly_observed"
                if all(request["token_evidence_quality"] == "exact" for request in requests)
                else "reliably_derived"
                if all(request["prompt_token_count"] is not None for request in requests)
                else "unresolved",
                "explicit prompt token IDs"
                if all(request["token_evidence_quality"] == "exact" for request in requests)
                else "telemetry token-count attributes"
                if all(request["prompt_token_count"] is not None for request in requests)
                else "prompt token evidence absent",
            ),
            "cached_input_tokens": evidence_field(
                max(
                    cached_token_grades,
                    key={
                        "directly_observed": 0,
                        "reliably_derived": 1,
                        "weakly_inferred": 2,
                        "unresolved": 3,
                    }.__getitem__,
                ),
                "OpenInference or GenAI cache-read attributes"
                if all(grade == "directly_observed" for grade in cached_token_grades)
                else "cache-read attribute absent from one or more LLM spans",
            ),
            "server_ttft": evidence_field(
                max(
                    server_ttft_grades,
                    key={
                        "directly_observed": 0,
                        "reliably_derived": 1,
                        "weakly_inferred": 2,
                        "unresolved": 3,
                    }.__getitem__,
                ),
                "explicit RunEconomy server TTFT extension"
                if all(grade == "directly_observed" for grade in server_ttft_grades)
                else "generic framework TTFT attribute with ambiguous clock boundary"
                if all(grade != "unresolved" for grade in server_ttft_grades)
                else "server TTFT absent from one or more LLM spans",
            ),
            "workflow_dependencies": evidence_field(
                "directly_observed"
                if dependency_evidence == "complete_explicit"
                else "weakly_inferred"
                if dependency_evidence == "partial_explicit"
                else "unresolved",
                "explicit runeconomy.depends_on declarations"
                if dependency_evidence != "absent"
                else "no explicit workflow dependencies",
            ),
            "backend_identity": (
                evidence_field(
                    "weakly_inferred",
                    "provider-declared identity; undisclosed fields are recorded explicitly",
                )
                if any(
                    request.get("identity_grade") == PROVIDER_DECLARED
                    for request in requests
                )
                else evidence_field(
                    "directly_observed" if identity_complete else "unresolved",
                    "trace identity attributes"
                    if identity_complete
                    else "one or more pinned identity fields are absent",
                )
            ),
            "task_identity": evidence_field(
                "directly_observed" if len(task_ids) == 1 else "unresolved",
                "runeconomy.task_id trace attribute"
                if len(task_ids) == 1
                else "task identity was not recorded in the trace",
            ),
            "workflow_latency": evidence_field(
                "reliably_derived", "monotonic span timestamps"
            ),
        },
    )
    Trace.from_dict(normalized)
    return normalized


def _iso_nanoseconds(value: Any, path: str) -> int:
    if not isinstance(value, str):
        raise TraceValidationError(f"{path} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TraceValidationError(f"{path} must be an ISO 8601 timestamp") from exc
    return int(parsed.timestamp() * 1_000_000_000)


def import_phoenix_json(
    path: Path,
    *,
    workload: str = "unknown",
    backend: str = "unknown",
    block_size: int = 16,
) -> dict[str, Any]:
    """Import the documented ``px trace get --format json`` structure.

    Phoenix parent IDs are retained as structural evidence. They are not
    converted into RunEconomy execution dependencies.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceValidationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if isinstance(payload, list):
        if len(payload) != 1:
            raise TraceValidationError(
                "Phoenix input must contain one trace; import a directory for multiple traces"
            )
        payload = payload[0]
    if not isinstance(payload, dict) or not isinstance(payload.get("spans"), list):
        raise TraceValidationError(f"{path}: expected a Phoenix trace object")
    trace_id = str(payload.get("traceId", payload.get("trace_id", "phoenix-trace")))
    prepared: list[tuple[dict[str, Any], int, int]] = []
    for index, span in enumerate(payload["spans"]):
        if not isinstance(span, dict):
            continue
        start = _iso_nanoseconds(span.get("start_time"), f"spans[{index}].start_time")
        end = _iso_nanoseconds(span.get("end_time"), f"spans[{index}].end_time")
        prepared.append((span, start, end))
    if not prepared:
        raise TraceValidationError("Phoenix trace contains no spans")
    origin = min(start for _, start, _ in prepared)
    requests: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    any_error = False
    for span, start, end in prepared:
        attributes = _attributes(span.get("attributes", {}))
        kind = str(span.get("span_kind", attributes.get("openinference.span.kind", ""))).upper()
        if kind not in {"LLM", "TOOL", "RETRIEVER"}:
            continue
        context = span.get("context", {})
        span_id = str(
            context.get("span_id")
            if isinstance(context, dict) and context.get("span_id")
            else span.get("span_id", f"span-{len(spans) + 1}")
        )
        if str(span.get("status_code", "")).upper() == "ERROR":
            any_error = True
        request_id: str | None = None
        if kind == "LLM":
            request_id = span_id
            prompt_count = _integer(
                _first(attributes, "llm.token_count.prompt", "gen_ai.usage.input_tokens")
            )
            available = [
                str(value)
                for key, value in attributes.items()
                if _TOOL_NAME.match(key) and isinstance(value, str)
            ]
            invoked = [
                str(value)
                for key, value in attributes.items()
                if _TOOL_CALL.match(key) and isinstance(value, str)
            ]
            requests.append(
                {
                    "request_id": request_id,
                    "sequence": len(requests),
                    "model": str(attributes.get("llm.model_name") or "unknown"),
                    "backend": str(attributes.get("llm.provider") or backend).lower(),
                    "prompt_token_ids": None,
                    "prompt_token_count": prompt_count,
                    "token_evidence_quality": "count_only" if prompt_count is not None else "unknown",
                    "segments": _openinference_segments(attributes, None),
                    "available_tools": list(dict.fromkeys([*available, *invoked])),
                    "invoked_tools": list(dict.fromkeys(invoked)),
                    "cache": {"enabled": True, "block_size": block_size},
                    "metrics": {
                        "input_tokens": prompt_count,
                        "output_tokens": _integer(attributes.get("llm.token_count.completion")),
                        "cached_input_tokens": _integer(attributes.get("llm.token_count.prompt_details.cache_read")),
                        "total_latency_ms": (end - start) / 1_000_000,
                    },
                }
            )
        spans.append(
            {
                "span_id": span_id,
                "kind": "inference" if kind == "LLM" else kind.lower(),
                "start_ms": (start - origin) / 1_000_000,
                "end_ms": (end - origin) / 1_000_000,
                "request_id": request_id,
                "depends_on": [],
                "parent_span_id": span.get("parent_id"),
            }
        )
    normalized = {
        "schema_version": "0.2",
        "trace_id": trace_id,
        "workload": workload,
        "success": False if any_error else None,
        "dependency_evidence": "absent",
        "span_parentage_evidence": "structural_only",
        "spans": spans,
        "requests": requests,
    }
    normalized["evidence_quality"] = build_evidence_quality(
        "phoenix_cli_json",
        {
            "prompt_tokens": evidence_field(
                "reliably_derived"
                if requests
                and all(request["prompt_token_count"] is not None for request in requests)
                else "unresolved",
                "Phoenix-exported token-count attributes"
                if requests
                and all(request["prompt_token_count"] is not None for request in requests)
                else "prompt token count absent",
            ),
            "cached_input_tokens": evidence_field(
                "directly_observed"
                if requests
                and all(
                    request["metrics"].get("cached_input_tokens") is not None
                    for request in requests
                )
                else "unresolved",
                "Phoenix-exported cache-read attributes"
                if requests
                and all(
                    request["metrics"].get("cached_input_tokens") is not None
                    for request in requests
                )
                else "cache-read attribute absent",
            ),
            "server_ttft": evidence_field(
                "unresolved", "Phoenix adapter does not assign server TTFT semantics"
            ),
            "workflow_dependencies": evidence_field(
                "unresolved",
                "Phoenix parentage is structural and is not an execution DAG",
            ),
            "backend_identity": evidence_field(
                "unresolved", "complete pinned backend identity is unavailable"
            ),
            "task_identity": evidence_field(
                "unresolved", "Phoenix trace does not establish RunEconomy task pairing"
            ),
            "workflow_latency": evidence_field(
                "reliably_derived", "Phoenix span timestamps"
            ),
        },
    )
    Trace.from_dict(normalized)
    return normalized


_LEDGER_EVIDENCE_FIELDS = {
    "session_id",
    "agent_id",
    "provider",
    "model",
    "cause",
    "prompt_tokens",
    "cached_tokens",
    "avoidable_tokens",
    "avoidable_cost",
    "task_success",
    "legit",
    "native_reason",
    "classifier_version",
    "task_id",
    "request_id",
    "completion_tokens",
    "fixes",
    "cost",
    "status",
    "error",
    "api",
    "latency_ms",
    "overhead_ms",
    "gap_seconds",
    "evidence",
    "secondary_causes",
    "divergence_role",
    "divergence_index",
    "detail",
    "cache_ttl_seconds",
    "unread_write_tokens",
    "schema_version",
    "recorded_at",
    "prefix_hash",
    "block_hash",
}


def _ledger_template(cause: str) -> tuple[str, str, str, str]:
    mappings = {
        "tool_reserialize": (
            "prompt.canonicalize_tools",
            "cache_reuse",
            "cached_tokens",
            "increase",
        ),
        "volatile_system": (
            "prompt.move_dynamic_metadata",
            "cache_reuse",
            "cached_tokens",
            "increase",
        ),
        "history_edit": (
            "history.preserve_prefix",
            "cache_reuse",
            "cached_tokens",
            "increase",
        ),
    }
    return mappings.get(cause, ("", "cache_reuse", "cached_tokens", "increase"))


def import_ledger_jsonl(
    path: Path, *, source_name: str = "cache_ledger"
) -> dict[str, Any]:
    """Import cache-ledger JSONL exports as findings, preserving only ledger evidence."""

    findings: list[dict[str, Any]] = []
    sessions: set[str] = set()
    for index, row in enumerate(_records(path)):
        cause = row.get("cause")
        if not isinstance(cause, str) or not cause or cause in {"none", "first_write"}:
            continue
        request_id = str(row.get("request_id") or f"ledger-row-{index + 1}")
        session_id = row.get("session_id")
        if isinstance(session_id, str):
            sessions.add(session_id)
        identity = f"{session_id or ''}\0{request_id}\0{cause}"
        finding_id = "finding-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        template, target, metric, direction = _ledger_template(cause)
        evidence = {
            key: value
            for key, value in row.items()
            if key in _LEDGER_EVIDENCE_FIELDS
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        avoidable = _integer(row.get("avoidable_tokens"))
        findings.append(
            {
                "finding_id": finding_id,
                "rule_id": f"{source_name}.{cause}",
                "request_id": request_id,
                "affected_request_ids": [request_id],
                "title": f"The imported ledger classified the request as {cause}",
                "evidence_level": "imported",
                "confidence": "source_reported",
                "summary": str(row.get("detail") or row.get("evidence") or cause),
                "suggested_experiment": str(row.get("fixes") or "Review the imported evidence and define a paired experiment."),
                "supporting_metrics": evidence,
                "on_critical_path": None,
                "estimated_reducible_tokens": avoidable,
                "estimated_reducible_ms": None,
                "optimization_target": target,
                "experiment_template_id": template,
                "experiment_parameters": {"source_evidence": evidence},
                "required_capabilities": ["task_evaluator", "replayable_workload"],
                "unsupported_reason": (
                    "The ledger export contains hashes, counts, and conclusions rather than a "
                    "replayable request; supply the baseline request, candidate change, "
                    "backend, sampling, acceptance gates, and evaluator."
                ),
                "causal_mechanism": str(
                    row.get("detail")
                    or f"The imported ledger attributed cache waste to {cause}."
                ),
                "predicted_metric": metric,
                "predicted_direction": direction,
                "attribution_quality": "block_level" if row.get("divergence_role") else "unknown",
                "source": f"{source_name}_export",
            }
        )
    return {
        "artifact_type": "runeconomy.imported_findings",
        "artifact_version": "0.1",
        "source": source_name,
        "trace_id": next(iter(sessions)) if len(sessions) == 1 else "ledger-import",
        "findings": findings,
        "limitations": [
            "No raw prompts or token arrays were reconstructed from hashes or counts.",
            "Critical-path membership is unknown without a corresponding workflow trace.",
            "Generated experiments remain unresolved until replay inputs and an evaluator are supplied.",
        ],
    }
