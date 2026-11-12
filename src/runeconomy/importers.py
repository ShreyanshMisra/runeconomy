"""Import exact-token vLLM capture records into normalized traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    required; semantic segments are retained only when explicitly supplied.
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
        segments = record.get("segments", [])
        request_model = original_request.get("model", model)
        request_payload: dict[str, Any] = {
            "request_id": request_id,
            "sequence": sequence,
            "model": request_model,
            "backend": "vllm",
            "prompt_token_ids": token_ids,
            "segments": segments,
            "available_tools": available_tools,
            "invoked_tools": invoked_tools,
            "cache": {"enabled": True, "block_size": block_size},
            "metrics": {
                "input_tokens": len(token_ids),
                "output_tokens": capture.get("output_tokens"),
                "cached_input_tokens": capture.get("cached_tokens"),
                "ttft_ms": capture.get("server_ttft_ms", capture.get("client_ttft_ms")),
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
        "schema_version": "0.1",
        "trace_id": trace_id,
        "workload": workload,
        "success": success,
        "spans": spans,
        "requests": requests,
    }
    Trace.from_dict(payload)
    return payload
