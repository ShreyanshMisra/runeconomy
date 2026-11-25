"""Off-path response telemetry for captured OpenAI-compatible exchanges.

Everything here runs on recorder threads after the client response is
complete. The input is the exact chunk sequence the proxy forwarded, each with
the monotonic time it arrived from the upstream. Nothing is estimated. A metric
the stream cannot establish is ``None``.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence


@dataclass(frozen=True)
class SSEEvent:
    """One dispatched server-sent event and the time its last byte arrived."""

    arrived_ns: int
    data: str
    event: str | None = None


class SSEParser:
    """Incremental WHATWG server-sent-event parser.

    It handles LF, CRLF and bare-CR line endings split across chunk
    boundaries, comment lines, multi-line ``data`` fields and a final event
    that has no trailing blank line.
    """

    def __init__(self) -> None:
        self._buffer = b""
        self._data: list[str] = []
        self._event: str | None = None
        self._pending_cr = False

    def feed(self, chunk: bytes, arrived_ns: int) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        data = self._buffer + chunk
        self._buffer = b""
        start = 0
        index = 0
        length = len(data)
        if self._pending_cr and data.startswith(b"\n"):
            start = index = 1
        self._pending_cr = False
        while index < length:
            byte = data[index]
            if byte == 0x0A or byte == 0x0D:
                line = data[start:index]
                if byte == 0x0D:
                    if index + 1 < length and data[index + 1] == 0x0A:
                        index += 1
                    elif index + 1 == length:
                        self._pending_cr = True
                self._line(line, arrived_ns, events)
                start = index + 1
            index += 1
        self._buffer = data[start:]
        return events

    def finish(self, arrived_ns: int) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        if self._buffer:
            self._line(self._buffer, arrived_ns, events)
            self._buffer = b""
        if self._data:
            self._dispatch(arrived_ns, events)
        return events

    def _line(self, raw: bytes, arrived_ns: int, events: list[SSEEvent]) -> None:
        line = raw.decode("utf-8", errors="replace")
        if line == "":
            self._dispatch(arrived_ns, events)
            return
        if line.startswith(":"):
            return
        name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value

    def _dispatch(self, arrived_ns: int, events: list[SSEEvent]) -> None:
        if self._data:
            events.append(
                SSEEvent(
                    arrived_ns=arrived_ns,
                    data="\n".join(self._data),
                    event=self._event,
                )
            )
        self._data = []
        self._event = None


class _Decoder:
    """Undo upstream content coding chunk by chunk, if any was applied."""

    def __init__(self, content_encoding: str | None) -> None:
        coding = (content_encoding or "identity").strip().lower()
        self.supported = coding in {"identity", "", "gzip", "x-gzip", "deflate"}
        self.coding = coding
        if coding in {"gzip", "x-gzip"}:
            self._inflater: Any = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif coding == "deflate":
            self._inflater = zlib.decompressobj()
        else:
            self._inflater = None
        self._deflate_raw_retry = coding == "deflate"
        self._pending = b""

    def decode(self, chunk: bytes) -> bytes:
        if self._inflater is None:
            return chunk
        try:
            return self._inflater.decompress(chunk)
        except zlib.error:
            if self._deflate_raw_retry:
                # Some servers send raw DEFLATE without the zlib wrapper.
                self._deflate_raw_retry = False
                self._inflater = zlib.decompressobj(-zlib.MAX_WBITS)
                return self._inflater.decompress(chunk)
            raise


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_intervals(values_ms: Sequence[float]) -> dict[str, Any] | None:
    """Summarize inter-event intervals with linear-interpolated quantiles."""

    if not values_ms:
        return None
    ordered = sorted(values_ms)
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(_quantile(ordered, 0.50), 3),
        "p90": round(_quantile(ordered, 0.90), 3),
        "p99": round(_quantile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or not math.isfinite(float(value)):
        return None
    return float(value)


def _non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


@dataclass
class ResponseTelemetry:
    """Facts extracted from one captured response, with unknowns left None."""

    stream: bool
    decodable: bool = True
    events: int = 0
    output_event_times_ns: list[int] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reasons: list[str] = field(default_factory=list)
    invoked_tools: list[str] = field(default_factory=list)
    response_id: str | None = None
    response_model: str | None = None
    server_metrics: dict[str, float] = field(default_factory=dict)
    terminal: str = "unknown"
    stream_error: Any = None
    assembled: dict[str, Any] | None = None
    output_sha256: str | None = None
    parse_errors: int = 0

    def usage_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


_SERVER_METRIC_KEYS = {
    "time_to_first_token_ms": "server_ttft_ms",
    "queue_time_ms": "server_queue_ms",
    "generation_time_ms": "server_generation_ms",
    "mean_itl_ms": "server_mean_itl_ms",
    "tokens_per_second": "server_tokens_per_second",
}


def _absorb_server_metrics(target: ResponseTelemetry, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return
    for source, name in _SERVER_METRIC_KEYS.items():
        value = _number(metrics.get(source))
        if value is not None:
            target.server_metrics[name] = value


def _absorb_chat_usage(target: ResponseTelemetry, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    target.prompt_tokens = _int(usage.get("prompt_tokens"))
    target.completion_tokens = _int(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details")
    target.cached_tokens = (
        _int(details.get("cached_tokens")) if isinstance(details, dict) else None
    )
    completion_details = usage.get("completion_tokens_details")
    target.reasoning_tokens = (
        _int(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else None
    )


def _absorb_responses_usage(target: ResponseTelemetry, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    target.prompt_tokens = _int(usage.get("input_tokens"))
    target.completion_tokens = _int(usage.get("output_tokens"))
    details = usage.get("input_tokens_details")
    target.cached_tokens = (
        _int(details.get("cached_tokens")) if isinstance(details, dict) else None
    )
    output_details = usage.get("output_tokens_details")
    target.reasoning_tokens = (
        _int(output_details.get("reasoning_tokens"))
        if isinstance(output_details, dict)
        else None
    )


def _digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _ChatAssembler:
    def __init__(self) -> None:
        self.choices: dict[int, dict[str, Any]] = {}

    def add(self, choice: dict[str, Any]) -> bool:
        index = choice.get("index", 0)
        index = index if isinstance(index, int) and not isinstance(index, bool) else 0
        slot = self.choices.setdefault(
            index,
            {"index": index, "message": {"role": "assistant"}, "finish_reason": None},
        )
        message = slot["message"]
        delta = choice.get("delta")
        output = False
        if isinstance(delta, dict):
            for key in ("content", "reasoning_content", "reasoning", "refusal"):
                piece = delta.get(key)
                if isinstance(piece, str) and piece:
                    message[key] = message.get(key, "") + piece
                    output = True
            role = delta.get("role")
            if isinstance(role, str):
                message["role"] = role
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                output = True
                calls = message.setdefault("tool_calls", [])
                for raw in tool_calls:
                    if not isinstance(raw, dict):
                        continue
                    call_index = raw.get("index", len(calls))
                    if not isinstance(call_index, int) or isinstance(call_index, bool):
                        call_index = len(calls)
                    while len(calls) <= call_index:
                        calls.append(
                            {"type": "function", "function": {"name": "", "arguments": ""}}
                        )
                    call = calls[call_index]
                    if isinstance(raw.get("id"), str):
                        call["id"] = raw["id"]
                    if isinstance(raw.get("type"), str):
                        call["type"] = raw["type"]
                    function = raw.get("function")
                    if isinstance(function, dict):
                        if isinstance(function.get("name"), str):
                            call["function"]["name"] += function["name"]
                        if isinstance(function.get("arguments"), str):
                            call["function"]["arguments"] += function["arguments"]
            function_call = delta.get("function_call")
            if isinstance(function_call, dict):
                output = True
                legacy = message.setdefault("function_call", {"name": "", "arguments": ""})
                if isinstance(function_call.get("name"), str):
                    legacy["name"] += function_call["name"]
                if isinstance(function_call.get("arguments"), str):
                    legacy["arguments"] += function_call["arguments"]
        text = choice.get("text")
        if isinstance(text, str) and text:
            slot["text"] = slot.get("text", "") + text
            output = True
        finish = choice.get("finish_reason")
        if isinstance(finish, str):
            slot["finish_reason"] = finish
        return output

    def result(self) -> list[dict[str, Any]]:
        return [self.choices[key] for key in sorted(self.choices)]


def _tool_names_from_choices(choices: Iterable[Any]) -> list[str]:
    names: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                function = call.get("function")
                if isinstance(function, dict) and isinstance(function.get("name"), str):
                    if function["name"]:
                        names.append(function["name"])
        legacy = message.get("function_call")
        if isinstance(legacy, dict) and isinstance(legacy.get("name"), str):
            if legacy["name"]:
                names.append(legacy["name"])
    return list(dict.fromkeys(names))


def _tool_names_from_response_output(output: Any) -> list[str]:
    names: list[str] = []
    for item in output if isinstance(output, list) else []:
        if (
            isinstance(item, dict)
            and item.get("type") in {"function_call", "custom_tool_call"}
            and isinstance(item.get("name"), str)
            and item["name"]
        ):
            names.append(item["name"])
    return list(dict.fromkeys(names))


def _finish_reasons(choices: Iterable[Any]) -> list[str]:
    return [
        choice["finish_reason"]
        for choice in choices
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str)
    ]


def _iter_decoded(
    chunks: Sequence[tuple[int, bytes]], decoder: _Decoder
) -> Iterator[tuple[int, bytes]]:
    for arrived_ns, chunk in chunks:
        yield arrived_ns, decoder.decode(chunk)


def analyze_response(
    route: str,
    *,
    status: int | None,
    content_type: str | None,
    content_encoding: str | None,
    chunks: Sequence[tuple[int, bytes]],
) -> ResponseTelemetry:
    """Extract usage, timing anchors, tools and an assembled response."""

    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    is_stream = media_type == "text/event-stream"
    telemetry = ResponseTelemetry(stream=is_stream)
    decoder = _Decoder(content_encoding)
    if not decoder.supported:
        telemetry.decodable = False
        telemetry.terminal = "undecodable_content_encoding"
        return telemetry
    try:
        if is_stream:
            _analyze_stream(route, chunks, decoder, telemetry)
        else:
            body = b"".join(chunk for _, chunk in _iter_decoded(chunks, decoder))
            _analyze_json(route, body, telemetry, status)
    except zlib.error:
        telemetry.decodable = False
        telemetry.terminal = "undecodable_content_encoding"
    return telemetry


def _analyze_json(
    route: str, body: bytes, telemetry: ResponseTelemetry, status: int | None
) -> None:
    try:
        payload = json.loads(body) if body else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        telemetry.parse_errors += 1
        telemetry.terminal = "unparseable_body"
        return
    if not isinstance(payload, dict):
        telemetry.terminal = "unparseable_body"
        return
    telemetry.assembled = payload
    error_status = status is not None and status >= 400
    error_only = (
        "error" in payload and "choices" not in payload and "output" not in payload
    )
    if error_status or error_only:
        telemetry.terminal = "error_body"
        telemetry.stream_error = payload.get("error")
        return
    telemetry.response_id = payload.get("id") if isinstance(payload.get("id"), str) else None
    telemetry.response_model = (
        payload.get("model") if isinstance(payload.get("model"), str) else None
    )
    _absorb_server_metrics(telemetry, payload)
    if route == "responses":
        _absorb_responses_usage(telemetry, payload.get("usage"))
        telemetry.invoked_tools = _tool_names_from_response_output(payload.get("output"))
        status_value = payload.get("status")
        telemetry.finish_reasons = [status_value] if isinstance(status_value, str) else []
        telemetry.output_sha256 = (
            _digest(payload.get("output")) if payload.get("output") is not None else None
        )
    else:
        _absorb_chat_usage(telemetry, payload.get("usage"))
        choices = payload.get("choices")
        choices = choices if isinstance(choices, list) else []
        telemetry.invoked_tools = _tool_names_from_choices(choices)
        telemetry.finish_reasons = _finish_reasons(choices)
        telemetry.output_sha256 = _digest(
            [
                {key: value for key, value in choice.items() if key != "logprobs"}
                for choice in choices
                if isinstance(choice, dict)
            ]
        )
    telemetry.terminal = "complete_body"


def _analyze_stream(
    route: str,
    chunks: Sequence[tuple[int, bytes]],
    decoder: _Decoder,
    telemetry: ResponseTelemetry,
) -> None:
    parser = SSEParser()
    events: list[SSEEvent] = []
    last_ns = chunks[-1][0] if chunks else 0
    for arrived_ns, chunk in _iter_decoded(chunks, decoder):
        events.extend(parser.feed(chunk, arrived_ns))
    events.extend(parser.finish(last_ns))
    assembler = _ChatAssembler()
    final_response: dict[str, Any] | None = None
    saw_done = False
    terminal_event: str | None = None
    for event in events:
        if event.data == "[DONE]":
            saw_done = True
            continue
        telemetry.events += 1
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError:
            telemetry.parse_errors += 1
            continue
        if not isinstance(payload, dict):
            continue
        if route == "responses":
            event_type = payload.get("type")
            if isinstance(event_type, str) and event_type.endswith(".delta"):
                if _non_empty(payload.get("delta")):
                    telemetry.output_event_times_ns.append(event.arrived_ns)
            if event_type in {"response.completed", "response.incomplete", "response.failed"}:
                terminal_event = event_type
                candidate = payload.get("response")
                if isinstance(candidate, dict):
                    final_response = candidate
            elif event_type == "error":
                telemetry.stream_error = payload.get("error", payload)
            continue
        if "error" in payload and not isinstance(payload.get("choices"), list):
            telemetry.stream_error = payload.get("error")
            continue
        if isinstance(payload.get("id"), str):
            telemetry.response_id = payload["id"]
        if isinstance(payload.get("model"), str):
            telemetry.response_model = payload["model"]
        if isinstance(payload.get("usage"), dict):
            _absorb_chat_usage(telemetry, payload["usage"])
        _absorb_server_metrics(telemetry, payload)
        choices = payload.get("choices")
        if isinstance(choices, list):
            output = False
            for choice in choices:
                if isinstance(choice, dict) and assembler.add(choice):
                    output = True
            if output:
                telemetry.output_event_times_ns.append(event.arrived_ns)

    if route == "responses":
        if final_response is not None:
            _absorb_responses_usage(telemetry, final_response.get("usage"))
            _absorb_server_metrics(telemetry, final_response)
            telemetry.invoked_tools = _tool_names_from_response_output(
                final_response.get("output")
            )
            status_value = final_response.get("status")
            telemetry.finish_reasons = (
                [status_value] if isinstance(status_value, str) else []
            )
            telemetry.response_id = (
                final_response.get("id")
                if isinstance(final_response.get("id"), str)
                else None
            )
            telemetry.response_model = (
                final_response.get("model")
                if isinstance(final_response.get("model"), str)
                else None
            )
            telemetry.assembled = final_response
            if final_response.get("output") is not None:
                telemetry.output_sha256 = _digest(final_response.get("output"))
        telemetry.terminal = (
            "stream_error"
            if telemetry.stream_error is not None
            else terminal_event or ("done_marker" if saw_done else "eof_without_terminal_event")
        )
        return

    choices_out = assembler.result()
    telemetry.finish_reasons = _finish_reasons(choices_out)
    telemetry.invoked_tools = _tool_names_from_choices(choices_out)
    telemetry.assembled = {
        "object": "chat.completion" if route == "chat_completions" else "text_completion",
        "id": telemetry.response_id,
        "model": telemetry.response_model,
        "choices": choices_out,
        "usage": None
        if telemetry.prompt_tokens is None and telemetry.completion_tokens is None
        else {
            key: value for key, value in telemetry.usage_dict().items() if value is not None
        },
        "assembled_from_stream": True,
    }
    if choices_out:
        telemetry.output_sha256 = _digest(choices_out)
    all_finished = bool(choices_out) and all(
        choice.get("finish_reason") is not None for choice in choices_out
    )
    if telemetry.stream_error is not None:
        telemetry.terminal = "stream_error"
    elif saw_done:
        telemetry.terminal = "done_marker"
    elif all_finished:
        telemetry.terminal = "finish_reason_without_done"
    else:
        telemetry.terminal = "eof_without_terminal_event"


def stream_timing(telemetry: ResponseTelemetry, *, received_ns: int) -> dict[str, Any]:
    """Client-visible streaming timing, measured at the proxy.

    ``client_ttft_ms`` runs from the moment the proxy held the complete client
    request to the arrival of the first output-bearing event. Role-only or
    empty deltas do not count. ``itl_ms`` summarizes intervals between
    successive output-bearing events. One event can carry more than one token,
    so ``tpot_ms`` is defined only when the upstream reported a completion
    token count. Non-streamed responses have no token-level timing.
    """

    times = telemetry.output_event_times_ns
    if not telemetry.stream or not times:
        return {
            "client_ttft_ms": None,
            "itl_ms": None,
            "tpot_ms": None,
            "output_event_count": len(times),
            "decode_window_ms": None,
        }
    first, last = times[0], times[-1]
    intervals = [(later - earlier) / 1e6 for earlier, later in zip(times, times[1:])]
    tpot = None
    tokens = telemetry.completion_tokens
    if tokens is not None and tokens >= 2:
        tpot = round((last - first) / 1e6 / (tokens - 1), 3)
    return {
        "client_ttft_ms": round((first - received_ns) / 1e6, 3),
        "itl_ms": summarize_intervals(intervals),
        "tpot_ms": tpot,
        "output_event_count": len(times),
        "decode_window_ms": round((last - first) / 1e6, 3),
    }
