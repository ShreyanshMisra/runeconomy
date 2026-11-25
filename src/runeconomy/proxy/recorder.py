"""Off-path session recording for the capture proxy.

The request path hands each finished exchange to :meth:`SessionRecorder.submit`,
which never blocks. Recorder threads then parse, redact and persist the
exchange, and resolve exact prompt tokens against the upstream. One writer
thread owns the file, so records are never interleaved.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import secrets
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .. import __version__
from ..privacy import _BEARER, _SENSITIVE_KEY
from .contract import (
    HEADER_ATTEMPT_ID,
    HEADER_DEPENDS_ON,
    HEADER_SPAN_ID,
    HEADER_TASK_ID,
    HEADER_TRACEPARENT,
    HEADER_WORKFLOW_ID,
    OUTCOME_COMPLETED,
    PROXY_SESSION_VERSION,
    RECORD_EXCHANGE,
    RECORD_SESSION,
    RECORD_SESSION_CLOSE,
    RECORD_TOKEN_RESOLUTION,
    RECORD_UPSTREAM_IDENTITY,
    RECORDED_REQUEST_HEADERS,
    RECORDED_RESPONSE_HEADERS,
    parse_depends_on,
    parse_traceparent,
    recorded_headers,
    safe_identifier,
)
from .stream import analyze_response, stream_timing
from .tokens import TokenResolver
from .upstream import OffPathClient, UpstreamTarget, probe_upstream


TOKEN_RESOLUTION_MODES = frozenset({"background", "on_close", "off"})
PAYLOAD_CAPTURE_MODES = frozenset({"full", "digest_only"})
_ERROR_EXCERPT_BYTES = 2048


class RecorderError(RuntimeError):
    """The recorder cannot start or was used outside its lifecycle."""


# -- redaction -------------------------------------------------------------

RedactionHook = Callable[[str, Any], Any]
"""``hook(part, value) -> value``. ``part`` is ``request_body`` or ``response_body``.

A hook receives a deep copy of the parsed JSON value and returns the value to
persist. It may mutate and return its argument. Hooks run on recorder threads,
never on the request path, and never change what reaches the upstream or the
client.
"""

_CREDENTIAL_VALUE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|xox[abposr]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16})\b"
)
REDACTED = "[REDACTED]"


def _builtin_redact(value: Any, path: str, notes: list[str]) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                isinstance(key, str)
                and isinstance(child, str)
                and child
                and _SENSITIVE_KEY.search(key)
            ):
                result[key] = REDACTED
                notes.append(child_path)
            else:
                result[key] = _builtin_redact(child, child_path, notes)
        return result
    if isinstance(value, list):
        return [
            _builtin_redact(child, f"{path}[{index}]", notes)
            for index, child in enumerate(value)
        ]
    if isinstance(value, str):
        masked = _BEARER.sub("Bearer " + REDACTED, value)
        masked = _CREDENTIAL_VALUE.sub(REDACTED, masked)
        if masked != value:
            notes.append(path)
        return masked
    return value


@dataclass(frozen=True)
class RedactionPolicy:
    """Built-in credential masking plus optional operator hooks.

    The built-in pass masks string values under credential-like keys, as well
    as bearer tokens and well-known API key formats inside any string. It does
    not touch JSON-schema objects such as a tool parameter called ``api_key``.
    Operator hooks run after it, in order.
    """

    hooks: tuple[RedactionHook, ...] = ()
    builtin: bool = True
    name: str = "builtin_credentials"

    def apply(self, part: str, value: Any) -> tuple[Any, dict[str, Any]]:
        notes: list[str] = []
        original = _canonical(value)
        result = _builtin_redact(value, "$", notes) if self.builtin else json.loads(original)
        for hook in self.hooks:
            result = hook(part, result)
        modified = _canonical(result) != original
        return result, {
            "policy": self.name,
            "builtin": self.builtin,
            "hooks": len(self.hooks),
            "modified": modified,
            "builtin_paths": notes[:64],
        }


def load_redaction_hook(spec: str) -> RedactionHook:
    """Load ``package.module:function`` as a redaction hook. Trusted local code only."""

    import importlib

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise RecorderError("redaction hook must be written as module:function")
    module = importlib.import_module(module_name)
    hook = getattr(module, attribute, None)
    if not callable(hook):
        raise RecorderError(f"redaction hook {spec!r} is not callable")
    return hook


# -- observations ----------------------------------------------------------


@dataclass
class Observation:
    """Everything the request path learned about one exchange.

    The request path only fills fields and appends chunk references. All
    interpretation happens later on recorder threads.
    """

    exchange_id: str
    sequence: int
    method: str
    path: str
    route: str | None
    received_unix_ns: int
    headers_received_ns: int
    received_ns: int
    request_headers: list[tuple[str, str]]
    request_body: bytes
    upstream_sent_ns: int | None = None
    response_headers_ns: int | None = None
    first_byte_ns: int | None = None
    completed_ns: int | None = None
    status: int | None = None
    response_headers: list[tuple[str, str]] = field(default_factory=list)
    chunks: list[tuple[int, bytes]] = field(default_factory=list)
    outcome: str = OUTCOME_COMPLETED
    error: str | None = None
    upstream_retried: bool = False


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _header_map(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in headers:
        key = name.lower()
        result[key] = f"{result[key]}, {value}" if key in result else value
    return result


def _round_ms(value_ns: int | None, anchor_ns: int) -> float | None:
    return None if value_ns is None else round((value_ns - anchor_ns) / 1e6, 3)


def _duration_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return round((end_ns - start_ns) / 1e6, 3)


def correlation_from_headers(headers: Mapping[str, str]) -> dict[str, Any]:
    """Interpret correlation headers without guessing anything they omit."""

    errors: list[str] = []
    workflow_raw = headers.get(HEADER_WORKFLOW_ID)
    workflow_id = safe_identifier(workflow_raw)
    if workflow_raw is not None and workflow_id is None:
        errors.append(f"{HEADER_WORKFLOW_ID} is not a safe identifier")
    span_raw = headers.get(HEADER_SPAN_ID)
    span_id = safe_identifier(span_raw)
    if span_raw is not None and span_id is None:
        errors.append(f"{HEADER_SPAN_ID} is not a safe identifier")
    asserted, depends_on, depends_error = parse_depends_on(headers.get(HEADER_DEPENDS_ON))
    if depends_error:
        errors.append(depends_error)
    traceparent, traceparent_error = parse_traceparent(headers.get(HEADER_TRACEPARENT))
    if traceparent_error:
        errors.append(traceparent_error)
    task_raw = headers.get(HEADER_TASK_ID)
    task_id = safe_identifier(task_raw)
    if task_raw is not None and task_id is None:
        errors.append(f"{HEADER_TASK_ID} is not a safe identifier")
    attempt_raw = headers.get(HEADER_ATTEMPT_ID)
    attempt_id = safe_identifier(attempt_raw)
    if attempt_raw is not None and attempt_id is None:
        errors.append(f"{HEADER_ATTEMPT_ID} is not a safe identifier")
    return {
        "workflow_id": workflow_id,
        "traceparent": None
        if traceparent is None
        else {
            "trace_id": traceparent.trace_id,
            "parent_id": traceparent.parent_id,
            "flags": traceparent.flags,
        },
        "span_id": span_id,
        "depends_on": {"asserted": asserted, "span_ids": list(depends_on)},
        "task_id": task_id,
        "attempt_id": attempt_id,
        "header_errors": errors,
    }


# -- recorder --------------------------------------------------------------

_FLUSH = object()


@dataclass(frozen=True)
class _Stop:
    summary: dict[str, Any]


class SessionRecorder:
    """Persist capture evidence as an append-only JSONL session."""

    def __init__(
        self,
        path: Path,
        *,
        upstream: UpstreamTarget,
        token_resolution: str = "background",
        payload_capture: str = "full",
        redaction: RedactionPolicy | None = None,
        max_pending: int = 1024,
        resolver_workers: int = 1,
        resolver_timeout: float = 30.0,
        resolver_authorization: str | None = None,
        segment_prompts: bool = True,
        max_segment_messages: int = 256,
        session_id: str | None = None,
        scope: Mapping[str, Any] | None = None,
        configuration: Mapping[str, Any] | None = None,
        on_record: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if token_resolution not in TOKEN_RESOLUTION_MODES:
            raise RecorderError(
                f"token_resolution must be one of {sorted(TOKEN_RESOLUTION_MODES)}"
            )
        if payload_capture not in PAYLOAD_CAPTURE_MODES:
            raise RecorderError(
                f"payload_capture must be one of {sorted(PAYLOAD_CAPTURE_MODES)}"
            )
        if max_pending <= 0 or resolver_workers <= 0:
            raise RecorderError("max_pending and resolver_workers must be positive")
        self.path = Path(path)
        self.upstream = upstream
        self.token_resolution = token_resolution
        self.payload_capture = payload_capture
        self.redaction = redaction or RedactionPolicy()
        self.max_pending = max_pending
        self.session_id = session_id or "proxy-" + secrets.token_hex(6)
        self.scope = dict(scope) if scope else {"kind": "session"}
        self.configuration = dict(configuration or {})
        self._resolver_authorization = resolver_authorization
        # Called on the writer thread after each record is persisted, so a CLI
        # can show live progress. It never runs on the request path, and an
        # exception it raises is swallowed.
        self._on_record = on_record
        self._client = OffPathClient(upstream, timeout=resolver_timeout)
        self._resolver = TokenResolver(
            self._client,
            segment=segment_prompts,
            max_segment_messages=max_segment_messages,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=resolver_workers, thread_name_prefix="runeconomy-resolve"
        )
        self._queue: queue.Queue[Any] = queue.Queue()
        self._pending_observations = 0
        self._lock = threading.Lock()
        self._sequence = 0
        self._writer: threading.Thread | None = None
        self._handle: Any = None
        self._flushed = threading.Event()
        self._closed = threading.Event()
        self._closing = False
        self._deferred: list[tuple[Callable[..., dict[str, Any]], tuple[Any, ...]]] = []
        self._futures: list[Future[Any]] = []
        self._probe_scheduled = False
        self.anchor_ns = time.perf_counter_ns()
        self.anchor_unix_ns = time.time_ns()
        self.counters = {
            "exchanges_observed": 0,
            "exchanges_recorded": 0,
            "observations_dropped": 0,
            "observations_after_close": 0,
            "passthrough_requests": 0,
            "token_resolution_exact": 0,
            "token_resolution_unknown": 0,
            "token_resolution_incomplete_at_close": 0,
            "record_errors": 0,
        }

    # -- lifecycle -------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._writer is not None

    def start(self) -> "SessionRecorder":
        if self._writer is not None:
            raise RecorderError("recorder already started")
        try:
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as exc:
            raise RecorderError(
                f"{self.path} already exists; capture sessions are never overwritten"
            ) from exc
        self._handle = os.fdopen(descriptor, "w", encoding="utf-8")
        self._write(self._session_record())
        self._writer = threading.Thread(
            target=self._run, name="runeconomy-recorder", daemon=True
        )
        self._writer.start()
        return self

    def next_identity(self) -> tuple[int, str]:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        return sequence, f"{self.session_id}-{sequence:06d}"

    def note_passthrough(self) -> None:
        with self._lock:
            self.counters["passthrough_requests"] += 1

    def submit(self, observation: Observation) -> bool:
        """Queue an observation without blocking. Returns False if it was dropped."""

        with self._lock:
            self.counters["exchanges_observed"] += 1
            if self._closing:
                self.counters["observations_after_close"] += 1
                return False
            if self._pending_observations >= self.max_pending:
                self.counters["observations_dropped"] += 1
                return False
            self._pending_observations += 1
            # Enqueue under the lock so that no observation can land behind
            # the close-time flush marker. The queue is unbounded, so this
            # never blocks; the bound is enforced by the counter above.
            self._queue.put_nowait(observation)
        return True

    def close(self, *, resolution_timeout: float = 120.0) -> dict[str, Any]:
        """Drain observations, finish token resolution and write the close record."""

        if self._writer is None:
            raise RecorderError("recorder was not started")
        with self._lock:
            if self._closing:
                self._closed.wait()
                return dict(self.counters)
            self._closing = True
        self._queue.put(_FLUSH)
        self._flushed.wait()
        if self.token_resolution == "on_close":
            for function, arguments in self._deferred:
                self._futures.append(self._executor.submit(self._job, function, arguments))
            self._deferred.clear()
        _, not_done = wait(self._futures, timeout=resolution_timeout)
        for future in not_done:
            future.cancel()
        with self._lock:
            self.counters["token_resolution_incomplete_at_close"] = len(not_done)
        self._executor.shutdown(wait=False, cancel_futures=True)
        summary = {
            "unavailable_backend_endpoints": self._resolver.unavailable_endpoints(),
        }
        self._queue.put(_Stop(summary))
        self._writer.join()
        self._client.close()
        self._closed.set()
        return dict(self.counters)

    # -- writer thread ---------------------------------------------------

    def _write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        self._handle.flush()
        if self._on_record is not None:
            try:
                self._on_record(record)
            except Exception:  # noqa: BLE001 - observers must not break capture
                pass

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _FLUSH:
                self._flushed.set()
                continue
            if isinstance(item, _Stop):
                with self._lock:
                    counters = dict(self.counters)
                self._write(
                    {
                        "record_type": RECORD_SESSION_CLOSE,
                        "session_version": PROXY_SESSION_VERSION,
                        "session_id": self.session_id,
                        "closed_unix_ns": time.time_ns(),
                        "closed_ms": _round_ms(time.perf_counter_ns(), self.anchor_ns),
                        "clean": True,
                        "counters": counters,
                        **item.summary,
                    }
                )
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self._handle.close()
                return
            try:
                if isinstance(item, Observation):
                    with self._lock:
                        self._pending_observations -= 1
                    self._record_observation(item)
                elif isinstance(item, dict):
                    self._write(item)
            except Exception as exc:  # noqa: BLE001 - recording must not die
                with self._lock:
                    self.counters["record_errors"] += 1
                try:
                    self._write(
                        {
                            "record_type": RECORD_EXCHANGE,
                            "session_id": self.session_id,
                            "exchange_id": getattr(item, "exchange_id", None),
                            "sequence": getattr(item, "sequence", None),
                            "record_error": type(exc).__name__,
                            "outcome": "record_error",
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass

    def _session_record(self) -> dict[str, Any]:
        return {
            "record_type": RECORD_SESSION,
            "session_version": PROXY_SESSION_VERSION,
            "session_id": self.session_id,
            "runeconomy_version": __version__,
            "started_unix_ns": self.anchor_unix_ns,
            "clock": {
                "timeline": "monotonic milliseconds since session start",
                "wall_anchor_unix_ns": self.anchor_unix_ns,
            },
            "upstream": self.upstream.describe(),
            "scope": self.scope,
            "capture": {
                "token_resolution": self.token_resolution,
                "payload_capture": self.payload_capture,
                "redaction_policy": self.redaction.name,
                "redaction_hooks": len(self.redaction.hooks),
                "max_pending_observations": self.max_pending,
                "recorded_request_headers": sorted(RECORDED_REQUEST_HEADERS),
                "recorded_response_headers": sorted(RECORDED_RESPONSE_HEADERS),
            },
            "configuration": self.configuration,
        }

    def _record_observation(self, observation: Observation) -> None:
        anchor = self.anchor_ns
        request_headers = _header_map(observation.request_headers)
        response_headers = _header_map(observation.response_headers)
        authorization = request_headers.get("authorization")
        correlation = correlation_from_headers(request_headers)

        raw_body = observation.request_body
        body: Any = None
        body_format = "empty" if not raw_body else "json"
        if raw_body:
            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body_format = "non_json"

        route = observation.route
        telemetry = analyze_response(
            route or "unknown",
            status=observation.status,
            content_type=response_headers.get("content-type"),
            content_encoding=response_headers.get("content-encoding"),
            chunks=observation.chunks,
        )
        timing = stream_timing(telemetry, received_ns=observation.received_ns)
        raw_response = b"".join(chunk for _, chunk in observation.chunks)

        request_payload: Any = None
        response_payload: Any = None
        request_redaction: dict[str, Any] | None = None
        response_redaction: dict[str, Any] | None = None
        error_excerpt: str | None = None
        if self.payload_capture == "full":
            if body is not None:
                request_payload, request_redaction = self.redaction.apply(
                    "request_body", json.loads(json.dumps(body))
                )
            if telemetry.assembled is not None:
                response_payload, response_redaction = self.redaction.apply(
                    "response_body", json.loads(json.dumps(telemetry.assembled, default=str))
                )
            elif observation.status is not None and observation.status >= 400 and raw_response:
                excerpt, _ = self.redaction.apply(
                    "response_body",
                    raw_response[:_ERROR_EXCERPT_BYTES].decode("utf-8", errors="replace"),
                )
                error_excerpt = excerpt if isinstance(excerpt, str) else None

        stream_requested = isinstance(body, dict) and body.get("stream") is True
        include_usage = (
            isinstance(body, dict)
            and isinstance(body.get("stream_options"), dict)
            and body["stream_options"].get("include_usage") is True
        )
        record = {
            "record_type": RECORD_EXCHANGE,
            "session_id": self.session_id,
            "exchange_id": observation.exchange_id,
            "sequence": observation.sequence,
            "method": observation.method,
            "path": observation.path,
            "route": route,
            "received_unix_ns": observation.received_unix_ns,
            "outcome": observation.outcome,
            "error": observation.error,
            "upstream_retried_stale_connection": observation.upstream_retried,
            "timing": {
                "headers_received_ms": _round_ms(observation.headers_received_ns, anchor),
                "request_received_ms": _round_ms(observation.received_ns, anchor),
                "upstream_sent_ms": _round_ms(observation.upstream_sent_ns, anchor),
                "response_headers_ms": _round_ms(observation.response_headers_ns, anchor),
                "first_byte_ms": _round_ms(observation.first_byte_ns, anchor),
                "first_output_ms": _round_ms(
                    telemetry.output_event_times_ns[0]
                    if telemetry.output_event_times_ns
                    else None,
                    anchor,
                ),
                "last_output_ms": _round_ms(
                    telemetry.output_event_times_ns[-1]
                    if telemetry.output_event_times_ns
                    else None,
                    anchor,
                ),
                "completed_ms": _round_ms(observation.completed_ns, anchor),
            },
            "correlation": correlation,
            "request": {
                "headers": recorded_headers(request_headers, RECORDED_REQUEST_HEADERS),
                "body_format": body_format,
                "body_bytes": len(raw_body),
                "body_sha256": _digest_bytes(raw_body) if raw_body else None,
                "model": body.get("model") if isinstance(body, dict) and isinstance(body.get("model"), str) else None,
                "stream": stream_requested,
                "stream_include_usage": include_usage,
                "body": request_payload,
                "redaction": request_redaction,
            },
            "response": {
                "status": observation.status,
                "headers": recorded_headers(response_headers, RECORDED_RESPONSE_HEADERS),
                "content_type": response_headers.get("content-type"),
                "content_encoding": response_headers.get("content-encoding"),
                "body_bytes": len(raw_response),
                "body_sha256": _digest_bytes(raw_response) if raw_response else None,
                "body": response_payload,
                "error_excerpt": error_excerpt,
                "redaction": response_redaction,
            },
            "telemetry": {
                "stream": telemetry.stream,
                "decodable": telemetry.decodable,
                "stream_termination": telemetry.terminal,
                "stream_error": telemetry.stream_error is not None,
                "parse_errors": telemetry.parse_errors,
                "event_count": telemetry.events,
                "usage": telemetry.usage_dict(),
                "server_metrics": telemetry.server_metrics,
                "finish_reasons": telemetry.finish_reasons,
                "invoked_tools": telemetry.invoked_tools,
                "response_id": telemetry.response_id,
                "response_model": telemetry.response_model,
                "output_sha256": telemetry.output_sha256,
                "client_ttft_ms": timing["client_ttft_ms"],
                "itl_ms": timing["itl_ms"],
                "tpot_ms": timing["tpot_ms"],
                "output_event_count": timing["output_event_count"],
                "decode_window_ms": timing["decode_window_ms"],
                "time_to_response_headers_ms": _duration_ms(
                    observation.received_ns, observation.response_headers_ns
                ),
                "proxy_overhead_ms": _duration_ms(
                    observation.received_ns, observation.upstream_sent_ns
                ),
                "total_latency_ms": _duration_ms(
                    observation.received_ns, observation.completed_ns
                ),
            },
        }
        self._write(record)
        with self._lock:
            self.counters["exchanges_recorded"] += 1

        resolvable = (
            route is not None
            and observation.outcome == OUTCOME_COMPLETED
            and observation.status is not None
            and 200 <= observation.status < 300
            and isinstance(body, dict)
        )
        if not resolvable or self.token_resolution == "off":
            return
        credential = self._resolver_authorization or authorization
        if not self._probe_scheduled:
            self._probe_scheduled = True
            self._schedule(self._probe, (credential,))
        self._schedule(
            self._resolve,
            (observation.exchange_id, route, body, credential, telemetry.prompt_tokens),
        )

    # -- background jobs -------------------------------------------------

    def _schedule(self, function: Callable[..., dict[str, Any]], arguments: tuple[Any, ...]) -> None:
        if self.token_resolution == "on_close":
            self._deferred.append((function, arguments))
            return
        self._futures.append(self._executor.submit(self._job, function, arguments))
        if len(self._futures) > 4096:
            self._futures = [item for item in self._futures if not item.done()]

    def _job(self, function: Callable[..., dict[str, Any]], arguments: Sequence[Any]) -> None:
        try:
            record = function(*arguments)
        except Exception as exc:  # noqa: BLE001 - a failed resolution is evidence
            record = None
            if function == self._resolve:
                record = {
                    "record_type": RECORD_TOKEN_RESOLUTION,
                    "session_id": self.session_id,
                    "exchange_id": arguments[0],
                    **TokenResolver._unknown(f"resolver raised {type(exc).__name__}"),
                }
                with self._lock:
                    self.counters["token_resolution_unknown"] += 1
        if record is not None:
            self._queue.put(record)

    def _probe(self, authorization: str | None) -> dict[str, Any]:
        return {
            "record_type": RECORD_UPSTREAM_IDENTITY,
            "session_id": self.session_id,
            "probed_ms": _round_ms(time.perf_counter_ns(), self.anchor_ns),
            **probe_upstream(self._client, authorization=authorization),
        }

    def _resolve(
        self,
        exchange_id: str,
        route: str,
        body: Any,
        authorization: str | None,
        usage_prompt_tokens: int | None,
    ) -> dict[str, Any]:
        result = self._resolver.resolve(
            route,
            body,
            authorization=authorization,
            usage_prompt_tokens=usage_prompt_tokens,
        )
        with self._lock:
            key = (
                "token_resolution_exact"
                if result.get("status") == "exact"
                else "token_resolution_unknown"
            )
            self.counters[key] += 1
        return {
            "record_type": RECORD_TOKEN_RESOLUTION,
            "session_id": self.session_id,
            "exchange_id": exchange_id,
            **result,
        }
