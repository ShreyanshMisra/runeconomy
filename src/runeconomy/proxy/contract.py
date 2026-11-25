"""Wire contract for RunEconomy zero-code capture.

This module is the specification for the local capture proxy, and for any
future production gateway that has to produce interchangeable evidence. The
invariants below are normative. An implementation that cannot keep one of them
must say so in the session record and must not emit evidence that implies it
did.

Streaming transparency
    1. Each upstream body chunk is written to the client as soon as it arrives.
       The proxy never waits for a complete response, an SSE event boundary or
       a token count before forwarding bytes.
    2. Status codes, payload bytes and end-to-end headers reach the client
       unchanged. Hop-by-hop headers are regenerated as RFC 9110 requires. On
       inference routes the proxy asks the upstream for ``identity`` content
       encoding so the telemetry copy is decodable. Clients accept identity
       encoding, so the payload they see is the same.
    3. The proxy never adds request fields such as ``stream_options``,
       ``include_usage`` or sampling defaults, and it never rewrites prompts.
       Evidence the upstream does not report stays unknown.
    4. If the upstream fails part-way through a streamed body, the client
       connection is closed without a terminating chunk, so the client sees
       an error rather than a truncated "success".

Off-path emission
    5. The request path does only constant-time work per chunk: forward it,
       note one monotonic timestamp and keep a reference to the bytes it
       already holds. It parses no payloads, writes no files, resolves no
       tokens and makes no extra network calls.
    6. Handing an observation to the recorder never blocks. If the bounded
       intake is full, the observation is dropped and counted, and the session
       close record reports the loss. Evidence is never lost silently.
    7. Token resolution, identity probes, parsing, redaction and file writes
       all happen on recorder threads after the client response is complete.

Evidence semantics
    8. Exact prompt token IDs come only from the serving backend's own render
       or tokenize endpoint, or from a request that already carried token IDs.
       The proxy never tokenizes locally. When the backend reports a prompt
       token count that disagrees with the resolved IDs, the IDs are discarded
       and the request is marked ``unknown``.
    9. Prompt segment boundaries are ``exact_token_boundary`` only when a
       prefix render is verified to be a literal token prefix of the complete
       render. Anything else is a gap or structural attribution.
   10. Workflow dependencies (``depends_on``) come only from an explicit header
       assertion. Timing, ordering, overlap or idle gaps never produce an
       edge. Idle gaps only group requests into workflows, and are labelled as
       such. If uncorrelated requests overlap in flight, their idle-gap group
       is explicitly ambiguous and cannot produce a runnable workspace.
   11. Credentials, cookies and non-allowlisted headers are never recorded.
       Bodies pass through the configured redaction hooks before they are
       persisted. Recorded data is never uploaded. The only network peer is
       the upstream that the client itself addressed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping


PROXY_SESSION_VERSION = "0.1"
SESSION_SCHEMA_RESOURCE = "proxy-session-0.1.schema.json"

RECORD_SESSION = "runeconomy.proxy.session"
RECORD_EXCHANGE = "runeconomy.proxy.exchange"
RECORD_TOKEN_RESOLUTION = "runeconomy.proxy.token_resolution"
RECORD_UPSTREAM_IDENTITY = "runeconomy.proxy.upstream_identity"
RECORD_SESSION_CLOSE = "runeconomy.proxy.session_close"
RECORD_TYPES = frozenset(
    {
        RECORD_SESSION,
        RECORD_EXCHANGE,
        RECORD_TOKEN_RESOLUTION,
        RECORD_UPSTREAM_IDENTITY,
        RECORD_SESSION_CLOSE,
    }
)

# Correlation headers. Every header is optional, and a request without any of
# them falls back to idle-gap grouping and has no asserted dependencies.
HEADER_WORKFLOW_ID = "x-runeconomy-workflow-id"
HEADER_SPAN_ID = "x-runeconomy-span-id"
HEADER_DEPENDS_ON = "x-runeconomy-depends-on"
HEADER_TASK_ID = "x-runeconomy-task-id"
HEADER_ATTEMPT_ID = "x-runeconomy-attempt-id"
HEADER_TRACEPARENT = "traceparent"
CORRELATION_HEADERS = frozenset(
    {
        HEADER_WORKFLOW_ID,
        HEADER_SPAN_ID,
        HEADER_DEPENDS_ON,
        HEADER_TASK_ID,
        HEADER_ATTEMPT_ID,
        HEADER_TRACEPARENT,
    }
)

# Header allowlists. A header that is not named here is never persisted.
RECORDED_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        "user-agent",
        "openai-beta",
        "x-stainless-lang",
        "x-stainless-package-version",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-retry-count",
        *CORRELATION_HEADERS,
    }
)
RECORDED_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "x-request-id",
        "openai-model",
        "openai-processing-ms",
    }
)
# Never recorded, even if an allowlist is later widened by mistake.
FORBIDDEN_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "api-key",
        "x-api-key",
        "openai-organization",
        "openai-project",
    }
)
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Inference routes whose exchanges become evidence. Other routes pass through
# untouched and are only counted.
INFERENCE_ROUTES: Mapping[str, str] = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/responses": "responses",
}

GROUPING_RUNECONOMY_HEADER = "runeconomy_header"
GROUPING_TRACEPARENT = "w3c_traceparent"
GROUPING_IDLE_GAP = "idle_gap"
GROUPING_ATTEMPT_SCOPE = "attempt_scope"

OUTCOME_COMPLETED = "completed"
OUTCOME_UPSTREAM_HTTP_ERROR = "upstream_http_error"
OUTCOME_UPSTREAM_UNREACHABLE = "upstream_unreachable"
OUTCOME_UPSTREAM_INTERRUPTED = "upstream_interrupted"
OUTCOME_CLIENT_DISCONNECTED = "client_disconnected"

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+=-]{0,127}")
_TRACEPARENT = re.compile(
    r"([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(-.*)?"
)
_EMPTY_ASSERTION = frozenset({"", "none", "[]"})


def safe_identifier(value: str | None) -> str | None:
    """Return ``value`` if it is a bounded, printable identifier, else None."""

    if value is None:
        return None
    stripped = value.strip()
    return stripped if _SAFE_IDENTIFIER.fullmatch(stripped) else None


@dataclass(frozen=True)
class TraceParent:
    trace_id: str
    parent_id: str
    flags: str


def parse_traceparent(value: str | None) -> tuple[TraceParent | None, str | None]:
    """Parse a W3C ``traceparent`` header.

    Returns ``(parsed, error)``. An invalid value is reported, not repaired.
    """

    if value is None:
        return None, None
    match = _TRACEPARENT.fullmatch(value.strip().lower())
    if match is None:
        return None, "traceparent is not a W3C trace-context value"
    version, trace_id, parent_id, flags, extra = match.groups()
    if version == "ff":
        return None, "traceparent version ff is invalid"
    if version == "00" and extra:
        return None, "traceparent version 00 must not carry extra fields"
    if trace_id == "0" * 32 or parent_id == "0" * 16:
        return None, "traceparent contains an all-zero identifier"
    return TraceParent(trace_id=trace_id, parent_id=parent_id, flags=flags), None


def parse_depends_on(
    value: str | None,
) -> tuple[bool, tuple[str, ...], str | None]:
    """Parse an explicit dependency assertion.

    Returns ``(asserted, span_ids, error)``. A missing header asserts nothing.
    A header whose value is empty, ``none`` or ``[]`` asserts that the span has
    no dependencies, which is different from not knowing. An invalid assertion
    counts as not asserted, so nothing is guessed from a malformed value.
    """

    if value is None:
        return False, (), None
    raw = value.strip()
    if raw.lower() in _EMPTY_ASSERTION:
        return True, (), None
    items: list[str]
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return False, (), "depends-on header is not a valid JSON array"
        if not isinstance(parsed, list) or any(
            not isinstance(item, str) for item in parsed
        ):
            return False, (), "depends-on JSON value must be an array of strings"
        items = parsed
    else:
        items = [item for item in raw.split(",")]
    identifiers: list[str] = []
    for item in items:
        identifier = safe_identifier(item)
        if identifier is None:
            return False, (), "depends-on header contains an invalid span identifier"
        identifiers.append(identifier)
    return True, tuple(dict.fromkeys(identifiers)), None


def route_for_path(path: str) -> str | None:
    """Return the inference route name for a request path, ignoring a query."""

    bare = path.split("?", 1)[0].rstrip("/") or "/"
    return INFERENCE_ROUTES.get(bare)


def recorded_headers(
    headers: Mapping[str, str], allowlist: frozenset[str]
) -> dict[str, str]:
    """Return only allowlisted, non-forbidden headers with lower-case names."""

    result: dict[str, str] = {}
    for name, value in headers.items():
        key = name.lower()
        if key in allowlist and key not in FORBIDDEN_HEADERS:
            result[key] = value
    return result
