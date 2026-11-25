"""Turn a recorded capture session into normalized traces and evidence bundles.

The session file is the durable evidence. Everything here is a deterministic
function of that file plus explicit operator input, so ``import`` can be
re-run with a different idle gap or replay context without capturing again.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..analysis import analyze_trace
from ..evidence_quality import build_evidence_quality, evidence_field
from ..experiment_init import AUTOMATABLE_TEMPLATES, initialize_experiment
from ..experiments import ExperimentValidationError
from ..importers import _structural_segments, _tool_name
from ..models import Trace, TraceValidationError
from .contract import (
    GROUPING_ATTEMPT_SCOPE,
    GROUPING_IDLE_GAP,
    GROUPING_RUNECONOMY_HEADER,
    GROUPING_TRACEPARENT,
    OUTCOME_COMPLETED,
    PROXY_SESSION_VERSION,
    RECORD_EXCHANGE,
    RECORD_SESSION,
    RECORD_SESSION_CLOSE,
    RECORD_TOKEN_RESOLUTION,
    RECORD_UPSTREAM_IDENTITY,
)


DEFAULT_IDLE_GAP_MS = 10_000.0
EVIDENCE_BUNDLE_VERSION = "0.1"
SOURCE_KIND = "runeconomy_proxy"
_IDENTITY_FIELDS = (
    "model_revision",
    "backend_version",
    "tokenizer",
    "tokenizer_revision",
    "chat_template_digest",
)
SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "length_penalty",
        "stop",
        "stop_token_ids",
        "n",
        "best_of",
        "use_beam_search",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "min_tokens",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    }
)
_RUNNER_CONTROLLED = ("stream", "stream_options", "cache_salt")
_STATEFUL = ("previous_response_id", "background")
_LITERAL_PLACEHOLDER = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
_OPERATOR_CONTEXT_KEYS = frozenset(
    {
        "backend",
        "sampling",
        "execution",
        "acceptance",
        "evaluator",
        "candidate_request",
        "candidate_tool_names",
        "baseline_request",
        "confirmed_capabilities",
        "experiment_id",
        "case_id",
        "constants",
        "experiment_template_id",
        "candidate_strategy",
    }
)
TRIAL_PLACEHOLDER = "{{RUNECONOMY_TRIAL_ID}}"
CANDIDATE_STRATEGY_MOVE_VOLATILE = "move_volatile_messages_after_system_block"


# -- loading ----------------------------------------------------------------


@dataclass
class ProxySession:
    path: Path
    sha256: str
    header: dict[str, Any]
    exchanges: list[dict[str, Any]]
    resolutions: dict[str, dict[str, Any]]
    identity: dict[str, Any] | None
    close: dict[str, Any] | None
    diagnostics: list[str] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        return str(self.header.get("session_id", "proxy-session"))

    @property
    def scope(self) -> dict[str, Any]:
        scope = self.header.get("scope")
        return scope if isinstance(scope, dict) else {"kind": "session"}


def is_proxy_session(path: Path) -> bool:
    """Return True if ``path`` starts with a RunEconomy proxy session record."""

    try:
        with Path(path).open("rb") as handle:
            first = handle.readline(1024 * 1024)
    except OSError:
        return False
    try:
        value = json.loads(first)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(value, dict) and value.get("record_type") == RECORD_SESSION


def load_session(path: Path) -> ProxySession:
    """Load a session, tolerating only a truncated final line from a crash."""

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TraceValidationError(f"cannot read {path}: {exc}") from exc
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    header: dict[str, Any] | None = None
    exchanges: dict[str, dict[str, Any]] = {}
    resolutions: dict[str, dict[str, Any]] = {}
    identity: dict[str, Any] | None = None
    close: dict[str, Any] | None = None
    diagnostics: list[str] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines):
                diagnostics.append(f"line {number}: truncated final record ignored")
                continue
            raise TraceValidationError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise TraceValidationError(f"{path}:{number}: record must be an object")
        kind = record.get("record_type")
        if number == 1 or header is None:
            if kind != RECORD_SESSION:
                raise TraceValidationError(
                    f"{path}: first record must be {RECORD_SESSION!r}"
                )
            if record.get("session_version") != PROXY_SESSION_VERSION:
                raise TraceValidationError(
                    f"{path}: unsupported proxy session version "
                    f"{record.get('session_version')!r}"
                )
            header = record
            continue
        if record.get("session_id") != header.get("session_id"):
            raise TraceValidationError(
                f"{path}:{number}: record belongs to a different session"
            )
        if kind == RECORD_EXCHANGE:
            exchange_id = record.get("exchange_id")
            if not isinstance(exchange_id, str):
                diagnostics.append(f"line {number}: exchange without exchange_id ignored")
                continue
            if exchange_id in exchanges:
                raise TraceValidationError(
                    f"{path}:{number}: duplicate exchange_id {exchange_id!r}"
                )
            exchanges[exchange_id] = record
        elif kind == RECORD_TOKEN_RESOLUTION:
            exchange_id = record.get("exchange_id")
            if isinstance(exchange_id, str):
                if exchange_id in resolutions:
                    diagnostics.append(
                        f"line {number}: later token resolution for {exchange_id} superseded the earlier one"
                    )
                resolutions[exchange_id] = record
        elif kind == RECORD_UPSTREAM_IDENTITY:
            if identity is None:
                identity = record
        elif kind == RECORD_SESSION_CLOSE:
            close = record
        else:
            diagnostics.append(f"line {number}: unknown record_type {kind!r} ignored")
    if header is None:
        raise TraceValidationError(f"{path}: no proxy session record found")
    if close is None:
        diagnostics.append("session has no close record; capture may have been interrupted")
    ordered = sorted(
        exchanges.values(),
        key=lambda item: (
            item.get("sequence") if isinstance(item.get("sequence"), int) else 0,
            str(item.get("exchange_id")),
        ),
    )
    return ProxySession(
        path=path,
        sha256=digest,
        header=header,
        exchanges=ordered,
        resolutions=resolutions,
        identity=identity,
        close=close,
        diagnostics=diagnostics,
    )


# -- grouping ---------------------------------------------------------------


@dataclass
class Workflow:
    workflow_id: str
    grouping: str
    grouping_key: str | None
    exchanges: list[dict[str, Any]]
    idle_gap_ms: float | None = None
    overlap_count: int = 0
    overlapping_exchange_ids: tuple[str, ...] = ()

    @property
    def ambiguous(self) -> bool:
        """Whether idle-gap grouping merged overlapping uncorrelated requests."""

        return self.grouping == GROUPING_IDLE_GAP and self.overlap_count > 0


def _timing(exchange: Mapping[str, Any], name: str) -> float | None:
    timing = exchange.get("timing")
    value = timing.get(name) if isinstance(timing, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _start(exchange: Mapping[str, Any]) -> float:
    return _timing(exchange, "request_received_ms") or 0.0


def _end(exchange: Mapping[str, Any]) -> float:
    return _timing(exchange, "completed_ms") or _start(exchange)


def _correlation(exchange: Mapping[str, Any]) -> dict[str, Any]:
    value = exchange.get("correlation")
    return value if isinstance(value, dict) else {}


def _is_inference(exchange: Mapping[str, Any]) -> bool:
    return isinstance(exchange.get("route"), str) and exchange.get("outcome") != "record_error"


def group_workflows(
    session: ProxySession, *, idle_gap_ms: float = DEFAULT_IDLE_GAP_MS
) -> list[Workflow]:
    """Group exchanges into workflows. Grouping never implies dependencies.

    The precedence is: attempt scope (one runner attempt is one workflow),
    then ``x-runeconomy-workflow-id``, then the W3C ``traceparent`` trace ID.
    Exchanges with none of these are split wherever no exchange was in flight
    for more than ``idle_gap_ms``.
    """

    if idle_gap_ms <= 0:
        raise TraceValidationError("idle_gap_ms must be positive")
    exchanges = [item for item in session.exchanges if _is_inference(item)]
    scope = session.scope
    if scope.get("kind") == "attempt":
        attempt = scope.get("attempt_id") or session.session_id
        return (
            [
                Workflow(
                    workflow_id=f"attempt-{attempt}",
                    grouping=GROUPING_ATTEMPT_SCOPE,
                    grouping_key=str(attempt),
                    exchanges=sorted(exchanges, key=_start),
                )
            ]
            if exchanges
            else []
        )
    keyed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    unkeyed: list[dict[str, Any]] = []
    for exchange in exchanges:
        correlation = _correlation(exchange)
        workflow_id = correlation.get("workflow_id")
        traceparent = correlation.get("traceparent")
        if isinstance(workflow_id, str):
            keyed.setdefault((GROUPING_RUNECONOMY_HEADER, workflow_id), []).append(exchange)
        elif isinstance(traceparent, dict) and isinstance(traceparent.get("trace_id"), str):
            keyed.setdefault((GROUPING_TRACEPARENT, traceparent["trace_id"]), []).append(
                exchange
            )
        else:
            unkeyed.append(exchange)
    workflows: list[Workflow] = []
    used_ids: set[str] = set()

    def unique(candidate: str) -> str:
        result = candidate
        counter = 2
        while result in used_ids:
            result = f"{candidate}-{counter}"
            counter += 1
        used_ids.add(result)
        return result

    for (method, key), members in keyed.items():
        workflows.append(
            Workflow(
                workflow_id=unique(key),
                grouping=method,
                grouping_key=key,
                exchanges=sorted(members, key=_start),
            )
        )
    unkeyed.sort(key=_start)
    groups: list[list[dict[str, Any]]] = []
    group_overlap_counts: list[int] = []
    group_overlapping_ids: list[set[str]] = []
    active: list[dict[str, Any]] = []
    horizon: float | None = None
    for exchange in unkeyed:
        start = _start(exchange)
        if not groups or horizon is None or start - horizon > idle_gap_ms:
            groups.append([])
            group_overlap_counts.append(0)
            group_overlapping_ids.append(set())
            active = []
            horizon = None
        active = [item for item in active if _end(item) > start]
        if active:
            group_overlap_counts[-1] += len(active)
            for item in (*active, exchange):
                exchange_id = item.get("exchange_id")
                if isinstance(exchange_id, str):
                    group_overlapping_ids[-1].add(exchange_id)
        groups[-1].append(exchange)
        if _end(exchange) > start:
            active.append(exchange)
        horizon = max(_end(exchange), horizon if horizon is not None else _end(exchange))
    for index, members in enumerate(groups, 1):
        workflows.append(
            Workflow(
                workflow_id=unique(f"{session.session_id}-idle-{index:03d}"),
                grouping=GROUPING_IDLE_GAP,
                grouping_key=None,
                exchanges=members,
                idle_gap_ms=idle_gap_ms,
                overlap_count=group_overlap_counts[index - 1],
                overlapping_exchange_ids=tuple(
                    sorted(group_overlapping_ids[index - 1])
                ),
            )
        )
    workflows.sort(key=lambda item: _start(item.exchanges[0]))
    return workflows


# -- volatile-content labelling ----------------------------------------------

_VOLATILE = re.compile(
    r"""(?ix)
    \d{4}-\d{2}-\d{2}(?:[T\ ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:z|[+-]\d{2}:?\d{2})?
    | \d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\ ?[ap]m)?
    | \d{1,2}/\d{1,2}/\d{2,4}
    | [0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}
    | (?=[0-9a-f]*\d)[0-9a-f]{8,}
    | \d+(?:\.\d+)?
    | [a-z]+[-_](?=[a-z0-9]*\d)[a-z0-9]{6,}
    | (?:mon|tue|wed|thu|fri|sat|sun)[a-z]{0,6}
    | (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]{0,6}
    """
)
_TOKENS = re.compile(r"\w+|[^\w\s]+|\s+")
_EDGE_PUNCTUATION = "()[]{}<>'\"`,.;:!?"
_MAX_VOLATILE_SPAN = 64
_MAX_VOLATILE_REGIONS = 8


def _expand(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen ``[start, end)`` to the surrounding word, minus edge punctuation."""

    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    while start < end and text[start] in _EDGE_PUNCTUATION:
        start += 1
    while end > start and text[end - 1] in _EDGE_PUNCTUATION:
        end -= 1
    return start, end


def _is_volatile(span: str) -> bool:
    if not span or len(span) > _MAX_VOLATILE_SPAN:
        return False
    return _VOLATILE.fullmatch(span) is not None


def volatile_spans(before: str, after: str) -> list[tuple[int, int]] | None:
    """Return the volatile character spans of ``after`` if that is all that changed.

    Volatile values are timestamps, dates, times, UUIDs, hex or numeric IDs,
    prefixed run IDs and day or month names. Each changed region is widened to
    the whitespace-delimited word around it, and that word must look volatile
    on both sides. Returns None for any other change, such as edited prose,
    and also when nothing changed.
    """

    if before == after:
        return None
    left = _TOKENS.findall(before)
    right = _TOKENS.findall(after)
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(left), len(right)) - prefix
        and left[len(left) - 1 - suffix] == right[len(right) - 1 - suffix]
    ):
        suffix += 1
    left_offsets = [0]
    for token in left:
        left_offsets.append(left_offsets[-1] + len(token))
    right_offsets = [0]
    for token in right:
        right_offsets.append(right_offsets[-1] + len(token))
    matcher = difflib.SequenceMatcher(
        None,
        left[prefix : len(left) - suffix],
        right[prefix : len(right) - suffix],
        autojunk=False,
    )
    spans: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if len(spans) >= _MAX_VOLATILE_REGIONS:
            return None
        left_start, left_end = _expand(
            before, left_offsets[prefix + i1], left_offsets[prefix + i2]
        )
        right_start, right_end = _expand(
            after, right_offsets[prefix + j1], right_offsets[prefix + j2]
        )
        if not (
            _is_volatile(before[left_start:left_end])
            and _is_volatile(after[right_start:right_end])
        ):
            return None
        if spans and right_start < spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], right_end))
        else:
            spans.append((right_start, right_end))
    return spans or None


def volatile_difference(before: str, after: str) -> tuple[bool, int]:
    """Return ``(volatile, regions)``. See :func:`volatile_spans`."""

    spans = volatile_spans(before, after)
    return (spans is not None, len(spans or ()))


def _message_text(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
                return None
            text = part.get("text")
            if not isinstance(text, str):
                return None
            pieces.append(text)
        return "".join(pieces)
    return None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _label_volatile_segments(
    requests: list[dict[str, Any]], bodies: Mapping[str, Any]
) -> int:
    """Relabel exact system-message segments whose only change is volatile values.

    A system or developer message is ``dynamic_metadata`` when the closest
    earlier request in the workflow had identical messages before it and the
    same message differed only in volatile values. The character spans of
    those values are kept as evidence, so a replay can vary them per trial.
    Returns the number of messages relabelled.
    """

    relabelled = 0
    for index, request in enumerate(requests):
        body = bodies.get(request["request_id"])
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list):
            continue
        candidates = sorted(
            {
                segment["message_index"]
                for segment in request["segments"]
                if segment.get("kind") == "system_instructions"
                and segment.get("attribution_quality") == "exact_token_boundary"
                and isinstance(segment.get("message_index"), int)
                and segment["message_index"] < len(messages)
            }
        )
        evidence: dict[int, dict[str, Any]] = {}
        for message_index in candidates:
            current = messages[message_index]
            current_text = _message_text(current)
            if current_text is None:
                continue
            prefix_key = _canonical(messages[:message_index])
            for earlier in reversed(requests[:index]):
                earlier_body = bodies.get(earlier["request_id"])
                earlier_messages = (
                    earlier_body.get("messages") if isinstance(earlier_body, dict) else None
                )
                if (
                    not isinstance(earlier_messages, list)
                    or len(earlier_messages) <= message_index
                    or _canonical(earlier_messages[:message_index]) != prefix_key
                ):
                    continue
                prior = earlier_messages[message_index]
                if not isinstance(prior, dict) or prior.get("role") != current.get("role"):
                    continue
                prior_text = _message_text(prior)
                if prior_text is None or prior_text == current_text:
                    continue
                spans = volatile_spans(prior_text, current_text)
                if spans is not None:
                    evidence[message_index] = {
                        "compared_with_request_id": earlier["request_id"],
                        "changed_regions": len(spans),
                        "character_spans": [list(item) for item in spans],
                        "content_is_string": isinstance(current.get("content"), str),
                        "method": "proxy.volatile_content_diff",
                    }
                break
        for segment in request["segments"]:
            found = evidence.get(segment.get("message_index"))
            if found is not None and segment.get("kind") == "system_instructions":
                segment["kind"] = "dynamic_metadata"
                segment["label"] = f"{segment['label']} (volatile values)"
                segment["volatile_evidence"] = found
        relabelled += len(evidence)
    return relabelled


# -- trace building ---------------------------------------------------------


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _body(exchange: Mapping[str, Any]) -> Any:
    request = exchange.get("request")
    return request.get("body") if isinstance(request, Mapping) else None


def _telemetry(exchange: Mapping[str, Any]) -> dict[str, Any]:
    value = exchange.get("telemetry")
    return value if isinstance(value, dict) else {}


def _included(exchange: Mapping[str, Any]) -> tuple[bool, str | None]:
    response = exchange.get("response")
    status = response.get("status") if isinstance(response, Mapping) else None
    if exchange.get("outcome") != OUTCOME_COMPLETED:
        return False, f"outcome {exchange.get('outcome')}"
    if not isinstance(status, int) or not 200 <= status < 300:
        return False, f"HTTP status {status}"
    return True, None


def _available_tools(body: Any) -> list[str]:
    tools = body.get("tools") if isinstance(body, dict) else None
    if not isinstance(tools, list):
        return []
    return list(dict.fromkeys(name for tool in tools if (name := _tool_name(tool))))


def _structural_for(body: Any, covered: set[int], has_tools_segment: bool) -> list[dict[str, Any]]:
    if not isinstance(body, dict):
        return [
            {
                "kind": "unknown",
                "label": "unattributed prompt",
                "start_token": None,
                "end_token": None,
                "attribution_quality": "unknown",
                "source": "proxy.capture",
            }
        ]
    structural = _structural_segments(body)
    result = []
    for item in structural:
        source = str(item.get("source", ""))
        match = re.fullmatch(r"request\.messages\[(\d+)\]", source)
        if match and int(match.group(1)) in covered:
            continue
        if source == "request.tools" and has_tools_segment:
            continue
        result.append(item)
    return result


def _identity_for(session: ProxySession) -> dict[str, Any]:
    identity = session.identity or {}
    return identity if isinstance(identity, dict) else {}


def _cache_config(session: ProxySession) -> dict[str, Any]:
    cache = _identity_for(session).get("cache_config")
    return cache if isinstance(cache, dict) else {}


def build_workflow_trace(
    session: ProxySession,
    workflow: Workflow,
    *,
    workload: str = "unknown",
    backend: str | None = None,
    block_size: int | None = None,
    identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one validated normalized trace (schema 0.2) for a workflow.

    ``identity`` lets the operator pin identity fields that APIs do not expose,
    such as model revision, tokenizer revision and chat-template digest. They
    are graded as operator-supplied, never as observed.
    """

    probe = _identity_for(session)
    cache = _cache_config(session)
    resolved_backend = backend or str(probe.get("backend") or "unknown")
    probed_block = _int(cache.get("block_size"))
    resolved_block = probed_block or block_size or 16
    block_source = (
        "vllm:cache_config_info"
        if probed_block
        else "import argument; not observed from the server"
        if block_size
        else "default assumption (16); not observed from the server"
    )
    prefix_caching = cache.get("enable_prefix_caching")
    operator_identity = {
        key: str(value)
        for key, value in (identity or {}).items()
        if key in _IDENTITY_FIELDS and value
    }
    probed_version = probe.get("backend_version")
    if isinstance(probed_version, str) and "backend_version" in operator_identity:
        if operator_identity["backend_version"] != probed_version:
            raise TraceValidationError(
                "operator-supplied backend_version conflicts with the upstream /version "
                f"report ({operator_identity['backend_version']!r} != {probed_version!r})"
            )

    included: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for exchange in workflow.exchanges:
        ok, reason = _included(exchange)
        if ok:
            included.append(exchange)
        else:
            exclusions.append({"exchange_id": exchange.get("exchange_id"), "reason": reason})
    if not included:
        raise TraceValidationError(
            f"workflow {workflow.workflow_id!r} contains no completed inference exchanges"
        )
    origin = _start(included[0])

    requests: list[dict[str, Any]] = []
    bodies: dict[str, Any] = {}
    timeline: list[dict[str, Any]] = []
    token_sources: list[str] = []
    verifications: list[str] = []
    dropped_invocations = 0
    for sequence, exchange in enumerate(included):
        exchange_id = str(exchange["exchange_id"])
        body = _body(exchange)
        bodies[exchange_id] = body
        telemetry = _telemetry(exchange)
        usage = telemetry.get("usage") if isinstance(telemetry.get("usage"), dict) else {}
        server = (
            telemetry.get("server_metrics")
            if isinstance(telemetry.get("server_metrics"), dict)
            else {}
        )
        resolution = session.resolutions.get(exchange_id)
        tokens: list[int] | None = None
        segments: list[dict[str, Any]] = []
        if resolution and resolution.get("status") == "exact" and isinstance(
            resolution.get("prompt_token_ids"), list
        ):
            tokens = list(resolution["prompt_token_ids"])
            segments = [dict(item) for item in resolution.get("segments") or []]
            token_sources.append(str(resolution.get("source")))
            verifications.append(str(resolution.get("verification")))
        else:
            token_sources.append(
                "unresolved: "
                + (
                    str(resolution.get("reason"))
                    if resolution
                    else "no token-resolution record"
                )
            )
            verifications.append("not_resolved")
        prompt_count = len(tokens) if tokens is not None else _int(usage.get("prompt_tokens"))
        quality = "exact" if tokens is not None else "count_only" if prompt_count is not None else "unknown"
        covered = {
            item["message_index"]
            for item in segments
            if isinstance(item.get("message_index"), int)
        }
        has_tools = any(item.get("kind") == "tools" for item in segments)
        if isinstance(body, dict) or not segments:
            segments.extend(_structural_for(body, covered, has_tools))
        available = _available_tools(body)
        invoked_raw = telemetry.get("invoked_tools") if isinstance(telemetry.get("invoked_tools"), list) else []
        invoked = [name for name in invoked_raw if isinstance(name, str) and name in available]
        dropped_invocations += len(invoked_raw) - len(invoked)
        model = (
            body.get("model")
            if isinstance(body, dict) and isinstance(body.get("model"), str)
            else telemetry.get("response_model")
            if isinstance(telemetry.get("response_model"), str)
            else "unknown"
        )
        salt = body.get("cache_salt") if isinstance(body, dict) else None
        request_payload: dict[str, Any] = {
            "request_id": exchange_id,
            "sequence": sequence,
            "model": model,
            "backend": resolved_backend,
            "prompt_token_ids": tokens,
            "prompt_token_count": prompt_count,
            "token_evidence_quality": quality,
            "segments": segments,
            "available_tools": available,
            "invoked_tools": list(dict.fromkeys(invoked)),
            "cache": {
                "enabled": prefix_caching if isinstance(prefix_caching, bool) else True,
                "block_size": resolved_block,
                "cache_salt_digest": (
                    "sha256:" + hashlib.sha256(salt.encode("utf-8")).hexdigest()
                    if isinstance(salt, str)
                    else None
                ),
            },
            "metrics": {
                "input_tokens": prompt_count,
                "output_tokens": _int(usage.get("completion_tokens")),
                "cached_input_tokens": _int(usage.get("cached_tokens")),
                "ttft_ms": _float(server.get("server_ttft_ms")),
                "client_ttft_ms": _float(telemetry.get("client_ttft_ms")),
                "queue_ms": _float(server.get("server_queue_ms")),
                "total_latency_ms": _float(telemetry.get("total_latency_ms")),
            },
            "proxy": {
                "exchange_id": exchange_id,
                "route": exchange.get("route"),
                "token_source": token_sources[-1],
                "token_verification": verifications[-1],
                "segmentation": (resolution or {}).get("segmentation"),
                "stream": telemetry.get("stream"),
                "stream_termination": telemetry.get("stream_termination"),
                "itl_ms": telemetry.get("itl_ms"),
                "tpot_ms": telemetry.get("tpot_ms"),
                "output_event_count": telemetry.get("output_event_count"),
                "proxy_overhead_ms": telemetry.get("proxy_overhead_ms"),
            },
        }
        if isinstance(probed_version, str):
            request_payload["backend_version"] = probed_version
        request_payload.update(operator_identity)
        requests.append(request_payload)
        correlation = _correlation(exchange)
        depends = correlation.get("depends_on") if isinstance(correlation.get("depends_on"), dict) else {}
        timeline.append(
            {
                "request_id": exchange_id,
                "span_id": correlation.get("span_id") or exchange_id,
                "start_ms": round(_start(exchange) - origin, 3),
                "end_ms": round(_end(exchange) - origin, 3),
                "dependency_asserted": bool(depends.get("asserted")),
                "asserted_depends_on": list(depends.get("span_ids") or []),
                "parent_span_id": (
                    correlation["traceparent"].get("parent_id")
                    if isinstance(correlation.get("traceparent"), dict)
                    else None
                ),
            }
        )

    relabelled = _label_volatile_segments(requests, bodies)
    spans, dependency_evidence, dependency_notes = _explicit_spans(timeline)

    task_ids = {
        _correlation(item).get("task_id")
        for item in included
        if isinstance(_correlation(item).get("task_id"), str)
    }
    attempt_ids = {
        _correlation(item).get("attempt_id")
        for item in included
        if isinstance(_correlation(item).get("attempt_id"), str)
    }
    scope = session.scope
    if scope.get("kind") == "attempt":
        if isinstance(scope.get("task_id"), str):
            task_ids.add(scope["task_id"])
        if isinstance(scope.get("attempt_id"), str):
            attempt_ids.add(scope["attempt_id"])
    if len(task_ids) > 1:
        raise TraceValidationError(
            f"workflow {workflow.workflow_id!r} carries conflicting task IDs"
        )
    if len(attempt_ids) > 1:
        raise TraceValidationError(
            f"workflow {workflow.workflow_id!r} carries conflicting attempt IDs"
        )

    payload: dict[str, Any] = {
        "schema_version": "0.2",
        "trace_id": workflow.workflow_id,
        "workload": workload,
        "success": None,
        "task_id": next(iter(task_ids)) if task_ids else None,
        "attempt_id": next(iter(attempt_ids)) if attempt_ids else None,
        "dependency_evidence": dependency_evidence,
        "workflow_grouping": {
            "method": workflow.grouping,
            "key": workflow.grouping_key,
            "idle_gap_ms": workflow.idle_gap_ms,
            "ambiguous": workflow.ambiguous,
            "overlap_count": workflow.overlap_count,
            "overlapping_exchange_ids": list(workflow.overlapping_exchange_ids),
        },
        "spans": spans,
        "requests": requests,
        "proxy_timeline": timeline,
        "proxy_session": {
            "session_id": session.session_id,
            "session_sha256": session.sha256,
            "wall_anchor_unix_ns": session.header.get("started_unix_ns"),
            "timeline_origin_ms": origin,
            "clean_close": bool(session.close and session.close.get("clean")),
            "observations_dropped": (
                (session.close or {}).get("counters", {}).get("observations_dropped")
                if session.close
                else None
            ),
            "excluded_exchanges": exclusions,
            "dependency_notes": dependency_notes,
            "volatile_segments_labelled": relabelled,
            "invoked_tools_not_offered": dropped_invocations,
            "block_size_source": block_source,
            "backend_evidence": probe.get("backend_evidence", "no upstream identity probe"),
            "workflow_grouping_ambiguity": {
                "detected": workflow.ambiguous,
                "reason": (
                    "overlapping in-flight requests had no workflow correlation header"
                    if workflow.ambiguous
                    else None
                ),
                "overlap_count": workflow.overlap_count,
                "overlapping_exchange_ids": list(workflow.overlapping_exchange_ids),
            },
        },
    }
    payload["evidence_quality"] = _evidence_quality(
        payload, requests, workflow, operator_identity, probe
    )
    Trace.from_dict(payload)
    return payload


def _explicit_spans(
    timeline: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Emit spans only when every request asserted its dependencies consistently.

    An empty ``depends_on`` claims independence, so spans are omitted rather
    than emitted with guessed edges when any assertion is missing or invalid.
    """

    notes: list[str] = []
    asserted = [item for item in timeline if item["dependency_asserted"]]
    if not asserted:
        return [], "absent", notes
    span_ids = [item["span_id"] for item in timeline]
    duplicates = sorted({value for value in span_ids if span_ids.count(value) > 1})
    if duplicates:
        notes.append(f"duplicate span identifiers: {duplicates[:5]}")
    by_span = {item["span_id"]: item for item in timeline}
    valid = not duplicates
    for item in asserted:
        for dependency in item["asserted_depends_on"]:
            target = by_span.get(dependency)
            if target is None:
                notes.append(
                    f"{item['span_id']} depends on {dependency}, which is not a completed "
                    "exchange in this workflow"
                )
                valid = False
            elif target["end_ms"] > item["start_ms"]:
                notes.append(
                    f"{item['span_id']} asserts a dependency on {dependency}, which had "
                    "not finished when it started; the assertion was not used"
                )
                valid = False
            elif dependency == item["span_id"]:
                notes.append(f"{item['span_id']} asserts a dependency on itself")
                valid = False
    if len(asserted) != len(timeline) or not valid:
        if len(asserted) != len(timeline):
            notes.append(
                f"{len(timeline) - len(asserted)} of {len(timeline)} exchanges did not assert "
                "dependencies"
            )
        return [], "partial_explicit", notes
    spans = [
        {
            "span_id": item["span_id"],
            "kind": "inference",
            "start_ms": item["start_ms"],
            "end_ms": item["end_ms"],
            "request_id": item["request_id"],
            "depends_on": list(item["asserted_depends_on"]),
        }
        for item in timeline
    ]
    return spans, "complete_explicit", notes


def _evidence_quality(
    payload: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    workflow: Workflow,
    operator_identity: Mapping[str, str],
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    exact = all(item["token_evidence_quality"] == "exact" for item in requests)
    counted = all(item["prompt_token_count"] is not None for item in requests)
    verified = sum(
        1 for item in requests if item["proxy"]["token_verification"] == "usage_prompt_tokens_match"
    )
    cached = all(item["metrics"]["cached_input_tokens"] is not None for item in requests)
    server_ttft = all(item["metrics"]["ttft_ms"] is not None for item in requests)
    client_ttft = all(item["metrics"]["client_ttft_ms"] is not None for item in requests)
    identity_complete = all(
        item.get(name) not in {None, "", "unknown"}
        for item in requests
        for name in ("model", "backend", *_IDENTITY_FIELDS)
    )
    dependency = payload["dependency_evidence"]
    grouping_grade = (
        (
            "unresolved",
            "overlapping in-flight requests had no x-runeconomy-workflow-id or "
            "W3C traceparent correlation; workflow membership is ambiguous",
        )
        if workflow.ambiguous
        else {
            GROUPING_RUNECONOMY_HEADER: (
                "directly_observed",
                "x-runeconomy-workflow-id header",
            ),
            GROUPING_TRACEPARENT: ("directly_observed", "W3C traceparent trace ID"),
            GROUPING_ATTEMPT_SCOPE: (
                "directly_observed",
                "RunEconomy runner attempt scope",
            ),
            GROUPING_IDLE_GAP: (
                "weakly_inferred",
                f"idle-gap fallback ({workflow.idle_gap_ms or 0:g} ms); no correlation header",
            ),
        }[workflow.grouping]
    )
    return build_evidence_quality(
        SOURCE_KIND,
        {
            "prompt_tokens": evidence_field(
                "directly_observed" if exact else "reliably_derived" if counted else "unresolved",
                "vLLM render or tokenize of the captured request"
                if exact
                else "server-reported usage counts"
                if counted
                else "prompt token evidence absent for one or more requests",
                detail=f"{verified} of {len(requests)} verified against server usage counts",
            ),
            "cached_input_tokens": evidence_field(
                "directly_observed" if cached else "unresolved",
                "server usage prompt_tokens_details.cached_tokens"
                if cached
                else "cached-token field absent from one or more responses",
            ),
            "server_ttft": evidence_field(
                "directly_observed" if server_ttft else "unresolved",
                "server-reported per-request metrics"
                if server_ttft
                else "server TTFT absent from one or more responses",
            ),
            "client_ttft": evidence_field(
                "directly_observed" if client_ttft else "unresolved",
                "proxy monotonic timer to the first output-bearing stream event"
                if client_ttft
                else "one or more responses were not streamed or carried no output",
            ),
            "workflow_dependencies": evidence_field(
                "directly_observed"
                if dependency == "complete_explicit"
                else "weakly_inferred"
                if dependency == "partial_explicit"
                else "unresolved",
                "explicit x-runeconomy-depends-on headers"
                if dependency != "absent"
                else "no dependency header; timing is never used to infer edges",
            ),
            "workflow_grouping": evidence_field(*grouping_grade),
            "backend_identity": evidence_field(
                "weakly_inferred" if identity_complete else "unresolved",
                "upstream self-report plus operator-supplied identity"
                if identity_complete
                else "model revision, tokenizer revision or chat-template digest not supplied",
                detail=str(probe.get("backend_evidence") or "no upstream identity probe"),
            ),
            "task_identity": evidence_field(
                "directly_observed" if payload.get("task_id") else "unresolved",
                "x-runeconomy-task-id header or runner attempt scope"
                if payload.get("task_id")
                else "task identity was not supplied",
            ),
            "request_latency": evidence_field(
                "directly_observed", "proxy monotonic timer (request received to last byte)"
            ),
        },
    )


# -- static replay context --------------------------------------------------


def _merge_observed(
    name: str,
    observed: Mapping[str, Any],
    supplied: Any,
    blockers: list[str],
) -> dict[str, Any]:
    result = dict(observed)
    if supplied is None:
        return result
    if not isinstance(supplied, Mapping):
        blockers.append(f"context.{name} must be an object")
        return result
    for key, value in supplied.items():
        if key in observed and isinstance(observed[key], Mapping) and isinstance(value, Mapping):
            merged = dict(observed[key])
            for inner_key, inner_value in value.items():
                if inner_key in merged and merged[inner_key] != inner_value:
                    blockers.append(
                        f"context.{name}.{key}.{inner_key} conflicts with the observed value"
                    )
                else:
                    merged[inner_key] = inner_value
            result[key] = merged
        elif key in observed and observed[key] != value:
            blockers.append(
                f"context.{name}.{key} conflicts with the observed value {observed[key]!r}"
            )
        else:
            result[key] = deepcopy(value)
    return result


def static_replay_parameters(
    finding: Mapping[str, Any],
    exchange: Mapping[str, Any],
    request: Mapping[str, Any],
    session: ProxySession,
    context: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    """Return experiment parameters the capture establishes, plus blockers.

    Observed values (backend address, model, version, block size, sampling and
    the exact recorded request) come from the session. Design decisions (the
    evaluator, gates, execution plan and any candidate rewrite) come only from
    the operator's ``context``. A conflict between the two is a blocker. It is
    never resolved silently.
    """

    context = dict(context or {})
    blockers: list[str] = []
    provenance: dict[str, list[str]] = {"observed": [], "operator_supplied": []}
    unknown_keys = sorted(set(context) - _OPERATOR_CONTEXT_KEYS)
    if unknown_keys:
        blockers.append("unknown replay context keys: " + ", ".join(unknown_keys))
    probe = _identity_for(session)
    cache = _cache_config(session)
    route = exchange.get("route")
    dialect = {"chat_completions": "chat_completions", "responses": "responses"}.get(str(route))
    if dialect is None:
        blockers.append(f"the static runner cannot replay the {route!r} route")
    if probe.get("backend") != "vllm":
        blockers.append(
            "static replay requires a vLLM upstream; the capture identified "
            f"{probe.get('backend', 'no backend')!r}"
        )
    upstream = session.header.get("upstream") if isinstance(session.header.get("upstream"), dict) else {}
    observed_backend: dict[str, Any] = {"kind": "vllm"}
    if isinstance(upstream.get("origin"), str):
        observed_backend["base_url"] = upstream["origin"] + str(upstream.get("base_path") or "")
    if dialect:
        observed_backend["api_dialect"] = dialect
    if request.get("model") not in {None, "unknown"}:
        observed_backend["model"] = request["model"]
    if isinstance(probe.get("backend_version"), str):
        observed_backend["backend_version"] = probe["backend_version"]
    if _int(cache.get("block_size")):
        observed_backend["block_size"] = cache["block_size"]
    if isinstance(cache.get("enable_prefix_caching"), bool):
        observed_backend["engine_config"] = {
            "enable_prefix_caching": cache["enable_prefix_caching"]
        }
    provenance["observed"].extend(f"backend.{key}" for key in observed_backend)
    supplied_backend = context.get("backend")
    if isinstance(supplied_backend, Mapping):
        provenance["operator_supplied"].extend(
            f"backend.{key}" for key in supplied_backend if key not in observed_backend
        )
    backend = _merge_observed("backend", observed_backend, supplied_backend, blockers)

    body = _body(exchange)
    request_record = exchange.get("request") if isinstance(exchange.get("request"), Mapping) else {}
    redaction = request_record.get("redaction") if isinstance(request_record.get("redaction"), Mapping) else {}
    baseline: dict[str, Any] | None = None
    observed_sampling: dict[str, Any] = {}
    if isinstance(context.get("baseline_request"), Mapping):
        baseline = deepcopy(dict(context["baseline_request"]))
        provenance["operator_supplied"].append("baseline_request")
    elif not isinstance(body, dict):
        blockers.append(
            "the request body was not recorded (payload_capture=digest_only); supply "
            "context.baseline_request"
        )
    elif redaction.get("modified"):
        blockers.append(
            "redaction changed the recorded request body, so it is not the executed "
            "request; supply context.baseline_request explicitly"
        )
    else:
        baseline = {
            key: deepcopy(value)
            for key, value in body.items()
            if key not in SAMPLING_KEYS and key not in _RUNNER_CONTROLLED
        }
        observed_sampling = {
            key: deepcopy(value) for key, value in body.items() if key in SAMPLING_KEYS
        }
        provenance["observed"].append("baseline_request")
        if observed_sampling:
            provenance["observed"].append("sampling")
    if baseline is not None:
        stateful = [key for key in _STATEFUL if key in baseline]
        if stateful:
            blockers.append(
                "the request depends on server-side state ("
                + ", ".join(stateful)
                + ") and cannot be replayed statically"
            )
        literal = _LITERAL_PLACEHOLDER.search(_canonical(baseline))
        if literal:
            blockers.append(
                f"the recorded request contains the literal text {literal.group(0)}, which "
                "the static runner treats as an unresolved placeholder"
            )
    if (
        baseline is not None
        and "baseline_request" not in context
        and finding.get("experiment_template_id") == "prompt.move_dynamic_metadata"
    ):
        volatile = _volatile_messages(request)
        if not volatile:
            blockers.append(
                "no volatile-value evidence locates the changing metadata; supply "
                "context.baseline_request with an explicit per-trial placeholder"
            )
        else:
            note = _apply_trial_placeholders(baseline, volatile, blockers)
            if note:
                provenance["observed"].append(note)
    strategy = context.get("candidate_strategy")
    if strategy is not None:
        if strategy != CANDIDATE_STRATEGY_MOVE_VOLATILE:
            blockers.append(f"unsupported context.candidate_strategy {strategy!r}")
        elif "candidate_request" in context:
            blockers.append("context supplies both candidate_request and candidate_strategy")
        elif finding.get("experiment_template_id") != "prompt.move_dynamic_metadata":
            blockers.append(
                f"candidate_strategy {strategy!r} applies only to prompt.move_dynamic_metadata"
            )
        elif baseline is not None:
            candidate = _move_volatile_candidate(
                baseline, sorted(_volatile_messages(request)), blockers
            )
            if candidate is not None:
                context["candidate_request"] = candidate
                provenance["generated"] = [
                    f"candidate_request by candidate_strategy {strategy!r}"
                ]
    sampling = _merge_observed("sampling", observed_sampling, context.get("sampling"), blockers)
    if isinstance(context.get("sampling"), Mapping):
        provenance["operator_supplied"].extend(
            f"sampling.{key}" for key in context["sampling"] if key not in observed_sampling
        )

    confirmed: list[str] = []
    if request.get("token_evidence_quality") == "exact":
        confirmed.append("exact_prompt_tokens")
    if request.get("metrics", {}).get("cached_input_tokens") is not None:
        confirmed.append("cache_telemetry")
    if context.get("evaluator") is not None:
        confirmed.append("task_evaluator")
    supplied_capabilities = context.get("confirmed_capabilities")
    if isinstance(supplied_capabilities, list):
        for item in supplied_capabilities:
            if isinstance(item, str) and item not in confirmed:
                confirmed.append(item)
                provenance["operator_supplied"].append(f"confirmed_capabilities.{item}")

    parameters: dict[str, Any] = {
        "backend": backend,
        "confirmed_capabilities": confirmed,
        "experiment_id": str(context.get("experiment_id") or f"proxy-{finding.get('finding_id')}"),
        "case_id": str(context.get("case_id") or exchange.get("exchange_id")),
    }
    if sampling:
        parameters["sampling"] = sampling
    if baseline is not None:
        parameters["baseline_request"] = baseline
    for key in ("execution", "acceptance", "evaluator", "candidate_request", "candidate_tool_names", "constants"):
        if key in context:
            parameters[key] = deepcopy(context[key])
            if key == "candidate_request" and provenance.get("generated"):
                continue
            provenance["operator_supplied"].append(key)
    acceptance = parameters.get("acceptance")
    if (
        isinstance(acceptance, dict)
        and "optimization_target" not in acceptance
        and isinstance(finding.get("optimization_target"), str)
    ):
        acceptance["optimization_target"] = finding["optimization_target"]
        provenance["observed"].append("acceptance.optimization_target (from the finding)")
    return parameters, blockers, provenance


def _volatile_messages(request: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for segment in request.get("segments", []):
        evidence = segment.get("volatile_evidence")
        index = segment.get("message_index")
        if isinstance(evidence, dict) and isinstance(index, int):
            found[index] = evidence
    return found


def _apply_trial_placeholders(
    baseline: dict[str, Any],
    volatile: Mapping[int, Mapping[str, Any]],
    blockers: list[str],
) -> str | None:
    """Replace observed volatile values with the per-trial placeholder.

    Replaying a recorded timestamp verbatim would freeze it. Every trial after
    the first would then hit the prefix cache, and the replay would no longer
    measure what the live agent does. The runner substitutes a fresh trial ID
    on each trial, which reproduces the observed variation.
    """

    messages = baseline.get("messages")
    if not isinstance(messages, list):
        blockers.append("baseline request has no message array for placeholder substitution")
        return None
    replaced = 0
    for index, evidence in sorted(volatile.items()):
        if index >= len(messages) or not isinstance(messages[index], dict):
            blockers.append(f"volatile message {index} is absent from the baseline request")
            return None
        content = messages[index].get("content")
        if not isinstance(content, str) or not evidence.get("content_is_string"):
            blockers.append(
                f"volatile values in message {index} are inside multi-part content; "
                "supply context.baseline_request with an explicit placeholder"
            )
            return None
        spans = sorted(
            (tuple(item) for item in evidence.get("character_spans", [])),
            reverse=True,
        )
        for start, end in spans:
            if not (0 <= start < end <= len(content)):
                blockers.append(f"volatile span in message {index} is out of range")
                return None
            content = content[:start] + TRIAL_PLACEHOLDER + content[end:]
            replaced += 1
        messages[index] = {**messages[index], "content": content}
    return (
        f"baseline_request: {replaced} observed volatile value(s) replaced by "
        f"{TRIAL_PLACEHOLDER} so each trial varies them as the agent did"
    )


def _move_volatile_candidate(
    baseline: Mapping[str, Any], volatile: Sequence[int], blockers: list[str]
) -> dict[str, Any] | None:
    """Move whole volatile system messages after the stable system messages.

    This is the only rewrite RunEconomy generates, and only when the operator
    asks for it. It keeps every message and its content, and it never moves a
    message past a conversation turn. It refuses when nothing can move.
    """

    messages = baseline.get("messages")
    if not isinstance(messages, list) or not volatile:
        blockers.append("candidate_strategy needs located volatile system messages")
        return None
    block_end = 0
    while block_end < len(messages) and isinstance(messages[block_end], dict) and messages[
        block_end
    ].get("role") in {"system", "developer"}:
        block_end += 1
    if any(index >= block_end for index in volatile):
        blockers.append(
            "a volatile message is outside the leading system block; moving it would "
            "reorder conversation turns"
        )
        return None
    stable = [messages[index] for index in range(block_end) if index not in volatile]
    moved = [messages[index] for index in volatile]
    if not stable:
        blockers.append(
            "every leading system message is volatile, so there is no stable prefix to "
            "move the metadata behind"
        )
        return None
    reordered = stable + moved + list(messages[block_end:])
    if reordered == list(messages):
        blockers.append("the volatile messages already follow the stable system messages")
        return None
    candidate = deepcopy(dict(baseline))
    candidate["messages"] = deepcopy(reordered)
    return candidate


def _enrich_with_replay(
    analysis: dict[str, Any],
    trace_payload: Mapping[str, Any],
    session: ProxySession,
    context: Mapping[str, Any] | None,
) -> int:
    """Attach replay parameters to findings. Returns the number now runnable."""

    exchanges = {str(item.get("exchange_id")): item for item in session.exchanges}
    requests = {item["request_id"]: item for item in trace_payload["requests"]}
    runnable = 0
    for finding in analysis.get("findings", []):
        if finding.get("experiment_template_id") not in AUTOMATABLE_TEMPLATES:
            continue
        request_id = finding.get("request_id")
        exchange = exchanges.get(str(request_id))
        request = requests.get(request_id)
        if exchange is None or request is None:
            continue
        applies = context is None or context.get("experiment_template_id") in {
            None,
            finding.get("experiment_template_id"),
        }
        additions, blockers, provenance = static_replay_parameters(
            finding, exchange, request, session, context if applies else None
        )
        parameters = finding.setdefault("experiment_parameters", {})
        parameters.update(additions)
        parameters["replay_provenance"] = {
            **provenance,
            "source_exchange_id": exchange.get("exchange_id"),
            "source_session_sha256": session.sha256,
        }
        finding["unsupported_reason"] = "; ".join(blockers) if blockers else None
        if blockers:
            continue
        try:
            generated = initialize_experiment(finding)
        except ExperimentValidationError as exc:
            finding["unsupported_reason"] = f"replay context is invalid: {exc}"
            continue
        if generated.get("generation_status") != "unresolved":
            runnable += 1
    return runnable


# -- evidence bundles -------------------------------------------------------


def _assessment(
    trace_payload: Mapping[str, Any], analysis: Mapping[str, Any], runnable: int
) -> dict[str, Any]:
    from ..evidence import _capability, assess_trace_evidence

    assessment = assess_trace_evidence(trace_payload, analysis, None)
    grouping = trace_payload.get("workflow_grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    ambiguous = grouping.get("ambiguous") is True
    replayable = [
        item
        for item in analysis.get("findings", [])
        if item.get("experiment_template_id") in AUTOMATABLE_TEMPLATES
    ]
    if ambiguous:
        replay_status = "missing"
        replay_evidence = (
            "overlapping in-flight requests without workflow correlation headers make "
            "idle-gap workflow membership ambiguous"
        )
        replay_next_action = (
            "recapture with a unique X-RunEconomy-Workflow-Id per concurrent workflow "
            "or propagate a distinct W3C traceparent trace ID"
        )
    elif runnable:
        replay_status = "available"
        replay_evidence = (
            f"{runnable} finding(s) resolve to a runnable static-replay manifest"
        )
        replay_next_action = None
    elif replayable:
        replay_status = "partial"
        replay_evidence = (
            f"{len(replayable)} finding(s) carry recorded replay requests but still "
            "need operator design fields (evaluator, gates, execution plan)"
        )
        replay_next_action = (
            "supply a replay context with backend identity, evaluator, execution and "
            "acceptance, then import again"
        )
    else:
        replay_status = "missing"
        replay_evidence = "no finding maps to an automatable static-replay template"
        replay_next_action = None
    assessment["capabilities"]["static_request_replay"] = _capability(
        replay_status,
        replay_evidence,
        next_action=replay_next_action,
    )
    assessment["workflow_grouping"] = {
        "method": grouping.get("method"),
        "ambiguous": ambiguous,
        "overlap_count": grouping.get("overlap_count", 0),
        "overlapping_exchange_ids": list(
            grouping.get("overlapping_exchange_ids", [])
        ),
    }
    assessment["runnable_experiment"] = bool(runnable) and not ambiguous
    assessment["recommended_mode"] = (
        "static_request" if runnable and not ambiguous else None
    )
    assessment["next_action"] = (
        "initialize a workspace from this evidence"
        if runnable and not ambiguous
        else assessment["capabilities"]["static_request_replay"].get("next_action")
    )
    return assessment


def _run_bundle(
    session: ProxySession,
    trace_payload: dict[str, Any],
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    trace = Trace.from_dict(trace_payload)
    analysis = analyze_trace(
        trace, dependency_evidence=trace_payload.get("dependency_evidence")
    )
    runnable = _enrich_with_replay(analysis, trace_payload, session, context)
    grouping = trace_payload.get("workflow_grouping")
    ambiguous = isinstance(grouping, Mapping) and grouping.get("ambiguous") is True
    if ambiguous:
        reason = (
            "capture workflow is ambiguous: overlapping in-flight requests had no "
            "X-RunEconomy-Workflow-Id or W3C traceparent correlation"
        )
        for finding in analysis.get("findings", []):
            if not isinstance(finding, dict):
                continue
            finding["capture_workflow_grouping"] = deepcopy(dict(grouping))
            existing = finding.get("unsupported_reason")
            finding["unsupported_reason"] = (
                f"{reason}; {existing}" if isinstance(existing, str) and existing else reason
            )
    return {
        "artifact_type": "runeconomy.evidence_bundle",
        "schema_version": EVIDENCE_BUNDLE_VERSION,
        "source": {
            "kind": SOURCE_KIND,
            "name": session.path.name,
            "sha256": session.sha256,
            "workflow_id": trace_payload["trace_id"],
        },
        "trace": trace_payload,
        "analysis": analysis,
        "capability_assessment": _assessment(trace_payload, analysis, runnable),
    }


def import_proxy_session(
    path: Path,
    *,
    workflow_id: str | None = None,
    idle_gap_ms: float = DEFAULT_IDLE_GAP_MS,
    workload: str = "unknown",
    backend: str | None = None,
    block_size: int | None = None,
    identity: Mapping[str, str] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Import a capture session as an evidence bundle.

    A session with one workflow, or a selected ``workflow_id``, produces a
    single-run bundle that ``experiment init --from-evidence`` reads directly.
    Several workflows produce a multi-run bundle with every finding at the top
    level, so ``--finding-id`` can select one.
    """

    path = Path(path)
    session = load_session(path)
    workflows = group_workflows(session, idle_gap_ms=idle_gap_ms)
    if workflow_id is not None:
        workflows = [item for item in workflows if item.workflow_id == workflow_id]
        if not workflows:
            raise TraceValidationError(f"workflow {workflow_id!r} was not found in {path}")
    runs: list[dict[str, Any]] = []
    skipped: list[str] = []
    for workflow in workflows:
        try:
            trace_payload = build_workflow_trace(
                session,
                workflow,
                workload=workload,
                backend=backend,
                block_size=block_size,
                identity=identity,
            )
        except TraceValidationError as exc:
            if "contains no completed inference exchanges" in str(exc):
                skipped.append(workflow.workflow_id)
                continue
            raise
        runs.append(_run_bundle(session, trace_payload, context))
    if not runs:
        raise TraceValidationError(
            f"{path}: no workflow contains a completed inference exchange"
        )
    limitations = list(session.diagnostics)
    if skipped:
        limitations.append(
            f"{len(skipped)} workflow(s) had no completed inference exchange and were skipped"
        )
    ambiguous_workflows = [
        run["trace"]["trace_id"]
        for run in runs
        if run.get("capability_assessment", {})
        .get("workflow_grouping", {})
        .get("ambiguous")
        is True
    ]
    warnings: list[str] = []
    if ambiguous_workflows:
        warning = (
            f"{len(ambiguous_workflows)} idle-gap workflow(s) contain overlapping "
            "in-flight requests without correlation headers; their workflow membership "
            "is ambiguous and runnable workspace generation is blocked. Recapture with "
            "X-RunEconomy-Workflow-Id or W3C traceparent."
        )
        warnings.append(warning)
        limitations.append(warning)
    summary = {
        "session_id": session.session_id,
        "scope": session.scope,
        "upstream": session.header.get("upstream"),
        "capture": session.header.get("capture"),
        "workflow_count": len(runs),
        "exchange_count": len(session.exchanges),
        "clean_close": bool(session.close and session.close.get("clean")),
        "counters": (session.close or {}).get("counters"),
        "limitations": limitations,
        "warnings": warnings,
        "ambiguous_workflow_count": len(ambiguous_workflows),
        "ambiguous_workflow_ids": ambiguous_workflows,
    }
    if len(runs) == 1:
        bundle = runs[0]
        bundle["proxy_session"] = summary
        return bundle
    keys = runs[0]["capability_assessment"]["capabilities"].keys()
    from ..evidence import _capability

    capabilities: dict[str, Any] = {}
    for key in keys:
        statuses = [run["capability_assessment"]["capabilities"][key]["status"] for run in runs]
        status = statuses[0] if len(set(statuses)) == 1 else "partial"
        capabilities[key] = _capability(
            status,
            f"{statuses.count('available')} of {len(statuses)} workflow(s) available",
        )
    runnable = any(run["capability_assessment"]["runnable_experiment"] for run in runs)
    return {
        "artifact_type": "runeconomy.evidence_bundle",
        "schema_version": EVIDENCE_BUNDLE_VERSION,
        "source": {"kind": SOURCE_KIND, "name": path.name, "sha256": session.sha256},
        "runs": runs,
        "findings": [
            finding for run in runs for finding in run.get("analysis", {}).get("findings", [])
        ],
        "capability_assessment": {
            "assessment_version": "0.1",
            "capabilities": capabilities,
            "runnable_experiment": runnable,
            "recommended_mode": "static_request" if runnable else None,
            "next_action": (
                "select a finding from an unambiguous correlation-header workflow "
                "with --finding-id and initialize a workspace"
                if runnable and ambiguous_workflows
                else "select one finding with --finding-id and initialize a workspace"
                if runnable
                else "recapture ambiguous concurrent workflows with correlation headers"
                if ambiguous_workflows
                else "select one workflow or finding explicitly before initializing an experiment"
            ),
            "workflow_grouping": {
                "ambiguous": bool(ambiguous_workflows),
                "ambiguous_workflow_count": len(ambiguous_workflows),
                "ambiguous_workflow_ids": ambiguous_workflows,
            },
        },
        "proxy_session": summary,
    }


# -- agent-process mode: OpenInference export -------------------------------


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        encoded: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _hex_id(value: str, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _segment_token_arrays(request: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    tokens = request.get("prompt_token_ids")
    if not isinstance(tokens, list):
        return None
    bounded = sorted(
        (
            item
            for item in request.get("segments", [])
            if item.get("start_token") is not None and item.get("end_token") is not None
        ),
        key=lambda item: item["start_token"],
    )
    if not bounded:
        return None
    result: list[dict[str, Any]] = []
    cursor = 0
    for item in bounded:
        if item["start_token"] > cursor:
            result.append(
                {
                    "kind": "unattributed",
                    "label": "unattributed template tokens",
                    "token_ids": tokens[cursor : item["start_token"]],
                }
            )
        result.append(
            {
                "kind": item["kind"],
                "label": item["label"],
                "token_ids": tokens[item["start_token"] : item["end_token"]],
            }
        )
        cursor = item["end_token"]
    if cursor < len(tokens):
        result.append(
            {
                "kind": "unattributed",
                "label": "unattributed template tokens",
                "token_ids": tokens[cursor:],
            }
        )
    return result


def trace_to_openinference(trace_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Encode a proxy trace as the OTLP JSON that agent-process mode imports.

    Comparison runs read ``RUNECONOMY_TRACE_PATH`` with the OpenInference
    importer. This writes that format with the RunEconomy extension attributes,
    so exact tokens, segments, cache counts and explicit dependencies survive
    the round trip. Dependencies are emitted only when the trace has complete
    explicit evidence.
    """

    trace_hex = _hex_id(str(trace_payload["trace_id"]), 32)
    session_info = trace_payload.get("proxy_session") or {}
    wall_anchor_unix_ns = int(session_info.get("wall_anchor_unix_ns") or 0)
    session_origin_ms = float(session_info.get("timeline_origin_ms") or 0.0)
    timeline = {item["request_id"]: item for item in trace_payload.get("proxy_timeline", [])}
    explicit = trace_payload.get("dependency_evidence") == "complete_explicit"
    span_hex = {
        item["span_id"]: _hex_id(f"{trace_payload['trace_id']}\0{item['span_id']}", 16)
        for item in trace_payload.get("proxy_timeline", [])
    }
    spans: list[dict[str, Any]] = []
    for request in trace_payload["requests"]:
        entry = timeline[request["request_id"]]
        start_ns = wall_anchor_unix_ns + int((session_origin_ms + entry["start_ms"]) * 1e6)
        end_ns = wall_anchor_unix_ns + int((session_origin_ms + entry["end_ms"]) * 1e6)
        metrics = request["metrics"]
        attributes = [
            _attribute("openinference.span.kind", "LLM"),
            _attribute("llm.model_name", request["model"]),
            _attribute("llm.provider", request["backend"]),
            _attribute("runeconomy.proxy.exchange_id", request["request_id"]),
        ]
        if request.get("prompt_token_ids") is not None:
            attributes.append(
                _attribute("runeconomy.prompt_token_ids", json.dumps(request["prompt_token_ids"]))
            )
            arrays = _segment_token_arrays(request)
            if arrays:
                attributes.append(
                    _attribute("runeconomy.prompt_segments", json.dumps(arrays))
                )
        if request.get("prompt_token_count") is not None:
            attributes.append(_attribute("llm.token_count.prompt", request["prompt_token_count"]))
        if metrics.get("output_tokens") is not None:
            attributes.append(_attribute("llm.token_count.completion", metrics["output_tokens"]))
        if metrics.get("cached_input_tokens") is not None:
            attributes.append(
                _attribute("runeconomy.cached_input_tokens", metrics["cached_input_tokens"])
            )
        if metrics.get("ttft_ms") is not None:
            attributes.append(_attribute("runeconomy.server_ttft_ms", float(metrics["ttft_ms"])))
        if metrics.get("client_ttft_ms") is not None:
            attributes.append(
                _attribute("runeconomy.client_ttft_ms", float(metrics["client_ttft_ms"]))
            )
        if metrics.get("queue_ms") is not None:
            attributes.append(_attribute("llm.queue_time_ms", float(metrics["queue_ms"])))
        for index, name in enumerate(request.get("available_tools", [])):
            attributes.append(_attribute(f"llm.tools.{index}.tool.name", name))
        for index, name in enumerate(request.get("invoked_tools", [])):
            attributes.append(
                _attribute(
                    f"llm.output_messages.0.message.tool_calls.{index}.tool_call.function.name",
                    name,
                )
            )
        for name in _IDENTITY_FIELDS:
            if request.get(name):
                key = "llm.backend_version" if name == "backend_version" else f"runeconomy.{name}"
                attributes.append(_attribute(key, request[name]))
        if trace_payload.get("task_id"):
            attributes.append(_attribute("runeconomy.task_id", trace_payload["task_id"]))
        if trace_payload.get("attempt_id"):
            attributes.append(_attribute("runeconomy.attempt_id", trace_payload["attempt_id"]))
        if explicit:
            attributes.append(
                _attribute(
                    "runeconomy.depends_on",
                    json.dumps([span_hex[item] for item in entry["asserted_depends_on"]]),
                )
            )
        spans.append(
            {
                "traceId": trace_hex,
                "spanId": span_hex[entry["span_id"]],
                "name": f"runeconomy.proxy.{request['proxy']['route']}",
                "startTimeUnixNano": str(start_ns),
                "endTimeUnixNano": str(max(end_ns, start_ns)),
                "attributes": attributes,
            }
        )
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [_attribute("service.name", "runeconomy-capture-proxy")]
                },
                "scopeSpans": [
                    {"scope": {"name": "runeconomy.proxy"}, "spans": spans}
                ],
            }
        ],
        "runeconomy": {
            "proxy_trace": {
                "schema_version": "0.1",
                "trace_id": trace_payload["trace_id"],
                "dependency_evidence": trace_payload.get("dependency_evidence"),
                "workflow_grouping": trace_payload.get("workflow_grouping"),
                "proxy_session": trace_payload.get("proxy_session"),
                "evidence_quality": trace_payload.get("evidence_quality"),
            }
        },
    }


def session_traces(
    path: Path,
    *,
    idle_gap_ms: float = DEFAULT_IDLE_GAP_MS,
    workload: str = "unknown",
    identity: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return every buildable normalized trace in a session, in start order."""

    session = load_session(path)
    traces: list[dict[str, Any]] = []
    for workflow in group_workflows(session, idle_gap_ms=idle_gap_ms):
        try:
            traces.append(
                build_workflow_trace(session, workflow, workload=workload, identity=identity)
            )
        except TraceValidationError as exc:
            if "contains no completed inference exchanges" not in str(exc):
                raise
    return traces
