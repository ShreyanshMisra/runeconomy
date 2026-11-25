"""Any OpenAI-compatible Chat Completions API on the adapter protocol.

This covers hosted providers and gateways (OpenAI, Fireworks, Together,
Baseten, Modal, llm-d and AIBrix gateways, and similar). The base URL is the
API base exactly as the provider documents it for the OpenAI SDK, which
usually ends in ``/v1``. The adapter posts to ``{base}/chat/completions`` and
lists ``{base}/models``. It calls nothing else.

What such an API can show is limited, and the adapter records only that:

* client-observed TTFT and latency, and provider-reported usage (including
  cached tokens when the provider reports them)
* the model the provider says served the request, and its
  ``system_fingerprint`` (kept as a digest; a drift indicator only)

Exact prompt tokens, server timing, cache reset and cache isolation are not
available. Prompt tokens are ``null``, never estimated. Identity is
``provider_declared``. A static experiment on this kind must preregister
``acceptance.identity_policy`` with ``minimum_grade: provider_declared``.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Mapping

from ..backends import BackendProbeError, Capability
from ..identity import PROVIDER_DECLARED, fingerprint_digest
from ..startup import StartupRetry
from ..vllm import CapturedResponse
from .base import (
    AdapterProfile,
    BackendRequestError,
    CapabilityEntry,
    CapabilityTable,
    CapabilityUnavailableError,
    ServingIdentity,
)
from .http import get, validate_base_url
from .report import build_report, mode


OPENAI_COMPAT_PROFILE = AdapterProfile(
    kind="openai_compat",
    label="OpenAI-compatible API",
    identity_grade=PROVIDER_DECLARED,
    api_dialects=frozenset({"chat_completions"}),
    cache_isolation_modes=frozenset({"shared"}),
    supports_reset_external=False,
    exact_prompt_tokens=False,
)
DEFAULT_UNDISCLOSED = (
    "backend_version",
    "tokenizer",
    "tokenizer_revision",
    "chat_template_digest",
)
_COMPLETED_REASONS = frozenset({"stop", "tool_calls", "function_call"})


@dataclass(frozen=True)
class ProviderCapturedResponse(CapturedResponse):
    """A capture whose prompt tokens are unknown (``None``).

    Adds what the provider declares about the serving request. The raw
    fingerprint is never stored, only its digest.
    """

    response_model: str | None = None
    provider_fingerprint_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("output_text", None)
        result["prompt_token_ids"] = None
        result["cache_coverage"] = self.cache_coverage
        result["client_server_ttft_gap_ms"] = None
        return result


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _events(response: Any) -> Iterator[dict[str, Any]]:
    data: list[str] = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                payload = "\n".join(data)
                data.clear()
                if payload != "[DONE]":
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise BackendRequestError(
                            "the OpenAI-compatible API returned a malformed SSE event"
                        ) from exc
                    if isinstance(event, dict):
                        yield event
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data) != "[DONE]":
        try:
            event = json.loads("\n".join(data))
        except json.JSONDecodeError as exc:
            raise BackendRequestError(
                "the OpenAI-compatible API returned a malformed final SSE event"
            ) from exc
        if isinstance(event, dict):
            yield event


class OpenAICompatAdapter:
    kind = "openai_compat"
    profile = OPENAI_COMPAT_PROFILE
    supports_cache_reset = False

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        startup_retry_seconds: float = 0.0,
        api_dialect: str = "chat_completions",
        on_startup_retry: Callable[[float], None] | None = None,
        provider: str | None = None,
        undisclosed_fields: tuple[str, ...] = DEFAULT_UNDISCLOSED,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if api_dialect != "chat_completions":
            raise CapabilityUnavailableError(
                "the openai_compat adapter replays the chat_completions dialect only"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.startup_retry_seconds = startup_retry_seconds
        self.on_startup_retry = on_startup_retry
        self.provider = provider or "openai_compat"
        self.undisclosed_fields = tuple(undisclosed_fields)
        self._clock = clock
        self._report: dict[str, Any] | None = None
        self._table: CapabilityTable | None = None
        self._identity: ServingIdentity | None = None

    # -- discovery ---------------------------------------------------------

    def discover(self) -> dict[str, Any]:
        validate_base_url(self.base_url, label="OpenAI-compatible API")
        startup = StartupRetry(self.startup_retry_seconds, on_retry=self.on_startup_retry)
        model_ids: list[str] = []
        try:
            status, body = get(
                self.base_url,
                "/models",
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
                label="OpenAI-compatible API",
                startup=startup,
            )
        except BackendProbeError as exc:
            if exc.category != "malformed_response":
                raise
            status, body = 0, None
        if status == 200 and isinstance(body, Mapping):
            model_ids = sorted(
                item["id"]
                for item in body.get("data", [])
                if isinstance(item, Mapping) and isinstance(item.get("id"), str)
            )
        listed = status == 200
        not_probed = "doctor never generates; verify with one smoke capture"
        table = CapabilityTable(
            backend_kind="openai_compat",
            entries=(
                CapabilityEntry("chat_completions", "unknown", not_probed),
                CapabilityEntry(
                    "responses", "unsupported", "the openai_compat adapter replays Chat Completions only"
                ),
                CapabilityEntry(
                    "exact_prompt_tokens",
                    "unsupported",
                    "an OpenAI-compatible API exposes no tokenizer or render endpoint",
                ),
                CapabilityEntry(
                    "cache_telemetry",
                    "unknown",
                    "provider-reported usage; verify cached_tokens with one smoke capture",
                ),
                CapabilityEntry(
                    "server_timing", "unsupported", "a hosted API reports no server TTFT or queue time"
                ),
                CapabilityEntry(
                    "cache_reset", "unsupported", "the provider controls its cache; no reset exists"
                ),
                CapabilityEntry(
                    "cache_salt", "unsupported", "no standard per-request cache namespace"
                ),
                CapabilityEntry(
                    "model_identity",
                    "supported" if listed else "unknown",
                    "/models listed served models" if listed else f"/models returned HTTP {status}",
                ),
                CapabilityEntry(
                    "prometheus_metrics", "unsupported", "a hosted API exposes no server metrics"
                ),
            ),
        )
        identity = ServingIdentity(
            backend_kind="openai_compat",
            grade=PROVIDER_DECLARED,
            model_ids=tuple(model_ids),
            undisclosed_fields=self.undisclosed_fields,
        )
        modes = [
            mode(
                "warm-cache",
                "supported" if listed else "unknown",
                safe=True,
                physical_cache_state="provider-managed; the initial state is unknown",
                note=(
                    "Replays requests against the provider as it serves them. Prompt "
                    "tokens are count-only and the starting cache state is unknown."
                ),
                next_action="Verify credentials and the base URL, then rerun doctor.",
            ),
            mode(
                "salt-isolated",
                "unsupported",
                safe=False,
                physical_cache_state="unchanged",
                note="Unavailable: there is no standard per-request cache namespace.",
                next_action="Use warm-cache mode.",
            ),
            mode(
                "physical-reset",
                "unsupported",
                safe=False,
                physical_cache_state="unchanged",
                note="Unavailable: a hosted provider's cache cannot be reset.",
                next_action="Use warm-cache mode.",
            ),
        ]
        actions = [
            "Pin a dated model snapshot, not a moving alias, in backend.model.",
            "Declare acceptance.identity_policy with minimum_grade provider_declared, "
            "and list the fields the provider does not disclose in backend.undisclosed_identity.",
            "Run one smoke capture to confirm Chat Completions streaming and whether usage reports cached tokens.",
        ]
        legacy = {
            "responses": Capability("unsupported", "the openai_compat adapter replays Chat Completions only"),
            "chat_completions": Capability("unknown", not_probed),
            "responses_render": Capability("unsupported", "no render endpoint on an OpenAI-compatible API"),
            "chat_completions_render": Capability(
                "unsupported", "no render endpoint on an OpenAI-compatible API"
            ),
            "tokenize": Capability("unsupported", "no tokenizer endpoint on an OpenAI-compatible API"),
            "cache_telemetry": Capability("unknown", "provider-reported usage; verify with a smoke capture"),
            "per_request_timing": Capability("unsupported", "no server timing on a hosted API"),
            "cache_reset": Capability("unsupported", "the provider controls its cache"),
            "cache_salt": Capability("unsupported", "no standard per-request cache namespace"),
        }
        self._table, self._identity = table, identity
        self._report = build_report(
            table=table,
            identity=identity,
            legacy=legacy,
            modes=modes,
            next_actions=actions,
            probes=["GET /models"],
        )
        return self._report

    def capability_table(self) -> CapabilityTable:
        if self._table is None:
            self.discover()
        assert self._table is not None
        return self._table

    def identity(self) -> ServingIdentity:
        if self._identity is None:
            self.discover()
        assert self._identity is not None
        return self._identity

    # -- execution ---------------------------------------------------------

    def capture(self, request_body: Mapping[str, Any]) -> ProviderCapturedResponse:
        if "cache_salt" in request_body:
            raise CapabilityUnavailableError("the openai_compat adapter does not support cache_salt")
        payload = dict(request_body)
        payload["stream"] = True
        options = payload.get("stream_options")
        options = dict(options) if isinstance(options, Mapping) else {}
        options["include_usage"] = True
        payload["stream_options"] = options
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        if self.api_key:
            request.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        started = self._clock()
        first_output: float | None = None
        ended: float | None = None
        usage: dict[str, Any] = {}
        model: str | None = None
        fingerprint: str | None = None
        response_id: str | None = None
        pieces: list[str] = []
        deltas: list[dict[str, Any]] = []
        finish_reason: str | None = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                for event in _events(response):
                    now = self._clock()
                    if isinstance(event.get("id"), str):
                        response_id = event["id"]
                    if isinstance(event.get("model"), str):
                        model = event["model"]
                    if isinstance(event.get("system_fingerprint"), str):
                        fingerprint = event["system_fingerprint"]
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict) and delta:
                            deltas.append(dict(delta))
                            content = delta.get("content")
                            tool = delta.get("tool_calls") not in (None, [], {})
                            if isinstance(content, str) and content:
                                pieces.append(content)
                            if first_output is None and (
                                (isinstance(content, str) and content) or tool
                            ):
                                first_output = now
                        if isinstance(choice.get("finish_reason"), str):
                            finish_reason = choice["finish_reason"]
                            ended = now
        except urllib.error.HTTPError as exc:
            raise BackendRequestError(
                f"the OpenAI-compatible API returned HTTP {exc.code}"
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise BackendRequestError(
                "the OpenAI-compatible API was unreachable or the stream was interrupted"
            ) from exc
        if finish_reason is None:
            raise BackendRequestError(
                "the OpenAI-compatible stream ended without a finish_reason"
            )
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        canonical = json.dumps(deltas, sort_keys=True, separators=(",", ":"))
        return ProviderCapturedResponse(
            prompt_token_ids=None,  # type: ignore[arg-type]
            input_tokens=_int(usage.get("prompt_tokens")),
            output_tokens=_int(usage.get("completion_tokens")),
            cached_tokens=_int(details.get("cached_tokens")),
            client_ttft_ms=None if first_output is None else (first_output - started) * 1000,
            total_latency_ms=((ended if ended is not None else self._clock()) - started) * 1000,
            server_ttft_ms=None,
            server_queue_ms=None,
            generation_time_ms=None,
            mean_itl_ms=None,
            tokens_per_second=None,
            response_id=response_id,
            status="completed" if finish_reason in _COMPLETED_REASONS else "incomplete",
            output_digest=hashlib.sha256(canonical.encode()).hexdigest() if deltas else None,
            output_text="".join(pieces) or None,
            response_model=model,
            provider_fingerprint_sha256=fingerprint_digest(fingerprint),
        )

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        raise CapabilityUnavailableError(
            "an OpenAI-compatible API cannot reset its cache; use warm, shared isolation"
        )

    def scrape_metrics(self) -> dict[str, Any] | None:
        return None

    def capability_record(self) -> dict[str, Any] | None:
        downgrades = [
            {
                "capability": name,
                "status": "unsupported",
                "effect": effect,
            }
            for name, effect in (
                (
                    "exact_prompt_tokens",
                    "Prompt tokens are count-only. Prefix-reuse predictions and "
                    "token-level mechanism evidence are unknown.",
                ),
                (
                    "server_timing",
                    "Server TTFT and queue time are unknown. Client timing remains measured.",
                ),
                (
                    "cache_reset",
                    "The warm run started from an unknown provider cache state; "
                    "no physical reset was possible.",
                ),
                ("cache_salt", "Salt-isolated cold runs are unavailable."),
                (
                    "prometheus_metrics",
                    "Server load telemetry is unknown.",
                ),
            )
        ]
        return {
            "backend_kind": "openai_compat",
            "capability_table_version": "0.1",
            "identity_grade": PROVIDER_DECLARED,
            "downgrades": downgrades,
        }
