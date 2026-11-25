"""vLLM on the adapter protocol, delegating to the existing client unchanged.

Capture, cache reset and discovery call :class:`runeconomy.vllm.VLLMClient`
and :func:`runeconomy.backends.discover_vllm_capabilities` exactly as the
static runner and ``doctor`` did before the protocol existed. Doctor JSON,
trial records and results are byte-for-byte what they were.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..backends import discover_vllm_capabilities
from ..identity import OPERATOR_VERIFIED
from ..vllm import CapturedResponse, VLLMClient
from . import prometheus
from .base import AdapterProfile, CapabilityEntry, CapabilityTable, ServingIdentity
from .http import get


VLLM_PROFILE = AdapterProfile(
    kind="vllm",
    label="vLLM",
    identity_grade=OPERATOR_VERIFIED,
    api_dialects=frozenset({"responses", "chat_completions"}),
    cache_isolation_modes=frozenset({"shared", "salt_isolated", "physical_reset"}),
    supports_reset_external=True,
    exact_prompt_tokens=True,
)


def _exact_tokens(report: Mapping[str, Any]) -> CapabilityEntry:
    def status(key: str) -> str:
        value = report.get(key)
        return value.get("status", "unknown") if isinstance(value, Mapping) else "unknown"

    paths = (
        (status("responses"), status("responses_render")),
        (status("chat_completions"), status("chat_completions_render")),
        (status("chat_completions"), status("tokenize")),
    )
    if any(api == "supported" and tokens == "supported" for api, tokens in paths):
        return CapabilityEntry(
            "exact_prompt_tokens", "supported", "render or /tokenize path for a supported API"
        )
    if all("unsupported" in pair for pair in paths):
        return CapabilityEntry(
            "exact_prompt_tokens", "unsupported", "no render or /tokenize path for a supported API"
        )
    return CapabilityEntry(
        "exact_prompt_tokens", "unknown", "render or /tokenize availability is unknown"
    )


def vllm_capability_table(report: Mapping[str, Any]) -> CapabilityTable:
    """Derive the protocol capability table from a vLLM doctor report."""

    def entry(name: str, key: str) -> CapabilityEntry:
        value = report.get(key)
        value = value if isinstance(value, Mapping) else {}
        return CapabilityEntry(
            name,
            str(value.get("status", "unknown")),
            str(value.get("evidence", "not reported")),
        )

    return CapabilityTable(
        backend_kind="vllm",
        entries=(
            entry("chat_completions", "chat_completions"),
            entry("responses", "responses"),
            _exact_tokens(report),
            entry("cache_telemetry", "cache_telemetry"),
            entry("server_timing", "per_request_timing"),
            entry("cache_reset", "cache_reset"),
            entry("cache_salt", "cache_salt"),
            CapabilityEntry(
                "model_identity",
                "supported" if report.get("model_ids") else "unknown",
                "/v1/models" if report.get("model_ids") else "no model IDs were listed",
            ),
            CapabilityEntry(
                "prometheus_metrics",
                "unknown",
                "doctor does not scrape /metrics; it is optional for vLLM",
            ),
        ),
    )


class VLLMAdapter:
    kind = "vllm"
    profile = VLLM_PROFILE
    supports_cache_reset = True

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        startup_retry_seconds: float = 0.0,
        api_dialect: str = "responses",
        on_startup_retry: Callable[[float], None] | None = None,
        client: VLLMClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.startup_retry_seconds = startup_retry_seconds
        self.on_startup_retry = on_startup_retry
        self.client = client or VLLMClient(
            base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            startup_retry_seconds=startup_retry_seconds,
            api_dialect=api_dialect,
            on_startup_retry=on_startup_retry,
        )
        self._report: dict[str, Any] | None = None

    def discover(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.startup_retry_seconds:
            options.update(
                {
                    "startup_retry_seconds": self.startup_retry_seconds,
                    "on_startup_retry": self.on_startup_retry,
                }
            )
        self._report = discover_vllm_capabilities(self.base_url, **options).to_dict()
        return self._report

    def capability_table(self) -> CapabilityTable:
        return vllm_capability_table(self._report or self.discover())

    def identity(self) -> ServingIdentity:
        report = self._report or self.discover()
        return ServingIdentity(
            backend_kind="vllm",
            grade=OPERATOR_VERIFIED,
            backend_version=report.get("backend_version"),
            model_ids=tuple(report.get("model_ids") or ()),
        )

    def capture(self, request_body: Mapping[str, Any]) -> CapturedResponse:
        return self.client.capture(request_body)

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        self.client.reset_prefix_cache(reset_external=reset_external)

    def scrape_metrics(self) -> dict[str, Any] | None:
        status, text = get(
            self.base_url,
            "/metrics",
            api_key=self.api_key,
            timeout_seconds=min(self.timeout_seconds, 10.0),
            label="vLLM",
            parse_json=False,
        )
        if status != 200 or not isinstance(text, str):
            return None
        return prometheus.select_metrics(text, prometheus.VLLM_METRICS)

    def capability_record(self) -> dict[str, Any] | None:
        # vLLM is the reference backend: results record no adapter section so
        # they stay identical to results recorded before the protocol existed.
        return None
