"""SGLang on the adapter protocol.

SGLang serves an OpenAI-compatible Chat Completions API, renders chat
messages through its tokenizer and template at ``/tokenize``, flushes its
radix (prefix) cache at ``POST /flush_cache``, reports model and server
metadata at ``/get_model_info`` and ``/get_server_info``, and exposes
Prometheus metrics at ``/metrics`` when started with ``--enable-metrics``.
Cached-token counts appear in usage only with ``--enable-cache-report``.

RunEconomy's SGLang adapter replays Chat Completions only. It reads no
per-request server timing, so server TTFT and queue time stay unknown.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..backends import BackendProbeError, Capability
from ..identity import OPERATOR_VERIFIED
from ..startup import StartupRetry
from ..vllm import CapturedResponse, VLLMClient, VLLMError
from . import prometheus
from .base import (
    AdapterProfile,
    BackendRequestError,
    CapabilityEntry,
    CapabilityTable,
    CapabilityUnavailableError,
    ServingIdentity,
)
from .http import get, post, validate_base_url
from .report import build_report, mode


SGLANG_PROFILE = AdapterProfile(
    kind="sglang",
    label="SGLang",
    identity_grade=OPERATOR_VERIFIED,
    api_dialects=frozenset({"chat_completions"}),
    cache_isolation_modes=frozenset({"shared", "physical_reset"}),
    supports_reset_external=False,
    exact_prompt_tokens=True,
)

_TOKENIZE_KEYS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
        "chat_template_kwargs",
    }
)


class _SGLangClient(VLLMClient):
    """The OpenAI-compatible capture path with SGLang's tokenizer endpoint."""

    def __init__(self, *args: Any, tokenize_path: str = "/tokenize", **kwargs: Any) -> None:
        super().__init__(*args, api_dialect="chat_completions", **kwargs)
        self.tokenize_path = tokenize_path

    def prompt_token_ids(self, request_body: Mapping[str, Any]) -> tuple[int, ...]:
        payload = {key: value for key, value in request_body.items() if key in _TOKENIZE_KEYS}
        result = self._post_json(self.tokenize_path, payload)
        tokens = result.get("tokens", result.get("input_ids"))
        if not isinstance(tokens, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in tokens
        ):
            raise BackendRequestError("SGLang tokenize response did not include integer tokens")
        return tuple(tokens)


def _relabel(exc: VLLMError) -> BackendRequestError:
    return BackendRequestError(str(exc).replace("vLLM", "SGLang"))


def _flag(info: Mapping[str, Any] | None, key: str) -> bool | None:
    if not isinstance(info, Mapping):
        return None
    value = info.get(key)
    if isinstance(value, bool):
        return value
    nested = info.get("server_args")
    if isinstance(nested, Mapping) and isinstance(nested.get(key), bool):
        return nested[key]
    return None


class SGLangAdapter:
    kind = "sglang"
    profile = SGLANG_PROFILE
    supports_cache_reset = True

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        startup_retry_seconds: float = 0.0,
        api_dialect: str = "chat_completions",
        on_startup_retry: Callable[[float], None] | None = None,
        tokenize_path: str = "/tokenize",
    ) -> None:
        if api_dialect != "chat_completions":
            raise CapabilityUnavailableError(
                "the SGLang adapter replays the chat_completions dialect only"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.startup_retry_seconds = startup_retry_seconds
        self.on_startup_retry = on_startup_retry
        self.client = _SGLangClient(
            self.base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            startup_retry_seconds=startup_retry_seconds,
            on_startup_retry=on_startup_retry,
            tokenize_path=tokenize_path,
        )
        self._report: dict[str, Any] | None = None
        self._table: CapabilityTable | None = None
        self._identity: ServingIdentity | None = None

    # -- discovery ---------------------------------------------------------

    def _metadata(self, startup: StartupRetry, *paths: str) -> Mapping[str, Any] | None:
        for path in paths:
            try:
                status, body = get(
                    self.base_url,
                    path,
                    api_key=self.api_key,
                    timeout_seconds=self.timeout_seconds,
                    label="SGLang",
                    startup=startup,
                )
            except BackendProbeError as exc:
                if exc.category in {"authentication", "connection", "timeout"}:
                    raise
                continue
            if status == 200 and isinstance(body, Mapping):
                return body
        return None

    def discover(self) -> dict[str, Any]:
        validate_base_url(self.base_url, label="SGLang")
        startup = StartupRetry(self.startup_retry_seconds, on_retry=self.on_startup_retry)
        openapi = self._metadata(startup, "/openapi.json")
        paths = set(openapi.get("paths", {})) if isinstance(openapi, Mapping) else set()
        advertised = openapi is not None
        model_info = self._metadata(startup, "/get_model_info", "/model_info")
        server_info = self._metadata(startup, "/get_server_info", "/server_info")
        models = self._metadata(startup, "/v1/models")
        model_ids = sorted(
            item["id"]
            for item in (models or {}).get("data", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        )
        version = server_info.get("version") if isinstance(server_info, Mapping) else None

        def path_status(*candidates: str) -> tuple[str, str]:
            if not advertised:
                return "unknown", "/openapi.json was unavailable; endpoint was not invoked"
            found = next((path for path in candidates if path in paths), None)
            if found:
                return "supported", f"{found} advertised by /openapi.json"
            return "unsupported", f"{' or '.join(candidates)} absent from /openapi.json"

        chat = path_status("/v1/chat/completions")
        tokenize = path_status("/tokenize", "/v1/tokenize")
        if tokenize[0] == "supported" and "/tokenize" not in paths:
            self.client.tokenize_path = "/v1/tokenize"
        flush = path_status("/flush_cache")
        cache_report = _flag(server_info, "enable_cache_report")
        metrics_enabled = _flag(server_info, "enable_metrics")
        radix_disabled = _flag(server_info, "disable_radix_cache")

        def flag_entry(name: str, flag: bool | None, option: str) -> CapabilityEntry:
            if flag is None:
                return CapabilityEntry(name, "unknown", f"{option} was not reported by /get_server_info")
            return CapabilityEntry(
                name,
                "supported" if flag else "unsupported",
                f"server started {'with' if flag else 'without'} {option}",
            )

        table = CapabilityTable(
            backend_kind="sglang",
            entries=(
                CapabilityEntry("chat_completions", *chat),
                CapabilityEntry(
                    "responses",
                    "unsupported",
                    "RunEconomy's SGLang adapter replays Chat Completions only",
                ),
                CapabilityEntry("exact_prompt_tokens", *tokenize),
                flag_entry("cache_telemetry", cache_report, "--enable-cache-report"),
                CapabilityEntry(
                    "server_timing",
                    "unsupported",
                    "the SGLang adapter reads no per-request server TTFT or queue time",
                ),
                CapabilityEntry(
                    "cache_reset",
                    flush[0],
                    flush[1] + ("; presence only, doctor never invokes it" if flush[0] == "supported" else ""),
                ),
                CapabilityEntry(
                    "cache_salt",
                    "unsupported",
                    "the SGLang adapter does not send cache_salt",
                ),
                CapabilityEntry(
                    "model_identity",
                    "supported" if model_info else "unknown",
                    "/get_model_info" if model_info else "/get_model_info was unavailable",
                ),
                flag_entry("prometheus_metrics", metrics_enabled, "--enable-metrics"),
            ),
        )
        identity = ServingIdentity(
            backend_kind="sglang",
            grade=OPERATOR_VERIFIED,
            backend_version=version if isinstance(version, str) else None,
            model_ids=tuple(model_ids),
            model_path=(
                model_info.get("model_path")
                if isinstance(model_info, Mapping) and isinstance(model_info.get("model_path"), str)
                else None
            ),
            tokenizer_path=(
                model_info.get("tokenizer_path")
                if isinstance(model_info, Mapping)
                and isinstance(model_info.get("tokenizer_path"), str)
                else None
            ),
        )
        exact = chat[0] == "supported" and tokenize[0] == "supported"
        modes = [
            mode(
                "warm-cache",
                "supported" if exact else "unknown" if "unknown" in (chat[0], tokenize[0]) else "unsupported",
                safe=True,
                physical_cache_state="unchanged",
                note="Measures normal radix-cache reuse with exact prompt tokens from /tokenize.",
                next_action="Expose /v1/chat/completions and /tokenize.",
            ),
            mode(
                "salt-isolated",
                "unsupported",
                safe=False,
                physical_cache_state="unchanged",
                note="Unavailable: the SGLang adapter has no per-request cache namespace.",
                next_action="Use warm-cache or physical-reset mode.",
            ),
            mode(
                "physical-reset",
                flush[0] if exact else "unsupported",
                safe=False,
                physical_cache_state="radix cache flushed before each trial",
                note=(
                    "Strong isolation for a private benchmark server. /flush_cache "
                    "fails while requests are running."
                ),
                next_action="Use only on a private benchmark server.",
            ),
        ]
        actions: list[str] = []
        if radix_disabled:
            actions.append(
                "The radix cache is disabled (--disable-radix-cache). Prefix-cache "
                "experiments will observe no reuse."
            )
        if cache_report is not True:
            actions.append(
                "Start SGLang with --enable-cache-report so cached-token counts are reported."
            )
        if metrics_enabled is not True:
            actions.append("Start SGLang with --enable-metrics to expose Prometheus telemetry.")
        actions.append(
            "Pin model revision, tokenizer revision, chat-template digest and "
            "engine configuration (including disable_radix_cache) in the manifest."
        )
        legacy = {
            "responses": Capability("unsupported", "the SGLang adapter replays Chat Completions only"),
            "chat_completions": Capability(chat[0], chat[1]),
            "responses_render": Capability("unsupported", "SGLang has no render endpoint"),
            "chat_completions_render": Capability(
                "unsupported", "SGLang has no render endpoint; /tokenize renders chat messages"
            ),
            "tokenize": Capability(tokenize[0], tokenize[1]),
            "cache_telemetry": Capability(
                table.status("cache_telemetry"), "--enable-cache-report server flag"
            ),
            "per_request_timing": Capability(
                "unsupported", "the SGLang adapter reads no per-request server timing"
            ),
            "cache_reset": Capability(flush[0], flush[1]),
            "cache_salt": Capability("unsupported", "the SGLang adapter does not send cache_salt"),
        }
        self._table, self._identity = table, identity
        self._report = build_report(
            table=table,
            identity=identity,
            legacy=legacy,
            modes=modes,
            next_actions=actions,
            probes=["metadata"],
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

    def capture(self, request_body: Mapping[str, Any]) -> CapturedResponse:
        if "cache_salt" in request_body:
            raise CapabilityUnavailableError("the SGLang adapter does not support cache_salt")
        try:
            return self.client.capture(request_body)
        except BackendRequestError:
            raise
        except VLLMError as exc:
            raise _relabel(exc) from exc

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        if reset_external:
            raise CapabilityUnavailableError(
                "SGLang has no external KV-cache reset; set execution.reset_external to false"
            )
        status, body = post(
            self.base_url,
            "/flush_cache",
            {},
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        if status != 200:
            raise BackendRequestError(
                f"SGLang /flush_cache returned HTTP {status}; ensure no request is "
                "running and the endpoint is reachable on this private server"
            )

    def scrape_metrics(self) -> dict[str, Any] | None:
        status, text = get(
            self.base_url,
            "/metrics",
            api_key=self.api_key,
            timeout_seconds=min(self.timeout_seconds, 10.0),
            label="SGLang",
            parse_json=False,
        )
        if status != 200 or not isinstance(text, str):
            return None
        return prometheus.select_metrics(text, prometheus.SGLANG_METRICS)

    def capability_record(self) -> dict[str, Any] | None:
        return {
            "backend_kind": "sglang",
            "capability_table_version": "0.1",
            "identity_grade": OPERATOR_VERIFIED,
            "downgrades": [
                {
                    "capability": "server_timing",
                    "status": "unsupported",
                    "effect": "Server TTFT and queue time are unknown. Client timing remains measured.",
                },
                {
                    "capability": "cache_salt",
                    "status": "unsupported",
                    "effect": "Salt-isolated cold runs are unavailable.",
                },
            ],
        }
