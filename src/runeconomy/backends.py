"""Backend capability discovery without running inference requests."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


class BackendProbeError(RuntimeError):
    """The backend could not be inspected safely."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "probe_failure",
        next_action: str = "Check the server logs and rerun `runeconomy doctor`.",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.next_action = next_action


DEFAULT_BENCHMARK_FLAGS = (
    "--enable-prefix-caching",
    "--enable-prompt-tokens-details",
    "--enable-per-request-metrics",
    "--enable-scale-out",
)


def missing_vllm_flags(
    source_text: str, required: tuple[str, ...] = DEFAULT_BENCHMARK_FLAGS
) -> tuple[str, ...]:
    """Return benchmark flags absent from an installed vLLM source tree."""

    return tuple(flag for flag in required if flag not in source_text)


@dataclass(frozen=True)
class Capability:
    status: str
    evidence: str
    detail: str | None = None
    required: bool = False
    next_action: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"supported", "unsupported", "unknown"}:
            raise ValueError(f"invalid capability status: {self.status}")


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    backend_version: str | None
    model_ids: tuple[str, ...]
    responses: Capability
    chat_completions: Capability
    responses_render: Capability
    chat_completions_render: Capability
    tokenize: Capability
    cache_telemetry: Capability
    per_request_timing: Capability
    cache_reset: Capability
    cache_salt: Capability
    model_revision: str | None = None
    tokenizer_identity: str | None = None
    chat_template_identity: str | None = None
    engine_configuration: str | None = None

    def readiness(self) -> dict[str, Any]:
        modes = self.experiment_modes()
        supported_modes = [
            mode["mode"]
            for mode in modes
            if mode["status"] == "supported" and mode["safe"]
        ]
        responses_path = (
            self.responses.status == "supported"
            and self.responses_render.status == "supported"
        )
        chat_path = (
            self.chat_completions.status == "supported"
            and (
                self.chat_completions_render.status == "supported"
                or self.tokenize.status == "supported"
            )
        )
        generation = (
            self.responses.status == "supported"
            or self.chat_completions.status == "supported"
        )
        exact_tokens = responses_path or chat_path
        blockers: list[str] = []
        actions: list[str] = []
        if not generation:
            blockers.append("No supported Responses or Chat Completions generation API was detected.")
            actions.append(
                "Expose `/v1/chat/completions` or `/v1/responses`, then rerun `runeconomy doctor`."
            )
        if not exact_tokens:
            blockers.append("No safe exact-token render or tokenize path was detected.")
            actions.append(
                "Enable `/tokenize` or start vLLM with `--enable-scale-out` so a render endpoint is available."
            )
        if supported_modes:
            overall = "ready"
            summary = f"Safe modes available: {', '.join(supported_modes)}."
        elif any(mode["status"] == "unknown" for mode in modes):
            overall = "unknown"
            summary = "Compatibility could not be established from the available metadata."
        else:
            overall = "blocked"
            summary = "No safe RunEconomy experiment mode is currently available."
        if not actions and (
            self.cache_telemetry.status == "unknown"
            or self.per_request_timing.status == "unknown"
        ):
            actions.append(
                "Run one non-production smoke capture, then inspect whether cached-token and server-timing fields are present."
            )
        return {
            "status": overall,
            "summary": summary,
            "supported_modes": supported_modes,
            "blockers": blockers,
            "next_actions": actions,
        }

    def experiment_modes(self) -> tuple[dict[str, Any], ...]:
        responses_path = (
            self.responses.status == "supported"
            and self.responses_render.status == "supported"
        )
        chat_path = (
            self.chat_completions.status == "supported"
            and (
                self.chat_completions_render.status == "supported"
                or self.tokenize.status == "supported"
            )
        )
        generation = (
            self.responses.status == "supported"
            or self.chat_completions.status == "supported"
        )
        exact_tokens = responses_path or chat_path
        modes = (
            {
                "mode": "warm-cache",
                "status": "supported" if generation and exact_tokens else "unsupported",
                "supported": generation and exact_tokens,
                "safe": generation and exact_tokens,
                "physical_cache_state": "unchanged",
                "note": "Measures normal cache reuse without clearing server state.",
                "next_action": (
                    "No additional server capability is required."
                    if generation and exact_tokens
                    else "Enable a generation API and an exact-token render or tokenize path."
                ),
            },
            {
                "mode": "salt-isolated",
                "status": (
                    "supported"
                    if generation and exact_tokens and self.cache_salt.status == "supported"
                    else "unknown"
                    if self.cache_salt.status == "unknown"
                    else "unsupported"
                ),
                "supported": generation and exact_tokens and self.cache_salt.status == "supported",
                "safe": generation and exact_tokens and self.cache_salt.status == "supported",
                "physical_cache_state": "unchanged",
                "note": (
                    "Uses a unique random cache namespace per trial. This prevents "
                    "cross-trial prefix reuse but does not evict cached blocks or remove "
                    "capacity and eviction effects."
                ),
                "next_action": (
                    "No additional server capability is required."
                    if generation and exact_tokens and self.cache_salt.status == "supported"
                    else "Use warm-cache mode, or upgrade/configure vLLM to accept `cache_salt`."
                ),
            },
            {
                "mode": "physical-reset",
                "status": (
                    "supported"
                    if generation and exact_tokens and self.cache_reset.status == "supported"
                    else "unknown"
                    if self.cache_reset.status == "unknown"
                    else "unsupported"
                ),
                "supported": generation and exact_tokens and self.cache_reset.status == "supported",
                "safe": False,
                "physical_cache_state": "reset before each trial",
                "note": (
                    "Strong isolation for a private benchmark server; the reset endpoint "
                    "is a development-mode capability and should not be exposed publicly."
                ),
                "next_action": (
                    "Use only on a private benchmark server; RunEconomy doctor never invokes the reset endpoint."
                    if self.cache_reset.status == "supported"
                    else "Use warm-cache or salt-isolated mode; do not expose development-mode reset publicly."
                ),
            },
        )
        return modes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = "0.2"
        result["model_ids"] = list(self.model_ids)
        result["experiment_modes"] = list(self.experiment_modes())
        result["readiness"] = self.readiness()
        result["probe_safety"] = {
            "read_only": True,
            "generation_invoked": False,
            "cache_reset_invoked": False,
            "probes": ["metadata", "non-inference render when advertised"],
        }
        identity_actions = {
            "model_revision": "Pin the exact model revision in the experiment manifest.",
            "tokenizer_identity": "Record the tokenizer name and revision in the experiment manifest.",
            "chat_template_identity": "Hash or pin the rendered chat template in the experiment manifest.",
            "engine_configuration": "Record cache, block-size, memory, sequence, and routing settings that can affect the result.",
        }
        result["identity"] = {
            key: {
                "status": "supported" if getattr(self, key) else "unknown",
                "value": getattr(self, key),
                "next_action": action,
            }
            for key, action in identity_actions.items()
        }
        return result


def sanitized_target(base_url: str) -> str:
    """Return a display-safe origin without credentials, path, query, or fragment."""

    parsed = urllib.parse.urlsplit(base_url)
    hostname = parsed.hostname or "unknown-host"
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, hostname, "", "", ""))


def _capability(paths: set[str], path: str, *, openapi: bool) -> Capability:
    if openapi:
        if path in paths:
            return Capability("supported", "advertised by /openapi.json")
        return Capability("unsupported", "absent from /openapi.json")
    return Capability("unknown", "/openapi.json was unavailable; endpoint was not invoked")


def _with_guidance(
    capability: Capability, *, required: bool, next_action: str
) -> Capability:
    return Capability(
        status=capability.status,
        evidence=capability.evidence,
        detail=capability.detail,
        required=required,
        next_action=capability.next_action or next_action,
    )


def _json_get(
    base_url: str,
    path: str,
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", headers=headers, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        exc.read(2048)
        if exc.code in {401, 403}:
            raise BackendProbeError(
                f"the server rejected the {path} metadata request with HTTP {exc.code}",
                category="authentication",
                next_action=(
                    "Set the API key in an environment variable and rerun with "
                    "`--api-key-env NAME`; verify that the key may access metadata endpoints."
                ),
            ) from exc
        return exc.code, None
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(reason, TimeoutError)
        raise BackendProbeError(
            "the vLLM endpoint did not respond before the timeout"
            if timed_out
            else "the vLLM endpoint could not be reached",
            category="timeout" if timed_out else "connection",
            next_action=(
                "Increase `--timeout`, verify the server is running, and rerun `runeconomy doctor`."
                if timed_out
                else "Verify the base URL, network access, and server process, then rerun `runeconomy doctor`."
            ),
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BackendProbeError(
            f"the server returned malformed JSON for {path}",
            category="malformed_response",
            next_action=(
                "Confirm the URL points to an OpenAI-compatible vLLM server rather than a proxy login or HTML error page."
            ),
        ) from exc


def _post_probe(
    base_url: str,
    path: str,
    payload: Mapping[str, Any],
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            try:
                payload = json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise BackendProbeError(
                    f"the server returned malformed JSON for {path}",
                    category="malformed_response",
                    next_action=(
                        "Confirm the endpoint is a vLLM render API and inspect the reverse-proxy response."
                    ),
                ) from exc
            if not isinstance(payload, Mapping):
                raise BackendProbeError(
                    f"the server returned a non-object JSON value for {path}",
                    category="malformed_response",
                    next_action="Confirm the endpoint is a compatible vLLM render API.",
                )
            return response.status, payload
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(2048).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        timed_out = isinstance(reason, TimeoutError)
        raise BackendProbeError(
            f"the safe capability probe for {path} timed out"
            if timed_out
            else f"the safe capability probe for {path} could not reach the server",
            category="timeout" if timed_out else "connection",
            next_action=(
                "Increase `--timeout`, verify server health, and rerun `runeconomy doctor`."
                if timed_out
                else "Verify network access and rerun `runeconomy doctor`."
            ),
        ) from exc


def _render_capability(
    advertised: Capability,
    *,
    base_url: str,
    path: str,
    payload: Mapping[str, Any] | None,
    api_key: str | None,
    timeout_seconds: float,
) -> Capability:
    if advertised.status != "supported":
        return advertised
    if payload is None:
        return Capability(
            "unknown",
            "advertised but not probed because no model ID was available",
            "A model ID is required for a safe render probe.",
        )
    status, body = _post_probe(
        base_url,
        path,
        payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    if status == 200:
        token_ids = body.get("token_ids", body.get("prompt_token_ids"))
        if token_ids is None and isinstance(body.get("engine_input"), Mapping):
            token_ids = body["engine_input"].get("prompt_token_ids")
        if isinstance(token_ids, list) and token_ids and all(
            isinstance(token, int) and not isinstance(token, bool)
            for token in token_ids
        ):
            return Capability("supported", "successful non-inference render probe")
        return Capability(
            "unknown",
            "render probe returned HTTP 200 without integer token IDs",
            "The response shape was not compatible with RunEconomy's renderer.",
        )
    if status in {403, 404} and (
        status == 404
        or (
            isinstance(body, str)
            and "scale-out endpoints are disabled" in body.lower()
        )
    ):
        return Capability(
            "unsupported",
            f"render probe returned HTTP {status}",
            "The route is advertised but scale-out rendering is unavailable.",
        )
    if status in {401, 403}:
        return Capability(
            "unknown",
            f"render probe returned HTTP {status}",
            "The render endpoint may require different authorization.",
            next_action=(
                "Verify the bearer key named by `--api-key-env` may call render endpoints, then rerun doctor."
            ),
        )
    return Capability(
        "unknown",
        f"render probe returned HTTP {status}",
        "The endpoint was advertised but could not be verified safely.",
    )


def discover_vllm_capabilities(
    base_url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> BackendCapabilities:
    """Inspect public metadata endpoints; never generate text or clear caches."""

    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BackendProbeError(
            "the base URL must be an absolute HTTP or HTTPS URL",
            category="configuration",
            next_action="Rerun with `--base-url http://HOST:PORT` or an HTTPS equivalent.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise BackendProbeError(
            "the base URL must not contain embedded credentials",
            category="configuration",
            next_action="Remove credentials from the URL and supply the bearer key with `--api-key-env NAME`.",
        )
    if parsed.query or parsed.fragment:
        raise BackendProbeError(
            "the base URL must not contain a query string or fragment",
            category="configuration",
            next_action="Pass the server root or reverse-proxy path without query parameters or a fragment.",
        )

    openapi_status, openapi = _json_get(
        base_url, "/openapi.json", api_key=api_key, timeout_seconds=timeout_seconds
    )
    if openapi_status >= 500:
        raise BackendProbeError(
            f"the server returned HTTP {openapi_status} for OpenAPI metadata",
            category="server_error",
            next_action="Inspect the vLLM or reverse-proxy logs, correct the server error, and rerun `runeconomy doctor`.",
        )
    openapi_available = openapi_status == 200 and isinstance(openapi, Mapping)
    if openapi_status == 200 and not isinstance(openapi, Mapping):
        raise BackendProbeError(
            "the OpenAPI document was JSON but not an object",
            category="malformed_response",
            next_action="Verify that the URL points to the root of a vLLM OpenAI-compatible server.",
        )
    raw_paths = openapi.get("paths", {}) if openapi_available else {}
    paths = set(raw_paths) if isinstance(raw_paths, Mapping) else set()

    version: str | None = None
    try:
        status, payload = _json_get(
            base_url, "/version", api_key=api_key, timeout_seconds=timeout_seconds
        )
        if status == 200 and isinstance(payload, Mapping):
            value = payload.get("version")
            version = value if isinstance(value, str) else None
    except BackendProbeError:
        pass

    model_ids: list[str] = []
    try:
        status, payload = _json_get(
            base_url, "/v1/models", api_key=api_key, timeout_seconds=timeout_seconds
        )
        if status == 200 and isinstance(payload, Mapping):
            data = payload.get("data")
            if isinstance(data, list):
                model_ids = sorted(
                    item["id"]
                    for item in data
                    if isinstance(item, Mapping) and isinstance(item.get("id"), str)
                )
    except BackendProbeError:
        pass

    schema_text = json.dumps(openapi, sort_keys=True) if openapi_available else ""
    cache_salt = (
        Capability("supported", "cache_salt is advertised in the OpenAPI schema")
        if '"cache_salt"' in schema_text
        else Capability(
            "unsupported" if openapi_available else "unknown",
            "cache_salt is absent from the OpenAPI schema"
            if openapi_available
            else "/openapi.json was unavailable",
        )
    )
    reset = _capability(paths, "/reset_prefix_cache", openapi=openapi_available)
    if reset.status == "supported":
        reset = Capability(
            "supported",
            reset.evidence,
            "Presence only; doctor does not invoke this destructive development endpoint.",
        )

    # These fields are response- and configuration-dependent. Endpoint discovery
    # alone cannot establish that a deployment returns them for real requests.
    telemetry = Capability(
        "unknown",
        "requires a completed generation response",
        "Run a smoke capture to verify cached-token fields.",
    )
    timing = Capability(
        "unknown",
        "requires a completed generation response",
        "Run a smoke capture to verify server-side timing metrics.",
    )

    model = model_ids[0] if model_ids else None
    responses_render = _render_capability(
        _capability(paths, "/v1/responses/render", openapi=openapi_available),
        base_url=base_url,
        path="/v1/responses/render",
        payload=None if model is None else {"model": model, "input": "capability probe"},
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    chat_render = _render_capability(
        _capability(paths, "/v1/chat/completions/render", openapi=openapi_available),
        base_url=base_url,
        path="/v1/chat/completions/render",
        payload=(
            None
            if model is None
            else {
                "model": model,
                "messages": [{"role": "user", "content": "capability probe"}],
            }
        ),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    return BackendCapabilities(
        backend="vllm",
        backend_version=version,
        model_ids=tuple(model_ids),
        responses=_with_guidance(
            _capability(paths, "/v1/responses", openapi=openapi_available),
            required=False,
            next_action=(
                "No action required when Chat Completions is supported; otherwise expose `/v1/responses`."
            ),
        ),
        chat_completions=_with_guidance(
            _capability(paths, "/v1/chat/completions", openapi=openapi_available),
            required=False,
            next_action=(
                "No action required when Responses is supported; otherwise expose `/v1/chat/completions`."
            ),
        ),
        responses_render=_with_guidance(
            responses_render,
            required=False,
            next_action="Use Chat Completions render or `/tokenize`, or enable vLLM scale-out rendering.",
        ),
        chat_completions_render=_with_guidance(
            chat_render,
            required=False,
            next_action="Use Responses render or `/tokenize`, or enable vLLM scale-out rendering.",
        ),
        tokenize=_with_guidance(
            _capability(paths, "/tokenize", openapi=openapi_available),
            required=False,
            next_action="Enable `/tokenize`, or provide a supported render endpoint.",
        ),
        cache_telemetry=_with_guidance(
            telemetry,
            required=False,
            next_action="Run one smoke capture and inspect cached-token fields; absence limits mechanism evidence, not execution.",
        ),
        per_request_timing=_with_guidance(
            timing,
            required=False,
            next_action="Run one smoke capture and inspect server timing fields; client timing remains available.",
        ),
        cache_reset=_with_guidance(
            reset,
            required=False,
            next_action="Use only on a private development server; prefer warm-cache or salt isolation otherwise.",
        ),
        cache_salt=_with_guidance(
            cache_salt,
            required=False,
            next_action="Use warm-cache mode, or configure a vLLM version that accepts `cache_salt`.",
        ),
    )
