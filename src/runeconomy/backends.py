"""Backend capability discovery without running inference requests."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class BackendProbeError(RuntimeError):
    """The backend could not be inspected safely."""


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

    def experiment_modes(self) -> tuple[dict[str, Any], ...]:
        generation = (
            self.responses.status == "supported"
            or self.chat_completions.status == "supported"
        )
        exact_tokens = (
            self.responses_render.status == "supported"
            or (
                self.chat_completions.status == "supported"
                and (
                    self.chat_completions_render.status == "supported"
                    or self.tokenize.status == "supported"
                )
            )
        )
        return (
            {
                "mode": "warm-cache",
                "supported": generation and exact_tokens,
                "physical_cache_state": "unchanged",
                "note": "Measures normal cache reuse without clearing server state.",
            },
            {
                "mode": "salt-isolated",
                "supported": generation and exact_tokens and self.cache_salt.status == "supported",
                "physical_cache_state": "unchanged",
                "note": (
                    "Uses a unique random cache namespace per trial. This prevents "
                    "cross-trial prefix reuse but does not evict cached blocks or remove "
                    "capacity and eviction effects."
                ),
            },
            {
                "mode": "physical-reset",
                "supported": generation and exact_tokens and self.cache_reset.status == "supported",
                "physical_cache_state": "reset before each trial",
                "note": (
                    "Strong isolation for a private benchmark server; the reset endpoint "
                    "is a development-mode capability and should not be exposed publicly."
                ),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_ids"] = list(self.model_ids)
        result["experiment_modes"] = list(self.experiment_modes())
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
        return exc.code, None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BackendProbeError(
            f"cannot reach vLLM at {sanitized_target(base_url)}: {exc}"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BackendProbeError(f"{path} returned invalid JSON") from exc


def _post_probe(
    base_url: str,
    path: str,
    payload: Mapping[str, Any],
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> tuple[int, str]:
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
            response.read(2048)
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(2048).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise BackendProbeError(
            f"cannot probe {path} at {sanitized_target(base_url)}: {exc}"
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
    if advertised.status != "supported" or payload is None:
        return advertised
    status, body = _post_probe(
        base_url,
        path,
        payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    if status == 200:
        return Capability("supported", "successful non-inference render probe")
    if status in {403, 404} and (
        status == 404 or "scale-out endpoints are disabled" in body.lower()
    ):
        return Capability(
            "unsupported",
            f"render probe returned HTTP {status}",
            "The route is advertised but scale-out rendering is unavailable.",
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

    openapi_status, openapi = _json_get(
        base_url, "/openapi.json", api_key=api_key, timeout_seconds=timeout_seconds
    )
    openapi_available = openapi_status == 200 and isinstance(openapi, Mapping)
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
        responses=_capability(paths, "/v1/responses", openapi=openapi_available),
        chat_completions=_capability(
            paths, "/v1/chat/completions", openapi=openapi_available
        ),
        responses_render=responses_render,
        chat_completions_render=chat_render,
        tokenize=_capability(paths, "/tokenize", openapi=openapi_available),
        cache_telemetry=telemetry,
        per_request_timing=timing,
        cache_reset=reset,
        cache_salt=cache_salt,
    )
