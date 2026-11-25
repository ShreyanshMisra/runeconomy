"""Serving-backend adapters: vLLM, SGLang and OpenAI-compatible APIs.

See :mod:`runeconomy.adapters.base` for the protocol and capability tables.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import (
    ADAPTER_PROTOCOL_VERSION,
    CAPABILITY_NAMES,
    CAPABILITY_TABLE_VERSION,
    AdapterProfile,
    BackendAdapter,
    BackendRequestError,
    CapabilityEntry,
    CapabilityTable,
    CapabilityUnavailableError,
    ServingIdentity,
)
from .openai_compat import (
    OPENAI_COMPAT_PROFILE,
    OpenAICompatAdapter,
    ProviderCapturedResponse,
)
from .report import DOCTOR_ADAPTER_SCHEMA_VERSION, format_report
from .sglang import SGLANG_PROFILE, SGLangAdapter
from .vllm_adapter import VLLM_PROFILE, VLLMAdapter, vllm_capability_table


BACKEND_KINDS = ("vllm", "sglang", "openai_compat")
PROFILES: dict[str, AdapterProfile] = {
    "vllm": VLLM_PROFILE,
    "sglang": SGLANG_PROFILE,
    "openai_compat": OPENAI_COMPAT_PROFILE,
}
_ADAPTERS: dict[str, type] = {
    "vllm": VLLMAdapter,
    "sglang": SGLangAdapter,
    "openai_compat": OpenAICompatAdapter,
}


def profile_for(kind: str) -> AdapterProfile:
    try:
        return PROFILES[kind]
    except KeyError as exc:
        raise ValueError(
            "backend kind must be one of " + ", ".join(BACKEND_KINDS)
        ) from exc


def create_adapter(
    kind: str,
    base_url: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 300.0,
    startup_retry_seconds: float = 0.0,
    api_dialect: str | None = None,
    on_startup_retry: Callable[[float], None] | None = None,
    **options: Any,
) -> BackendAdapter:
    """Construct the adapter for one backend kind."""

    profile_for(kind)
    arguments: dict[str, Any] = {
        "api_key": api_key,
        "timeout_seconds": timeout_seconds,
        "startup_retry_seconds": startup_retry_seconds,
        "on_startup_retry": on_startup_retry,
        **options,
    }
    if api_dialect is not None:
        arguments["api_dialect"] = api_dialect
    return _ADAPTERS[kind](base_url, **arguments)


__all__ = [
    "ADAPTER_PROTOCOL_VERSION",
    "BACKEND_KINDS",
    "CAPABILITY_NAMES",
    "CAPABILITY_TABLE_VERSION",
    "DOCTOR_ADAPTER_SCHEMA_VERSION",
    "PROFILES",
    "AdapterProfile",
    "BackendAdapter",
    "BackendRequestError",
    "CapabilityEntry",
    "CapabilityTable",
    "CapabilityUnavailableError",
    "OpenAICompatAdapter",
    "ProviderCapturedResponse",
    "SGLangAdapter",
    "ServingIdentity",
    "VLLMAdapter",
    "create_adapter",
    "format_report",
    "profile_for",
    "vllm_capability_table",
]
