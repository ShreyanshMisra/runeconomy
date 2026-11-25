"""The serving-backend adapter protocol and capability tables.

A :class:`BackendAdapter` is everything RunEconomy needs from a serving
backend:

* **capability discovery**: a read-only probe that never generates text and
  never mutates server state (``discover``)
* **identity**: what the backend reports about itself, with the grade that
  says who stands behind it (``identity``, see :mod:`runeconomy.identity`)
* **request execution**: one streamed generation (``capture``)
* **cache reset** (optional): a physical prefix-cache reset
  (``reset_prefix_cache``); adapters that cannot reset set
  ``supports_cache_reset = False`` and raise :class:`CapabilityUnavailableError`
* **metrics** (optional): one Prometheus scrape (``scrape_metrics``)

Every adapter publishes a capability table with one fixed set of names. A
capability is ``supported``, ``unsupported`` or ``unknown``, and every
non-supported entry states what it costs the experiment. Nothing is silently
assumed: an unknown capability is recorded as unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ..vllm import CapturedResponse, VLLMError


ADAPTER_PROTOCOL_VERSION = "0.1"
CAPABILITY_TABLE_VERSION = "0.1"
CAPABILITY_STATUSES = ("supported", "unsupported", "unknown")

# name -> (label, what the experiment loses when the capability is absent)
CAPABILITY_DEFINITIONS: dict[str, tuple[str, str]] = {
    "chat_completions": (
        "Chat Completions API",
        "Static replay requires Chat Completions on this backend kind.",
    ),
    "responses": (
        "Responses API",
        "The Responses dialect is unavailable; use Chat Completions.",
    ),
    "exact_prompt_tokens": (
        "Exact prompt tokens",
        "Prompt tokens are count-only. Prefix-reuse predictions and token-level "
        "mechanism evidence are unknown.",
    ),
    "cache_telemetry": (
        "Cache telemetry",
        "Cached-token counts are unknown unless the response usage reports them.",
    ),
    "server_timing": (
        "Server timing",
        "Server TTFT and queue time are unknown. Client timing remains measured.",
    ),
    "cache_reset": (
        "Cache reset",
        "Cold physical-reset isolation is unavailable, and a warm run starts "
        "from an unknown cache state.",
    ),
    "cache_salt": (
        "Cache salt",
        "Salt-isolated cold runs are unavailable.",
    ),
    "model_identity": (
        "Model metadata",
        "Served-model identity cannot be read from the backend.",
    ),
    "prometheus_metrics": (
        "Prometheus metrics",
        "Server load telemetry (queue depth, cache utilization) is unknown.",
    ),
}
CAPABILITY_NAMES = tuple(CAPABILITY_DEFINITIONS)


class CapabilityUnavailableError(RuntimeError):
    """An adapter was asked for a capability its backend kind does not have."""


class BackendRequestError(VLLMError):
    """A request to a non-vLLM serving backend failed.

    It subclasses :class:`runeconomy.vllm.VLLMError` so the CLI's existing
    request-failure classification and exit codes apply unchanged.
    """


@dataclass(frozen=True)
class CapabilityEntry:
    name: str
    status: str
    evidence: str

    def __post_init__(self) -> None:
        if self.name not in CAPABILITY_DEFINITIONS:
            raise ValueError(f"unknown capability {self.name!r}")
        if self.status not in CAPABILITY_STATUSES:
            raise ValueError(f"invalid capability status {self.status!r}")

    def to_dict(self) -> dict[str, str]:
        label, effect = CAPABILITY_DEFINITIONS[self.name]
        value = {
            "name": self.name,
            "label": label,
            "status": self.status,
            "evidence": self.evidence,
        }
        if self.status != "supported":
            value["effect"] = effect
        return value


@dataclass(frozen=True)
class CapabilityTable:
    backend_kind: str
    entries: tuple[CapabilityEntry, ...]

    def __post_init__(self) -> None:
        names = [entry.name for entry in self.entries]
        if sorted(names) != sorted(CAPABILITY_NAMES) or len(set(names)) != len(names):
            raise ValueError("a capability table must list every capability exactly once")

    def status(self, name: str) -> str:
        return next(entry.status for entry in self.entries if entry.name == name)

    def downgrades(self) -> list[dict[str, str]]:
        return [entry.to_dict() for entry in self.entries if entry.status != "supported"]

    def to_dict(self) -> dict[str, Any]:
        order = {name: index for index, name in enumerate(CAPABILITY_NAMES)}
        return {
            "capability_table_version": CAPABILITY_TABLE_VERSION,
            "backend_kind": self.backend_kind,
            "capabilities": [
                entry.to_dict()
                for entry in sorted(self.entries, key=lambda entry: order[entry.name])
            ],
        }


@dataclass(frozen=True)
class AdapterProfile:
    """What a backend kind can do at most, independent of any deployment.

    The static manifest is validated against the profile before anything runs.
    Discovery may further downgrade a deployment, never upgrade it.
    """

    kind: str
    label: str
    identity_grade: str
    api_dialects: frozenset[str]
    cache_isolation_modes: frozenset[str]
    supports_reset_external: bool
    exact_prompt_tokens: bool
    # Static capability limits that hold for every deployment of this kind.
    fixed_capabilities: Mapping[str, tuple[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingIdentity:
    """What a backend reports about itself, with the grade behind it."""

    backend_kind: str
    grade: str
    backend_version: str | None = None
    model_ids: tuple[str, ...] = ()
    model_path: str | None = None
    tokenizer_path: str | None = None
    undisclosed_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "grade": self.grade,
            "backend_version": self.backend_version,
            "model_ids": list(self.model_ids),
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "undisclosed_fields": list(self.undisclosed_fields),
        }


@runtime_checkable
class BackendAdapter(Protocol):
    """A serving backend, as the static runner and ``doctor`` see it."""

    kind: str
    profile: AdapterProfile
    supports_cache_reset: bool

    def discover(self) -> dict[str, Any]:
        """Read-only capability discovery, returned as a doctor report."""

    def identity(self) -> ServingIdentity:
        """Served identity from metadata, with its grade."""

    def capture(self, request_body: Mapping[str, Any]) -> CapturedResponse:
        """Run one streamed generation and capture its evidence."""

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        """Physically reset the prefix cache, or raise CapabilityUnavailableError."""

    def scrape_metrics(self) -> dict[str, Any] | None:
        """One Prometheus scrape of known server metrics, or ``None``."""

    def capability_record(self) -> dict[str, Any] | None:
        """Capability downgrades to record in a result, or ``None``."""
