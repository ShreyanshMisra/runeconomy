"""Versioned cost models that turn attempt records into economics.

A cost model is declared in the manifest (``cost_model``, schema 0.4) and is
therefore covered by the manifest fingerprint and by ``experiment freeze``.
It is the ``attempt_economics`` callable the runners already accept: each
population attempt receives ``record["economics"]`` before any decision is
made, and the ``usd_per_1k_successful_workflows`` and
``successful_workflows_per_gpu_hour`` metric gates read it.

Two kinds are supported.

``gpu_hourly``
    Reserved accelerator time priced per GPU-hour. An attempt is charged its
    wall time multiplied by ``gpus_per_variant``. Under concurrent load the
    reserved time is shared, so the charge is divided by the attempt's
    time-weighted mean in-flight count (``record.load.mean_in_flight``); the
    charges of one block then sum to roughly the block's reserved GPU time.
    This is a fair-share attribution, not a measurement of kernel time.

``per_token``
    Provider list prices per million input, cached-input and output tokens.
    An attempt with unknown token counts has unknown cost: it is never
    priced as zero.

The model's canonical SHA-256 is recorded with every economics attachment, so
a result names exactly which prices produced its cost gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


COST_MODEL_SCHEMA_VERSION = "0.1"
COST_MODEL_KINDS = ("gpu_hourly", "per_token")
_GPU_FIELDS = {"schema_version", "kind", "usd_per_gpu_hour", "gpus_per_variant", "source"}
_TOKEN_FIELDS = {
    "schema_version",
    "kind",
    "usd_per_1m_input_tokens",
    "usd_per_1m_cached_input_tokens",
    "usd_per_1m_output_tokens",
    "source",
}


class CostModelError(ValueError):
    """A cost-model declaration that cannot be applied."""


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CostModelError(f"cost_model.{field} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise CostModelError(
            f"cost_model.{field} must be {'positive' if positive else 'non-negative'}"
        )
    return float(value)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass(frozen=True)
class CostModel:
    kind: str
    payload: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        canonical = json.dumps(dict(self.payload), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def __call__(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.economics(record)

    def economics(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if self.kind == "gpu_hourly":
            return self._gpu_hourly(record)
        return self._per_token(record)

    def _gpu_hourly(self, record: Mapping[str, Any]) -> dict[str, Any]:
        duration = attempt_wall_time_ms(record)
        share = 1.0
        load = record.get("load")
        if isinstance(load, Mapping):
            mean = _finite(load.get("mean_in_flight"))
            if mean is None or mean <= 0:
                duration = None
            else:
                share = max(1.0, mean)
        gpus = float(self.payload["gpus_per_variant"])
        gpu_time = None if duration is None else duration * gpus / share
        cost = (
            None
            if gpu_time is None
            else gpu_time / 3_600_000.0 * float(self.payload["usd_per_gpu_hour"])
        )
        return {
            "cost_usd": cost,
            "gpu_time_ms": gpu_time,
            "cost_model_sha256": self.sha256,
            "attribution": (
                "fair_share_reserved_gpu_time" if isinstance(load, Mapping) else "reserved_gpu_time"
            ),
            "wall_time_ms": duration,
            "concurrency_share": share,
        }

    def _per_token(self, record: Mapping[str, Any]) -> dict[str, Any]:
        tokens = attempt_tokens(record)
        input_tokens, cached, output = tokens["input_tokens"], tokens["cached_tokens"], tokens["output_tokens"]
        cost: float | None = None
        if input_tokens is not None and output is not None:
            cached_price = self.payload.get("usd_per_1m_cached_input_tokens")
            cached_count = cached or 0
            if cached and cached_price is None:
                cached_price = self.payload["usd_per_1m_input_tokens"]
            uncached = max(0, input_tokens - cached_count)
            cost = (
                uncached * float(self.payload["usd_per_1m_input_tokens"])
                + cached_count * float(cached_price or 0.0)
                + output * float(self.payload["usd_per_1m_output_tokens"])
            ) / 1_000_000.0
        return {
            "cost_usd": cost,
            "gpu_time_ms": None,
            "cost_model_sha256": self.sha256,
            "attribution": "provider_list_price",
            **tokens,
        }


def attempt_wall_time_ms(record: Mapping[str, Any]) -> float | None:
    """Wall time of one attempt (agent process or static request/workflow)."""

    load = record.get("load")
    if isinstance(load, Mapping):
        started, ended = _finite(load.get("started_ms")), _finite(load.get("ended_ms"))
        if started is not None and ended is not None and ended >= started:
            return ended - started
    metrics = record.get("metrics")
    sources = [metrics] if isinstance(metrics, Mapping) else []
    sources.append(record)
    for source in sources:
        for key in ("workflow_latency_ms", "process_wall_time_ms", "total_latency_ms", "latency_ms"):
            value = _finite(source.get(key))
            if value is not None:
                return value
    return None


def attempt_tokens(record: Mapping[str, Any]) -> dict[str, int | None]:
    """Input, cached-input and output token counts, or None when unknown."""

    metrics = record.get("metrics")
    sources = [metrics] if isinstance(metrics, Mapping) else []
    sources.append(record)
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        sources.append(usage)

    def first(*keys: str) -> int | None:
        for source in sources:
            for key in keys:
                value = _integer(source.get(key))
                if value is not None:
                    return value
        return None

    return {
        "input_tokens": first("input_tokens", "prompt_tokens", "total_input_tokens"),
        "cached_tokens": first("cached_tokens", "cached_input_tokens", "total_cached_tokens", "actual_cached_tokens"),
        "output_tokens": first("output_tokens", "completion_tokens", "total_output_tokens"),
    }


def parse_cost_model(value: Any) -> CostModel | None:
    """Validate a manifest ``cost_model`` declaration."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CostModelError("cost_model must be an object")
    if value.get("schema_version") != COST_MODEL_SCHEMA_VERSION:
        raise CostModelError(
            f"cost_model.schema_version must be {COST_MODEL_SCHEMA_VERSION!r}"
        )
    kind = value.get("kind")
    if kind not in COST_MODEL_KINDS:
        raise CostModelError("cost_model.kind must be gpu_hourly or per_token")
    if "source" in value and (not isinstance(value["source"], str) or not value["source"].strip()):
        raise CostModelError("cost_model.source must be a non-empty string when supplied")
    if kind == "gpu_hourly":
        unknown = set(value) - _GPU_FIELDS
        if unknown:
            raise CostModelError("cost_model has unsupported fields: " + ", ".join(sorted(unknown)))
        _number(value.get("usd_per_gpu_hour"), "usd_per_gpu_hour", positive=True)
        gpus = value.get("gpus_per_variant", 1)
        if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus <= 0:
            raise CostModelError("cost_model.gpus_per_variant must be a positive integer")
        payload = {**dict(value), "gpus_per_variant": gpus}
    else:
        unknown = set(value) - _TOKEN_FIELDS
        if unknown:
            raise CostModelError("cost_model has unsupported fields: " + ", ".join(sorted(unknown)))
        _number(value.get("usd_per_1m_input_tokens"), "usd_per_1m_input_tokens")
        _number(value.get("usd_per_1m_output_tokens"), "usd_per_1m_output_tokens")
        if value.get("usd_per_1m_cached_input_tokens") is not None:
            _number(value["usd_per_1m_cached_input_tokens"], "usd_per_1m_cached_input_tokens")
        payload = dict(value)
    return CostModel(kind=str(kind), payload=payload)
