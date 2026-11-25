"""Server telemetry sampled during concurrent runs.

:class:`ServerMetricsSampler` implements the optional
:class:`runeconomy.execution.LoadObserver` hooks. While a load block runs it
polls one or more Prometheus scrapes on a background thread, then attaches a
block summary (``telemetry.server_metrics``) and, for each trial, the most
recent sample from the server that served it (``server_telemetry``).

Telemetry is descriptive. It never enters a gate, a pairing decision or a
verdict, and a scrape failure leaves the affected values unknown rather than
stopping the run. Only the allowlisted numeric metrics from
:mod:`runeconomy.adapters.prometheus` are kept; label values are dropped.
"""

from __future__ import annotations

import math
import threading
import time
import urllib.request
from typing import Any, Callable, Mapping

from .adapters.prometheus import SGLANG_METRICS, VLLM_METRICS, select_metrics


TELEMETRY_SCHEMA_VERSION = "0.1"
GAUGES = (
    "running_requests",
    "waiting_requests",
    "kv_cache_usage_ratio",
    "generation_throughput_tokens_per_s",
)
COUNTERS = ("preemptions_total", "prefix_cache_hits_total", "prefix_cache_queries_total")
SERVER_TARGET = "server"

Scrape = Callable[[], Mapping[str, Any] | None]


def prometheus_url_scraper(url: str, *, timeout_seconds: float = 2.0) -> Scrape:
    """Scrape a ``/metrics`` URL and map vLLM or SGLang names."""

    mapping = merged_metric_sources(VLLM_METRICS, SGLANG_METRICS)

    def scrape() -> Mapping[str, Any] | None:
        request = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            text = response.read().decode("utf-8", errors="replace")
        return select_metrics(text, mapping)

    return scrape


def merged_metric_sources(
    *mappings: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Union source names per RunEconomy metric, in argument order.

    Engines share RunEconomy names (``running_requests`` comes from
    ``vllm:num_requests_running`` or ``sglang:num_running_reqs``). A dict
    merge would let the last engine's sources replace the first's, leaving
    that engine's gauges permanently unknown.
    """

    merged: dict[str, tuple[str, ...]] = {}
    for mapping in mappings:
        for name, sources in mapping.items():
            existing = merged.get(name, ())
            merged[name] = existing + tuple(source for source in sources if source not in existing)
    return merged


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


class _Target:
    def __init__(self, scrape: Scrape) -> None:
        self.scrape = scrape
        self.samples: list[tuple[float, dict[str, float | None]]] = []
        self.errors = 0
        self.latest: dict[str, float | None] | None = None


class ServerMetricsSampler:
    """Poll Prometheus scrapes while each load block runs."""

    def __init__(
        self,
        targets: Mapping[str, Scrape],
        *,
        interval_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not targets:
            raise ValueError("at least one metrics target is required")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self.clock = clock
        self._targets = {name: _Target(scrape) for name, scrape in targets.items()}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_all(self) -> None:
        for target in self._targets.values():
            try:
                values = target.scrape()
            except Exception:  # A missing sample is unknown, never fatal.
                values = None
            with self._lock:
                if values is None:
                    target.errors += 1
                    continue
                sample = {name: _finite(value) for name, value in values.items()}
                target.samples.append((self.clock(), sample))
                target.latest = sample

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_all()

    def on_block_start(self, info: Mapping[str, Any]) -> None:
        with self._lock:
            for target in self._targets.values():
                target.samples = []
                target.errors = 0
        self._stop.clear()
        self._sample_all()
        self._thread = threading.Thread(target=self._loop, name="runeconomy-metrics", daemon=True)
        self._thread.start()

    def on_block_end(self, info: Mapping[str, Any]) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._sample_all()
        with self._lock:
            summaries = {name: self._summarize(target) for name, target in self._targets.items()}
        return {"server_metrics": {"schema_version": TELEMETRY_SCHEMA_VERSION, "targets": summaries}}

    def on_trial_end(self, trial: Any, record: Mapping[str, Any]) -> dict[str, Any] | None:
        variant = record.get("variant") if isinstance(record, Mapping) else None
        name = variant if variant in self._targets else SERVER_TARGET
        with self._lock:
            target = self._targets.get(name)
            latest = None if target is None or target.latest is None else dict(target.latest)
        if latest is None:
            return None
        return {"metrics_target": name, **{key: latest.get(key) for key in GAUGES}}

    def _summarize(self, target: _Target) -> dict[str, Any]:
        samples = [values for _, values in target.samples]
        if not samples:
            return {"status": "unavailable", "samples": 0, "scrape_errors": target.errors}
        gauges: dict[str, Any] = {}
        for name in GAUGES:
            values = [sample[name] for sample in samples if sample.get(name) is not None]
            gauges[name] = (
                None
                if not values
                else {
                    "mean": sum(values) / len(values),
                    "max": max(values),
                    "p95": _nearest_rank(values, 0.95),
                }
            )
        counters: dict[str, float | None] = {}
        for name in COUNTERS:
            first, last = samples[0].get(name), samples[-1].get(name)
            counters[f"{name}_delta"] = (
                None if first is None or last is None or last < first else last - first
            )
        hits = counters.get("prefix_cache_hits_total_delta")
        queries = counters.get("prefix_cache_queries_total_delta")
        return {
            "status": "measured",
            "samples": len(samples),
            "scrape_errors": target.errors,
            "interval_seconds": self.interval_seconds,
            "gauges": gauges,
            "counters": counters,
            "prefix_cache_hit_rate_in_block": (
                hits / queries if hits is not None and queries else None
            ),
        }
