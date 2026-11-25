"""Minimal Prometheus text-exposition parsing for server telemetry.

Only a fixed allowlist of numeric server metrics is kept, mapped to stable
RunEconomy names. Label values, which can carry model paths or hostnames, are
dropped: samples of one metric are summed across label sets.
"""

from __future__ import annotations

import math
import re
from typing import Mapping


_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)(?:\s+\S+)?$"
)

# RunEconomy name -> candidate source metric names, first match wins.
VLLM_METRICS: Mapping[str, tuple[str, ...]] = {
    "running_requests": ("vllm:num_requests_running",),
    "waiting_requests": ("vllm:num_requests_waiting",),
    "kv_cache_usage_ratio": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "preemptions_total": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "prefix_cache_hits_total": ("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    "prefix_cache_queries_total": (
        "vllm:prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_queries_total",
    ),
}
SGLANG_METRICS: Mapping[str, tuple[str, ...]] = {
    "running_requests": ("sglang:num_running_reqs",),
    "waiting_requests": ("sglang:num_queue_reqs",),
    "kv_cache_usage_ratio": ("sglang:token_usage",),
    "prefix_cache_hit_rate": ("sglang:cache_hit_rate",),
    "generation_throughput_tokens_per_s": ("sglang:gen_throughput",),
}


def parse_samples(text: str) -> dict[str, float]:
    """Sum finite samples per metric name across label sets."""

    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        name = match.group("name")
        totals[name] = totals.get(name, 0.0) + value
    return totals


def select_metrics(
    text: str, mapping: Mapping[str, tuple[str, ...]]
) -> dict[str, float | None]:
    """Map known source metrics to RunEconomy names; absent metrics are None."""

    samples = parse_samples(text)
    selected: dict[str, float | None] = {}
    for target, sources in mapping.items():
        selected[target] = next(
            (samples[source] for source in sources if source in samples), None
        )
    hits = selected.get("prefix_cache_hits_total")
    queries = selected.get("prefix_cache_queries_total")
    if hits is not None and queries:
        # Lifetime hit rate from counters; it replaces a directly published
        # rate only when both counters are present.
        selected["prefix_cache_hit_rate"] = hits / queries
    elif "prefix_cache_hits_total" in mapping:
        selected.setdefault("prefix_cache_hit_rate", None)
    return selected
