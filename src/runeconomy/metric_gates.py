"""Preregistered tail-latency and unit-economics gates with paired intervals.

``docs/load-pairing-and-tail-gates.md`` section 4 is the normative spec. Each
gate bounds the candidate-minus-baseline difference ``D`` in the metric's
*worse* direction with a two-sided 95% interval ``[L, U]``:

* superiority passes iff ``U < -margin``;
* non-inferiority passes iff ``U < margin``;
* a gate reports a regression iff ``L > margin`` (non-inferiority) or
  ``L > 0`` (superiority).

Two methods are implemented.

``quantile_order_statistic``
    Distribution-free Bonferroni combination of four one-sided order-statistic
    bounds, each at ``alpha / 4``. Valid for any dependence between arms and
    conservative under task-stratified sampling. Workflow-level quantiles only.

``paired_cluster_bootstrap``
    Percentile bootstrap that resamples tasks, carrying every repetition and
    both arms together. Approximate; needs the larger documented minimum n.

Unbounded interval ends are reported as ``None`` with an explicit
``*_unbounded`` flag so artifacts stay strict JSON.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .paired_inference import binomial_cdf


ORDER_STATISTIC = "quantile_order_statistic"
CLUSTER_BOOTSTRAP = "paired_cluster_bootstrap"
METHODS = frozenset({ORDER_STATISTIC, CLUSTER_BOOTSTRAP})
SUCCESSFUL_PAIRS = "successful_pairs"
ALL_ATTEMPTS_CONSERVATIVE = "all_attempts_conservative"
POPULATIONS = frozenset({SUCCESSFUL_PAIRS, ALL_ATTEMPTS_CONSERVATIVE})
SUPERIORITY = "superiority"
NONINFERIORITY = "noninferiority"
QUANTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
CONFIDENCE = 0.95
ALPHA = 1.0 - CONFIDENCE
BOOTSTRAP_REPLICATES = 4000
MIN_BOOTSTRAP_CLUSTERS = 20
MIN_BOOTSTRAP_TAIL_OBSERVATIONS = 10
MIN_BOOTSTRAP_OBSERVATIONS = 20


class MetricGateError(ValueError):
    pass


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    level: str  # workflow | request | ratio
    higher_is_better: bool
    unit: str
    description: str


METRICS: dict[str, MetricDefinition] = {
    definition.name: definition
    for definition in (
        MetricDefinition(
            "workflow_latency_ms",
            "workflow",
            False,
            "ms",
            "end-to-end latency of one unit (agent process wall time, or the "
            "sum of a static workflow's request latencies)",
        ),
        MetricDefinition(
            "client_ttft_ms", "request", False, "ms", "client-observed time to first token"
        ),
        MetricDefinition(
            "server_ttft_ms", "request", False, "ms", "server-reported time to first token"
        ),
        MetricDefinition(
            "tpot_ms",
            "request",
            False,
            "ms",
            "client-observed time per output token after the first: "
            "(total_latency_ms - client_ttft_ms) / (output_tokens - 1)",
        ),
        MetricDefinition(
            "usd_per_1k_successful_workflows",
            "ratio",
            False,
            "usd",
            "1000 x total cost of all population attempts / successful attempts",
        ),
        MetricDefinition(
            "successful_workflows_per_gpu_hour",
            "ratio",
            True,
            "workflows/gpu-hour",
            "goodput: successful attempts / total GPU-hours of all population attempts",
        ),
    )
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class MetricGate:
    metric: str
    statistic: str
    semantics: str
    method: str
    population: str
    margin: float | None = None
    relative_margin: float | None = None

    @property
    def definition(self) -> MetricDefinition:
        return METRICS[self.metric]

    @property
    def quantile(self) -> float | None:
        return QUANTILES.get(self.statistic)

    @property
    def name(self) -> str:
        statistic = "" if self.statistic == "ratio" else f"{self.statistic}_"
        return f"{statistic}{self.metric}_{self.semantics}"

    @classmethod
    def from_value(cls, value: Any, location: str) -> "MetricGate":
        if not isinstance(value, Mapping):
            raise MetricGateError(f"{location} must be an object")
        allowed = {
            "metric",
            "statistic",
            "semantics",
            "method",
            "population",
            "margin",
            "relative_margin",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise MetricGateError(f"{location} has unsupported fields: " + ", ".join(unknown))
        metric = value.get("metric")
        if metric not in METRICS:
            raise MetricGateError(
                f"{location}.metric must be one of " + ", ".join(sorted(METRICS))
            )
        definition = METRICS[metric]
        if definition.level == "ratio":
            statistic = value.get("statistic", "ratio")
            if statistic != "ratio":
                raise MetricGateError(f"{location}.statistic must be 'ratio' for {metric}")
        else:
            statistic = value.get("statistic")
            if statistic not in {"mean", *QUANTILES}:
                raise MetricGateError(
                    f"{location}.statistic must be mean, p50, p90, p95, or p99"
                )
        semantics = value.get("semantics")
        if semantics not in {SUPERIORITY, NONINFERIORITY}:
            raise MetricGateError(
                f"{location}.semantics must be superiority or noninferiority"
            )
        default_method = (
            ORDER_STATISTIC
            if definition.level == "workflow" and statistic in QUANTILES
            else CLUSTER_BOOTSTRAP
        )
        method = value.get("method", default_method)
        if method not in METHODS:
            raise MetricGateError(
                f"{location}.method must be quantile_order_statistic or "
                "paired_cluster_bootstrap"
            )
        if method == ORDER_STATISTIC and (
            definition.level != "workflow" or statistic not in QUANTILES
        ):
            raise MetricGateError(
                f"{location}.method quantile_order_statistic applies only to "
                "quantiles of workflow-level metrics; request-level and ratio "
                "metrics are clustered and require paired_cluster_bootstrap"
            )
        population = value.get("population", SUCCESSFUL_PAIRS)
        if population not in POPULATIONS:
            raise MetricGateError(
                f"{location}.population must be successful_pairs or "
                "all_attempts_conservative"
            )
        if definition.level == "ratio" and "population" in value:
            raise MetricGateError(
                f"{location}.population is fixed for ratio metrics: all population attempts"
            )
        if definition.level == "ratio":
            population = ALL_ATTEMPTS_CONSERVATIVE
        if "margin" in value and "relative_margin" in value:
            raise MetricGateError(f"{location} must declare margin or relative_margin, not both")
        margin = value.get("margin")
        relative = value.get("relative_margin")
        if margin is not None and (not _finite_number(margin) or float(margin) < 0):
            raise MetricGateError(f"{location}.margin must be a non-negative number")
        if relative is not None and (
            not _finite_number(relative) or not 0 <= float(relative) < 1
        ):
            raise MetricGateError(f"{location}.relative_margin must be in [0, 1)")
        if margin is None and relative is None:
            margin = 0.0
        return cls(
            metric=metric,
            statistic=statistic,
            semantics=semantics,
            method=method,
            population=population,
            margin=None if margin is None else float(margin),
            relative_margin=None if relative is None else float(relative),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metric": self.metric,
            "statistic": self.statistic,
            "semantics": self.semantics,
            "method": self.method,
        }
        if self.definition.level != "ratio":
            # Ratio metrics always use every population attempt; the field is
            # implied and rejected on input, so it is omitted here too.
            payload["population"] = self.population
        if self.relative_margin is not None:
            payload["relative_margin"] = self.relative_margin
        else:
            payload["margin"] = self.margin
        return payload


def parse_metric_gates(value: Any, location: str = "acceptance.metric_gates") -> tuple[MetricGate, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise MetricGateError(f"{location} must be an array")
    gates = tuple(
        MetricGate.from_value(item, f"{location}[{index}]") for index, item in enumerate(value)
    )
    names = [gate.name for gate in gates]
    if len(set(names)) != len(names):
        raise MetricGateError(f"{location} declares the same gate twice")
    return gates


# --------------------------------------------------------------------------
# Order statistics


def order_statistic_upper_rank(n: int, q: float, a: float) -> int | None:
    """Smallest 1-based rank k with P(X_(k) >= Q_q) >= 1 - a, or None."""

    if n <= 0:
        return None
    start = max(1, math.floor(n * q))
    for k in range(start, n + 1):
        if binomial_cdf(k - 1, n, q) >= 1.0 - a:
            return k
    return None


def order_statistic_lower_rank(n: int, q: float, a: float) -> int | None:
    """Largest 1-based rank j with P(X_(j) <= Q_q) >= 1 - a, or None."""

    if n <= 0:
        return None
    start = min(n, math.ceil(n * q))
    for j in range(start, 0, -1):
        if 1.0 - binomial_cdf(j - 1, n, q) >= 1.0 - a:
            return j
    return None


def order_statistic_minimum_n(q: float, alpha: float = ALPHA) -> int:
    """Smallest n for which both marginal bounds exist at alpha / 4."""

    a = alpha / 4.0
    upper = math.ceil(math.log(a) / math.log(q) - 1e-12)
    lower = math.ceil(math.log(a) / math.log(1.0 - q) - 1e-12)
    return max(upper, lower, 1)


def quantile_bounds(values: Sequence[float], q: float, a: float) -> tuple[float, float]:
    """One-sided ``1 - a`` lower and upper bounds for the q-quantile."""

    ordered = sorted(values)
    n = len(ordered)
    lower_rank = order_statistic_lower_rank(n, q, a)
    upper_rank = order_statistic_upper_rank(n, q, a)
    lower = -math.inf if lower_rank is None else ordered[lower_rank - 1]
    upper = math.inf if upper_rank is None else ordered[upper_rank - 1]
    return lower, upper


def nearest_rank(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


# --------------------------------------------------------------------------
# Observations


@dataclass(frozen=True)
class Observation:
    """One population attempt reduced to what metric gates need."""

    cluster: str
    repetition: int
    variant: str
    success: bool | None
    workflow_latency_ms: float | None
    requests: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    cost_usd: float | None = None
    gpu_time_ms: float | None = None

    def values(self, metric: str) -> tuple[float, ...] | None:
        if metric == "workflow_latency_ms":
            return None if self.workflow_latency_ms is None else (self.workflow_latency_ms,)
        return tuple(self.requests.get(metric, ()))


def tpot_ms(total_latency_ms: Any, client_ttft_ms: Any, output_tokens: Any) -> float | None:
    if (
        not _finite_number(total_latency_ms)
        or not _finite_number(client_ttft_ms)
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens < 2
    ):
        return None
    value = (float(total_latency_ms) - float(client_ttft_ms)) / (output_tokens - 1)
    return value if value >= 0 else None


def economics_of(record: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Return (cost_usd, gpu_time_ms), preferring a cost model's attachment."""

    economics = record.get("economics")
    source = economics if isinstance(economics, Mapping) else record.get("metrics")
    source = source if isinstance(source, Mapping) else {}
    cost = source.get("cost_usd")
    gpu = source.get("gpu_time_ms")
    return (
        float(cost) if _finite_number(cost) and float(cost) >= 0 else None,
        float(gpu) if _finite_number(gpu) and float(gpu) >= 0 else None,
    )


# --------------------------------------------------------------------------
# Statistics on a (resampled) collection of clusters


_Cluster = dict[str, list[Observation]]


def _clusters(observations: Iterable[Observation]) -> dict[str, _Cluster]:
    clusters: dict[str, _Cluster] = {}
    for item in observations:
        clusters.setdefault(item.cluster, {"baseline": [], "candidate": []})
        if item.variant in ("baseline", "candidate"):
            clusters[item.cluster][item.variant].append(item)
    return clusters


def _successful_pair_values(
    cluster: _Cluster, metric: str
) -> tuple[list[float], list[float]]:
    by_repetition = {
        variant: {item.repetition: item for item in cluster[variant]}
        for variant in ("baseline", "candidate")
    }
    baseline: list[float] = []
    candidate: list[float] = []
    for repetition in sorted(by_repetition["baseline"]):
        left = by_repetition["baseline"][repetition]
        right = by_repetition["candidate"].get(repetition)
        if right is None or left.success is not True or right.success is not True:
            continue
        left_values = left.values(metric)
        right_values = right.values(metric)
        if not left_values or not right_values:
            continue
        baseline.extend(left_values)
        candidate.extend(right_values)
    return baseline, candidate


def _conservative_values(cluster: _Cluster, metric: str) -> tuple[list[float], list[float]]:
    baseline: list[float] = []
    candidate: list[float] = []
    for item in cluster["baseline"]:
        values = item.values(metric)
        if values:
            baseline.extend(values)
    for item in cluster["candidate"]:
        values = item.values(metric)
        candidate.extend(values if values else (math.inf,))
    return baseline, candidate


def _cluster_summaries(gate: MetricGate, clusters: Sequence[_Cluster]) -> list[Any]:
    """Reduce each cluster once so bootstrap replicates only re-aggregate."""

    if gate.definition.level == "ratio":
        summaries: list[Any] = []
        for cluster in clusters:
            row: list[Any] = []
            for variant in ("baseline", "candidate"):
                total: float | None = 0.0
                successes = 0
                for item in cluster[variant]:
                    value = (
                        item.cost_usd
                        if gate.metric == "usd_per_1k_successful_workflows"
                        else item.gpu_time_ms
                    )
                    if value is None or total is None:
                        total = None
                    else:
                        total += value
                    successes += item.success is True
                row.extend((total, successes, len(cluster[variant])))
            summaries.append(tuple(row))
        return summaries
    extractor = (
        _successful_pair_values
        if gate.population == SUCCESSFUL_PAIRS
        else _conservative_values
    )
    return [extractor(cluster, gate.metric) for cluster in clusters]


def _ratio_value(metric: str, total: float, successes: int) -> float:
    if metric == "usd_per_1k_successful_workflows":
        return math.inf if successes == 0 else 1000.0 * total / successes
    return math.nan if total <= 0 else successes / (total / 3_600_000.0)


def _statistic(gate: MetricGate, values: Sequence[float]) -> float | None:
    if not values:
        return None
    if gate.statistic == "mean":
        return statistics.fmean(values) if all(math.isfinite(v) for v in values) else math.inf
    return nearest_rank(values, QUANTILES[gate.statistic])


def _point(gate: MetricGate, summaries: Sequence[Any]) -> tuple[float | None, float | None]:
    if gate.definition.level == "ratio":
        result: list[float | None] = []
        for offset in (0, 3):
            if any(row[offset] is None for row in summaries):
                result.append(None)
                continue
            total = sum(row[offset] for row in summaries)
            successes = sum(row[offset + 1] for row in summaries)
            result.append(_ratio_value(gate.metric, total, successes))
        return result[0], result[1]
    baseline = [value for row in summaries for value in row[0]]
    candidate = [value for row in summaries for value in row[1]]
    return _statistic(gate, baseline), _statistic(gate, candidate)


def _worse_difference(gate: MetricGate, baseline: float, candidate: float) -> float:
    """Candidate-minus-baseline in the worse direction (relative if declared).

    Undefined values (NaN, or a zero relative base) count against the
    candidate, so an undefined replicate can never help a gate pass.
    """

    higher = gate.definition.higher_is_better
    with_errors = math.inf
    try:
        if gate.relative_margin is not None:
            if baseline == 0:
                return with_errors
            ratio = candidate / baseline
            value = (1.0 - ratio) if higher else (ratio - 1.0)
        else:
            value = (baseline - candidate) if higher else (candidate - baseline)
    except (ArithmeticError, ValueError):
        return with_errors
    return with_errors if math.isnan(value) else value


def _json_bound(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else value


def minimum_sample(gate: MetricGate) -> dict[str, Any]:
    """The documented minimum analyzable sample for one gate."""

    if gate.method == ORDER_STATISTIC:
        assert gate.quantile is not None
        return {
            "observations": order_statistic_minimum_n(gate.quantile),
            "clusters": None,
            "rule": "smallest n with q^n <= alpha/4 and (1-q)^n <= alpha/4",
        }
    if gate.quantile is not None:
        return {
            "observations": max(
                MIN_BOOTSTRAP_OBSERVATIONS,
                math.ceil(MIN_BOOTSTRAP_TAIL_OBSERVATIONS / (1.0 - gate.quantile) - 1e-9),
            ),
            "clusters": MIN_BOOTSTRAP_CLUSTERS,
            "rule": "at least 10 observations above the quantile and 20 clusters",
        }
    return {
        "observations": None,
        "clusters": MIN_BOOTSTRAP_CLUSTERS,
        "rule": "at least 20 clusters (tasks)",
    }


def _seed_for(seed: int, gate: MetricGate) -> int:
    digest = hashlib.sha256(f"{seed}:{gate.name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    count = len(ordered)
    low = ordered[max(0, math.ceil(0.025 * count) - 1)]
    high = ordered[max(0, math.ceil(0.975 * count) - 1)]
    return low, high


def _limits(gate: MetricGate) -> tuple[float, float]:
    """Return (pass_limit_for_upper, regression_limit_for_lower)."""

    margin = gate.relative_margin if gate.relative_margin is not None else gate.margin
    assert margin is not None
    if gate.semantics == SUPERIORITY:
        return -margin, 0.0
    return margin, margin


def evaluate_metric_gate(
    gate: MetricGate, observations: Sequence[Observation], *, seed: int
) -> dict[str, Any]:
    """Evaluate one gate and return a decision check."""

    clusters_by_id = _clusters(observations)
    cluster_ids = sorted(clusters_by_id)
    clusters = [clusters_by_id[key] for key in cluster_ids]
    summaries = _cluster_summaries(gate, clusters)
    baseline_point, candidate_point = _point(gate, summaries)
    minimum = minimum_sample(gate)
    if gate.definition.level == "ratio":
        analyzable = min(
            sum(row[2] for row in summaries), sum(row[5] for row in summaries)
        )
        baseline_values = candidate_values = []
    else:
        baseline_values = [value for row in summaries for value in row[0]]
        candidate_values = [value for row in summaries for value in row[1]]
        analyzable = min(len(baseline_values), len(candidate_values))
    unknown_inputs = gate.definition.level == "ratio" and (
        baseline_point is None or candidate_point is None
    )
    pass_limit, regression_limit = _limits(gate)
    observed: dict[str, Any] = {
        "baseline": _json_bound(baseline_point),
        "candidate": _json_bound(candidate_point),
        "candidate_unbounded": candidate_point is not None and math.isinf(candidate_point),
        "analyzable_observations": analyzable,
        "clusters": len(clusters),
        "population": gate.population,
        "method": gate.method,
        "minimum_sample": minimum,
        "difference_scale": "relative" if gate.relative_margin is not None else "absolute",
        "difference_direction": "positive means the candidate is worse",
    }
    threshold = {
        "semantics": gate.semantics,
        "margin": gate.margin,
        "relative_margin": gate.relative_margin,
        "rule": (
            f"upper bound of the two-sided 95% interval < {pass_limit:g}"
            + (" (relative)" if gate.relative_margin is not None else f" {gate.definition.unit}")
        ),
    }
    check: dict[str, Any] = {
        "name": gate.name,
        "metric": gate.metric,
        "statistic": gate.statistic,
        "required": True,
        "passed": False,
        "regression": False,
        "observed": observed,
        "threshold": threshold,
    }
    if unknown_inputs:
        observed["status"] = "unknown_inputs"
        observed["reason"] = (
            "cost or GPU-time was unknown for at least one population attempt; "
            "unknown is not treated as zero"
        )
        return check
    too_few_observations = (
        minimum["observations"] is not None and analyzable < minimum["observations"]
    )
    too_few_clusters = minimum["clusters"] is not None and len(clusters) < minimum["clusters"]
    if too_few_observations or too_few_clusters:
        observed["status"] = "insufficient_sample"
        observed["reason"] = (
            "the analyzable sample is below the documented minimum for this method"
        )
        return check

    if gate.method == ORDER_STATISTIC:
        assert gate.quantile is not None
        a = ALPHA / 4.0
        b_low, b_high = quantile_bounds(baseline_values, gate.quantile, a)
        c_low, c_high = quantile_bounds(candidate_values, gate.quantile, a)
        if gate.relative_margin is not None:
            # Order-statistic gates exist only for lower-is-better latency, so
            # the worse-direction relative change is candidate/baseline - 1.
            lower = -1.0 if math.isinf(b_high) else (
                -math.inf if b_high <= 0 else c_low / b_high - 1.0
            )
            upper = math.inf if b_low <= 0 else c_high / b_low - 1.0
        else:
            lower = c_low - b_high
            upper = c_high - b_low
        observed["marginal_bounds"] = {
            "baseline": [_json_bound(b_low), _json_bound(b_high)],
            "candidate": [_json_bound(c_low), _json_bound(c_high)],
            "each_one_sided_level": 1.0 - a,
        }
        observed["method_detail"] = (
            "distribution-free order-statistic bounds, Bonferroni-combined "
            "(four one-sided bounds at alpha/4)"
        )
    else:
        rng = random.Random(_seed_for(seed, gate))
        replicates: list[float] = []
        for _ in range(BOOTSTRAP_REPLICATES):
            count = len(summaries)
            sample = [summaries[rng.randrange(count)] for _ in range(count)]
            left, right = _point(gate, sample)
            if left is None or right is None:
                replicates.append(math.inf)
                continue
            replicates.append(_worse_difference(gate, left, right))
        lower, upper = _percentile_interval(replicates)
        observed["method_detail"] = (
            f"percentile bootstrap over {BOOTSTRAP_REPLICATES} task-cluster "
            "resamples, both arms and all repetitions resampled together"
        )
        observed["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    point = (
        None
        if baseline_point is None or candidate_point is None
        else _worse_difference(gate, baseline_point, candidate_point)
    )
    observed["candidate_worse_by"] = _json_bound(point)
    observed["confidence_interval"] = [_json_bound(lower), _json_bound(upper)]
    observed["lower_unbounded"] = math.isinf(lower)
    observed["upper_unbounded"] = math.isinf(upper)
    observed["confidence_level"] = CONFIDENCE
    observed["status"] = "evaluated"
    check["passed"] = bool(math.isfinite(upper) and upper < pass_limit)
    check["regression"] = bool(lower > regression_limit)
    return check


def evaluate_metric_gates(
    gates: Sequence[MetricGate], observations: Sequence[Observation], *, seed: int
) -> list[dict[str, Any]]:
    return [evaluate_metric_gate(gate, observations, seed=seed) for gate in gates]


# --------------------------------------------------------------------------
# Load-delivery integrity


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def load_integrity_checks(
    records: Sequence[Mapping[str, Any]],
    *,
    open_loop: bool,
    gate_level: float,
    max_dispatch_lag_ms: float | None,
) -> list[dict[str, Any]]:
    """Checks on whether the declared load was delivered on the gate stage."""

    by_variant: dict[str, list[Mapping[str, Any]]] = {"baseline": [], "candidate": []}
    for record in records:
        load = record.get("load")
        variant = record.get("variant")
        if isinstance(load, Mapping) and variant in by_variant:
            by_variant[str(variant)].append(load)
    checks: list[dict[str, Any]] = []
    if open_loop:
        assert max_dispatch_lag_ms is not None
        for variant, name in (
            ("baseline", "baseline_arrival_schedule_integrity"),
            ("candidate", "candidate_offered_load_completion"),
        ):
            lags = [
                float(item["dispatch_lag_ms"])
                for item in by_variant[variant]
                if _finite_number(item.get("dispatch_lag_ms"))
            ]
            p95 = nearest_rank(lags, 0.95)
            checks.append(
                {
                    "name": name,
                    "required": True,
                    "passed": bool(p95 is not None and p95 <= max_dispatch_lag_ms),
                    "observed": {
                        "trials": len(lags),
                        "p95_dispatch_lag_ms": p95,
                        "max_dispatch_lag_ms": max(lags) if lags else None,
                    },
                    "threshold": {
                        "max_p95_dispatch_lag_ms": max_dispatch_lag_ms,
                        "rule": "p95 of arrival-to-start lag on the gate stage",
                    },
                }
            )
    medians = {
        variant: _median(
            [
                float(item["mean_in_flight"])
                for item in items
                if _finite_number(item.get("mean_in_flight"))
            ]
        )
        for variant, items in by_variant.items()
    }
    balanced = all(value is not None for value in medians.values())
    if balanced and not open_loop:
        balanced = all(float(value) >= 0.9 * gate_level for value in medians.values())  # type: ignore[arg-type]
    checks.append(
        {
            "name": "observed_concurrency_balance",
            "required": False,
            "passed": bool(balanced),
            "observed": {
                "median_mean_in_flight": medians,
                "declared_level": gate_level,
            },
            "threshold": (
                "both variants' median time-weighted in-flight >= 90% of the declared concurrency"
                if not open_loop
                else "time-weighted in-flight recorded for both variants (descriptive)"
            ),
        }
    )
    return checks


def merge_load_and_metric_checks(
    decision: Mapping[str, Any],
    *,
    integrity_checks: Sequence[Mapping[str, Any]],
    metric_checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fold load-integrity and metric-gate checks into a legacy decision.

    Ordering follows the existing verdict precedence: a baseline that did not
    receive the declared load invalidates the comparison; a candidate that
    could not absorb it, or a metric gate whose interval lies wholly on the
    worse side, is a regression; a metric gate that merely did not pass turns
    ``verified`` into ``inconclusive``.
    """

    status = str(decision.get("status"))
    reason = str(decision.get("reason"))
    checks = [dict(check) for check in decision.get("checks", [])]
    baseline_integrity = [
        dict(check) for check in integrity_checks if check["name"] == "baseline_arrival_schedule_integrity"
    ]
    other_integrity = [
        dict(check) for check in integrity_checks if check["name"] != "baseline_arrival_schedule_integrity"
    ]
    completion_positions = [
        index for index, check in enumerate(checks) if "completion" in str(check.get("name"))
    ]
    insert_at = (completion_positions[-1] + 1) if completion_positions else len(checks)
    merged = (
        baseline_integrity
        + checks[:insert_at]
        + [check for check in other_integrity if check["name"] == "candidate_offered_load_completion"]
        + checks[insert_at:]
        + [check for check in other_integrity if check["name"] != "candidate_offered_load_completion"]
        + [dict(check) for check in metric_checks]
    )
    if any(check.get("passed") is False for check in baseline_integrity):
        status = "invalid"
        reason = "the baseline did not receive the declared arrival schedule, so the load was not established"
    elif status != "invalid":
        offered = [
            check
            for check in other_integrity
            if check["name"] == "candidate_offered_load_completion"
        ]
        regressions = [check["name"] for check in metric_checks if check.get("regression")]
        unmet = [check["name"] for check in metric_checks if not check.get("passed")]
        if any(check.get("passed") is False for check in offered):
            status = "regression"
            reason = "the candidate could not absorb the declared offered load on schedule"
        elif status != "regression" and regressions:
            status = "regression"
            reason = "metric gate interval lies wholly on the worse side: " + ", ".join(regressions)
        elif status == "verified" and unmet:
            status = "inconclusive"
            reason = "quality and latency gates passed, but metric gate(s) were not established: " + ", ".join(unmet)
    return {**decision, "status": status, "reason": reason, "checks": merged}
