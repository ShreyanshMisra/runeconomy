from __future__ import annotations

import json
import math
import random
import unittest
from unittest.mock import patch

from runeconomy import metric_gates as mg
from runeconomy.metric_gates import (
    MetricGate,
    MetricGateError,
    Observation,
    evaluate_metric_gate,
    load_integrity_checks,
    merge_load_and_metric_checks,
    minimum_sample,
    order_statistic_lower_rank,
    order_statistic_minimum_n,
    order_statistic_upper_rank,
    parse_metric_gates,
    quantile_bounds,
)


def _gate(**overrides) -> MetricGate:
    value = {
        "metric": "workflow_latency_ms",
        "statistic": "p95",
        "semantics": "noninferiority",
        "margin": 0,
        **overrides,
    }
    if "relative_margin" in overrides:
        value.pop("margin")
    value = {key: item for key, item in value.items() if item is not None}
    return MetricGate.from_value(value, "gate")


def _pairs(
    baseline: list[float],
    candidate: list[float],
    *,
    repetitions: int = 1,
    cost: tuple[float, float] | None = None,
    success: tuple[bool, bool] = (True, True),
) -> list[Observation]:
    observations: list[Observation] = []
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        cluster = f"task-{index // repetitions}"
        repetition = index % repetitions
        for variant, value, ok, spend in (
            ("baseline", left, success[0], None if cost is None else cost[0]),
            ("candidate", right, success[1], None if cost is None else cost[1]),
        ):
            observations.append(
                Observation(
                    cluster=cluster,
                    repetition=repetition,
                    variant=variant,
                    success=ok,
                    workflow_latency_ms=value,
                    cost_usd=spend,
                    gpu_time_ms=None if spend is None else 1000.0,
                )
            )
    return observations


class OrderStatisticTests(unittest.TestCase):
    def test_documented_minimum_sample_table(self) -> None:
        self.assertEqual(order_statistic_minimum_n(0.50), 7)
        self.assertEqual(order_statistic_minimum_n(0.90), 42)
        self.assertEqual(order_statistic_minimum_n(0.95), 86)
        self.assertEqual(order_statistic_minimum_n(0.99), 437)
        a = mg.ALPHA / 4
        self.assertIsNone(order_statistic_upper_rank(85, 0.95, a))
        self.assertIsNotNone(order_statistic_upper_rank(86, 0.95, a))

    def test_bootstrap_minimum_samples_are_documented(self) -> None:
        p95 = minimum_sample(
            _gate(statistic="p95", method="paired_cluster_bootstrap")
        )
        p99 = minimum_sample(
            _gate(statistic="p99", method="paired_cluster_bootstrap")
        )
        mean = minimum_sample(_gate(statistic="mean"))
        self.assertEqual((p95["observations"], p95["clusters"]), (200, 20))
        self.assertEqual(p99["observations"], 1000)
        self.assertEqual((mean["observations"], mean["clusters"]), (None, 20))

    def test_ranks_satisfy_their_defining_binomial_inequalities(self) -> None:
        for n in (86, 150, 400):
            for q in (0.5, 0.95):
                a = 0.0125
                k = order_statistic_upper_rank(n, q, a)
                j = order_statistic_lower_rank(n, q, a)
                assert k is not None and j is not None
                self.assertGreaterEqual(mg.binomial_cdf(k - 1, n, q), 1 - a)
                if k > 1:
                    self.assertLess(mg.binomial_cdf(k - 2, n, q), 1 - a)
                self.assertGreaterEqual(1 - mg.binomial_cdf(j - 1, n, q), 1 - a)
                self.assertLessEqual(j, k)

    def test_simulated_marginal_coverage_is_at_least_nominal(self) -> None:
        # Heavy-tailed latency-like data; the true p95 of lognormal(0, 1).
        true_p95 = math.exp(1.6448536269514722)
        rng = random.Random(20261125)
        a = 0.0125
        reps = 1500
        upper_misses = lower_misses = 0
        for _ in range(reps):
            values = [rng.lognormvariate(0.0, 1.0) for _ in range(120)]
            lower, upper = quantile_bounds(values, 0.95, a)
            upper_misses += upper < true_p95
            lower_misses += lower > true_p95
        tolerance = 3 * math.sqrt(a * (1 - a) / reps)
        self.assertLessEqual(upper_misses / reps, a + tolerance)
        self.assertLessEqual(lower_misses / reps, a + tolerance)

    def test_simulated_false_superiority_rate_is_controlled(self) -> None:
        rng = random.Random(7)
        gate = _gate(statistic="p95", semantics="superiority")
        reps = 400
        false_passes = 0
        for _ in range(reps):
            baseline = [rng.lognormvariate(0.0, 0.5) for _ in range(120)]
            candidate = [rng.lognormvariate(0.0, 0.5) for _ in range(120)]
            check = evaluate_metric_gate(gate, _pairs(baseline, candidate), seed=1)
            false_passes += check["passed"]
        # Nominal one-sided level is 2.5%; Bonferroni makes it conservative.
        self.assertLessEqual(false_passes / reps, 0.025 + 3 * math.sqrt(0.025 * 0.975 / reps))

    def test_clear_tail_improvement_passes_and_clear_regression_is_flagged(self) -> None:
        rng = random.Random(3)
        baseline = [rng.lognormvariate(0.0, 0.3) * 1000 for _ in range(200)]
        faster = [value * 0.6 for value in baseline]
        slower = [value * 1.6 for value in baseline]
        improved = evaluate_metric_gate(
            _gate(semantics="superiority"), _pairs(baseline, faster), seed=1
        )
        regressed = evaluate_metric_gate(_gate(), _pairs(baseline, slower), seed=1)
        self.assertTrue(improved["passed"])
        self.assertFalse(improved["regression"])
        self.assertFalse(regressed["passed"])
        self.assertTrue(regressed["regression"])
        self.assertEqual(improved["observed"]["status"], "evaluated")
        json.dumps(improved, allow_nan=False)

    def test_insufficient_sample_fails_without_claiming_regression(self) -> None:
        check = evaluate_metric_gate(_gate(), _pairs([1.0] * 50, [9.0] * 50), seed=1)
        self.assertFalse(check["passed"])
        self.assertFalse(check["regression"])
        self.assertEqual(check["observed"]["status"], "insufficient_sample")

    def test_relative_margin_is_scaled_to_the_baseline(self) -> None:
        baseline = [100.0 + index % 10 for index in range(200)]
        candidate = [value * 1.03 for value in baseline]
        within = evaluate_metric_gate(
            _gate(margin=None, relative_margin=0.10), _pairs(baseline, candidate), seed=1
        )
        self.assertTrue(within["passed"])
        self.assertEqual(within["observed"]["difference_scale"], "relative")


class BootstrapTests(unittest.TestCase):
    def test_mean_noninferiority_size_at_the_boundary(self) -> None:
        rng = random.Random(11)
        margin = 5.0
        gate = _gate(statistic="mean", margin=margin)
        reps = 120
        false_passes = 0
        with patch.object(mg, "BOOTSTRAP_REPLICATES", 400):
            for _ in range(reps):
                observations: list[Observation] = []
                for task in range(30):
                    task_effect = rng.gauss(0, 20)
                    for repetition in range(2):
                        base = 100 + task_effect + rng.gauss(0, 10)
                        cand = base + margin + rng.gauss(0, 10)
                        for variant, value in (("baseline", base), ("candidate", cand)):
                            observations.append(
                                Observation(f"t{task}", repetition, variant, True, value)
                            )
                false_passes += evaluate_metric_gate(gate, observations, seed=rng.randrange(10**9))["passed"]
        # Approximate method: nominal 2.5%; allow simulation noise and the
        # percentile bootstrap's small-sample optimism.
        self.assertLessEqual(false_passes / reps, 0.08)

    def test_cluster_resampling_keeps_pairs_together(self) -> None:
        observations = _pairs([100.0] * 40, [90.0] * 40, repetitions=2)
        with patch.object(mg, "BOOTSTRAP_REPLICATES", 200):
            check = evaluate_metric_gate(
                _gate(statistic="mean", semantics="superiority"), observations, seed=4
            )
        self.assertEqual(check["observed"]["clusters"], 20)
        self.assertEqual(check["observed"]["confidence_interval"], [-10.0, -10.0])
        self.assertTrue(check["passed"])

    def test_cost_per_successful_workflow_counts_failed_attempt_costs(self) -> None:
        # Same spend per attempt, but the candidate fails half its tasks, so
        # each candidate success costs twice as much.
        observations: list[Observation] = []
        for task in range(40):
            observations.append(Observation(f"t{task}", 0, "baseline", True, 10.0, cost_usd=0.01))
            observations.append(
                Observation(f"t{task}", 0, "candidate", task % 2 == 0, 10.0, cost_usd=0.01)
            )
        gate = MetricGate.from_value(
            {
                "metric": "usd_per_1k_successful_workflows",
                "semantics": "noninferiority",
                "relative_margin": 0.10,
            },
            "gate",
        )
        with patch.object(mg, "BOOTSTRAP_REPLICATES", 300):
            check = evaluate_metric_gate(gate, observations, seed=2)
        self.assertAlmostEqual(check["observed"]["baseline"], 10.0)
        self.assertAlmostEqual(check["observed"]["candidate"], 20.0)
        self.assertFalse(check["passed"])
        self.assertTrue(check["regression"])

    def test_goodput_is_higher_is_better(self) -> None:
        observations: list[Observation] = []
        for task in range(30):
            observations.append(
                Observation(f"t{task}", 0, "baseline", True, 1.0, gpu_time_ms=2000.0)
            )
            observations.append(
                Observation(f"t{task}", 0, "candidate", True, 1.0, gpu_time_ms=1000.0)
            )
        gate = MetricGate.from_value(
            {"metric": "successful_workflows_per_gpu_hour", "semantics": "superiority"},
            "gate",
        )
        with patch.object(mg, "BOOTSTRAP_REPLICATES", 200):
            check = evaluate_metric_gate(gate, observations, seed=2)
        self.assertAlmostEqual(check["observed"]["candidate"], 3600.0)
        self.assertAlmostEqual(check["observed"]["baseline"], 1800.0)
        self.assertTrue(check["passed"])

    def test_unknown_cost_is_not_zero(self) -> None:
        observations = _pairs([1.0] * 30, [1.0] * 30, cost=(0.01, 0.01))
        observations[-1] = Observation("task-29", 0, "candidate", True, 1.0, cost_usd=None)
        gate = MetricGate.from_value(
            {"metric": "usd_per_1k_successful_workflows", "semantics": "noninferiority"},
            "gate",
        )
        check = evaluate_metric_gate(gate, observations, seed=1)
        self.assertEqual(check["observed"]["status"], "unknown_inputs")
        self.assertFalse(check["passed"])
        self.assertFalse(check["regression"])

    def test_conservative_population_imputes_unmeasured_candidate_tail(self) -> None:
        observations = _pairs([100.0] * 200, [80.0] * 200)
        observations = [
            Observation(item.cluster, item.repetition, item.variant, False, None)
            if item.variant == "candidate" and int(item.cluster.split("-")[1]) % 10 == 0
            else item
            for item in observations
        ]
        successful = evaluate_metric_gate(_gate(semantics="superiority"), observations, seed=1)
        conservative = evaluate_metric_gate(
            _gate(semantics="superiority", population="all_attempts_conservative"),
            observations,
            seed=1,
        )
        self.assertTrue(successful["passed"])
        self.assertFalse(conservative["passed"])
        self.assertTrue(conservative["observed"]["upper_unbounded"])


class ParsingAndMergeTests(unittest.TestCase):
    def test_methods_are_restricted_to_valid_metric_levels(self) -> None:
        with self.assertRaisesRegex(MetricGateError, "clustered"):
            _gate(metric="client_ttft_ms", method="quantile_order_statistic")
        with self.assertRaisesRegex(MetricGateError, "quantile_order_statistic"):
            _gate(statistic="mean", method="quantile_order_statistic")
        with self.assertRaisesRegex(MetricGateError, "population is fixed"):
            MetricGate.from_value(
                {
                    "metric": "usd_per_1k_successful_workflows",
                    "semantics": "noninferiority",
                    "population": "successful_pairs",
                },
                "gate",
            )
        with self.assertRaisesRegex(MetricGateError, "same gate twice"):
            parse_metric_gates([_gate().to_dict(), _gate().to_dict()])
        self.assertEqual(_gate(metric="tpot_ms").method, "paired_cluster_bootstrap")
        self.assertEqual(_gate().method, "quantile_order_statistic")

    def test_round_trip(self) -> None:
        gate = _gate(relative_margin=0.05, margin=None)
        self.assertEqual(MetricGate.from_value(gate.to_dict(), "gate"), gate)

    def test_merge_precedence(self) -> None:
        base = {
            "status": "verified",
            "reason": "ok",
            "checks": [{"name": "completion_rate", "passed": True}],
        }
        unmet = {"name": "p95_x", "passed": False, "regression": False, "required": True}
        regressed = {**unmet, "regression": True}
        self.assertEqual(
            merge_load_and_metric_checks(base, integrity_checks=[], metric_checks=[unmet])["status"],
            "inconclusive",
        )
        self.assertEqual(
            merge_load_and_metric_checks(base, integrity_checks=[], metric_checks=[regressed])["status"],
            "regression",
        )
        integrity = load_integrity_checks(
            [
                {"variant": "baseline", "load": {"dispatch_lag_ms": 900.0, "mean_in_flight": 2}},
                {"variant": "candidate", "load": {"dispatch_lag_ms": 0.0, "mean_in_flight": 2}},
            ],
            open_loop=True,
            gate_level=2.0,
            max_dispatch_lag_ms=250.0,
        )
        merged = merge_load_and_metric_checks(
            {**base, "status": "regression"}, integrity_checks=integrity, metric_checks=[]
        )
        self.assertEqual(merged["status"], "invalid")
        self.assertEqual(merged["checks"][0]["name"], "baseline_arrival_schedule_integrity")
        candidate_late = load_integrity_checks(
            [
                {"variant": "baseline", "load": {"dispatch_lag_ms": 0.0}},
                {"variant": "candidate", "load": {"dispatch_lag_ms": 900.0}},
            ],
            open_loop=True,
            gate_level=2.0,
            max_dispatch_lag_ms=250.0,
        )
        merged = merge_load_and_metric_checks(base, integrity_checks=candidate_late, metric_checks=[])
        self.assertEqual(merged["status"], "regression")
        names = [check["name"] for check in merged["checks"]]
        self.assertEqual(names.index("candidate_offered_load_completion"), 2)


if __name__ == "__main__":
    unittest.main()
