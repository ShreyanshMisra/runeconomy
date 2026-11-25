"""Correctness, exact-size, and simulation-based coverage tests for paired methods."""

from __future__ import annotations

import math
import random
import unittest

from runeconomy import paired_inference as pi
from runeconomy.experiments import (
    _clopper_pearson_interval,
    _one_sided_binomial_lower,
    _one_sided_binomial_upper,
    _paired_success_statistics,
)


ALPHA = 0.05


def _trinomial_log_pmf(x: int, y: int, n: int, q01: float, q10: float) -> float:
    rest = n - x - y
    value = (
        math.lgamma(n + 1)
        - math.lgamma(x + 1)
        - math.lgamma(y + 1)
        - math.lgamma(rest + 1)
    )
    for count, probability in ((x, q01), (y, q10), (rest, 1.0 - q01 - q10)):
        if count:
            if probability <= 0:
                return -math.inf
            value += count * math.log(probability)
    return value


def _enumerated_rejection_probability(
    n: int, delta0: float, critical: float, q10: float, q01: float
) -> float:
    total = 0.0
    for x in range(n + 1):
        for y in range(n + 1 - x):
            if pi.tango_score(x, y, n, delta0) >= critical - 1e-10:
                log_pmf = _trinomial_log_pmf(x, y, n, q01, q10)
                if log_pmf > -math.inf:
                    total += math.exp(log_pmf)
    return total


def _legacy_rejects(x: int, y: int, n: int, margin: float) -> bool:
    lower = _one_sided_binomial_lower(x, n, alpha=ALPHA / 2) - _one_sided_binomial_upper(
        y, n, alpha=ALPHA / 2
    )
    return lower >= -margin


def _simulate_counts(
    rng: random.Random, n: int, q10: float, q01: float
) -> tuple[int, int]:
    candidate_only = baseline_only = 0
    for _ in range(n):
        draw = rng.random()
        if draw < q01:
            candidate_only += 1
        elif draw < q01 + q10:
            baseline_only += 1
    return candidate_only, baseline_only


class ScoreStatisticTests(unittest.TestCase):
    def test_closed_form_restricted_mle_matches_numeric_maximum(self) -> None:
        rng = random.Random(11)
        for _ in range(300):
            n = rng.randint(1, 60)
            x = rng.randint(0, n)
            y = rng.randint(0, n - x)
            delta0 = rng.uniform(-0.8, 0.8)
            lower, upper = max(0.0, -delta0), (1.0 - delta0) / 2.0

            def log_likelihood(a: float) -> float:
                return _trinomial_log_pmf(x, y, n, a + delta0, a)

            grid = [lower + (upper - lower) * i / 2000 for i in range(2001)]
            best = max(grid, key=log_likelihood)
            variance = 2.0 * best + delta0 - delta0 * delta0
            if variance <= 1e-9:
                continue
            numeric = (x - y - n * delta0) / math.sqrt(n * variance)
            self.assertAlmostEqual(
                pi.tango_score(x, y, n, delta0), numeric, delta=0.02 + abs(numeric) * 0.01
            )

    def test_score_is_barnard_convex(self) -> None:
        for n in (5, 17, 40, 90):
            for delta0 in (-0.4, -0.1, -0.05, -0.01, 0.0, 0.05):
                for x in range(n + 1):
                    for y in range(n + 1 - x):
                        score = pi.tango_score(x, y, n, delta0)
                        if x + y < n:
                            self.assertGreaterEqual(
                                pi.tango_score(x + 1, y, n, delta0), score - 1e-9
                            )
                        if y:
                            self.assertGreaterEqual(
                                pi.tango_score(x, y - 1, n, delta0), score - 1e-9
                            )

    def test_all_agreement_score_has_closed_form(self) -> None:
        for n in (10, 40, 180):
            for margin in (0.02, 0.05, 0.1):
                expected = math.sqrt(n * margin / (1.0 - margin))
                self.assertAlmostEqual(pi.tango_score(0, 0, n, -margin), expected, places=9)


class ExactComputationTests(unittest.TestCase):
    def test_rejection_probability_matches_trinomial_enumeration(self) -> None:
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randint(4, 70)
            delta0 = rng.uniform(-0.3, 0.1)
            critical = rng.uniform(-1.0, 3.0)
            lower, upper = pi._nuisance_range(delta0)
            nuisance = rng.uniform(lower + 1e-6, upper - 1e-6)
            thresholds = pi._rejection_thresholds(n, delta0, critical)
            self.assertAlmostEqual(
                pi._rejection_probability(n, thresholds, nuisance, nuisance + delta0),
                _enumerated_rejection_probability(
                    n, delta0, critical, nuisance, nuisance + delta0
                ),
                places=11,
            )

    def test_rejection_probability_is_exact_at_nuisance_endpoints(self) -> None:
        for n, delta0, critical in ((12, -0.1, 1.0), (30, -0.05, 1.7), (25, 0.0, 0.5)):
            thresholds = pi._rejection_thresholds(n, delta0, critical)
            lower, upper = pi._nuisance_range(delta0)
            for q10, q01 in (
                (lower, lower + delta0),  # no candidate-only mass when delta0 < 0
                (upper, upper + delta0),  # every pair discordant
                (0.0, 0.3),  # every discordant pair is candidate-only
                (0.0, 0.0),  # no discordance
            ):
                q01 = max(q01, 0.0)
                self.assertAlmostEqual(
                    pi._rejection_probability(n, thresholds, q10, q01),
                    _enumerated_rejection_probability(n, delta0, critical, q10, q01),
                    places=11,
                )

    def test_default_nuisance_supremum_matches_dense_grid(self) -> None:
        for n in (20, 57, 120):
            for delta0 in (-0.02, -0.05, 0.0):
                for critical in (1.65, 1.75):
                    thresholds = pi._rejection_thresholds(n, delta0, critical)
                    default = pi._null_supremum(n, delta0, thresholds)
                    dense = pi._null_supremum(
                        n, delta0, thresholds, resolution=32 * n, refine=5
                    )
                    self.assertLessEqual(dense - default, 1e-6)

    def test_exact_size_is_at_most_alpha_on_and_inside_the_null(self) -> None:
        """Exact size by full enumeration at nuisance values off the search grid."""

        rng = random.Random(29)
        for n, margin in ((15, 0.1), (30, 0.05), (48, 0.1), (60, 0.0)):
            critical = pi.score_exact_critical_value(n, -margin, ALPHA)
            for _ in range(12):
                # Null boundary and interior: true delta <= -margin.
                delta = -margin - rng.choice([0.0, 0.0, 0.02, 0.1])
                lower, upper = pi._nuisance_range(delta)
                nuisance = rng.uniform(lower, upper)
                size = _enumerated_rejection_probability(
                    n, -margin, critical, nuisance, max(nuisance + delta, 0.0)
                )
                self.assertLessEqual(size, ALPHA + 1e-9, (n, margin, delta, nuisance))

    def test_p_value_and_critical_value_define_the_same_rejection_region(self) -> None:
        n, margin = 26, 0.1
        critical = pi.score_exact_critical_value(n, -margin, ALPHA)
        for x in range(n + 1):
            for y in range(n + 1 - x):
                by_p_value = pi.score_exact_rejects(x, y, n, margin, alpha=ALPHA)
                by_critical = pi.tango_score(x, y, n, -margin) >= critical - 1e-10
                self.assertEqual(by_p_value, by_critical, (x, y))

    def test_critical_value_is_the_smallest_valid_level(self) -> None:
        for n, margin in ((40, 0.05), (90, 0.05), (120, 0.1)):
            critical = pi.score_exact_critical_value(n, -margin, ALPHA)
            thresholds = pi._rejection_thresholds(n, -margin, critical)
            self.assertLessEqual(pi._null_supremum(n, -margin, thresholds), ALPHA)
            smallest = pi._minimum_region_score(n, -margin, thresholds)
            # Admitting the next boundary cell would exceed alpha.
            looser = pi._rejection_thresholds(n, -margin, smallest - 1e-6)
            below = max(
                pi.tango_score(k - 1, d - k + 1, n, -margin)
                for d, k in enumerate(looser)
                if 1 <= k <= d + 1 and k - 1 <= d
            )
            wider = pi._rejection_thresholds(n, -margin, below - 1e-9)
            self.assertGreater(pi._null_supremum(n, -margin, wider), ALPHA)

    def test_lower_bound_agrees_with_the_gate_decision(self) -> None:
        n, margin = 80, 0.05
        for x, y in ((0, 0), (0, 1), (0, 3), (2, 4), (5, 2), (1, 7)):
            lower = pi.score_exact_lower_bound(x, y, n, alpha=ALPHA, anchor=-margin)
            self.assertEqual(
                lower >= -margin, pi.score_exact_rejects(x, y, n, margin, alpha=ALPHA)
            )
            self.assertLessEqual(lower, (x - y) / n)

    def test_statistics_report_method_and_preserve_outcome_counts(self) -> None:
        statistics = pi.paired_score_exact_statistics(
            {"both_success": 175, "both_failure": 1, "baseline_only": 2, "candidate_only": 2},
            margin=0.05,
        )
        self.assertEqual(statistics["method_id"], "paired_score_exact")
        self.assertEqual(statistics["paired_count"], 180)
        self.assertEqual(statistics["candidate_minus_baseline_mean"], 0.0)
        self.assertTrue(statistics["rejects_inferiority"])
        self.assertGreaterEqual(statistics["one_sided_lower_bound"], -0.05)
        empty = pi.paired_score_exact_statistics({}, margin=0.05)
        self.assertIsNone(empty["one_sided_lower_bound"])
        self.assertFalse(empty["rejects_inferiority"])

    def test_planning_binomial_helpers_match_decision_bounds(self) -> None:
        for successes, trials in ((0, 10), (10, 10), (38, 40), (170, 180), (95, 100)):
            interval = _clopper_pearson_interval(successes, trials)
            self.assertAlmostEqual(
                pi.binomial_lower_bound(successes, trials, 0.025), interval[0], places=9
            )
            self.assertAlmostEqual(
                pi.binomial_upper_bound(successes, trials, 0.025), interval[1], places=9
            )
        self.assertEqual(pi.minimum_successes_for_absolute_gate(180, 0.90), 170)
        self.assertEqual(pi.minimum_successes_for_absolute_gate(40, 0.80), 38)
        self.assertIsNone(pi.minimum_successes_for_absolute_gate(10, 0.80))

    def test_completion_tolerance_is_the_smallest_integer_meeting_target(self) -> None:
        tolerance = pi.completion_tolerance(180, 0.0075, 0.95)
        self.assertEqual(tolerance, 3)
        self.assertGreaterEqual(pi.binomial_cdf(3, 180, 0.0075), 0.95)
        self.assertLess(pi.binomial_cdf(2, 180, 0.0075), 0.95)
        self.assertEqual(pi.completion_tolerance(180, 0.0, 0.95), 0)


class SimulationCoverageTests(unittest.TestCase):
    """Monte Carlo checks of the decision functions the gates actually call."""

    def test_simulated_type_one_error_at_the_null_boundary(self) -> None:
        rng = random.Random(20261125)
        repetitions = 2000
        tolerance = 3.0 * math.sqrt(ALPHA * (1 - ALPHA) / repetitions)
        settings = (
            (60, 0.05, 0.08, 0.03),  # q10 - q01 = margin
            (100, 0.05, 0.05, 0.0),  # all discordance against the candidate
            (100, 0.10, 0.20, 0.10),
        )
        for n, margin, q10, q01 in settings:
            cache: dict[tuple[int, int], tuple[bool, bool]] = {}
            score_rejections = legacy_rejections = 0
            for _ in range(repetitions):
                key = _simulate_counts(rng, n, q10, q01)
                if key not in cache:
                    cache[key] = (
                        pi.score_exact_rejects(*key, n, margin, alpha=ALPHA),
                        _legacy_rejects(*key, n, margin),
                    )
                score, legacy = cache[key]
                score_rejections += score
                legacy_rejections += legacy
            score_rate = score_rejections / repetitions
            legacy_rate = legacy_rejections / repetitions
            self.assertLessEqual(score_rate, ALPHA + tolerance, (n, margin, q10, q01))
            self.assertLessEqual(legacy_rate, ALPHA + tolerance, (n, margin, q10, q01))
            # The documented property: the legacy method spends far less than
            # alpha, which is the power the score method recovers.
            self.assertLessEqual(legacy_rate, score_rate + tolerance)

    def test_simulated_lower_bound_coverage(self) -> None:
        rng = random.Random(7)
        repetitions = 400
        n = 40
        tolerance = 3.0 * math.sqrt(0.95 * 0.05 / repetitions)
        for q10, q01 in ((0.10, 0.05), (0.03, 0.03)):
            truth = q01 - q10
            cache: dict[tuple[int, int], tuple[float, float]] = {}
            score_covered = legacy_covered = 0
            for _ in range(repetitions):
                key = _simulate_counts(rng, n, q10, q01)
                if key not in cache:
                    x, y = key
                    pairs = (
                        [(False, True)] * x
                        + [(True, False)] * y
                        + [(True, True)] * (n - x - y)
                    )
                    cache[key] = (
                        pi.score_exact_lower_bound(x, y, n, alpha=ALPHA, tolerance=1e-4),
                        _paired_success_statistics(pairs)["one_sided_lower_bound"],
                    )
                score_lower, legacy_lower = cache[key]
                score_covered += score_lower <= truth
                legacy_covered += legacy_lower <= truth
            self.assertGreaterEqual(score_covered / repetitions, 0.95 - tolerance)
            self.assertGreaterEqual(legacy_covered / repetitions, 0.95 - tolerance)

    def test_score_method_has_more_power_at_realistic_alternatives(self) -> None:
        for success, discordance, n in ((0.95, 0.04, 180), (0.90, 0.06, 200), (0.90, 0.10, 300)):
            cells = pi.paired_cell_probabilities(
                expected_success_rate=success, expected_discordance_rate=discordance
            )
            legacy = pi.gate_power(pairs=n, method="paired_exact", margin=0.05, cells=cells)
            score = pi.gate_power(
                pairs=n, method="paired_score_exact", margin=0.05, cells=cells
            )
            self.assertGreater(score["noninferiority"], legacy["noninferiority"] + 0.1)

    def test_exact_power_matches_simulation(self) -> None:
        rng = random.Random(5)
        n, margin, q10, q01 = 120, 0.05, 0.02, 0.02
        cells = {
            "baseline_only": q10,
            "candidate_only": q01,
            "both_success": 0.92,
            "both_failure": 0.04,
        }
        exact = pi.gate_power(
            pairs=n, method="paired_score_exact", margin=margin, cells=cells
        )["noninferiority"]
        repetitions = 3000
        cache: dict[tuple[int, int], bool] = {}
        rejections = 0
        for _ in range(repetitions):
            key = _simulate_counts(rng, n, q10, q01)
            if key not in cache:
                cache[key] = pi.score_exact_rejects(*key, n, margin, alpha=ALPHA)
            rejections += cache[key]
        standard_error = math.sqrt(exact * (1 - exact) / repetitions)
        self.assertAlmostEqual(rejections / repetitions, exact, delta=4 * standard_error)


class MethodPropertiesTests(unittest.TestCase):
    def test_every_method_documents_its_coverage(self) -> None:
        for method in ("paired_exact", "paired_score_exact", "point_estimate"):
            properties = pi.method_properties(method)
            self.assertIn("coverage", properties)
            self.assertIn("label", properties)
        self.assertTrue(pi.method_properties("paired_exact")["default"])
        self.assertFalse(pi.method_properties("paired_score_exact")["default"])
        with self.assertRaises(pi.PairedInferenceError):
            pi.method_properties("mcnemar")

    def test_cell_probabilities_reject_inconsistent_assumptions(self) -> None:
        with self.assertRaises(pi.PairedInferenceError):
            pi.paired_cell_probabilities(
                expected_success_rate=0.9, expected_discordance_rate=0.02, expected_difference=0.05
            )
        with self.assertRaises(pi.PairedInferenceError):
            pi.paired_cell_probabilities(
                expected_success_rate=0.01, expected_discordance_rate=0.5
            )


if __name__ == "__main__":
    unittest.main()
