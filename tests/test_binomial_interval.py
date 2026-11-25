"""Exact binomial interval: equivalence with the original sum and large-n safety.

The original ``_binomial_cdf`` summed ``comb(n, k) * p**k * q**(n - k)``.
Converting ``comb(n, k)`` to a float overflows once it passes ~1e308, so any
interval with roughly 1,030 or more trials raised ``OverflowError``. The
replacement evaluates the regularized incomplete beta function in log space.
These tests pin it to the original algorithm wherever that algorithm worked
and check it against independent references where it did not.
"""

from __future__ import annotations

import math
import unittest

from runeconomy.experiments import (
    _binomial_cdf,
    _clopper_pearson_interval,
    _one_sided_binomial_lower,
    _one_sided_binomial_upper,
    _paired_success_statistics,
)


# Two runs of the same bisection over CDFs that differ only by rounding land
# within a few ulps of each other; the observed worst case is ~6e-15.
EQUIVALENCE_TOLERANCE = 1e-12


def _original_binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Verbatim copy of the pre-fix implementation, kept as the reference."""

    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    return sum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes + 1)
    )


def _original_inverse_binomial_cdf(successes: int, trials: int, target: float) -> float:
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _original_binomial_cdf(successes, trials, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _original_clopper_pearson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> list[float] | None:
    if trials <= 0:
        return None
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if successes == 0
        else _original_inverse_binomial_cdf(successes - 1, trials, 1.0 - tail)
    )
    upper = (
        1.0
        if successes == trials
        else _original_inverse_binomial_cdf(successes, trials, tail)
    )
    return [lower, upper]


def _log_space_binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Independent reference: sum of log-space pmf terms, valid for any n."""

    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_n = math.lgamma(trials + 1.0)
    return math.fsum(
        math.exp(
            log_n
            - math.lgamma(count + 1.0)
            - math.lgamma(trials - count + 1.0)
            + count * log_p
            + (trials - count) * log_q
        )
        for count in range(successes + 1)
    )


class OriginalAlgorithmEquivalenceTests(unittest.TestCase):
    """Wherever the original code returned a value, the fix returns the same one."""

    def assertIntervalsMatch(
        self, successes: int, trials: int, confidence: float
    ) -> None:
        expected = _original_clopper_pearson_interval(
            successes, trials, confidence=confidence
        )
        actual = _clopper_pearson_interval(successes, trials, confidence=confidence)
        assert expected is not None and actual is not None
        for bound in (0, 1):
            self.assertAlmostEqual(
                actual[bound],
                expected[bound],
                delta=EQUIVALENCE_TOLERANCE,
                msg=f"successes={successes} trials={trials} confidence={confidence}",
            )

    def test_every_success_count_for_small_samples(self) -> None:
        # 0.975 is the per-margin level _paired_success_statistics uses.
        for trials in range(1, 41):
            for successes in range(trials + 1):
                for confidence in (0.95, 0.975):
                    self.assertIntervalsMatch(successes, trials, confidence)

    def test_representative_counts_up_to_the_old_overflow_limit(self) -> None:
        cases = [
            (successes, trials)
            for trials in (97, 180, 500)
            for successes in (0, 1, 2, trials // 10, trials // 2, trials - 2, trials - 1, trials)
        ]
        for successes, trials in cases:
            for confidence in (0.95, 0.8):
                self.assertIntervalsMatch(successes, trials, confidence)
        # 1029 is the largest n the original survived for every success count;
        # the reference is slow there, so check one confidence level.
        for successes in (0, 3, 514, 1029):
            self.assertIntervalsMatch(successes, 1029, 0.95)

    def test_cdf_matches_the_original_sum(self) -> None:
        for trials in (1, 2, 7, 40, 180, 600):
            for successes in sorted({0, 1, trials // 3, trials // 2, trials - 1}):
                for probability in (1e-9, 0.001, 0.1, 0.37, 0.5, 0.9, 0.999):
                    expected = _original_binomial_cdf(successes, trials, probability)
                    actual = _binomial_cdf(successes, trials, probability)
                    self.assertAlmostEqual(
                        actual,
                        expected,
                        delta=1e-12,
                        msg=f"k={successes} n={trials} p={probability}",
                    )

    def test_cdf_edges_match_the_original(self) -> None:
        # Bisection midpoints can round to exactly 1.0; both must agree there.
        for trials in (1, 5, 50):
            for successes in range(-1, trials + 2):
                for probability in (0.0, 1.0):
                    self.assertEqual(
                        _binomial_cdf(successes, trials, probability),
                        _original_binomial_cdf(successes, trials, probability),
                    )

    def test_original_algorithm_overflows_at_1030_trials(self) -> None:
        # Documents the bug this module fixes.
        with self.assertRaises(OverflowError):
            _original_clopper_pearson_interval(515, 1030)


class LargeSampleTests(unittest.TestCase):
    """Sizes the original could not handle at all."""

    def test_interval_is_finite_and_ordered_for_large_samples(self) -> None:
        for trials in (1030, 1031, 5000, 100_000, 1_000_000, 10_000_000):
            previous_lower = -1.0
            for successes in sorted(
                {0, 1, 2, trials // 100, trials // 2, trials - 2, trials - 1, trials}
            ):
                interval = _clopper_pearson_interval(successes, trials)
                assert interval is not None
                lower, upper = interval
                self.assertTrue(math.isfinite(lower) and math.isfinite(upper))
                self.assertLessEqual(0.0, lower)
                self.assertLessEqual(lower, successes / trials)
                self.assertLessEqual(successes / trials, upper)
                self.assertLessEqual(upper, 1.0)
                self.assertGreaterEqual(lower, previous_lower)
                previous_lower = lower

    def test_boundary_counts_match_closed_forms(self) -> None:
        # For k = 0 the upper bound solves q**n = alpha/2 exactly, and k = n
        # mirrors it, so both have closed forms at any n.
        for trials in (1030, 5000, 100_000, 1_000_000):
            for confidence in (0.95, 0.99):
                tail = (1.0 - confidence) / 2.0
                closed_form = -math.expm1(math.log(tail) / trials)
                none = _clopper_pearson_interval(0, trials, confidence=confidence)
                every = _clopper_pearson_interval(trials, trials, confidence=confidence)
                assert none is not None and every is not None
                self.assertEqual(none[0], 0.0)
                self.assertAlmostEqual(none[1], closed_form, delta=1e-13)
                self.assertAlmostEqual(every[0], 1.0 - closed_form, delta=1e-13)
                self.assertEqual(every[1], 1.0)

    def test_bounds_solve_the_defining_tail_equations(self) -> None:
        # Checked with an independent log-space summation, not the code under test.
        tail = 0.025
        for successes, trials in ((515, 1030), (40, 2000), (4750, 5000), (12_345, 20_000)):
            interval = _clopper_pearson_interval(successes, trials)
            assert interval is not None
            lower, upper = interval
            self.assertAlmostEqual(
                1.0 - _log_space_binomial_cdf(successes - 1, trials, lower),
                tail,
                delta=1e-9,
            )
            self.assertAlmostEqual(
                _log_space_binomial_cdf(successes, trials, upper), tail, delta=1e-9
            )

    def test_cdf_matches_log_space_reference_beyond_the_old_limit(self) -> None:
        for trials in (1030, 5000, 20_000):
            for successes in sorted({0, 1, trials // 20, trials // 2, trials - 5, trials - 1}):
                mean_probability = (successes + 0.5) / trials
                for probability in (
                    mean_probability * 0.9,
                    mean_probability,
                    min(0.999999, mean_probability * 1.05),
                ):
                    self.assertAlmostEqual(
                        _binomial_cdf(successes, trials, probability),
                        _log_space_binomial_cdf(successes, trials, probability),
                        delta=1e-10,
                        msg=f"k={successes} n={trials} p={probability}",
                    )

    def test_wide_interval_approaches_the_normal_approximation(self) -> None:
        trials = 1_000_000
        interval = _clopper_pearson_interval(trials // 2, trials)
        assert interval is not None
        half_width = 1.959963984540054 * math.sqrt(0.25 / trials)
        self.assertAlmostEqual(interval[0], 0.5 - half_width, delta=2e-6)
        self.assertAlmostEqual(interval[1], 0.5 + half_width, delta=2e-6)

    def test_paired_statistics_and_one_sided_bounds_no_longer_overflow(self) -> None:
        pairs = (
            [(True, True)] * 1900
            + [(False, False)] * 60
            + [(False, True)] * 25
            + [(True, False)] * 15
        )
        statistics = _paired_success_statistics(pairs)
        self.assertEqual(statistics["paired_count"], 2000)
        low, high = statistics["confidence_interval"]
        self.assertLess(low, 0.005)
        self.assertGreater(high, 0.005)
        self.assertLessEqual(statistics["one_sided_lower_bound"], 0.005)
        self.assertLess(_one_sided_binomial_lower(25, 2000, alpha=0.025), 25 / 2000)
        self.assertGreater(_one_sided_binomial_upper(15, 2000, alpha=0.025), 15 / 2000)


if __name__ == "__main__":
    unittest.main()
