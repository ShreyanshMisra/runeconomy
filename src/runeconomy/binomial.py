"""Overflow-safe exact binomial probabilities and confidence bounds.

The binomial CDF is evaluated through the regularized incomplete beta
function, so large binomial coefficients are never materialized as floats.
Both decision code and prospective planning import this module; keeping the
numerical implementation here prevents their exact bounds from drifting.
"""

from __future__ import annotations

import math
from functools import lru_cache


_BETA_FRACTION_TINY = 1e-300
_BETA_FRACTION_EPSILON = 1e-15


class IncompleteBetaConvergenceError(ArithmeticError):
    """Raised when the incomplete-beta continued fraction does not converge."""


def _incomplete_beta_fraction(
    a: float,
    b: float,
    x: float,
    *,
    iteration_limit: int | None = None,
) -> float:
    """Evaluate the continued fraction of ``I_x(a, b)`` by modified Lentz.

    The caller picks the side of ``(a + 1) / (a + b + 2)`` on which the
    fraction converges in ``O(sqrt(max(a, b)))`` terms. ``iteration_limit`` is
    exposed only to make the failure path deterministic in tests; production
    calls use a size-dependent limit.
    """

    tiny = _BETA_FRACTION_TINY
    sum_ab, a_plus, a_minus = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - sum_ab * x / a_plus
    d = 1.0 / (tiny if abs(d) < tiny else d)
    result = d
    term_limit = (
        1000 + 20 * math.isqrt(int(max(a, b)) + 1)
        if iteration_limit is None
        else iteration_limit
    )
    for m in range(1, term_limit + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((a_minus + m2) * (a + m2))
        d = 1.0 + numerator * d
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        result *= d * c
        numerator = -(a + m) * (sum_ab + m) * x / ((a + m2) * (a_plus + m2))
        d = 1.0 + numerator * d
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= _BETA_FRACTION_EPSILON:
            return result
    raise IncompleteBetaConvergenceError(
        "incomplete beta continued fraction did not converge "
        f"after {term_limit} iterations for a={a}, b={b}, x={x}"
    )


def binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return ``P(Binomial(trials, probability) <= successes)``.

    Uses ``P(X <= k) = I_{1-p}(n - k, k + 1)`` with the regularized
    incomplete beta evaluated in log space. Directly summing
    ``comb(n, k) * p**k * q**(n-k)`` overflows once the binomial coefficient
    exceeds a float (around ``n = 1030``); this form remains finite at the
    sample sizes supported by the planner.
    """

    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    a = float(trials - successes)
    b = float(successes + 1)
    q = 1.0 - probability
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log1p(-probability)
        + b * math.log(probability)
    )
    front = math.exp(log_front)
    if q < (a + 1.0) / (a + b + 2.0):
        value = front * _incomplete_beta_fraction(a, b, q) / a
    else:
        value = 1.0 - front * _incomplete_beta_fraction(b, a, probability) / b
    return min(1.0, max(0.0, value))


def inverse_binomial_cdf(successes: int, trials: int, target: float) -> float:
    """Invert the binomial CDF in ``probability`` using monotone bisection."""

    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if binomial_cdf(successes, trials, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def clopper_pearson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> list[float] | None:
    """Return an exact two-sided binomial confidence interval."""

    if trials <= 0:
        return None
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if successes == 0
        else inverse_binomial_cdf(successes - 1, trials, 1.0 - tail)
    )
    upper = (
        1.0 if successes == trials else inverse_binomial_cdf(successes, trials, tail)
    )
    return [lower, upper]


@lru_cache(maxsize=65536)
def binomial_lower_bound(successes: int, trials: int, alpha: float) -> float:
    """One-sided exact Clopper-Pearson lower bound at level ``1 - alpha``."""

    if successes == 0:
        return 0.0
    return inverse_binomial_cdf(successes - 1, trials, 1.0 - alpha)


@lru_cache(maxsize=65536)
def binomial_upper_bound(successes: int, trials: int, alpha: float) -> float:
    """One-sided exact Clopper-Pearson upper bound at level ``1 - alpha``."""

    if successes == trials:
        return 1.0
    return inverse_binomial_cdf(successes, trials, alpha)
