"""Paired task-success non-inferiority: methods, exact computation, and power.

Every comparison pairs one baseline attempt with one candidate attempt on the
same task and repetition. Each pair lands in one of four cells: both succeed,
both fail, baseline-only success (``y``), or candidate-only success (``x``).
The paired success difference is

    Delta = P(candidate-only) - P(baseline-only) = q01 - q10,

and the preregistered non-inferiority gate tests ``H0: Delta <= -margin``
against ``H1: Delta > -margin``.

Two preregisterable methods are available.

``paired_exact`` (default for every contract that omits the field)
    Exact one-sided Clopper-Pearson bounds for ``q01`` and ``q10`` are each
    computed at ``alpha / 2`` and joined with Bonferroni. Coverage is at least
    the nominal level for every sample size and every nuisance value, but the
    bound ignores the pairing and spends half of ``alpha`` on each marginal. In
    practice it is very conservative. Its size at the null boundary is usually
    far below ``alpha``, so it costs a lot of power.

``paired_score_exact`` (opt-in, prospective)
    An exact unconditional test (no asymptotics) ordered by the Tango (1998)
    restricted-maximum-likelihood score statistic for a paired difference, as
    studied by Hsueh, Liu and Chen (2001). The p-value is the supremum, over
    every nuisance value on the null boundary ``Delta = -margin``, of the exact
    trinomial probability of a score at least as extreme as the one observed.
    Type I error is therefore at most ``alpha`` for every sample size and every
    discordance rate, up to the numerical accuracy of the supremum. The score
    statistic is Barnard-convex (it increases with ``x`` and decreases with
    ``y``), so the boundary is the least favourable part of the composite null.
    The supremum is evaluated on a deterministic grid of ``max(64, n) + 1``
    nuisance values with golden-section refinement of the leading local maxima;
    the test suite checks it against a 32n-point grid and checks exact size
    off the grid. The one-sided lower bound inverts the same test, probing the
    preregistered margin first so that the reported bound and the gate decision
    can never disagree. Size is close to ``alpha`` rather than far below it,
    which is where the extra power comes from.

``point_estimate`` (static manifests only) applies no sampling-error control
and is not a verification method.

References: Tango T. (1998), Statistics in Medicine 17, 891-908. Hsueh H.-M.,
Liu J.-P. and Chen J. J. (2001), Biometrics 57, 478-483.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Iterable, Mapping

from .binomial import (
    binomial_cdf,
    binomial_lower_bound,
    binomial_upper_bound,
)

PAIRED_EXACT = "paired_exact"
PAIRED_SCORE_EXACT = "paired_score_exact"
POINT_ESTIMATE = "point_estimate"
DEFAULT_TASK_SUCCESS_METHOD = PAIRED_EXACT
PAIRED_METHODS = frozenset({PAIRED_EXACT, PAIRED_SCORE_EXACT})

PAIRED_EXACT_LABEL = "exact marginal binomial bounds with Bonferroni coverage"
PAIRED_SCORE_EXACT_LABEL = (
    "exact unconditional non-inferiority test ordered by the Tango paired score"
)

METHOD_PROPERTIES: dict[str, dict[str, Any]] = {
    PAIRED_EXACT: {
        "method": PAIRED_EXACT,
        "label": PAIRED_EXACT_LABEL,
        "default": True,
        "test": (
            "one-sided Clopper-Pearson bounds for candidate-only and baseline-only "
            "probabilities at alpha/2 each, combined with Bonferroni"
        ),
        "coverage": (
            "at least nominal for every sample size and nuisance value; very "
            "conservative because pairing is ignored and alpha is split in half"
        ),
        "power": "low; zero discordant pairs at n=40 still bound the loss at -8.8 pp",
    },
    PAIRED_SCORE_EXACT: {
        "method": PAIRED_SCORE_EXACT,
        "label": PAIRED_SCORE_EXACT_LABEL,
        "default": False,
        "test": (
            "exact unconditional test of H0: delta <= -margin ordered by the Tango "
            "restricted-MLE score statistic; p-value is the supremum over the "
            "nuisance discordance on the null boundary"
        ),
        "coverage": (
            "type I error at most alpha for every sample size and discordance rate, "
            "up to the numerical accuracy of the nuisance supremum (deterministic "
            "grid of max(64, n)+1 values with golden-section refinement); size is "
            "close to alpha rather than far below it"
        ),
        "power": "substantially higher than paired_exact at realistic discordance",
        "references": [
            "Tango (1998), Statistics in Medicine 17:891-908",
            "Hsueh, Liu and Chen (2001), Biometrics 57:478-483",
        ],
    },
    POINT_ESTIMATE: {
        "method": POINT_ESTIMATE,
        "label": "point estimate without sampling-error control",
        "default": False,
        "test": "candidate success rate >= baseline success rate - margin",
        "coverage": "none; not a verification method",
        "power": "not applicable",
    },
}

_TIE_TOLERANCE = 1e-10
_COARSE_RESOLUTION = 32
_COARSE_GOLDEN_ITERATIONS = 12
_REFINED_PEAKS = 3
_GOLDEN_ITERATIONS = 30
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


class PairedInferenceError(ValueError):
    """Raised when a paired-inference request is outside the supported domain."""


def method_properties(method: str) -> dict[str, Any]:
    """Return a copy of the documented properties of one success method."""

    if method not in METHOD_PROPERTIES:
        raise PairedInferenceError(f"unsupported task_success_method {method!r}")
    return dict(METHOD_PROPERTIES[method])


def _validate_margin(margin: float) -> float:
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not 0.0 <= float(margin) < 1.0
    ):
        raise PairedInferenceError("non-inferiority margin must be in [0, 1)")
    return float(margin)


def _validate_alpha(alpha: float) -> float:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 0.5
    ):
        raise PairedInferenceError("alpha must be in (0, 0.5)")
    return float(alpha)


def _validate_counts(candidate_only: int, baseline_only: int, pairs: int) -> None:
    for name, value in (
        ("candidate_only", candidate_only),
        ("baseline_only", baseline_only),
        ("pairs", pairs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PairedInferenceError(f"{name} must be a non-negative integer")
    if pairs <= 0:
        raise PairedInferenceError("pairs must be positive")
    if candidate_only + baseline_only > pairs:
        raise PairedInferenceError("discordant pairs cannot exceed total pairs")


def outcome_counts(pairs: Iterable[tuple[bool, bool]]) -> dict[str, int]:
    """Count paired (baseline_success, candidate_success) outcomes."""

    counts = {"both_success": 0, "both_failure": 0, "candidate_only": 0, "baseline_only": 0}
    for baseline, candidate in pairs:
        if baseline and candidate:
            counts["both_success"] += 1
        elif not baseline and not candidate:
            counts["both_failure"] += 1
        elif candidate:
            counts["candidate_only"] += 1
        else:
            counts["baseline_only"] += 1
    return counts


# ---------------------------------------------------------------------------
# Tango score statistic and exact unconditional test
# ---------------------------------------------------------------------------


def tango_score(
    candidate_only: int, baseline_only: int, pairs: int, delta0: float
) -> float:
    """Return the Tango score statistic for ``H0: Delta = delta0``.

    ``a`` is the restricted maximum-likelihood estimate of the baseline-only
    probability given ``q01 = a + delta0``. It solves
    ``2n a^2 - B a - C = 0`` with ``B = (x + y)(1 - delta0) - 2 delta0 (n - x)``
    and ``C = y delta0 (1 - delta0)``.
    """

    x, y, n = candidate_only, baseline_only, pairs
    linear = (x + y) * (1.0 - delta0) - 2.0 * delta0 * (n - x)
    constant = y * delta0 * (1.0 - delta0)
    discriminant = max(linear * linear + 8.0 * n * constant, 0.0)
    estimate = (linear + math.sqrt(discriminant)) / (4.0 * n)
    estimate = min(max(estimate, max(0.0, -delta0)), (1.0 - delta0) / 2.0)
    variance = 2.0 * estimate + delta0 - delta0 * delta0
    numerator = x - y - n * delta0
    if variance <= 0.0:
        if abs(numerator) <= 1e-12:
            return 0.0
        return math.copysign(math.inf, numerator)
    return numerator / math.sqrt(n * variance)


@lru_cache(maxsize=64)
def _log_factorials(limit: int) -> tuple[float, ...]:
    return tuple(math.lgamma(index + 1.0) for index in range(limit + 1))


def _rejection_thresholds(pairs: int, delta0: float, critical: float) -> list[int]:
    """Return ``k[d]``, the smallest candidate-only count rejecting at ``d``.

    ``d`` is the number of discordant pairs. Barnard convexity makes the
    rejection region an upper set in ``x`` for each ``d``, and the thresholds
    are non-decreasing with ``k[d + 1] <= k[d] + 1``. ``k[d] = d + 1`` means no
    split of ``d`` discordant pairs rejects.
    """

    limit = critical - _TIE_TOLERANCE
    thresholds: list[int] = []
    previous = 0
    for discordant in range(pairs + 1):
        threshold = previous
        while (
            threshold <= discordant
            and tango_score(threshold, discordant - threshold, pairs, delta0) < limit
        ):
            threshold += 1
        thresholds.append(threshold)
        previous = threshold
    return thresholds


def _xlog(count: int, log_value: float) -> float:
    return 0.0 if count == 0 else count * log_value


def _safe_log(value: float) -> float:
    return math.log(value) if value > 0.0 else -math.inf


def _safe_log1m(value: float) -> float:
    return math.log1p(-value) if value < 1.0 else -math.inf


def _rejection_probability(
    pairs: int,
    thresholds: list[int],
    baseline_only_probability: float,
    candidate_only_probability: float,
) -> float:
    """Exact trinomial probability of the rejection region in ``O(n)``.

    The number of discordant pairs ``d`` is binomial with the discordance rate,
    and the candidate-only count given ``d`` is binomial with the conditional
    candidate share. The conditional upper tails are carried from ``d`` to
    ``d + 1`` with the identity ``T(d+1, k) = T(d, k) + pi * pmf(d, k-1)``.
    Only discordant counts within 12 standard deviations (plus 12) of the mean
    are visited; the omitted binomial mass is far below 1e-20.
    """

    n = pairs
    discordance = baseline_only_probability + candidate_only_probability
    discordance = min(max(discordance, 0.0), 1.0)
    if discordance <= 0.0:
        return 1.0 if thresholds[0] <= 0 else 0.0
    share = min(max(candidate_only_probability / discordance, 0.0), 1.0)
    factorials = _log_factorials(n)
    if discordance >= 1.0:
        # Every pair is discordant, so d = n.
        threshold = thresholds[n]
        if threshold <= 0:
            return 1.0
        if threshold > n:
            return 0.0
        return 1.0 - binomial_cdf(threshold - 1, n, share)
    window = _window(
        n * discordance, math.sqrt(n * discordance * (1.0 - discordance)), 0, n
    )
    log_disc = math.log(discordance)
    log_conc = math.log1p(-discordance)
    log_n = factorials[n]
    exp = math.exp
    if share <= 0.0 or share >= 1.0:
        # Degenerate split: x = 0 always, or x = d always.
        total = 0.0
        for discordant in window:
            threshold = thresholds[discordant]
            rejects = threshold <= 0 if share <= 0.0 else threshold <= discordant
            if rejects:
                total += exp(
                    log_n
                    - factorials[discordant]
                    - factorials[n - discordant]
                    + discordant * log_disc
                    + (n - discordant) * log_conc
                )
        return min(max(total, 0.0), 1.0)

    log_share = math.log(share)
    log_other = math.log1p(-share)

    start = window.start
    threshold = thresholds[start]
    if threshold <= 0:
        tail = 1.0
    elif threshold > start:
        tail = 0.0
    else:
        tail = 1.0 - binomial_cdf(threshold - 1, start, share)
    total = 0.0
    for discordant in window:
        if discordant > start:
            previous = discordant - 1
            carried = threshold - 1
            if 0 <= carried <= previous:
                tail += share * exp(
                    factorials[previous]
                    - factorials[carried]
                    - factorials[previous - carried]
                    + carried * log_share
                    + (previous - carried) * log_other
                )
            target = thresholds[discordant]
            while threshold < target:
                if 0 <= threshold <= discordant:
                    tail -= exp(
                        factorials[discordant]
                        - factorials[threshold]
                        - factorials[discordant - threshold]
                        + threshold * log_share
                        + (discordant - threshold) * log_other
                    )
                threshold += 1
            if threshold <= 0:
                tail = 1.0
            elif threshold > discordant:
                tail = 0.0
        if tail <= 0.0:
            continue
        total += exp(
            log_n
            - factorials[discordant]
            - factorials[n - discordant]
            + discordant * log_disc
            + (n - discordant) * log_conc
        ) * (tail if tail < 1.0 else 1.0)
    return min(max(total, 0.0), 1.0)


def _nuisance_range(delta0: float) -> tuple[float, float]:
    return max(0.0, -delta0), (1.0 - delta0) / 2.0


def _null_supremum(
    pairs: int,
    delta0: float,
    thresholds: list[int],
    *,
    resolution: int | None = None,
    refine: int = _REFINED_PEAKS,
    golden_iterations: int = _GOLDEN_ITERATIONS,
) -> float:
    """Return the supremum of the rejection probability on ``Delta = delta0``."""

    lower, upper = _nuisance_range(delta0)
    if upper <= lower:
        return _rejection_probability(pairs, thresholds, lower, lower + delta0)
    points = resolution if resolution is not None else max(64, pairs)

    def probability(nuisance: float) -> float:
        nuisance = min(max(nuisance, lower), upper)
        return _rejection_probability(
            pairs, thresholds, nuisance, max(nuisance + delta0, 0.0)
        )

    grid = [lower + (upper - lower) * index / points for index in range(points + 1)]
    values = [probability(value) for value in grid]
    best = max(values)
    peaks = [
        index
        for index in range(len(values))
        if (index == 0 or values[index] >= values[index - 1])
        and (index == len(values) - 1 or values[index] >= values[index + 1])
    ]
    peaks.sort(key=values.__getitem__, reverse=True)
    for index in peaks[:refine]:
        left = grid[max(index - 1, 0)]
        right = grid[min(index + 1, len(grid) - 1)]
        first = right - _GOLDEN * (right - left)
        second = left + _GOLDEN * (right - left)
        first_value = probability(first)
        second_value = probability(second)
        for _ in range(golden_iterations):
            if first_value >= second_value:
                right, second, second_value = second, first, first_value
                first = right - _GOLDEN * (right - left)
                first_value = probability(first)
            else:
                left, first, first_value = first, second, second_value
                second = left + _GOLDEN * (right - left)
                second_value = probability(second)
            best = max(best, first_value, second_value)
    return min(best, 1.0)


def score_exact_p_value(
    candidate_only: int,
    baseline_only: int,
    pairs: int,
    delta0: float,
    *,
    resolution: int | None = None,
) -> float:
    """Exact unconditional p-value for ``H0: Delta <= delta0``."""

    _validate_counts(candidate_only, baseline_only, pairs)
    if not -1.0 < delta0 < 1.0:
        raise PairedInferenceError("delta0 must be in (-1, 1)")
    observed = tango_score(candidate_only, baseline_only, pairs, delta0)
    thresholds = _rejection_thresholds(pairs, delta0, observed)
    return _null_supremum(pairs, delta0, thresholds, resolution=resolution)


def _minimum_region_score(
    pairs: int, delta0: float, thresholds: list[int]
) -> float | None:
    scores = [
        tango_score(threshold, discordant - threshold, pairs, delta0)
        for discordant, threshold in enumerate(thresholds)
        if threshold <= discordant
    ]
    return min(scores) if scores else None


_CRITICAL_VALUES: dict[tuple[int, float, float], float] = {}
_CRITICAL_VALUE_CACHE_LIMIT = 8192


def score_exact_critical_value(pairs: int, delta0: float, alpha: float) -> float:
    """Smallest score whose exact rejection region has size at most ``alpha``.

    A coarse supremum (a lower bound on the true supremum) brackets the value
    by bisection; the fine supremum then removes boundary cells one score
    level at a time until the region is valid. The returned value is the
    smallest score inside the final region, so it is canonical: the bracket
    (which may be warm-started from a cached neighbouring ``pairs``) affects
    speed only, never the result.
    """

    if pairs <= 0:
        raise PairedInferenceError("pairs must be positive")
    alpha = _validate_alpha(alpha)
    key = (pairs, float(delta0), alpha)
    if key in _CRITICAL_VALUES:
        return _CRITICAL_VALUES[key]

    # The bisection grid grows with n so that the coarse supremum stays close
    # to the fine one; otherwise many fine step-ups are needed afterwards.
    bisection_resolution = max(_COARSE_RESOLUTION, pairs // 8)

    def coarse(thresholds: list[int]) -> float:
        return _null_supremum(
            pairs,
            delta0,
            thresholds,
            resolution=bisection_resolution,
            refine=1,
            golden_iterations=_COARSE_GOLDEN_ITERATIONS,
        )

    full_low = tango_score(0, pairs, pairs, delta0) - 1.0
    full_high = tango_score(pairs, 0, pairs, delta0) + 1.0
    hints = [
        _CRITICAL_VALUES[(pairs + offset, key[1], alpha)]
        for offset in (-1, 1, -2, 2, -3, 3)
        if (pairs + offset, key[1], alpha) in _CRITICAL_VALUES
    ]
    centre = hints[0] if hints else _normal_quantile(1.0 - alpha)
    width = 0.1 if hints else 0.6
    low, high = max(full_low, centre - width), min(full_high, centre + width)
    low_region = _rejection_thresholds(pairs, delta0, low)
    high_region = _rejection_thresholds(pairs, delta0, high)
    # Verify the bracket (coarse(low) > alpha >= coarse(high)); widen if not.
    if coarse(high_region) > alpha:
        high = full_high
        high_region = _rejection_thresholds(pairs, delta0, high)
    if coarse(low_region) <= alpha:
        low = full_low
        low_region = _rejection_thresholds(pairs, delta0, low)
    for _ in range(80):
        if low_region == high_region or high - low < 1e-9:
            break
        middle = (low + high) / 2.0
        region = _rejection_thresholds(pairs, delta0, middle)
        if coarse(region) <= alpha:
            high, high_region = middle, region
        else:
            low, low_region = middle, region
    critical = high
    while True:
        thresholds = _rejection_thresholds(pairs, delta0, critical)
        if _null_supremum(pairs, delta0, thresholds) <= alpha:
            smallest = _minimum_region_score(pairs, delta0, thresholds)
            result = critical if smallest is None else smallest
            break
        smallest = _minimum_region_score(pairs, delta0, thresholds)
        if smallest is None:
            result = critical
            break
        critical = max(critical, smallest) + 4.0 * _TIE_TOLERANCE
    if len(_CRITICAL_VALUES) >= _CRITICAL_VALUE_CACHE_LIMIT:
        _CRITICAL_VALUES.clear()
    _CRITICAL_VALUES[key] = result
    return result


def score_exact_rejects(
    candidate_only: int,
    baseline_only: int,
    pairs: int,
    margin: float,
    *,
    alpha: float = 0.05,
) -> bool:
    """Return whether ``H0: Delta <= -margin`` is rejected at level ``alpha``.

    Equivalent to ``score_exact_p_value(...) <= alpha``. A coarse supremum is
    a lower bound on the fine one, so coarse non-rejection is final.
    """

    _validate_counts(candidate_only, baseline_only, pairs)
    delta0 = -_validate_margin(margin)
    alpha = _validate_alpha(alpha)
    observed = tango_score(candidate_only, baseline_only, pairs, delta0)
    thresholds = _rejection_thresholds(pairs, delta0, observed)
    if (
        _null_supremum(
            pairs, delta0, thresholds, resolution=_COARSE_RESOLUTION, refine=1
        )
        > alpha
    ):
        return False
    return _null_supremum(pairs, delta0, thresholds) <= alpha


def score_exact_lower_bound(
    candidate_only: int,
    baseline_only: int,
    pairs: int,
    *,
    alpha: float = 0.05,
    anchor: float | None = None,
    tolerance: float = 1e-6,
) -> float:
    """One-sided ``1 - alpha`` lower bound obtained by inverting the exact test.

    The bound is the largest rejected null value found by bisection, so it is
    conservative relative to the true infimum of non-rejected values. When an
    ``anchor`` (the negative preregistered margin) is supplied, it is probed
    first; the bound is then at least the anchor exactly when the gate rejects.
    """

    _validate_counts(candidate_only, baseline_only, pairs)
    alpha = _validate_alpha(alpha)
    estimate = (candidate_only - baseline_only) / pairs
    floor = -1.0 + 1e-9

    def rejects(delta0: float) -> bool:
        # The coarse supremum never exceeds the fine one, so a coarse
        # non-rejection is final and only coarse rejections need confirming.
        observed = tango_score(candidate_only, baseline_only, pairs, delta0)
        thresholds = _rejection_thresholds(pairs, delta0, observed)
        coarse = _null_supremum(
            pairs, delta0, thresholds, resolution=_COARSE_RESOLUTION, refine=1
        )
        if coarse > alpha:
            return False
        return _null_supremum(pairs, delta0, thresholds) <= alpha

    if estimate <= floor or not rejects(floor):
        return -1.0
    low, high = floor, min(estimate, 1.0 - 1e-9)
    if rejects(high):
        return high
    if anchor is not None and low < anchor < high:
        if rejects(anchor):
            low = anchor
        else:
            high = anchor
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if rejects(middle):
            low = middle
        else:
            high = middle
    return low


def paired_score_exact_statistics(
    counts: Mapping[str, int],
    *,
    margin: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize paired outcomes under the preregistered score-exact method."""

    margin = _validate_margin(margin)
    alpha = _validate_alpha(1.0 - confidence)
    normalized = {
        name: int(counts.get(name, 0))
        for name in ("both_success", "both_failure", "candidate_only", "baseline_only")
    }
    total = sum(normalized.values())
    base = {
        "method": PAIRED_SCORE_EXACT_LABEL,
        "method_id": PAIRED_SCORE_EXACT,
        "noninferiority_margin": margin,
        "noninferiority_alpha": alpha,
        "confidence_level": confidence,
        "outcome_counts": normalized,
        "confidence_interval": None,
        "interval_note": (
            "this method reports a one-sided lower bound that inverts the exact "
            "test; no two-sided interval is preregistered"
        ),
    }
    if not total:
        return {
            **base,
            "paired_count": 0,
            "candidate_minus_baseline_mean": None,
            "one_sided_lower_bound": None,
            "noninferiority_p_value": None,
            "rejects_inferiority": False,
        }
    candidate_only = normalized["candidate_only"]
    baseline_only = normalized["baseline_only"]
    p_value = score_exact_p_value(candidate_only, baseline_only, total, -margin)
    lower = score_exact_lower_bound(
        candidate_only, baseline_only, total, alpha=alpha, anchor=-margin
    )
    return {
        **base,
        "paired_count": total,
        "candidate_minus_baseline_mean": (candidate_only - baseline_only) / total,
        "one_sided_lower_bound": lower,
        "noninferiority_p_value": p_value,
        "rejects_inferiority": p_value <= alpha,
    }


# ---------------------------------------------------------------------------
# Binomial helpers for planning
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def minimum_successes_for_absolute_gate(
    trials: int, threshold: float, confidence: float = 0.95
) -> int | None:
    """Smallest success count whose two-sided exact interval clears ``threshold``.

    Mirrors the absolute gate, which compares the lower end of the two-sided
    ``confidence`` Clopper-Pearson interval with the preregistered minimum.
    """

    if threshold <= 0.0:
        return 0
    tail = (1.0 - confidence) / 2.0
    if binomial_lower_bound(trials, trials, tail) < threshold:
        return None
    low, high = 0, trials
    while low < high:
        middle = (low + high) // 2
        if binomial_lower_bound(middle, trials, tail) >= threshold:
            high = middle
        else:
            low = middle + 1
    return low


def completion_tolerance(
    attempts: int, noncompletion_rate: float, pass_probability: float
) -> int:
    """Smallest integer tolerance ``k`` with ``P(Bin(attempts, r) <= k) >= target``."""

    if noncompletion_rate <= 0.0:
        return 0
    for tolerance in range(attempts + 1):
        if binomial_cdf(tolerance, attempts, noncompletion_rate) >= pass_probability:
            return tolerance
    return attempts


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


def paired_cell_probabilities(
    *,
    expected_success_rate: float,
    expected_discordance_rate: float,
    expected_difference: float = 0.0,
) -> dict[str, float]:
    """Translate planning assumptions into the four paired cell probabilities.

    ``expected_success_rate`` is the baseline success rate. The candidate rate
    is the baseline rate plus ``expected_difference``.
    """

    success = float(expected_success_rate)
    discordance = float(expected_discordance_rate)
    difference = float(expected_difference)
    if not 0.0 < success <= 1.0:
        raise PairedInferenceError("expected_success_rate must be in (0, 1]")
    if not 0.0 <= discordance <= 1.0:
        raise PairedInferenceError("expected_discordance_rate must be in [0, 1]")
    if abs(difference) > discordance:
        raise PairedInferenceError(
            "the absolute expected difference cannot exceed the discordance rate"
        )
    baseline_only = (discordance - difference) / 2.0
    candidate_only = (discordance + difference) / 2.0
    both_success = success - baseline_only
    both_failure = 1.0 - both_success - baseline_only - candidate_only
    if both_success < -1e-12 or both_failure < -1e-12:
        raise PairedInferenceError(
            "the success rate, discordance, and difference are mutually inconsistent"
        )
    return {
        "both_success": max(both_success, 0.0),
        "both_failure": max(both_failure, 0.0),
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
    }


def _paired_exact_reject_thresholds(
    pairs: int, margin: float, alpha: float, baseline_only_values: range
) -> dict[int, int]:
    """For each baseline-only count, the minimal candidate-only count rejecting."""

    half = alpha / 2.0
    thresholds: dict[int, int] = {}
    for baseline_only in baseline_only_values:
        upper = binomial_upper_bound(baseline_only, pairs, half)
        limit = upper - margin
        maximum = pairs - baseline_only
        if binomial_lower_bound(maximum, pairs, half) < limit:
            thresholds[baseline_only] = maximum + 1
            continue
        low, high = 0, maximum
        while low < high:
            middle = (low + high) // 2
            if binomial_lower_bound(middle, pairs, half) >= limit:
                high = middle
            else:
                low = middle + 1
        thresholds[baseline_only] = low
    return thresholds


def _window(mean: float, deviation: float, low: int, high: int) -> range:
    span = 12.0 * deviation + 12.0
    start = max(low, int(math.floor(mean - span)))
    stop = min(high, int(math.ceil(mean + span)))
    return range(start, stop + 1)


def _binomial_pmf(trials: int, successes: int, log_p: float, log_q: float) -> float:
    if successes < 0 or successes > trials:
        return 0.0
    factorials = _log_factorials(trials)
    value = (
        factorials[trials]
        - factorials[successes]
        - factorials[trials - successes]
        + _xlog(successes, log_p)
        + _xlog(trials - successes, log_q)
    )
    return 0.0 if value == -math.inf else math.exp(value)


def gate_power(
    *,
    pairs: int,
    method: str,
    margin: float,
    cells: Mapping[str, float],
    min_task_success_rate: float = 0.0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Exact probability that the paired and absolute success gates pass.

    ``noninferiority`` is the probability that the preregistered paired method
    rejects inferiority. ``quality_gates`` additionally requires both variants
    to clear the absolute Clopper-Pearson gate. Both are exact sums over the
    multinomial outcome space (negligible-mass cells beyond 12 standard
    deviations are skipped).
    """

    if method not in PAIRED_METHODS:
        raise PairedInferenceError(f"power is defined only for {sorted(PAIRED_METHODS)}")
    margin = _validate_margin(margin)
    alpha = _validate_alpha(alpha)
    n = pairs
    q10 = float(cells["baseline_only"])
    q01 = float(cells["candidate_only"])
    q11 = float(cells["both_success"])
    q00 = float(cells["both_failure"])
    discordance = q10 + q01
    share = q01 / discordance if discordance > 0 else 0.0
    concordant = q11 + q00
    success_share = q11 / concordant if concordant > 0 else 0.0
    minimum = minimum_successes_for_absolute_gate(n, float(min_task_success_rate))
    log_disc, log_conc = _safe_log(discordance), _safe_log1m(discordance)
    log_share, log_other = _safe_log(share), _safe_log1m(share)

    score_thresholds: list[int] | None = None
    exact_thresholds: dict[int, int] = {}
    if method == PAIRED_SCORE_EXACT:
        critical = score_exact_critical_value(n, -margin, alpha)
        score_thresholds = _rejection_thresholds(n, -margin, critical)
    else:
        exact_thresholds = _paired_exact_reject_thresholds(
            n,
            margin,
            alpha,
            _window(n * q10, math.sqrt(n * q10 * (1.0 - q10)), 0, n),
        )

    def rejects(candidate_only: int, baseline_only: int) -> bool:
        if score_thresholds is not None:
            return candidate_only >= score_thresholds[candidate_only + baseline_only]
        threshold = exact_thresholds.get(baseline_only)
        return threshold is not None and candidate_only >= threshold

    noninferiority = 0.0
    quality = 0.0
    discordant_window = _window(
        n * discordance, math.sqrt(n * discordance * (1.0 - discordance)), 0, n
    )
    for discordant in discordant_window:
        weight = _binomial_pmf(n, discordant, log_disc, log_conc)
        if weight == 0.0:
            continue
        concordant_pairs = n - discordant
        tails: dict[int, float] = {}
        split_window = _window(
            discordant * share,
            math.sqrt(discordant * share * (1.0 - share)),
            0,
            discordant,
        )
        for candidate_only in split_window:
            baseline_only = discordant - candidate_only
            if not rejects(candidate_only, baseline_only):
                continue
            probability = weight * _binomial_pmf(
                discordant, candidate_only, log_share, log_other
            )
            noninferiority += probability
            if minimum is None:
                continue
            needed = minimum - min(candidate_only, baseline_only)
            if needed not in tails:
                if needed <= 0:
                    tails[needed] = 1.0
                elif needed > concordant_pairs:
                    tails[needed] = 0.0
                else:
                    tails[needed] = 1.0 - binomial_cdf(
                        needed - 1, concordant_pairs, success_share
                    )
            quality += probability * tails[needed]
    return {
        "noninferiority": min(noninferiority, 1.0),
        "quality_gates": min(quality, 1.0) if minimum is not None else 0.0,
        "minimum_successes_per_variant": minimum,
    }


def asymptotic_score_pairs(
    *,
    margin: float,
    cells: Mapping[str, float],
    alpha: float = 0.05,
    power: float = 0.8,
) -> int | None:
    """Normal-approximation starting point for the exact sample-size search."""

    difference = float(cells["candidate_only"]) - float(cells["baseline_only"])
    discordance = float(cells["candidate_only"]) + float(cells["baseline_only"])
    distance = difference + margin
    if distance <= 0:
        return None
    variance = max(discordance - difference * difference, 1e-9)
    z_alpha = _normal_quantile(1.0 - alpha)
    z_beta = _normal_quantile(power)
    return max(2, math.ceil((z_alpha + z_beta) ** 2 * variance / distance**2))


def _normal_quantile(probability: float) -> float:
    """Acklam's rational approximation, accurate to about 1e-9."""

    if not 0.0 < probability < 1.0:
        raise PairedInferenceError("probability must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low = 0.02425
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if probability > 1.0 - low:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))
