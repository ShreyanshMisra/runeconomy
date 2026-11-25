"""Braintrust/autoevals-style scorers, written as Braintrust users write them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Score:  # mirrors autoevals.Score
    name: str
    score: float | None
    metadata: dict | None = None


def ExactMatch(input, output, expected, **kwargs):  # noqa: N802 - Braintrust naming
    return Score(name="ExactMatch", score=1.0 if output.strip() == str(expected) else 0.0)


async def AsyncPrefix(input, output, expected, **kwargs):  # noqa: N802
    return Score(name="AsyncPrefix", score=1.0 if output.startswith(str(expected)[:2]) else 0.0)


def Abstains(input, output, expected, **kwargs):  # noqa: N802
    return Score(name="Abstains", score=None)


def plain_number(input, output, expected):
    return 0.75
