"""Inspect AI scorers used by the evaluator adapter tests."""

from inspect_ai.scorer import CORRECT, INCORRECT, PARTIAL, Score, accuracy, scorer


@scorer(metrics=[accuracy()])
def graded(partial_on: str = "maybe"):
    async def score(state, target):
        text = state.output.completion
        if text == partial_on:
            return Score(value=PARTIAL, answer=text)
        return Score(value=CORRECT if text in target.target else INCORRECT, answer=text)

    return score


@scorer(metrics=[accuracy()])
def by_key():
    async def score(state, target):
        return Score(value={"exact": 1.0 if state.output.completion == target.text else 0.0, "length": 3})

    return score


@scorer(metrics=[accuracy()])
def explodes():
    async def score(state, target):
        raise RuntimeError("boom")

    return score
