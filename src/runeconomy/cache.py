"""Backend-specific static cache eligibility models."""

from __future__ import annotations

from .models import InferenceRequest


def longest_common_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the number of identical leading token IDs."""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def eligible_prefix_tokens(request: InferenceRequest, shared_tokens: int) -> int:
    """Return a conservative static upper bound for reusable prefix tokens.

    vLLM reuses full cache blocks. It also recomputes at least the final prompt
    token, so the upper bound excludes the current request's final token before
    rounding down. Unknown backends currently use token-level eligibility and
    are explicitly reported as generic by the analyzer.
    """
    if not request.cache.enabled or not request.prompt_token_ids:
        return 0
    bounded = min(shared_tokens, len(request.prompt_token_ids) - 1)
    if bounded <= 0:
        return 0
    if request.backend == "vllm":
        block_size = request.cache.block_size
        return (bounded // block_size) * block_size
    return bounded
