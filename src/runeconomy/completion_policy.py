"""Preregistered completion semantics for common-mode paired failures.

A common-mode paired failure is a task on which both the baseline and the
candidate attempt (same task, same repetition) fail the same way: the same
declared task-level failure status with the same failure provenance
(component, stage, and code). Such a pair says something about the task, not
about either variant.

Completion policy schema 0.1 offers two classifications.

``variant_non_completion`` (legacy; used by every contract without a policy)
    Each failed attempt counts against its variant's completion gate.

``task_outcome`` (opt-in, prospective)
    Common-mode paired failures stay in the task-success population as a
    both-failure pair, so they still lower each variant's absolute success
    rate and are neutral in the paired difference. They do not count as
    variant non-completion. Only declared task-level statuses are eligible.
    Infrastructure failures and timeouts carry no evaluated task outcome, so
    reclassifying them would invent an outcome. They always remain
    non-completion.

The policy is part of the frozen acceptance contract. It applies only to
contracts that declare it before execution and never to a recorded result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


COMPLETION_POLICY_SCHEMA_VERSION = "0.1"
VARIANT_NON_COMPLETION = "variant_non_completion"
TASK_OUTCOME = "task_outcome"
COMMON_MODE_CLASSIFICATIONS = frozenset({VARIANT_NON_COMPLETION, TASK_OUTCOME})
COMMON_MODE_ELIGIBLE_STATUSES = frozenset(
    {"application_failure", "model_output_failure"}
)


class CompletionPolicyError(ValueError):
    """Raised when a declared completion policy is malformed."""


@dataclass(frozen=True)
class CompletionPolicy:
    common_mode_paired_failures: str = VARIANT_NON_COMPLETION
    schema_version: str = COMPLETION_POLICY_SCHEMA_VERSION

    @classmethod
    def from_value(cls, value: Any) -> "CompletionPolicy":
        """Parse a declared policy; ``None`` means the legacy semantics."""

        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise CompletionPolicyError("acceptance.completion_policy must be an object")
        allowed = {"schema_version", "common_mode_paired_failures"}
        if not set(value).issubset(allowed):
            raise CompletionPolicyError(
                "acceptance.completion_policy contains unsupported fields"
            )
        if value.get("schema_version") != COMPLETION_POLICY_SCHEMA_VERSION:
            raise CompletionPolicyError(
                "acceptance.completion_policy.schema_version must be "
                f"{COMPLETION_POLICY_SCHEMA_VERSION!r}"
            )
        classification = value.get("common_mode_paired_failures")
        if classification not in COMMON_MODE_CLASSIFICATIONS:
            raise CompletionPolicyError(
                "acceptance.completion_policy.common_mode_paired_failures must be "
                "variant_non_completion or task_outcome"
            )
        return cls(common_mode_paired_failures=str(classification))

    @property
    def reclassifies_common_mode(self) -> bool:
        return self.common_mode_paired_failures == TASK_OUTCOME

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "common_mode_paired_failures": self.common_mode_paired_failures,
        }


def _failure_signature(attempt: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    status = attempt.get("status")
    if status not in COMMON_MODE_ELIGIBLE_STATUSES:
        return None
    if attempt.get("task_success") is not False:
        return None
    agent_status = attempt.get("agent_status")
    if not isinstance(agent_status, Mapping):
        return None
    provenance = agent_status.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    labels = tuple(provenance.get(name) for name in ("component", "stage", "code"))
    if not all(isinstance(label, str) and label for label in labels):
        return None
    return (str(status), *labels)  # type: ignore[return-value]


def common_mode_failure_pairs(
    attempts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return pairs whose two attempts failed with an identical signature."""

    by_key: dict[tuple[Any, Any, str], Mapping[str, Any]] = {}
    for attempt in attempts:
        by_key[(attempt.get("task_id"), attempt.get("repetition"), str(attempt.get("variant")))] = attempt
    pairs: list[dict[str, Any]] = []
    keys = sorted(
        {(task_id, repetition) for task_id, repetition, _ in by_key},
        key=lambda item: (str(item[0]), str(item[1])),
    )
    for task_id, repetition in keys:
        baseline = by_key.get((task_id, repetition, "baseline"))
        candidate = by_key.get((task_id, repetition, "candidate"))
        if baseline is None or candidate is None:
            continue
        signature = _failure_signature(baseline)
        if signature is None or signature != _failure_signature(candidate):
            continue
        pairs.append(
            {
                "task_id": task_id,
                "repetition": repetition,
                "status": signature[0],
                "provenance": {
                    "component": signature[1],
                    "stage": signature[2],
                    "code": signature[3],
                },
            }
        )
    return pairs


def completion_policy_record(
    policy: CompletionPolicy, pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Describe the frozen policy and the pairs it observed."""

    return {
        **policy.to_dict(),
        "eligible_statuses": sorted(COMMON_MODE_ELIGIBLE_STATUSES),
        "same_failure_definition": (
            "same task and repetition, same declared task-level failure status, "
            "and identical provenance component, stage, and code"
        ),
        "common_mode_pair_count": len(pairs),
        "reclassified_as_task_outcome": (
            len(pairs) if policy.reclassifies_common_mode else 0
        ),
        "pairs": pairs,
    }
