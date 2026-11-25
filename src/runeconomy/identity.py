"""Graded serving identity and the claims a verdict can support.

Every inference request in an experiment carries a serving identity: backend,
backend version, model, model revision, tokenizer, tokenizer revision, and
chat-template digest. Legacy contracts require every field to hold a value and
reject only empty strings and ``"unknown"``. Those contracts keep exactly that
behavior.

Identity policy schema ``0.1`` is an opt-in, preregistered acceptance field
(``acceptance.identity_policy``) that grades identity instead:

``operator_verified``
    The operator controls the serving artifact (self-hosted weights, tokenizer,
    template, engine) and pins every field to a concrete value. A verdict can
    be attributed to that pinned configuration.

``provider_declared``
    Some or all fields come from a provider the operator does not control: a
    hosted API reports a model snapshot, but not its tokenizer, template, or
    serving stack. Fields the provider does not publish are recorded as
    ``null`` and listed explicitly as undisclosed. A verdict describes the
    provider service as observed during the run. It cannot be reproduced
    against, or attributed to, a pinned configuration.

Under a declared policy, placeholder strings such as ``"undisclosed"`` or
``"provider-managed"`` are not identity values. They count as missing, so the
only way to record an unknown field is an explicit undisclosed declaration on
a ``provider_declared`` request. A missing field that is not declared remains
incomplete identity, and the verdict is ``invalid``.

Provider ``system_fingerprint`` values are a drift indicator only. They are
summarized per variant as counts (never values), are not part of identity,
and never change validity or the verdict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_POLICY_SCHEMA_VERSION = "0.1"
IDENTITY_ASSURANCE_SCHEMA_VERSION = "0.1"

OPERATOR_VERIFIED = "operator_verified"
PROVIDER_DECLARED = "provider_declared"
# Strongest first. A variant's grade is the weakest grade of its requests.
IDENTITY_GRADES = (OPERATOR_VERIFIED, PROVIDER_DECLARED)
_GRADE_RANK = {grade: rank for rank, grade in enumerate(IDENTITY_GRADES)}

IDENTITY_FIELDS = (
    "backend",
    "backend_version",
    "model",
    "model_revision",
    "tokenizer",
    "tokenizer_revision",
    "chat_template_digest",
)
# Fields a provider may leave undisclosed. Backend and model name are always
# required: without them there is nothing to attribute a result to.
DISCLOSABLE_FIELDS = frozenset(IDENTITY_FIELDS) - {"backend", "model"}

PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "unknown",
        "undisclosed",
        "provider-managed",
        "provider_managed",
        "n/a",
        "na",
        "none",
        "null",
        "unset",
    }
)

# OpenInference/OTLP span attributes.
TRACE_GRADE_ATTRIBUTE = "runeconomy.identity_grade"
TRACE_UNDISCLOSED_ATTRIBUTE = "runeconomy.identity_undisclosed"
TRACE_FINGERPRINT_ATTRIBUTE = "runeconomy.provider_system_fingerprint"

SCOPE_PINNED = "pinned_configuration"
SCOPE_OBSERVED_SERVICE = "observed_service_window"


class IdentityPolicyError(ValueError):
    """Raised when a declared identity policy or declaration is malformed."""


@dataclass(frozen=True)
class IdentityPolicy:
    minimum_grade: str = OPERATOR_VERIFIED
    schema_version: str = IDENTITY_POLICY_SCHEMA_VERSION

    @classmethod
    def from_value(cls, value: Any) -> "IdentityPolicy":
        if not isinstance(value, Mapping):
            raise IdentityPolicyError("acceptance.identity_policy must be an object")
        if not set(value).issubset({"schema_version", "minimum_grade"}):
            raise IdentityPolicyError(
                "acceptance.identity_policy contains unsupported fields"
            )
        if value.get("schema_version") != IDENTITY_POLICY_SCHEMA_VERSION:
            raise IdentityPolicyError(
                "acceptance.identity_policy.schema_version must be "
                f"{IDENTITY_POLICY_SCHEMA_VERSION!r}"
            )
        grade = value.get("minimum_grade")
        if grade not in _GRADE_RANK:
            raise IdentityPolicyError(
                "acceptance.identity_policy.minimum_grade must be "
                "operator_verified or provider_declared"
            )
        return cls(minimum_grade=str(grade))

    def accepts(self, grade: str) -> bool:
        return _GRADE_RANK.get(grade, len(IDENTITY_GRADES)) <= _GRADE_RANK[
            self.minimum_grade
        ]

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "minimum_grade": self.minimum_grade,
        }


def parse_identity_policy(value: Any) -> IdentityPolicy | None:
    """``None`` means no policy: the legacy identity semantics apply."""

    return None if value is None else IdentityPolicy.from_value(value)


def is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() in PLACEHOLDER_VALUES


def parse_grade(value: Any) -> str | None:
    if value is None:
        return None
    if value not in _GRADE_RANK:
        raise IdentityPolicyError(
            f"{TRACE_GRADE_ATTRIBUTE} must be operator_verified or provider_declared"
        )
    return str(value)


def parse_undisclosed(value: Any) -> list[str] | None:
    """Parse a JSON-array attribute (or list) of undisclosed field names."""

    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise IdentityPolicyError(
                f"{TRACE_UNDISCLOSED_ATTRIBUTE} must be a JSON array of field names"
            ) from exc
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise IdentityPolicyError(
            f"{TRACE_UNDISCLOSED_ATTRIBUTE} must be a JSON array of field names"
        )
    unknown = sorted(set(value) - DISCLOSABLE_FIELDS)
    if unknown:
        raise IdentityPolicyError(
            f"{TRACE_UNDISCLOSED_ATTRIBUTE} names fields that cannot be undisclosed: "
            + ", ".join(unknown)
        )
    return sorted(set(value))


def fingerprint_digest(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def assess_request_identity(
    identity: Mapping[str, Any],
    *,
    grade: str | None,
    undisclosed: Sequence[str] | None,
    fields: Sequence[str] = IDENTITY_FIELDS,
) -> dict[str, Any]:
    """Assess one request's identity under a declared policy.

    A request without a grade declaration is ``operator_verified``: that is
    the meaning legacy identity always had. Undisclosed fields are honored
    only on ``provider_declared`` requests.
    """

    resolved_grade = grade or OPERATOR_VERIFIED
    declared_undisclosed = set(undisclosed or ())
    missing: list[str] = []
    placeholders: list[str] = []
    undisclosed_fields: list[str] = []
    for name in fields:
        value = identity.get(name)
        if is_placeholder(value):
            placeholders.append(name)
        elif value is None:
            if resolved_grade == PROVIDER_DECLARED and name in declared_undisclosed:
                undisclosed_fields.append(name)
            else:
                missing.append(name)
    contradicted = sorted(
        name
        for name in declared_undisclosed
        if name in fields and identity.get(name) is not None and not is_placeholder(identity.get(name))
    )
    return {
        "grade": resolved_grade,
        "complete": not missing and not placeholders and not contradicted,
        "missing_fields": missing,
        "placeholder_fields": placeholders,
        "undisclosed_fields": undisclosed_fields,
        "contradicted_undisclosed_fields": contradicted,
    }


def weakest_grade(grades: Iterable[str]) -> str | None:
    ordered = sorted(set(grades), key=lambda grade: _GRADE_RANK[grade])
    return ordered[-1] if ordered else None


def fingerprint_summary(
    request_count: int, digests: Iterable[str | None]
) -> dict[str, Any]:
    """Privacy-safe drift indicator. It never affects identity or validity."""

    reported = [digest for digest in digests if digest is not None]
    distinct = len(set(reported))
    return {
        "request_count": request_count,
        "reported_count": len(reported),
        "distinct_count": distinct,
        "changed_during_run": distinct > 1,
        "affects_identity": False,
        "affects_validity": False,
    }


def variant_identity_summary(
    policy: IdentityPolicy,
    assessments: Sequence[Mapping[str, Any]],
    fingerprint_digests: Sequence[str | None],
) -> dict[str, Any]:
    grade = weakest_grade(item["grade"] for item in assessments)

    def union(key: str) -> list[str]:
        return sorted({name for item in assessments for name in item[key]})

    complete = bool(assessments) and all(item["complete"] for item in assessments)
    meets = grade is not None and policy.accepts(grade)
    return {
        "grade": grade,
        "request_count": len(assessments),
        "complete": complete,
        "meets_minimum_grade": meets,
        "valid": complete and meets,
        "missing_fields": union("missing_fields"),
        "placeholder_fields": union("placeholder_fields"),
        "undisclosed_fields": union("undisclosed_fields"),
        "contradicted_undisclosed_fields": union("contradicted_undisclosed_fields"),
        "provider_fingerprint": fingerprint_summary(
            len(assessments), fingerprint_digests
        ),
    }


_LIMIT_STATEMENTS = {
    "identity.provider_declared": (
        "Identity is provider-declared, not operator-verified. The verdict "
        "describes the provider service as observed during this run. It cannot "
        "be reproduced against, or attributed to, a pinned model or serving "
        "configuration."
    ),
    "identity.undisclosed_fields": (
        "The provider did not disclose some identity fields. Changes to them "
        "during or after the run cannot be detected."
    ),
    "identity.provider_fingerprint_changed": (
        "The provider reported more than one system fingerprint. This is a "
        "drift indicator only. It does not affect identity or validity."
    ),
}


def identity_assurance_record(
    policy: IdentityPolicy, variants: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Describe graded identity and the claim scope it supports."""

    limits: list[dict[str, Any]] = []
    for variant in ("baseline", "candidate"):
        summary = variants.get(variant)
        if not isinstance(summary, Mapping):
            continue
        if summary.get("grade") == PROVIDER_DECLARED:
            limits.append(
                {
                    "code": "identity.provider_declared",
                    "variant": variant,
                    "statement": _LIMIT_STATEMENTS["identity.provider_declared"],
                }
            )
        if summary.get("undisclosed_fields"):
            limits.append(
                {
                    "code": "identity.undisclosed_fields",
                    "variant": variant,
                    "fields": list(summary["undisclosed_fields"]),
                    "statement": _LIMIT_STATEMENTS["identity.undisclosed_fields"],
                }
            )
        fingerprint = summary.get("provider_fingerprint")
        if isinstance(fingerprint, Mapping) and fingerprint.get("changed_during_run"):
            limits.append(
                {
                    "code": "identity.provider_fingerprint_changed",
                    "variant": variant,
                    "distinct_count": fingerprint.get("distinct_count"),
                    "statement": _LIMIT_STATEMENTS[
                        "identity.provider_fingerprint_changed"
                    ],
                }
            )
    grades = {
        variant: summary.get("grade")
        for variant, summary in variants.items()
        if isinstance(summary, Mapping)
    }
    pinned = all(grade == OPERATOR_VERIFIED for grade in grades.values())
    return {
        "schema_version": IDENTITY_ASSURANCE_SCHEMA_VERSION,
        "policy": policy.to_dict(),
        "grades": dict(sorted(grades.items())),
        "claim_scope": SCOPE_PINNED if pinned else SCOPE_OBSERVED_SERVICE,
        "claim_limits": limits,
        "variants": {key: dict(value) for key, value in sorted(variants.items())},
    }
