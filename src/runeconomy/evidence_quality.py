"""Machine-readable evidence strength for imported and executed measurements."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


EVIDENCE_QUALITY_SCHEMA_VERSION = "0.1"
EVIDENCE_GRADES = (
    "directly_observed",
    "reliably_derived",
    "weakly_inferred",
    "unresolved",
)
_GRADE_RANK = {grade: index for index, grade in enumerate(EVIDENCE_GRADES)}


def evidence_field(
    grade: str, source: str, *, detail: str | None = None
) -> dict[str, str]:
    """Build one validated, content-safe field assessment."""

    if grade not in _GRADE_RANK:
        raise ValueError(f"unsupported evidence grade {grade!r}")
    value = {"grade": grade, "source": source}
    if detail:
        value["detail"] = detail
    return value


def build_evidence_quality(
    source_type: str, fields: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    """Build a versioned evidence-quality record.

    The overall grade is deliberately the weakest recorded field. Callers must
    not omit an unresolved field merely to obtain a stronger aggregate label.
    """

    normalized: dict[str, dict[str, str]] = {}
    for name, assessment in sorted(fields.items()):
        grade = assessment.get("grade")
        source = assessment.get("source")
        if grade not in _GRADE_RANK or not isinstance(source, str) or not source:
            raise ValueError(f"invalid evidence quality for {name}")
        normalized[name] = dict(assessment)
    grades = [item["grade"] for item in normalized.values()]
    overall = (
        max(grades, key=_GRADE_RANK.__getitem__) if grades else "unresolved"
    )
    counts = Counter(grades)
    return {
        "schema_version": EVIDENCE_QUALITY_SCHEMA_VERSION,
        "source_type": source_type,
        "overall_grade": overall,
        "grade_counts": {
            grade: counts.get(grade, 0) for grade in EVIDENCE_GRADES
        },
        "fields": normalized,
    }


def aggregate_evidence_quality(
    records: Sequence[Mapping[str, Any]], *, source_type: str
) -> dict[str, Any]:
    """Aggregate records without allowing stronger evidence to mask weaker data."""

    field_names = sorted(
        {
            str(name)
            for record in records
            for name in (
                record.get("fields", {})
                if isinstance(record.get("fields"), Mapping)
                else {}
            )
        }
    )
    fields: dict[str, dict[str, str]] = {}
    for name in field_names:
        assessments = [
            record["fields"][name]
            for record in records
            if isinstance(record.get("fields"), Mapping)
            and isinstance(record["fields"].get(name), Mapping)
        ]
        if len(assessments) != len(records):
            fields[name] = evidence_field(
                "unresolved",
                "not present in every evidence record",
            )
            continue
        weakest = max(
            assessments,
            key=lambda item: _GRADE_RANK.get(str(item.get("grade")), 99),
        )
        distinct_sources = sorted(
            {str(item.get("source", "unknown")) for item in assessments}
        )
        fields[name] = evidence_field(
            str(weakest["grade"]),
            "; ".join(distinct_sources),
            detail=f"aggregated across {len(records)} record(s)",
        )
    if not records:
        fields["evidence_records"] = evidence_field(
            "unresolved", "no evidence records"
        )
    return build_evidence_quality(source_type, fields)
