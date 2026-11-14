"""Versioned, content-safe status records for closed-loop agent attempts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ATTEMPT_STATUS_SCHEMA_VERSION = "0.1"
ATTEMPT_STATUSES = frozenset(
    {
        "completed",
        "task_failure",
        "model_output_failure",
        "application_failure",
        "infrastructure_failure",
        "timeout",
    }
)
PROVENANCE_COMPONENTS = frozenset(
    {"agent", "application", "infrastructure", "model_output", "tool"}
)
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class AttemptStatusError(ValueError):
    """Raised when an attempt status file violates the public contract."""


@dataclass(frozen=True)
class FailureProvenance:
    component: str
    stage: str
    code: str

    @classmethod
    def from_dict(cls, value: Any) -> FailureProvenance:
        if not isinstance(value, dict) or set(value) != {
            "component",
            "stage",
            "code",
        }:
            raise AttemptStatusError(
                "provenance must contain exactly component, stage, and code"
            )
        component = value["component"]
        if component not in PROVENANCE_COMPONENTS:
            raise AttemptStatusError("provenance.component is unsupported")
        labels: dict[str, str] = {}
        for name in ("stage", "code"):
            item = value[name]
            if not isinstance(item, str) or _SAFE_LABEL.fullmatch(item) is None:
                raise AttemptStatusError(
                    f"provenance.{name} must be a content-free stable label"
                )
            labels[name] = item
        return cls(component=component, **labels)

    def to_dict(self) -> dict[str, str]:
        return {
            "component": self.component,
            "stage": self.stage,
            "code": self.code,
        }


@dataclass(frozen=True)
class AttemptStatus:
    status: str
    provenance: FailureProvenance | None = None
    schema_version: str = ATTEMPT_STATUS_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Any) -> AttemptStatus:
        if not isinstance(value, dict):
            raise AttemptStatusError("attempt status must be a JSON object")
        allowed = {"schema_version", "status", "provenance"}
        if not set(value).issubset(allowed):
            raise AttemptStatusError("attempt status contains unsupported fields")
        if value.get("schema_version") != ATTEMPT_STATUS_SCHEMA_VERSION:
            raise AttemptStatusError("unsupported attempt status schema_version")
        status = value.get("status")
        if status not in ATTEMPT_STATUSES:
            raise AttemptStatusError("unsupported attempt status")
        raw_provenance = value.get("provenance")
        provenance = (
            FailureProvenance.from_dict(raw_provenance)
            if raw_provenance is not None
            else None
        )
        if status == "completed" and provenance is not None:
            raise AttemptStatusError("completed status must not include provenance")
        if status != "completed" and provenance is None:
            raise AttemptStatusError("failure status requires provenance")
        return cls(status=status, provenance=provenance)

    @classmethod
    def from_path(cls, path: Path) -> AttemptStatus:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise AttemptStatusError("attempt status could not be read") from exc
        except json.JSONDecodeError as exc:
            raise AttemptStatusError("attempt status is not valid JSON") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
        }
        if self.provenance is not None:
            value["provenance"] = self.provenance.to_dict()
        return value


def write_attempt_status(
    path: str | Path,
    status: str,
    *,
    component: str | None = None,
    stage: str | None = None,
    code: str | None = None,
) -> None:
    """Atomically write a validated status without generated/error text."""

    provenance = None
    supplied = (component, stage, code)
    if any(item is not None for item in supplied):
        if not all(item is not None for item in supplied):
            raise AttemptStatusError(
                "component, stage, and code must be supplied together"
            )
        provenance = FailureProvenance(
            component=str(component), stage=str(stage), code=str(code)
        )
    record = AttemptStatus(status=status, provenance=provenance)
    record = AttemptStatus.from_dict(record.to_dict())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
