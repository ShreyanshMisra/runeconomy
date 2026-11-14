"""Conservative checks for accidentally exported credentials."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Mapping


_SENSITIVE_KEY = re.compile(
    r"(^|_)(authorization|api_?key|access_?token|proxy_?token|secret|password)($|_)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+\-/]+=*", re.IGNORECASE)


def sensitive_export_paths(value: Any, path: str = "$") -> tuple[str, ...]:
    """Return locations that should be reviewed before publishing an artifact."""

    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if (
                isinstance(key, str)
                and not key.lower().endswith("_env")
                and _SENSITIVE_KEY.search(key)
            ):
                findings.append(child_path)
            findings.extend(sensitive_export_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(sensitive_export_paths(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        if _BEARER.search(value):
            findings.append(path)
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme in {"http", "https"} and (
            parsed.username is not None or parsed.password is not None or parsed.query
        ):
            findings.append(path)
    return tuple(dict.fromkeys(findings))


def assert_safe_for_export(value: Any) -> None:
    findings = sensitive_export_paths(value)
    if findings:
        raise ValueError(
            "artifact may contain credentials or sensitive URL parameters at: "
            + ", ".join(findings)
        )
