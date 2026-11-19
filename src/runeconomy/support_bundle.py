"""Privacy-minimized, local support bundles built from strict allowlists."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import zipfile
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .errors import (
    ERROR_CODE_VERSION,
    descriptor_for_code,
    validation_status_code,
)


SUPPORT_BUNDLE_SCHEMA_VERSION = "0.1"
SUPPORT_BUNDLE_FILES = (
    "manifest.json",
    "environment.json",
    "workspace.json",
    "capabilities.json",
    "validation.json",
    "errors.json",
)
_EXPECTED_WORKSPACE_ROLES = {
    "workspace_descriptor": "workspace.json",
    "experiment_manifest": "manifest.json",
    "intervention_description": "intervention.md",
    "workspace_readme": "README.md",
}
_MAX_JSON_BYTES = 2 * 1024 * 1024
_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,127}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_CAPABILITY_STATUSES = frozenset(
    {"available", "partial", "missing", "supported", "unsupported", "unknown"}
)
_KNOWN_FIELD_ROOTS = frozenset(
    {
        "acceptance",
        "backend",
        "baseline",
        "baseline_request",
        "candidate",
        "candidate_request",
        "cases",
        "confirmed_capabilities",
        "dataset",
        "evaluator",
        "execution",
        "experiment_context",
        "sampling",
        "unknown",
    }
)
_EXCLUDED_DATA_CLASSES = (
    "raw traces and imported source evidence",
    "prompts, messages, generated responses, evaluator inputs, and task answers",
    "replay requests and prompt-token arrays",
    "credentials, authorization headers, cookies, and environment values",
    "private endpoints, query strings, URL credentials, and IP addresses",
    "checkpoint records and experiment results",
    "source repository contents and arbitrary workspace files",
    "absolute paths, usernames, home directories, and private directory names",
    "verbatim exception payloads",
)
_CAPABILITY_KEYS = (
    "basic_diagnosis",
    "structural_prompt_attribution",
    "exact_token_cache_analysis",
    "critical_path_analysis",
    "static_request_replay",
    "complete_agent_comparison",
    "public_verification_claim",
)
_DOCTOR_CAPABILITY_KEYS = (
    "responses",
    "chat_completions",
    "responses_render",
    "chat_completions_render",
    "tokenize",
    "cache_telemetry",
    "per_request_timing",
    "cache_reset",
    "cache_salt",
)
_WORKSPACE_MODES = frozenset({"static_request", "agent_process", "unknown"})
_WORKSPACE_STATUSES = frozenset({"runnable", "unresolved", "invalid"})
_VALIDATION_STATUSES = frozenset(
    {
        "runnable",
        "unresolved",
        "invalid",
        "unsupported_backend",
        "claim_evidence_incomplete",
    }
)
_READINESS_STATUSES = frozenset({"ready", "partial", "blocked", "error", "unknown"})
_VERSION_LABEL = re.compile(r"v?\d+(?:\.\d+)*(?:[-+._][A-Za-z0-9]+)*")


class SupportBundleError(ValueError):
    """A support bundle could not be created without weakening its safety contract."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _size_category(size: int) -> str:
    if size == 0:
        return "empty"
    if size < 64 * 1024:
        return "small"
    if size < 1024 * 1024:
        return "medium"
    return "large"


def _safe_label(value: Any, *, fallback: str = "unknown") -> str:
    return value if isinstance(value, str) and _SAFE_LABEL.fullmatch(value) else fallback


def _known_value(value: Any, allowed: frozenset[str], *, fallback: str) -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def _safe_version(value: Any) -> str:
    return (
        value
        if isinstance(value, str) and _VERSION_LABEL.fullmatch(value)
        else "unknown"
    )


def _safe_digest(value: Any) -> str | None:
    return value if isinstance(value, str) and _SHA256.fullmatch(value) else None


def _safe_field_path(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = re.sub(r"\[\d+\]", "[]", value)
    root = normalized.split(".", 1)[0]
    if root not in _KNOWN_FIELD_ROOTS:
        return "other"
    if re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,160}", normalized) is None:
        return root
    return normalized


def _field_category(value: Any) -> str:
    path = _safe_field_path(value)
    return path.split(".", 1)[0] if path not in {"other", "unknown"} else path


def _installation_type() -> str:
    try:
        distribution = importlib.metadata.distribution("runeconomy")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            value = json.loads(direct_url)
            directory = value.get("dir_info") if isinstance(value, Mapping) else None
            if isinstance(directory, Mapping) and directory.get("editable") is True:
                return "editable"
            return "local_distribution"
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        return "unknown"
    return "installed_distribution"


def _safe_os_family() -> str:
    return {
        "darwin": "macos",
        "linux": "linux",
        "windows": "windows",
    }.get(platform.system().lower(), "other")


def _safe_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    return "other"


def _path_role(path: Path) -> dict[str, Any]:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return {"presence": "absent"}
    if stat.S_ISLNK(details.st_mode):
        return {"presence": "symlink"}
    if not stat.S_ISREG(details.st_mode):
        return {"presence": "unexpected_type"}
    return {
        "presence": "present",
        "size_category": _size_category(details.st_size),
        "sha256": _digest(path),
    }


def _load_json_role(path: Path, *, role: str) -> tuple[dict[str, Any] | None, str | None]:
    details = _path_role(path)
    if details["presence"] in {"symlink", "unexpected_type"}:
        raise SupportBundleError(
            f"{role} must be a regular file and must not be a symlink",
            error_code="runeconomy.support.unsafe_input",
        )
    if details["presence"] == "absent":
        return None, "absent"
    if path.stat().st_size > _MAX_JSON_BYTES:
        return None, "too_large"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "malformed"
    return (dict(value), None) if isinstance(value, Mapping) else (None, "not_object")


def _artifact_roles(workspace: Path) -> tuple[list[dict[str, Any]], int, int]:
    roles = []
    expected_names = set(_EXPECTED_WORKSPACE_ROLES.values())
    for role, filename in _EXPECTED_WORKSPACE_ROLES.items():
        roles.append({"role": role, **_path_role(workspace / filename)})
    unexpected = 0
    symlinks = 0
    try:
        entries = list(workspace.iterdir())
    except OSError as exc:
        raise SupportBundleError(
            "the workspace directory could not be inspected",
            error_code="runeconomy.support.unsafe_input",
        ) from exc
    for entry in entries:
        if entry.name in expected_names:
            continue
        unexpected += 1
        try:
            if entry.is_symlink():
                symlinks += 1
        except OSError:
            pass
    return roles, unexpected, symlinks


def _safe_error(code: str, stage: str) -> dict[str, str]:
    descriptor = descriptor_for_code(code)
    return {
        "code": descriptor.code,
        "category": descriptor.category,
        "stage": _safe_label(stage),
        "summary": descriptor.summary,
        "next_action_code": descriptor.next_action_code,
    }


def _workspace_section(workspace: Path) -> tuple[dict[str, Any], list[dict[str, str]], Any]:
    roles, unexpected_count, symlink_count = _artifact_roles(workspace)
    descriptor, descriptor_problem = _load_json_role(
        workspace / "workspace.json", role="workspace descriptor"
    )
    manifest, manifest_problem = _load_json_role(
        workspace / "manifest.json", role="experiment manifest"
    )
    errors: list[dict[str, str]] = []
    recognized = bool(
        descriptor
        and descriptor.get("artifact_type") == "runeconomy.experiment_workspace"
    )
    workspace_version = _safe_version(
        descriptor.get("workspace_version") if descriptor else None
    )
    if descriptor_problem or not recognized:
        status = "invalid"
        errors.append(_safe_error("runeconomy.workspace.invalid", "support.workspace"))
    elif workspace_version != "0.1":
        status = "unsupported_schema"
        errors.append(
            _safe_error("runeconomy.workspace.unsupported_schema", "support.workspace")
        )
    else:
        status = _known_value(
            descriptor.get("generation_status"),
            _WORKSPACE_STATUSES,
            fallback="invalid",
        )
        if status == "unresolved":
            errors.append(
                _safe_error("runeconomy.workspace.unresolved", "support.workspace")
            )
    if manifest_problem:
        errors.append(_safe_error("runeconomy.workspace.invalid", "support.manifest"))
        if status == "runnable":
            status = "invalid"
    unresolved_fields = []
    if descriptor and isinstance(descriptor.get("unresolved_fields"), list):
        for item in descriptor["unresolved_fields"]:
            if not isinstance(item, Mapping):
                continue
            path = _safe_field_path(item.get("path"))
            unresolved_fields.append(
                {
                    "field_path": path,
                    "category": _field_category(path),
                    "path_sha256": "sha256:"
                    + hashlib.sha256(str(item.get("path", "unknown")).encode()).hexdigest(),
                }
            )
    source_digest = None
    if descriptor and isinstance(descriptor.get("source_evidence"), Mapping):
        source_digest = _safe_digest(descriptor["source_evidence"].get("sha256"))
    fingerprints = {
        role["role"]: role.get("sha256")
        for role in roles
        if role.get("sha256") is not None
    }
    if source_digest:
        fingerprints["source_evidence"] = source_digest
    section = {
        "availability": "available" if recognized else "invalid",
        "workspace_schema_version": workspace_version,
        "mode": _known_value(
            descriptor.get("mode") if descriptor else None,
            _WORKSPACE_MODES,
            fallback="unknown",
        ),
        "status": status,
        "artifact_roles": roles,
        "fingerprints": fingerprints,
        "unresolved_fields": unresolved_fields,
        "unexpected_entries": {
            "count": unexpected_count,
            "symlink_count": symlink_count,
            "handling": "ignored",
        },
    }
    return section, errors, descriptor


def _import_capabilities(descriptor: Any) -> dict[str, Any]:
    assessment = (
        descriptor.get("capability_assessment")
        if isinstance(descriptor, Mapping)
        else None
    )
    capabilities = assessment.get("capabilities") if isinstance(assessment, Mapping) else None
    if not isinstance(capabilities, Mapping):
        return {"availability": "unavailable", "capabilities": {}}
    sanitized = {}
    for key in _CAPABILITY_KEYS:
        value = capabilities.get(key)
        status = value.get("status") if isinstance(value, Mapping) else "unknown"
        status = status if status in _CAPABILITY_STATUSES else "unknown"
        sanitized[key] = {
            "status": status,
            "next_action_code": f"runeconomy.action.capability.{key}.{status}",
        }
    return {
        "availability": "available",
        "assessment_version": _safe_version(assessment.get("assessment_version")),
        "runnable_experiment": assessment.get("runnable_experiment")
        if isinstance(assessment.get("runnable_experiment"), bool)
        else None,
        "recommended_mode": _known_value(
            assessment.get("recommended_mode"),
            _WORKSPACE_MODES,
            fallback="unknown",
        ),
        "capabilities": sanitized,
    }


def _saved_doctor(path: Path | None) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if path is None:
        return {"availability": "unavailable"}, []
    value, problem = _load_json_role(path, role="saved doctor report")
    if problem or value is None:
        return {"availability": "invalid"}, [
            _safe_error("runeconomy.backend.probe_failed", "support.doctor_report")
        ]
    if value.get("status") == "error":
        error = value.get("error")
        code = (
            error.get("code")
            if isinstance(error, Mapping)
            else "runeconomy.backend.probe_failed"
        )
        descriptor = descriptor_for_code(str(code))
        return {
            "availability": "available",
            "status": "error",
            "error_code": descriptor.code,
        }, [_safe_error(descriptor.code, "doctor")]
    capabilities = {}
    for key in _DOCTOR_CAPABILITY_KEYS:
        item = value.get(key)
        status = item.get("status") if isinstance(item, Mapping) else "unknown"
        capabilities[key] = {
            "status": status if status in _CAPABILITY_STATUSES else "unknown"
        }
    readiness = value.get("readiness")
    readiness_status = (
        readiness.get("status") if isinstance(readiness, Mapping) else "unknown"
    )
    return {
        "availability": "available",
        "schema_version": _safe_version(value.get("schema_version")),
        "backend": "vllm" if value.get("backend") == "vllm" else "unknown",
        "backend_version": _safe_version(value.get("backend_version")),
        "readiness_status": _known_value(
            readiness_status, _READINESS_STATUSES, fallback="unknown"
        ),
        "identity_presence": {
            key: bool(value.get(key))
            for key in (
                "model_revision",
                "tokenizer_identity",
                "chat_template_identity",
                "engine_configuration",
            )
        },
        "capabilities": capabilities,
    }, []


def _validation_issue_code(issue: Mapping[str, Any], status: str) -> str:
    code = issue.get("code")
    if isinstance(code, str) and descriptor_for_code(code).code == code:
        return code
    return validation_status_code(status) or "runeconomy.workspace.invalid"


def _saved_validation(
    path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if path is None:
        return {"availability": "unavailable"}, []
    value, problem = _load_json_role(path, role="saved validation report")
    if problem or value is None:
        return {"availability": "invalid"}, [
            _safe_error("runeconomy.workspace.invalid", "support.validation_report")
        ]
    status = _known_value(
        value.get("status"), _VALIDATION_STATUSES, fallback="invalid"
    )
    issues = []
    errors = []
    for raw in value.get("issues", []) if isinstance(value.get("issues"), list) else []:
        if not isinstance(raw, Mapping):
            continue
        code = _validation_issue_code(raw, status)
        issue = {
            "code": code,
            "field_path": _safe_field_path(raw.get("field_path")),
            "severity": "error" if raw.get("execution_blocking") is True else "warning",
            "claim_impact": "weakening" if raw.get("claim_weakening") is True else "none",
            "next_action_code": descriptor_for_code(code).next_action_code,
        }
        issues.append(issue)
        errors.append(_safe_error(code, "experiment.validate"))
    return {
        "availability": "available",
        "validation_version": _safe_version(value.get("validation_version")),
        "status": status,
        "runnable": value.get("runnable") if isinstance(value.get("runnable"), bool) else None,
        "mode": _known_value(
            value.get("mode"), _WORKSPACE_MODES, fallback="unknown"
        ),
        "issues": issues,
    }, errors


def _manifest() -> dict[str, Any]:
    return {
        "artifact_type": "runeconomy.support_bundle",
        "schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
        "error_code_version": ERROR_CODE_VERSION,
        "included_sections": list(SUPPORT_BUNDLE_FILES[1:]),
        "excluded_data_classes": list(_EXCLUDED_DATA_CLASSES),
        "privacy_notice": (
            "This archive is privacy-minimized, not guaranteed anonymous. Inspect every file before sharing. "
            "RunEconomy generated it locally and did not upload it."
        ),
        "determinism": {
            "creation_timestamp_included": False,
            "fixed_archive_metadata": True,
            "machine_dependent_fields": [
                "runeconomy_version",
                "python_version",
                "operating_system_family",
                "architecture",
                "installation_type",
                "input_fingerprints",
            ],
        },
    }


def _environment() -> dict[str, Any]:
    return {
        "runeconomy_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": _safe_label(platform.python_implementation().lower()),
        "operating_system_family": _safe_os_family(),
        "architecture": _safe_architecture(),
        "installation_type": _installation_type(),
    }


def _validate_archive(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = tuple(archive.namelist())
            if names != SUPPORT_BUNDLE_FILES:
                raise ValueError("archive members differ from the support allowlist")
            documents = {
                name: json.loads(archive.read(name).decode("utf-8")) for name in names
            }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile, ValueError) as exc:
        raise SupportBundleError(
            "the generated support archive failed structural validation",
            error_code="runeconomy.support.invalid_bundle",
        ) from exc
    manifest = documents["manifest.json"]
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SUPPORT_BUNDLE_SCHEMA_VERSION:
        raise SupportBundleError(
            "the generated support archive has an unsupported manifest",
            error_code="runeconomy.support.invalid_bundle",
        )
    serialized = json.dumps(documents, sort_keys=True)
    forbidden = (
        r"https?://",
        r"\bBearer\s+",
        r"Authorization\s*:",
        r"(?:^|[\"\s])/(?:Users|home|private|var/folders)/",
        r"[A-Za-z]:\\Users\\",
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    )
    if any(re.search(pattern, serialized, re.IGNORECASE) for pattern in forbidden):
        raise SupportBundleError(
            "the generated support archive contains a forbidden private-data pattern",
            error_code="runeconomy.support.invalid_bundle",
        )


def create_support_bundle(
    workspace: Path,
    output: Path,
    *,
    doctor_report: Path | None = None,
    validation_report: Path | None = None,
) -> dict[str, Any]:
    """Create and validate one local support archive without following symlinks."""

    if workspace.is_symlink() or not workspace.is_dir():
        raise SupportBundleError(
            "the support input must be a regular workspace directory, not a symlink",
            error_code="runeconomy.support.unsafe_input",
        )
    if output.suffix.lower() != ".zip":
        raise SupportBundleError(
            "the support output must use a .zip extension",
            error_code="runeconomy.support.unsafe_input",
        )
    if os.path.lexists(output):
        raise SupportBundleError(
            "the support output already exists; choose a new archive path",
            error_code="runeconomy.support.output_exists",
        )
    if not output.parent.is_dir():
        raise SupportBundleError(
            "the support output parent directory does not exist",
            error_code="runeconomy.support.unsafe_input",
        )
    temporary = output.with_name(output.name + ".tmp")
    if os.path.lexists(temporary):
        raise SupportBundleError(
            "a temporary support output already exists; inspect it before retrying",
            error_code="runeconomy.support.output_exists",
        )

    workspace_section, errors, descriptor = _workspace_section(workspace)
    doctor_section, doctor_errors = _saved_doctor(doctor_report)
    validation_section, validation_errors = _saved_validation(validation_report)
    errors.extend(doctor_errors)
    errors.extend(validation_errors)
    unique_errors = []
    seen = set()
    for error in errors:
        key = (error["code"], error["stage"])
        if key not in seen:
            seen.add(key)
            unique_errors.append(error)
    documents = {
        "manifest.json": _manifest(),
        "environment.json": _environment(),
        "workspace.json": workspace_section,
        "capabilities.json": {
            "import_evidence": _import_capabilities(descriptor),
            "saved_backend_assessment": doctor_section,
        },
        "validation.json": validation_section,
        "errors.json": {
            "error_code_version": ERROR_CODE_VERSION,
            "errors": unique_errors,
        },
    }
    try:
        with zipfile.ZipFile(
            temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in SUPPORT_BUNDLE_FILES:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100600 << 16
                archive.writestr(info, _json_bytes(documents[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        _validate_archive(temporary)
        temporary.replace(output)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {
        "schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
        "status": "created",
        "included_sections": list(SUPPORT_BUNDLE_FILES),
        "excluded_data_classes": list(_EXCLUDED_DATA_CLASSES),
        "error_count": len(unique_errors),
    }


def validate_support_bundle(path: Path) -> None:
    """Public validation hook used by tests and installed-product smoke checks."""

    _validate_archive(path)
