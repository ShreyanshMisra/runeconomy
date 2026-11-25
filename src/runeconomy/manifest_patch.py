"""Schema-validated manifest patches with an append-only provenance log.

``runeconomy experiment patch`` lets an operator or coding agent change a
workspace manifest without hand-editing JSON. Every patch is applied to an
in-memory copy, validated with the same validator that ``experiment
validate`` and ``experiment run`` use, and written only when the result is
valid. Each accepted patch appends one entry to ``provenance.jsonl`` that
records the operations, the previous and new values, the manifest digests
before and after, and the digest of the previous entry, so the history is a
tamper-evident chain.

Patches never resolve an unresolved workspace (that requires evidence-backed
re-initialization) and never touch a frozen workspace: once ``experiment
freeze`` has written a preregistration record, the contract is fixed and a
change needs a new workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .experiments import ExperimentManifest, ExperimentValidationError
from .workspaces import load_workspace, validate_agent_manifest


PROVENANCE_FILE = "provenance.jsonl"
PROVENANCE_SCHEMA_VERSION = "0.1"
PATCH_FILE_VERSION = "0.1"
PREREGISTRATION_FILE = "preregistration.json"

# Fields that identify what kind of artifact the manifest is. Changing them
# would silently turn one contract into another, so they are never patchable.
PROTECTED_PATHS = frozenset({("artifact_type",), ("mode",)})
# Agent-process file references whose digests are re-pinned automatically
# when the path changes, or on request with ``--repin``.
DIGEST_REFERENCES = ("dataset", "evaluator")

_SEGMENT = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


class ManifestPatchError(ExperimentValidationError):
    """A patch request that cannot be applied safely."""

    def __init__(self, message: str, *, error_code: str = "runeconomy.patch.invalid") -> None:
        super().__init__(message)
        self.error_code = error_code
        # Composed by RunEconomy from field paths and validator text; it never
        # contains manifest values, so the CLI may show it as error.details.
        self.public_details = {"reason": message}


@dataclass(frozen=True)
class PatchOperation:
    op: str
    path: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.op, "path": self.path}
        if self.op == "set":
            payload["value"] = deepcopy(self.value)
        return payload


def parse_path(path: str) -> tuple[str | int, ...]:
    """Parse ``a.b[0].c``, ``a.b.0.c`` or a JSON Pointer ``/a/b/0/c``."""

    if not isinstance(path, str) or not path.strip():
        raise ManifestPatchError("patch path must be a non-empty string")
    if path.startswith("/"):
        segments: list[str | int] = []
        for raw in path[1:].split("/"):
            token = raw.replace("~1", "/").replace("~0", "~")
            if token == "":
                raise ManifestPatchError(f"patch path {path!r} has an empty segment")
            segments.append(int(token) if token.isdigit() else token)
        return tuple(segments)
    segments = []
    for part in path.split("."):
        if part == "":
            raise ManifestPatchError(f"patch path {path!r} has an empty segment")
        position = 0
        indexed = False
        for match in _SEGMENT.finditer(part):
            name, index = match.groups()
            if match.start() != position or (index is None and indexed):
                raise ManifestPatchError(f"patch path {path!r} is malformed")
            indexed = indexed or index is not None
            if index is not None:
                segments.append(int(index))
            elif name.isdigit():
                segments.append(int(name))
            else:
                segments.append(name)
            position = match.end()
        if position != len(part):
            raise ManifestPatchError(f"patch path {path!r} is malformed")
    return tuple(segments)


def _display(segments: tuple[str | int, ...]) -> str:
    text = ""
    for segment in segments:
        if isinstance(segment, int):
            text += f"[{segment}]"
        else:
            text += ("." if text else "") + segment
    return text


_MISSING = object()


def _lookup(document: Any, segments: tuple[str | int, ...]) -> Any:
    current = document
    for segment in segments:
        if isinstance(segment, int) and isinstance(current, list):
            if segment >= len(current):
                return _MISSING
            current = current[segment]
        elif isinstance(segment, str) and isinstance(current, dict):
            if segment not in current:
                return _MISSING
            current = current[segment]
        else:
            return _MISSING
    return current


def _parent(document: Any, segments: tuple[str | int, ...], *, create: bool) -> Any:
    current = document
    for depth, segment in enumerate(segments[:-1]):
        following = segments[depth + 1]
        if isinstance(segment, int):
            if not isinstance(current, list) or segment >= len(current):
                raise ManifestPatchError(
                    f"{_display(segments[: depth + 1])} is not an existing array element"
                )
            current = current[segment]
            continue
        if not isinstance(current, dict):
            raise ManifestPatchError(f"{_display(segments[:depth])} is not an object")
        if segment not in current or current[segment] is None:
            if not create or isinstance(following, int):
                raise ManifestPatchError(f"{_display(segments[: depth + 1])} does not exist")
            current[segment] = {}
        current = current[segment]
    return current


def _apply(document: dict[str, Any], operation: PatchOperation) -> dict[str, Any]:
    segments = parse_path(operation.path)
    if segments in PROTECTED_PATHS:
        raise ManifestPatchError(
            f"{operation.path} identifies the artifact kind and cannot be patched; "
            "create a new workspace with `runeconomy experiment init` instead"
        )
    previous = _lookup(document, segments)
    parent = _parent(document, segments, create=operation.op == "set")
    leaf = segments[-1]
    if operation.op == "set":
        if isinstance(leaf, int):
            if not isinstance(parent, list) or leaf > len(parent):
                raise ManifestPatchError(
                    f"{operation.path} must index an existing array element or append at its end"
                )
            if leaf == len(parent):
                parent.append(deepcopy(operation.value))
            else:
                parent[leaf] = deepcopy(operation.value)
        else:
            if not isinstance(parent, dict):
                raise ManifestPatchError(f"the parent of {operation.path} is not an object")
            parent[leaf] = deepcopy(operation.value)
    elif operation.op == "unset":
        if previous is _MISSING:
            raise ManifestPatchError(f"{operation.path} does not exist and cannot be unset")
        del parent[leaf]
    else:
        raise ManifestPatchError(f"unsupported patch op {operation.op!r}; use set or unset")
    record: dict[str, Any] = {
        "op": operation.op,
        "path": _display(segments),
        "previous_present": previous is not _MISSING,
        "source": "operator",
    }
    if previous is not _MISSING:
        record["previous"] = deepcopy(previous)
    if operation.op == "set":
        record["value"] = deepcopy(operation.value)
    return record


def load_patch_file(path: Path) -> list[PatchOperation]:
    """Read a patch file: ``{"patch_version": "0.1", "operations": [...]}``."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestPatchError(f"cannot read patch file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestPatchError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if isinstance(payload, list):
        operations = payload
    elif isinstance(payload, Mapping):
        version = payload.get("patch_version", PATCH_FILE_VERSION)
        if version != PATCH_FILE_VERSION:
            raise ManifestPatchError(
                f"unsupported patch_version {version!r}; expected {PATCH_FILE_VERSION!r}"
            )
        operations = payload.get("operations")
    else:
        operations = None
    if not isinstance(operations, list):
        raise ManifestPatchError("patch file must contain an operations array")
    parsed = []
    for index, item in enumerate(operations):
        if not isinstance(item, Mapping):
            raise ManifestPatchError(f"operations[{index}] must be an object")
        op = item.get("op")
        if op in {"replace", "add"}:
            op = "set"
        elif op == "remove":
            op = "unset"
        if op not in {"set", "unset"}:
            raise ManifestPatchError(f"operations[{index}].op must be set or unset")
        if not isinstance(item.get("path"), str):
            raise ManifestPatchError(f"operations[{index}].path must be a string")
        if op == "set" and "value" not in item:
            raise ManifestPatchError(f"operations[{index}] set requires a value")
        parsed.append(PatchOperation(op, item["path"], deepcopy(item.get("value"))))
    return parsed


def parse_assignment(text: str, *, raw_string: bool = False) -> PatchOperation:
    """Parse a CLI ``PATH=VALUE`` assignment. VALUE is JSON unless ``raw_string``."""

    path, separator, value = text.partition("=")
    if not separator or not path:
        raise ManifestPatchError(f"expected PATH=VALUE, got {text!r}")
    if raw_string:
        return PatchOperation("set", path, value)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ManifestPatchError(
            f"value for {path} is not valid JSON ({exc.msg}); quote strings as JSON "
            "or use --set-string PATH=TEXT"
        ) from exc
    return PatchOperation("set", path, decoded)


def digest_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ManifestPatchError(f"cannot write {path}: {exc}") from exc


def read_provenance(workspace: Path) -> list[tuple[dict[str, Any], str]]:
    """Return ``(entry, entry_sha256)`` pairs and verify the hash chain."""

    path = workspace / PROVENANCE_FILE
    if not path.exists():
        return []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise ManifestPatchError(f"cannot read {path}: {exc}") from exc
    entries: list[tuple[dict[str, Any], str]] = []
    previous: str | None = None
    for number, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestPatchError(
                f"{PROVENANCE_FILE}:{number} is not valid JSON", error_code="runeconomy.provenance.broken"
            ) from exc
        if not isinstance(entry, dict) or entry.get("previous_entry_sha256") != previous:
            raise ManifestPatchError(
                f"{PROVENANCE_FILE}:{number} breaks the provenance hash chain",
                error_code="runeconomy.provenance.broken",
            )
        previous = digest_bytes(line)
        entries.append((entry, previous))
    return entries


def append_provenance(workspace: Path, entry: Mapping[str, Any]) -> str:
    """Append one chained entry and return its digest."""

    history = read_provenance(workspace)
    payload = {
        "artifact_type": "runeconomy.workspace_provenance_entry",
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "sequence": len(history) + 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "runeconomy_version": __version__,
        **dict(entry),
        "previous_entry_sha256": history[-1][1] if history else None,
    }
    line = _canonical(payload)
    path = workspace / PROVENANCE_FILE
    try:
        with path.open("ab") as handle:
            handle.write(line + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ManifestPatchError(f"cannot append to {path}: {exc}") from exc
    return digest_bytes(line)


def _validate(manifest: dict[str, Any], mode: str, workspace: Path) -> None:
    if mode == "agent_process":
        validate_agent_manifest(manifest, workspace)
    elif mode == "static_request":
        ExperimentManifest.from_dict(manifest)
    else:
        raise ManifestPatchError(f"unsupported workspace mode {mode!r}")


def _repin(manifest: dict[str, Any], workspace: Path, label: str) -> dict[str, Any] | None:
    reference = manifest.get(label)
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        return None
    path = Path(reference["path"])
    if not path.is_absolute():
        path = workspace / path
    try:
        digest = digest_bytes(path.read_bytes())
    except OSError as exc:
        raise ManifestPatchError(f"cannot read {label}.path {reference['path']}: {exc}") from exc
    previous = reference.get("sha256")
    if previous == digest:
        return None
    reference["sha256"] = digest
    return {
        "op": "set",
        "path": f"{label}.sha256",
        "previous_present": previous is not None,
        **({"previous": previous} if previous is not None else {}),
        "value": digest,
        "source": "derived_digest",
    }


def patch_workspace(
    workspace: Path,
    operations: list[PatchOperation],
    *,
    reason: str | None = None,
    repin: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply, validate, write, and record one patch. Returns the CLI payload."""

    if not workspace.is_dir():
        raise ManifestPatchError(
            f"{workspace} is not an experiment workspace directory; "
            "experiment patch edits workspaces created by `runeconomy experiment init`"
        )
    if not operations and not repin:
        raise ManifestPatchError("no patch operations were supplied")
    if (workspace / PREREGISTRATION_FILE).exists():
        raise ManifestPatchError(
            f"{workspace} is frozen ({PREREGISTRATION_FILE} exists); a frozen contract "
            "cannot change. Create a new workspace for a new design.",
            error_code="runeconomy.workspace.frozen",
        )
    descriptor, manifest = load_workspace(workspace)
    if descriptor.get("generation_status") != "runnable":
        raise ManifestPatchError(
            f"{workspace} is unresolved; supply the missing evidence-backed values in the "
            "source evidence and run `runeconomy experiment init` again. Patching cannot "
            "make an unresolved workspace runnable.",
            error_code="runeconomy.workspace.unresolved",
        )
    mode = str(descriptor.get("mode"))
    manifest_path = workspace / "manifest.json"
    before_bytes = manifest_path.read_bytes()
    before_digest = digest_bytes(before_bytes)
    history = read_provenance(workspace)
    untracked = bool(history) and history[-1][0].get("manifest_sha256_after") != before_digest

    patched = deepcopy(manifest)
    changes = [_apply(patched, operation) for operation in operations]
    if mode == "agent_process":
        touched = {change["path"] for change in changes}
        for label in DIGEST_REFERENCES:
            if repin or f"{label}.path" in touched or label in touched:
                derived = _repin(patched, workspace, label)
                if derived is not None:
                    changes.append(derived)
    elif repin:
        raise ManifestPatchError("--repin applies only to agent-process dataset and evaluator files")
    try:
        _validate(patched, mode, workspace)
    except ManifestPatchError:
        raise
    except ExperimentValidationError as exc:
        raise ManifestPatchError(
            f"patched manifest is invalid and was not written: {exc}"
        ) from exc

    after_bytes = _manifest_bytes(patched)
    after_digest = digest_bytes(after_bytes)
    changed = after_digest != before_digest
    entry_digest: str | None = None
    if changed and not dry_run:
        _atomic_write(manifest_path, after_bytes)
        entry_digest = append_provenance(
            workspace,
            {
                "event": "patch",
                "reason": reason,
                "operations": changes,
                "manifest_sha256_before": before_digest,
                "manifest_sha256_after": after_digest,
                "untracked_change_before": untracked,
            },
        )
    return {
        "status": "dry_run" if dry_run else "patched" if changed else "unchanged",
        "workspace": str(workspace),
        "mode": mode,
        "written": changed and not dry_run,
        "operations": changes,
        "manifest_sha256_before": before_digest,
        "manifest_sha256_after": after_digest,
        "untracked_change_before": untracked,
        "provenance": {
            "file": PROVENANCE_FILE,
            "entry_sha256": entry_digest,
            "entries": len(history) + (1 if entry_digest else 0),
        },
        "next_command": f"runeconomy experiment validate {workspace}",
    }
