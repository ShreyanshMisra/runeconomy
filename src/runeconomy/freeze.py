"""Freeze an experiment workspace and write its preregistration record.

``runeconomy experiment freeze`` turns a reviewed design into a fixed
contract. It refuses designs the execution validator or the planner cannot
accept, then writes ``preregistration.json`` beside the manifest. The record
locks, by SHA-256:

* the manifest, workspace descriptor and intervention text;
* every agent-process input file (dataset and evaluator) and any extra file
  the operator names with ``--lock`` (agent code, runners, allocators);
* the acceptance gates and the execution design, both copied verbatim and
  digested canonically so a reader can see exactly what was preregistered;
* the planner's filled preregistration (claim, population, gates, method, n,
  tolerance, cost estimate, stopping rule) and the head of the patch
  provenance chain.

Anything that influences the run but cannot be digest-locked (for example a
Python evaluator resolved by import path, or the code an agent command
launches when it is not named with ``--lock``) is listed under
``unlocked_inputs`` rather than silently treated as frozen.

After freezing, ``experiment patch`` refuses the workspace, ``experiment
validate`` reports drift as invalid, and ``experiment run`` refuses to start
when any locked digest no longer matches.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .experiments import ExperimentValidationError
from .manifest_patch import (
    PREREGISTRATION_FILE,
    PROVENANCE_FILE,
    _canonical,
    append_provenance,
    digest_bytes,
    read_provenance,
)
from .workspaces import load_workspace


RECORD_ARTIFACT_TYPE = "runeconomy.preregistration_record"
RECORD_SCHEMA_VERSION = "0.1"
READY_STATE = "ready_to_freeze"
INCOMPLETE_STATES = frozenset({"needs_design_input", "ready_for_calibration"})
LOCKED_WORKSPACE_FILES = (
    ("manifest", "manifest.json"),
    ("workspace_descriptor", "workspace.json"),
    ("intervention", "intervention.md"),
)


class FreezeError(ExperimentValidationError):
    """A freeze request or a frozen workspace that fails its locks."""

    def __init__(
        self, message: str, *, error_code: str, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.public_details = {"reason": message, **dict(details or {})}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_digest(path: Path) -> str | None:
    try:
        return digest_bytes(path.read_bytes())
    except OSError:
        return None


def _resolve(workspace: Path, reference: str) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else workspace / path


def _reference(path: Path, workspace: Path) -> str:
    """Workspace-relative path (``..`` allowed) so a bundle verifies anywhere."""

    try:
        return os.path.relpath(path.resolve(), workspace.resolve())
    except ValueError:  # different drives on Windows
        return str(path.resolve())


def _executable_evaluators(spec: Any, location: str) -> list[dict[str, str]]:
    if not isinstance(spec, Mapping):
        return []
    kind = spec.get("kind")
    if kind == "python":
        return [
            {
                "field": f"{location}.callable",
                "reason": "a python evaluator is resolved by import path at run time; its code is not digest-locked",
            }
        ]
    if kind == "command":
        return [
            {
                "field": f"{location}.command",
                "reason": "a command evaluator runs local code that is not digest-locked",
            }
        ]
    from .evaluators import is_executable_evaluator

    if kind != "all" and is_executable_evaluator(spec):
        return [
            {
                "field": location,
                "reason": f"a {kind} evaluator runs local code or a network judge that is not digest-locked",
            }
        ]
    found: list[dict[str, str]] = []
    if kind == "all":
        for index, child in enumerate(spec.get("evaluators", []) or []):
            found.extend(_executable_evaluators(child, f"{location}.evaluators[{index}]"))
    return found


def _unlocked_inputs(
    manifest: Mapping[str, Any], mode: str, locked_paths: set[str]
) -> list[dict[str, str]]:
    unlocked: list[dict[str, str]] = []
    if mode == "agent_process":
        for variant in ("baseline", "candidate"):
            unlocked.append(
                {
                    "field": f"{variant}.command",
                    "reason": (
                        "the command string is locked, but the code it launches is locked only "
                        "for files named with --lock"
                        + (f" ({len(locked_paths)} file(s) locked)" if locked_paths else "")
                    ),
                }
            )
        evaluator = manifest.get("evaluator")
        if isinstance(evaluator, Mapping) and str(evaluator.get("path", "")).endswith(".py"):
            unlocked.append(
                {
                    "field": "evaluator.path",
                    "reason": "the evaluator file is digest-locked; modules it imports are not",
                }
            )
    else:
        unlocked.append(
            {
                "field": "backend",
                "reason": "the serving stack behind backend.base_url is not locked; identity is checked at run time",
            }
        )
        for index, case in enumerate(manifest.get("cases", []) or []):
            if isinstance(case, Mapping):
                unlocked.extend(_executable_evaluators(case.get("evaluator"), f"cases[{index}].evaluator"))
    return unlocked


def _read_record(workspace: Path) -> tuple[dict[str, Any], str] | None:
    path = workspace / PREREGISTRATION_FILE
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(
            f"{PREREGISTRATION_FILE} cannot be read: {exc}", error_code="runeconomy.freeze.drift"
        ) from exc
    if not isinstance(record, dict) or record.get("artifact_type") != RECORD_ARTIFACT_TYPE:
        raise FreezeError(
            f"{PREREGISTRATION_FILE} is not a RunEconomy preregistration record",
            error_code="runeconomy.freeze.drift",
        )
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise FreezeError(
            f"unsupported preregistration record schema_version {record.get('schema_version')!r}",
            error_code="runeconomy.workspace.unsupported_schema",
        )
    return record, digest_bytes(raw)


def verify_freeze(workspace: Path) -> dict[str, Any]:
    """Compare every locked digest with the workspace as it is now."""

    if not workspace.is_dir():
        return {"status": "not_frozen", "record_sha256": None, "mismatches": []}
    loaded = _read_record(workspace)
    if loaded is None:
        return {"status": "not_frozen", "record_sha256": None, "mismatches": []}
    record, record_digest = loaded
    locks = record.get("locks", {})
    mismatches: list[dict[str, Any]] = []

    def check(name: str, path: Path, expected: Any) -> None:
        observed = _file_digest(path)
        if observed != expected:
            mismatches.append({"lock": name, "expected": expected, "observed": observed})

    for name, _ in LOCKED_WORKSPACE_FILES:
        entry = locks.get(name, {})
        check(name, workspace / str(entry.get("file", "")), entry.get("sha256"))
    for name, entry in sorted((locks.get("inputs") or {}).items()):
        check(f"inputs.{name}", _resolve(workspace, str(entry.get("path", ""))), entry.get("sha256"))
    for index, entry in enumerate(locks.get("additional_files") or []):
        check(
            f"additional_files[{index}]",
            _resolve(workspace, str(entry.get("path", ""))),
            entry.get("sha256"),
        )
    try:
        _, manifest = load_workspace(workspace)
    except ExperimentValidationError:
        manifest = {}
    for name, field in (("gates", "acceptance"), ("execution", "execution")):
        entry = locks.get(name, {})
        observed = digest_bytes(_canonical(manifest.get(field)))
        if observed != entry.get("sha256"):
            mismatches.append({"lock": name, "expected": entry.get("sha256"), "observed": observed})
    return {
        "status": "drifted" if mismatches else "intact",
        "record_sha256": record_digest,
        "frozen_at": record.get("frozen_at"),
        "design_completeness": record.get("design_completeness"),
        "mismatches": mismatches,
    }


def require_intact_freeze(workspace: Path, *, require_frozen: bool = False) -> dict[str, Any]:
    """Raise if a frozen workspace drifted, or if freezing is required and absent."""

    state = verify_freeze(workspace)
    if state["status"] == "drifted":
        names = ", ".join(item["lock"] for item in state["mismatches"])
        raise FreezeError(
            f"frozen workspace {workspace} drifted from its preregistration record ({names}); "
            "restore the locked files or create and freeze a new workspace",
            error_code="runeconomy.freeze.drift",
            details={"mismatched_locks": [item["lock"] for item in state["mismatches"]]},
        )
    if require_frozen and state["status"] != "intact":
        raise FreezeError(
            f"{workspace} has no preregistration record; run `runeconomy experiment freeze {workspace}` first",
            error_code="runeconomy.freeze.required",
        )
    return state


def _summary(workspace: Path, record: Mapping[str, Any], record_digest: str, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "workspace": str(workspace),
        "mode": record["workspace"]["mode"],
        "record": PREREGISTRATION_FILE,
        "record_sha256": record_digest,
        "frozen_at": record["frozen_at"],
        "design_completeness": record["design_completeness"],
        "planner_state": record["plan"]["state"],
        "locks": {
            "manifest_sha256": record["locks"]["manifest"]["sha256"],
            "gates_sha256": record["locks"]["gates"]["sha256"],
            "execution_sha256": record["locks"]["execution"]["sha256"],
            "inputs": {
                name: entry["sha256"] for name, entry in sorted(record["locks"]["inputs"].items())
            },
            "additional_files": [deepcopy(item) for item in record["locks"]["additional_files"]],
        },
        "unknown_fields": list(record["preregistration"].get("unknown_fields", [])),
        "unlocked_inputs": deepcopy(record["unlocked_inputs"]),
        "next_command": f"runeconomy experiment run {workspace} --output results/ --require-frozen",
    }


def freeze_workspace(
    workspace: Path,
    *,
    lock_files: list[Path] | None = None,
    allow_incomplete_design: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Validate, plan, lock, and write the preregistration record."""

    from .planning import plan_experiment
    from .workspaces import validate_experiment_artifact

    if not workspace.is_dir():
        raise FreezeError(
            f"{workspace} is not an experiment workspace directory",
            error_code="runeconomy.workspace.invalid",
        )
    existing = _read_record(workspace)
    if existing is not None:
        state = require_intact_freeze(workspace)
        record, record_digest = existing
        return _summary(workspace, record, record_digest, "already_frozen") | {
            "freeze_verification": state["status"]
        }

    validation = validate_experiment_artifact(workspace)
    if not validation.get("runnable"):
        codes = ", ".join(sorted({str(issue.get("code")) for issue in validation.get("issues", [])}))
        raise FreezeError(
            f"cannot freeze a workspace whose validation status is {validation.get('status')} ({codes}); "
            f"run `runeconomy experiment validate {workspace}`",
            error_code=(
                "runeconomy.workspace.unresolved"
                if validation.get("status") == "unresolved"
                else "runeconomy.freeze.not_ready"
            ),
            details={"validation_status": validation.get("status"), "issue_codes": codes.split(", ") if codes else []},
        )
    plan = plan_experiment(workspace)
    planner_state = str(plan.get("state"))
    if planner_state == "blocked" or (
        planner_state != READY_STATE
        and not (allow_incomplete_design and planner_state in INCOMPLETE_STATES)
    ):
        codes = sorted({str(issue.get("code")) for issue in plan.get("issues", [])})
        raise FreezeError(
            f"the planner state is {planner_state}, not {READY_STATE}"
            + (f" (issues: {', '.join(codes)})" if codes else "")
            + f"; review `runeconomy experiment plan {workspace}`, patch the design, "
            + (
                "and retry"
                if planner_state == "blocked"
                else "or pass --allow-incomplete-design to record an incomplete preregistration"
            ),
            error_code="runeconomy.freeze.not_ready",
            details={"planner_state": planner_state, "issue_codes": codes},
        )

    descriptor, manifest = load_workspace(workspace)
    mode = str(descriptor.get("mode"))
    locks: dict[str, Any] = {}
    for name, filename in LOCKED_WORKSPACE_FILES:
        digest = _file_digest(workspace / filename)
        if digest is None:
            raise FreezeError(
                f"cannot read {filename} to lock it", error_code="runeconomy.workspace.invalid"
            )
        locks[name] = {"file": filename, "sha256": digest}
    inputs: dict[str, Any] = {}
    if mode == "agent_process":
        for label in ("dataset", "evaluator"):
            reference = manifest[label]
            observed = _file_digest(_resolve(workspace, reference["path"]))
            if observed != reference.get("sha256"):
                raise FreezeError(
                    f"{label}.sha256 does not match {reference['path']}",
                    error_code="runeconomy.workspace.invalid",
                )
            inputs[label] = {"path": reference["path"], "sha256": observed}
    locks["inputs"] = inputs
    additional: list[dict[str, str]] = []
    for path in lock_files or []:
        digest = _file_digest(path)
        if digest is None or not path.is_file():
            raise FreezeError(f"--lock file does not exist: {path}", error_code="runeconomy.workspace.invalid")
        additional.append({"path": _reference(path, workspace), "sha256": digest})
    locks["additional_files"] = sorted(additional, key=lambda item: item["path"])
    for name, field in (("gates", "acceptance"), ("execution", "execution")):
        value = deepcopy(manifest.get(field))
        locks[name] = {"field": field, "value": value, "sha256": digest_bytes(_canonical(value))}

    history = read_provenance(workspace)
    preregistration = deepcopy(plan.get("preregistration") or {})
    preregistration.pop("status", None)
    preregistration.pop("freeze_note", None)
    preregistration.pop("artifact_type", None)
    record = {
        "artifact_type": RECORD_ARTIFACT_TYPE,
        "schema_version": RECORD_SCHEMA_VERSION,
        "status": "frozen",
        "frozen_at": _now(),
        "runeconomy_version": __version__,
        "heldout_executed": False,
        "reason": reason,
        "workspace": {
            "mode": mode,
            "experiment_id": manifest.get("experiment_id", workspace.name),
            "manifest_schema_version": manifest.get("schema_version"),
        },
        "design_completeness": "complete" if planner_state == READY_STATE else "incomplete",
        "locks": locks,
        "unlocked_inputs": _unlocked_inputs(manifest, mode, {item["path"] for item in additional}),
        "preregistration": preregistration,
        "plan": {
            "schema_version": plan.get("schema_version"),
            "state": planner_state,
            "sha256": digest_bytes(_canonical(plan)),
            "issue_codes": sorted({str(issue.get("code")) for issue in plan.get("issues", [])}),
            "next_action_code": plan.get("next_action_code"),
        },
        "provenance": {
            "file": PROVENANCE_FILE,
            "entries_before_freeze": len(history),
            "head_entry_sha256": history[-1][1] if history else None,
        },
        "rules": {
            "patch_after_freeze": "refused",
            "run_on_drift": "refused",
            "change_requires": "a new workspace and a new freeze",
        },
    }
    content = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = workspace / PREREGISTRATION_FILE
    try:
        descriptor_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FreezeError(
            f"{PREREGISTRATION_FILE} appeared while freezing; inspect it before retrying",
            error_code="runeconomy.freeze.drift",
        ) from exc
    with os.fdopen(descriptor_fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    record_digest = digest_bytes(content)
    append_provenance(
        workspace,
        {
            "event": "freeze",
            "reason": reason,
            "record_sha256": record_digest,
            "manifest_sha256_before": locks["manifest"]["sha256"],
            "manifest_sha256_after": locks["manifest"]["sha256"],
        },
    )
    return _summary(workspace, record, record_digest, "frozen") | {"freeze_verification": "intact"}
