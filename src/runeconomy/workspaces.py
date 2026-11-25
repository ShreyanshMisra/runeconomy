"""Experiment workspaces shared by static and agent-process comparisons."""

from __future__ import annotations

import os

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .completion_policy import CompletionPolicy, CompletionPolicyError
from .cost_model import CostModelError, parse_cost_model
from .identity import IdentityPolicyError, parse_identity_policy
from .experiment_init import initialize_experiment, select_finding
from .errors import classify_exception
from .execution import (
    CONCURRENT_ARRIVAL_PATTERNS,
    SEQUENTIAL,
    LoadSpec,
    LoadSpecError,
)
from .experiments import ExperimentManifest, ExperimentValidationError
from .metric_gates import MetricGateError, parse_metric_gates
from .paired_inference import PAIRED_EXACT, PAIRED_METHODS


WORKSPACE_VERSION = "0.1"
AGENT_MANIFEST_VERSION = "0.2"
# Schema 0.3 adds the preregistered task_success_method and completion_policy
# acceptance fields. Earlier versions reject them so the new semantics can
# only be declared prospectively, never applied to an existing contract.
AGENT_MANIFEST_PREREGISTERED_SEMANTICS_VERSION = "0.3"
# Schema 0.4 adds concurrent arrival patterns (execution.arrival_pattern and
# execution.load) and preregistered acceptance.metric_gates, on top of 0.3.
AGENT_MANIFEST_LOAD_VERSION = "0.4"
SUPPORTED_AGENT_MANIFEST_VERSIONS = {"0.1", "0.2", "0.3", "0.4"}


def _digest_path(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ExperimentValidationError(f"cannot read {path}: {exc}") from exc
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_workspace_files(
    output: Path,
    *,
    descriptor: dict[str, Any],
    manifest: Mapping[str, Any],
    intervention: str,
) -> dict[str, Any]:
    if output.exists():
        raise ExperimentValidationError(
            f"workspace output {output} already exists; choose a new directory"
        )
    output.mkdir(parents=True)
    _write_json(output / "manifest.json", manifest)
    (output / "intervention.md").write_text(intervention.rstrip() + "\n", encoding="utf-8")
    readme = "\n".join(
        [
            "# RunEconomy experiment workspace",
            "",
            f"Mode: `{descriptor['mode']}`",
            f"Status: `{descriptor['generation_status']}`",
            "",
            "Raw source evidence is not copied into this workspace. The source name and digest are recorded in `workspace.json`.",
            "",
            "## Next command",
            "",
            "```bash",
            descriptor["next_command"],
            "```",
        ]
    )
    (output / "README.md").write_text(readme + "\n", encoding="utf-8")
    _write_json(output / "workspace.json", descriptor)
    return descriptor


def create_workspace_from_evidence(
    evidence: Mapping[str, Any],
    *,
    evidence_path: Path,
    output: Path,
    finding_id: str | None = None,
) -> dict[str, Any]:
    analysis = evidence.get("analysis")
    finding_source: Any = analysis if isinstance(analysis, Mapping) else evidence
    finding = select_finding(finding_source, finding_id)
    finding_grouping = finding.get("capture_workflow_grouping")
    if isinstance(finding_grouping, Mapping) and finding_grouping.get(
        "ambiguous"
    ) is True:
        raise ExperimentValidationError(
            "cannot create a runnable workspace from an ambiguous capture workflow: "
            "overlapping in-flight requests lacked X-RunEconomy-Workflow-Id or W3C "
            "traceparent correlation; recapture with a unique workflow or trace ID"
        )
    selected_id = finding.get("finding_id")
    candidate_runs = evidence.get("runs")
    runs = (
        [item for item in candidate_runs if isinstance(item, Mapping)]
        if isinstance(candidate_runs, list)
        else [evidence]
    )
    for run in runs:
        run_analysis = run.get("analysis")
        run_findings = (
            run_analysis.get("findings", [])
            if isinstance(run_analysis, Mapping)
            else run.get("findings", [])
        )
        contains_selected = any(
            isinstance(item, Mapping) and item.get("finding_id") == selected_id
            for item in run_findings
        )
        grouping = run.get("trace", {}).get("workflow_grouping", {})
        if contains_selected and isinstance(grouping, Mapping) and grouping.get(
            "ambiguous"
        ) is True:
            raise ExperimentValidationError(
                "cannot create a runnable workspace from an ambiguous capture workflow: "
                "overlapping in-flight requests lacked X-RunEconomy-Workflow-Id or W3C "
                "traceparent correlation; recapture with a unique workflow or trace ID"
            )
    artifact = initialize_experiment(finding)
    unresolved = deepcopy(artifact.get("unresolved_fields", []))
    runnable = artifact.get("generation_status") != "unresolved"
    source = evidence.get("source", {})
    source_digest = source.get("sha256")
    if not isinstance(source_digest, str):
        source_digest = _digest_path(evidence_path)
    descriptor = {
        "artifact_type": "runeconomy.experiment_workspace",
        "workspace_version": WORKSPACE_VERSION,
        "generation_status": "runnable" if runnable else "unresolved",
        "mode": "static_request",
        "source_evidence": {
            "name": evidence_path.name,
            "sha256": source_digest,
            "kind": source.get("kind", "unknown"),
        },
        "selected_finding": {
            key: deepcopy(finding.get(key))
            for key in (
                "finding_id",
                "rule_id",
                "title",
                "optimization_target",
                "experiment_template_id",
                "causal_mechanism",
                "predicted_metric",
                "predicted_direction",
            )
        },
        "intervention": {
            "template_id": finding.get("experiment_template_id"),
            "summary": finding.get("suggested_experiment"),
            "control": "Baseline and candidate differ only by the declared intervention.",
        },
        "files": {
            "manifest": "manifest.json",
            "intervention": "intervention.md",
            "readme": "README.md",
        },
        "capability_assessment": deepcopy(evidence.get("capability_assessment")),
        "unresolved_fields": unresolved,
        "next_command": f"runeconomy experiment validate {output}",
    }
    intervention = "\n".join(
        [
            "# Intervention",
            "",
            str(finding.get("title") or "Imported finding"),
            "",
            str(finding.get("suggested_experiment") or "No intervention description was supplied."),
            "",
            "## Expected mechanism",
            "",
            str(finding.get("causal_mechanism") or "Unknown"),
            "",
            "## Predicted direction",
            "",
            f"{finding.get('predicted_metric') or 'Unknown'}: {finding.get('predicted_direction') or 'unknown'}",
        ]
    )
    return _write_workspace_files(
        output,
        descriptor=descriptor,
        manifest=artifact,
        intervention=intervention,
    )


def create_workspace_from_finding(
    finding_source: Mapping[str, Any],
    *,
    source_path: Path,
    output: Path,
    finding_id: str | None = None,
) -> dict[str, Any]:
    evidence = {
        "artifact_type": "runeconomy.evidence_bundle",
        "schema_version": "0.1",
        "source": {
            "kind": "structured_finding",
            "name": source_path.name,
            "sha256": _digest_path(source_path),
        },
        "findings": finding_source.get("findings", [finding_source]),
        "capability_assessment": None,
    }
    return create_workspace_from_evidence(
        evidence,
        evidence_path=source_path,
        output=output,
        finding_id=finding_id,
    )


def _relative_reference(path: Path, workspace: Path) -> str:
    """Reference an input relative to the workspace, even outside it.

    A relative reference keeps the workspace portable and keeps the local
    home directory out of files people share or freeze. Only a path on a
    different drive (Windows) has no relative form and stays absolute.
    """

    try:
        return os.path.relpath(path.resolve(), workspace.resolve())
    except ValueError:
        return str(path.resolve())


def create_agent_workspace(
    *,
    output: Path,
    baseline_command: str,
    candidate_command: str,
    dataset_path: Path,
    evaluator_path: Path,
    repetitions: int = 1,
    order: str = "randomized",
    seed: int = 0,
    timeout_seconds: float = 300.0,
    evaluator_timeout_seconds: float = 30.0,
    max_task_success_regression: float = 0.05,
    min_task_success_rate: float = 0.8,
    min_completed_rate: float = 1.0,
    optimization_target: str = "successful_workflow_latency",
    latency_semantics: str = "superiority",
    max_latency_regression_ms: float = 0.0,
    cost_ceiling_usd: float | None = None,
    task_success_method: str | None = None,
    completion_policy: Mapping[str, Any] | None = None,
    identity_policy: Mapping[str, Any] | None = None,
    arrival_pattern: str | None = None,
    load: Mapping[str, Any] | None = None,
    metric_gates: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    for path, label in ((dataset_path, "dataset"), (evaluator_path, "evaluator")):
        if not path.is_file():
            raise ExperimentValidationError(f"{label} file does not exist: {path}")
    manifest = {
        "artifact_type": "runeconomy.agent_process_experiment",
        "schema_version": AGENT_MANIFEST_VERSION,
        "mode": "agent_process",
        "baseline": {"command": baseline_command},
        "candidate": {"command": candidate_command},
        "dataset": {
            "path": _relative_reference(dataset_path, output),
            "sha256": _digest_path(dataset_path),
        },
        "evaluator": {
            "path": _relative_reference(evaluator_path, output),
            "sha256": _digest_path(evaluator_path),
        },
        "execution": {
            "repetitions": repetitions,
            "order": order,
            "seed": seed,
            "timeout_seconds": timeout_seconds,
            "evaluator_timeout_seconds": evaluator_timeout_seconds,
        },
        "acceptance": {
            "optimization_target": optimization_target,
            "latency_semantics": latency_semantics,
            "max_latency_regression_ms": max_latency_regression_ms,
            "max_task_success_regression": max_task_success_regression,
            "min_task_success_rate": min_task_success_rate,
            "min_completed_rate": min_completed_rate,
        },
        "cost_ceiling_usd": cost_ceiling_usd,
    }
    if (
        task_success_method is not None
        or completion_policy is not None
        or identity_policy is not None
    ):
        manifest["schema_version"] = AGENT_MANIFEST_PREREGISTERED_SEMANTICS_VERSION
        if task_success_method is not None:
            manifest["acceptance"]["task_success_method"] = task_success_method
        if completion_policy is not None:
            manifest["acceptance"]["completion_policy"] = dict(completion_policy)
        if identity_policy is not None:
            manifest["acceptance"]["identity_policy"] = dict(identity_policy)
    if (
        (arrival_pattern not in (None, SEQUENTIAL))
        or load is not None
        or metric_gates is not None
    ):
        manifest["schema_version"] = AGENT_MANIFEST_LOAD_VERSION
        if arrival_pattern is not None:
            manifest["execution"]["arrival_pattern"] = arrival_pattern
        if load is not None:
            manifest["execution"]["load"] = dict(load)
        if metric_gates is not None:
            manifest["acceptance"]["metric_gates"] = [dict(gate) for gate in metric_gates]
    validate_agent_manifest(manifest, output)
    descriptor = {
        "artifact_type": "runeconomy.experiment_workspace",
        "workspace_version": WORKSPACE_VERSION,
        "generation_status": "runnable",
        "mode": "agent_process",
        "source_evidence": None,
        "selected_finding": None,
        "intervention": {
            "template_id": (
                "engine.change_latency"
                if optimization_target == "engine_change_latency"
                else "agent.process_change"
            ),
            "summary": (
                "Compare successful-workflow latency across distinct serving backends."
                if optimization_target == "engine_change_latency"
                else "Compare complete baseline and candidate agent processes."
            ),
            "control": "Both variants receive the same task-level dataset and evaluator.",
        },
        "files": {
            "manifest": "manifest.json",
            "intervention": "intervention.md",
            "readme": "README.md",
        },
        "unresolved_fields": [],
        "next_command": f"runeconomy experiment validate {output}",
    }
    return _write_workspace_files(
        output,
        descriptor=descriptor,
        manifest=manifest,
        intervention=(
            "# Intervention\n\nCompare the declared baseline and candidate agent commands.\n\n"
            "The complete process, tool, retrieval, inference, status, and trace evidence remains variant-local."
        ),
    )


def load_workspace(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor_path = path / "workspace.json"
    manifest_path = path / "manifest.json"
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentValidationError(f"cannot read workspace {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExperimentValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid workspace JSON: {exc.msg}"
        ) from exc
    if not isinstance(descriptor, dict) or descriptor.get("artifact_type") != "runeconomy.experiment_workspace":
        raise ExperimentValidationError(f"{descriptor_path} is not a RunEconomy workspace descriptor")
    if descriptor.get("workspace_version") != WORKSPACE_VERSION:
        raise ExperimentValidationError(
            f"unsupported workspace_version {descriptor.get('workspace_version')!r}"
        )
    if not isinstance(manifest, dict):
        raise ExperimentValidationError(f"{manifest_path} must contain a JSON object")
    return descriptor, manifest


def _resolve_reference(workspace: Path, value: str) -> Path:
    """Resolve a workspace reference, including ``..`` before the directory exists.

    References are written relative to the resolved workspace, so joining
    them to the resolved workspace and normalizing lexically is exact.
    """

    path = Path(value)
    if path.is_absolute():
        return path
    return Path(os.path.normpath(workspace.resolve() / path))


def validate_agent_manifest(manifest: Mapping[str, Any], workspace: Path) -> None:
    if manifest.get("artifact_type") != "runeconomy.agent_process_experiment":
        raise ExperimentValidationError("agent manifest artifact_type is invalid")
    if manifest.get("schema_version") not in SUPPORTED_AGENT_MANIFEST_VERSIONS:
        raise ExperimentValidationError("unsupported agent-process manifest version")
    for variant in ("baseline", "candidate"):
        value = manifest.get(variant)
        if not isinstance(value, Mapping) or not isinstance(value.get("command"), str) or not value["command"].strip():
            raise ExperimentValidationError(f"{variant}.command must be a non-empty string")
    for label in ("dataset", "evaluator"):
        value = manifest.get(label)
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            raise ExperimentValidationError(f"{label}.path must be a string")
        path = _resolve_reference(workspace, value["path"])
        if not path.is_file():
            raise ExperimentValidationError(f"{label}.path does not exist: {value['path']}")
        if value.get("sha256") != _digest_path(path):
            raise ExperimentValidationError(f"{label}.sha256 does not match {value['path']}")
    execution = manifest.get("execution")
    acceptance = manifest.get("acceptance")
    if not isinstance(execution, Mapping):
        raise ExperimentValidationError("execution must be an object")
    if execution.get("order") not in {"randomized", "alternating"}:
        raise ExperimentValidationError("execution.order must be randomized or alternating")
    if not isinstance(acceptance, Mapping):
        raise ExperimentValidationError("acceptance must be an object")
    optimization_target = acceptance.get(
        "optimization_target", "successful_workflow_latency"
    )
    if optimization_target not in {
        "successful_workflow_latency",
        "engine_change_latency",
    }:
        raise ExperimentValidationError(
            "acceptance.optimization_target must be successful_workflow_latency or engine_change_latency"
        )
    latency_semantics = acceptance.get("latency_semantics", "superiority")
    if latency_semantics not in {"superiority", "noninferiority"}:
        raise ExperimentValidationError(
            "acceptance.latency_semantics must be superiority or noninferiority"
        )
    latency_margin = acceptance.get("max_latency_regression_ms", 0.0)
    if (
        isinstance(latency_margin, bool)
        or not isinstance(latency_margin, (int, float))
        or latency_margin < 0
    ):
        raise ExperimentValidationError(
            "acceptance.max_latency_regression_ms must be non-negative"
        )
    repetitions = execution.get("repetitions", 1)
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions <= 0
    ):
        raise ExperimentValidationError("execution.repetitions must be a positive integer")
    seed = execution.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ExperimentValidationError("execution.seed must be an integer")
    for field, default in (
        ("timeout_seconds", 300.0),
        ("evaluator_timeout_seconds", 30.0),
    ):
        value = execution.get(field, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ExperimentValidationError(f"execution.{field} must be positive")
    for field, default in (
        ("max_task_success_regression", 0.05),
        ("min_task_success_rate", 0.8),
        ("min_completed_rate", 1.0),
    ):
        value = acceptance.get(field, default)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= float(value) <= 1
        ):
            raise ExperimentValidationError(
                f"acceptance.{field} must be between 0 and 1"
            )
    method = acceptance.get("task_success_method", PAIRED_EXACT)
    if method not in PAIRED_METHODS:
        raise ExperimentValidationError(
            "acceptance.task_success_method must be paired_exact or paired_score_exact"
        )
    prospective = manifest.get("schema_version") in {
        AGENT_MANIFEST_PREREGISTERED_SEMANTICS_VERSION,
        AGENT_MANIFEST_LOAD_VERSION,
    }
    if method != PAIRED_EXACT and not prospective:
        raise ExperimentValidationError(
            "acceptance.task_success_method paired_score_exact requires agent "
            "manifest schema_version '0.3'"
        )
    if "completion_policy" in acceptance:
        if not prospective:
            raise ExperimentValidationError(
                "acceptance.completion_policy requires agent manifest "
                "schema_version '0.3'"
            )
        try:
            CompletionPolicy.from_value(acceptance["completion_policy"])
        except CompletionPolicyError as exc:
            raise ExperimentValidationError(str(exc)) from exc
    if "identity_policy" in acceptance:
        if not prospective:
            raise ExperimentValidationError(
                "acceptance.identity_policy requires agent manifest "
                "schema_version '0.3'"
            )
        try:
            parse_identity_policy(acceptance["identity_policy"])
        except IdentityPolicyError as exc:
            raise ExperimentValidationError(str(exc)) from exc
    load_version = manifest.get("schema_version") == AGENT_MANIFEST_LOAD_VERSION
    agent_load_spec(manifest)
    if (
        execution.get("arrival_pattern", SEQUENTIAL) != SEQUENTIAL
        or "load" in execution
    ) and not load_version:
        raise ExperimentValidationError(
            "execution.arrival_pattern and execution.load require agent manifest "
            "schema_version '0.4'"
        )
    if "metric_gates" in acceptance:
        if not load_version:
            raise ExperimentValidationError(
                "acceptance.metric_gates requires agent manifest schema_version '0.4'"
            )
        try:
            parse_metric_gates(acceptance["metric_gates"])
        except MetricGateError as exc:
            raise ExperimentValidationError(str(exc)) from exc
    if manifest.get("cost_model") is not None:
        if not load_version:
            raise ExperimentValidationError(
                "cost_model requires agent manifest schema_version '0.4'"
            )
        try:
            parse_cost_model(manifest["cost_model"])
        except CostModelError as exc:
            raise ExperimentValidationError(str(exc)) from exc
    if manifest.get("design") is not None and not isinstance(
        manifest.get("design"), Mapping
    ):
        raise ExperimentValidationError("design must be an object when supplied")


def agent_load_spec(manifest: Mapping[str, Any]) -> LoadSpec | None:
    """Parse an agent manifest's load declaration, or ``None`` if sequential."""

    execution = manifest.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    pattern = execution.get("arrival_pattern", SEQUENTIAL)
    if pattern == SEQUENTIAL:
        if "load" in execution:
            raise ExperimentValidationError(
                "execution.load requires a concurrent execution.arrival_pattern"
            )
        return None
    if pattern not in CONCURRENT_ARRIVAL_PATTERNS:
        raise ExperimentValidationError(
            "execution.arrival_pattern must be sequential, closed_loop, "
            "constant_rps, or poisson"
        )
    try:
        return LoadSpec.from_value(pattern, execution.get("load"))
    except LoadSpecError as exc:
        raise ExperimentValidationError(str(exc)) from exc


def _issue(
    artifact: str,
    field: str,
    reason: str,
    *,
    code: str,
    execution_blocking: bool,
    claim_weakening: bool,
    next_action: str,
    command: str | None = None,
) -> dict[str, Any]:
    value = {
        "code": code,
        "artifact_path": artifact,
        "field_path": field,
        "reason": reason,
        "execution_blocking": execution_blocking,
        "claim_weakening": claim_weakening,
        "next_action": next_action,
    }
    if command:
        value["command"] = command
    return value


def validate_experiment_artifact(
    path: Path, *, doctor_report: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a corrective validation report without executing the experiment."""

    issues: list[dict[str, Any]] = []
    mode = "static_request"
    experiment_id: str | None = None
    try:
        descriptor: dict[str, Any] | None = None
        parsed: ExperimentManifest | None = None
        if path.is_dir():
            from .freeze import require_intact_freeze

            # A frozen workspace whose locked digests drifted is invalid even
            # when the drifted manifest would otherwise parse.
            require_intact_freeze(path)
            descriptor, manifest = load_workspace(path)
            mode = str(descriptor.get("mode", "unknown"))
            unresolved = descriptor.get("unresolved_fields", [])
            if descriptor.get("generation_status") == "unresolved":
                for item in unresolved if isinstance(unresolved, list) else []:
                    if isinstance(item, Mapping):
                        issues.append(
                            _issue(
                                "manifest.json",
                                str(item.get("path", "unknown")),
                                str(item.get("reason", "required field is unresolved")),
                                code="runeconomy.workspace.unresolved",
                                execution_blocking=True,
                                claim_weakening=True,
                                next_action="supply only evidence-backed values, then validate again",
                                command=f"runeconomy experiment validate {path}",
                            )
                        )
                return {
                    "validation_version": "0.1",
                    "status": "unresolved",
                    "runnable": False,
                    "mode": mode,
                    "experiment_id": None,
                    "issues": issues,
                }
            if mode == "agent_process":
                validate_agent_manifest(manifest, path)
                experiment_id = path.name
            elif mode == "static_request":
                parsed = ExperimentManifest.from_dict(manifest)
                experiment_id = parsed.experiment_id
            else:
                raise ExperimentValidationError(f"unsupported workspace mode {mode!r}")
        else:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise ExperimentValidationError(f"cannot read {path}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ExperimentValidationError(
                    f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
                ) from exc
            if isinstance(raw, Mapping) and raw.get("artifact_type") == "runeconomy.experiment_skeleton":
                for item in raw.get("unresolved_fields", []):
                    if isinstance(item, Mapping):
                        issues.append(
                            _issue(
                                str(path),
                                str(item.get("path", "unknown")),
                                str(item.get("reason", "required field is unresolved")),
                                code="runeconomy.workspace.unresolved",
                                execution_blocking=True,
                                claim_weakening=True,
                                next_action="supply only evidence-backed values, then validate again",
                                command=f"runeconomy experiment validate {path}",
                            )
                        )
                return {
                    "validation_version": "0.1",
                    "status": "unresolved",
                    "runnable": False,
                    "mode": mode,
                    "experiment_id": None,
                    "issues": issues,
                }
            parsed = ExperimentManifest.from_dict(raw)
            experiment_id = parsed.experiment_id
    except ExperimentValidationError as exc:
        issues.append(
            _issue(
                str(path),
                "unknown",
                str(exc),
                code=classify_exception(
                    exc, stage="experiment.validate"
                ).code,
                execution_blocking=True,
                claim_weakening=True,
                next_action="correct the named artifact or field, then validate again",
                command=f"runeconomy experiment validate {path}",
            )
        )
        return {
            "validation_version": "0.1",
            "status": "invalid",
            "runnable": False,
            "mode": mode,
            "experiment_id": experiment_id,
            "issues": issues,
        }
    if doctor_report is not None and mode == "static_request" and parsed is not None:
        dialect_key = (
            "chat_completions"
            if parsed.backend.api_dialect == "chat_completions"
            else "responses"
        )
        capability = doctor_report.get(dialect_key)
        capability_status = (
            capability.get("status") if isinstance(capability, Mapping) else "unknown"
        )
        if capability_status != "supported":
            issues.append(
                _issue(
                    "doctor-report.json",
                    dialect_key,
                    f"the saved backend diagnosis reports {capability_status} for the required API dialect",
                    code="runeconomy.backend.unsupported",
                    execution_blocking=True,
                    claim_weakening=True,
                    next_action="correct the backend capability or select a supported API dialect",
                    command="runeconomy doctor --base-url \"$VLLM_BASE_URL\" --api-key-env VLLM_API_KEY --json",
                )
            )
            return {
                "validation_version": "0.1",
                "status": "unsupported_backend",
                "runnable": False,
                "mode": mode,
                "experiment_id": experiment_id,
                "issues": issues,
            }
    if (
        descriptor is not None
        and descriptor.get("intended_claim") == "public_verification"
        and mode == "static_request"
        and parsed is not None
        and not parsed.acceptance.require_environment_identity
    ):
        issues.append(
            _issue(
                "manifest.json",
                "acceptance.require_environment_identity",
                "a public verification claim requires complete stable serving identity",
                code="runeconomy.claim.incomplete",
                execution_blocking=False,
                claim_weakening=True,
                next_action="require environment identity or narrow the intended claim",
            )
        )
        return {
            "validation_version": "0.1",
            "status": "claim_evidence_incomplete",
            "runnable": True,
            "mode": mode,
            "experiment_id": experiment_id,
            "issues": issues,
            "next_command": f"runeconomy experiment run {path} --output results/",
        }
    return {
        "validation_version": "0.1",
        "status": "runnable",
        "runnable": True,
        "mode": mode,
        "experiment_id": experiment_id,
        "issues": [],
        "next_command": f"runeconomy experiment run {path} --output results/",
    }


def resolve_agent_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    descriptor, manifest = load_workspace(path)
    if descriptor.get("mode") != "agent_process":
        raise ExperimentValidationError(f"{path} is not an agent-process workspace")
    validate_agent_manifest(manifest, path)
    return manifest, path


def resolve_static_manifest(path: Path) -> ExperimentManifest:
    if path.is_dir():
        descriptor, manifest = load_workspace(path)
        if descriptor.get("mode") != "static_request":
            raise ExperimentValidationError(f"{path} is not a static-request workspace")
        if descriptor.get("generation_status") != "runnable":
            raise ExperimentValidationError(
                f"{path} is unresolved; run `runeconomy experiment validate {path}`"
            )
        return ExperimentManifest.from_dict(manifest)
    return ExperimentManifest.from_path(path)


def agent_run_arguments(manifest: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    execution = manifest["execution"]
    acceptance = manifest["acceptance"]
    return {
        "baseline_command": manifest["baseline"]["command"],
        "candidate_command": manifest["candidate"]["command"],
        "dataset_path": _resolve_reference(workspace, manifest["dataset"]["path"]),
        "evaluator_path": _resolve_reference(workspace, manifest["evaluator"]["path"]),
        "repetitions": int(execution.get("repetitions", 1)),
        "order": str(execution.get("order", "randomized")),
        "seed": int(execution.get("seed", 0)),
        "timeout_seconds": float(execution.get("timeout_seconds", 300.0)),
        "evaluator_timeout_seconds": float(execution.get("evaluator_timeout_seconds", 30.0)),
        "max_task_success_regression": float(acceptance.get("max_task_success_regression", 0.05)),
        "min_task_success_rate": float(acceptance.get("min_task_success_rate", 0.8)),
        "min_completed_rate": float(acceptance.get("min_completed_rate", 1.0)),
        "optimization_target": str(
            acceptance.get("optimization_target", "successful_workflow_latency")
        ),
        "latency_semantics": str(
            acceptance.get("latency_semantics", "superiority")
        ),
        "max_latency_regression_ms": float(
            acceptance.get("max_latency_regression_ms", 0.0)
        ),
        "task_success_method": str(
            acceptance.get("task_success_method", PAIRED_EXACT)
        ),
        "completion_policy": (
            dict(acceptance["completion_policy"])
            if isinstance(acceptance.get("completion_policy"), Mapping)
            else None
        ),
        **(
            {"identity_policy": dict(acceptance["identity_policy"])}
            if isinstance(acceptance.get("identity_policy"), Mapping)
            else {}
        ),
        **(
            {}
            if execution.get("arrival_pattern", SEQUENTIAL) == SEQUENTIAL
            else {
                "arrival_pattern": str(execution["arrival_pattern"]),
                "load": dict(execution["load"]),
            }
        ),
        **(
            {"metric_gates": [dict(gate) for gate in acceptance["metric_gates"]]}
            if isinstance(acceptance.get("metric_gates"), list)
            else {}
        ),
        **(
            {"attempt_economics": parse_cost_model(manifest["cost_model"])}
            if manifest.get("cost_model") is not None
            else {}
        ),
    }
