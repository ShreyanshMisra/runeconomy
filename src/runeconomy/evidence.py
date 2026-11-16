"""Versioned imported-evidence bundles and conservative capability grading."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .analysis import analyze_trace
from .importers import (
    import_ledger_jsonl,
    import_openinference,
    import_phoenix_json,
    import_vllm_jsonl,
)
from .models import Trace, TraceValidationError


EVIDENCE_BUNDLE_VERSION = "0.1"


def _digest_path(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise TraceValidationError(f"cannot read {path}: {exc}") from exc
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _capability(
    status: str, evidence: str, *, next_action: str | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {"status": status, "evidence": evidence}
    if next_action:
        value["next_action"] = next_action
    return value


def _experiment_context(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    direct = payload.get("runeconomy.experiment_context")
    if isinstance(direct, Mapping):
        return deepcopy(dict(direct))
    namespace = payload.get("runeconomy")
    if isinstance(namespace, Mapping) and isinstance(
        namespace.get("experiment_context"), Mapping
    ):
        return deepcopy(dict(namespace["experiment_context"]))
    return None


def _static_context_missing(context: Mapping[str, Any] | None) -> list[str]:
    if context is None:
        return ["experiment_context"]
    required = (
        "backend",
        "sampling",
        "execution",
        "acceptance",
        "evaluator",
        "baseline_request",
        "candidate_request",
        "confirmed_capabilities",
    )
    return [name for name in required if name not in context]


def assess_trace_evidence(
    trace_payload: Mapping[str, Any],
    analysis: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe only the uses established by the imported trace and extensions."""

    requests = trace_payload.get("requests", [])
    spans = trace_payload.get("spans", [])
    exact = bool(requests) and all(
        isinstance(item, Mapping)
        and item.get("token_evidence_quality") == "exact"
        and isinstance(item.get("prompt_token_ids"), list)
        for item in requests
    )
    structural = bool(requests) and all(
        isinstance(item, Mapping)
        and isinstance(item.get("segments"), list)
        and any(
            isinstance(segment, Mapping)
            and segment.get("attribution_quality") != "unknown"
            for segment in item.get("segments", [])
        )
        for item in requests
    )
    cache = bool(requests) and all(
        isinstance(item, Mapping)
        and isinstance(item.get("metrics"), Mapping)
        and item["metrics"].get("cached_input_tokens") is not None
        for item in requests
    )
    explicit_dag = (
        trace_payload.get("dependency_evidence") == "complete_explicit"
        and bool(spans)
        and analysis.get("critical_path") is not None
    )
    missing_context = _static_context_missing(context)
    has_supported_finding = any(
        isinstance(item, Mapping)
        and item.get("experiment_template_id") == "prompt.move_dynamic_metadata"
        for item in analysis.get("findings", [])
    )
    static_runnable = not missing_context and has_supported_finding
    agent_context = context if isinstance(context, Mapping) else {}
    agent_runnable = all(
        key in agent_context
        for key in ("baseline_command", "candidate_command", "dataset", "evaluator")
    )
    capabilities = {
        "basic_diagnosis": _capability(
            "available" if requests else "missing",
            f"{len(requests)} inference request(s) and {len(spans)} work span(s) imported",
        ),
        "structural_prompt_attribution": _capability(
            "available" if structural else "missing",
            (
                "request roles or blocks provide structural segment evidence"
                if structural
                else "no supported structural prompt segments were established"
            ),
            next_action=None if structural else "export message roles or rendered segments",
        ),
        "exact_token_cache_analysis": _capability(
            "available" if exact and cache else "partial" if exact or cache else "missing",
            (
                "exact prompt token IDs and cache telemetry are present"
                if exact and cache
                else "exact token IDs or cache telemetry are incomplete"
            ),
            next_action=(
                None
                if exact and cache
                else "capture exact prompt token IDs and observed cached input tokens"
            ),
        ),
        "critical_path_analysis": _capability(
            "available" if explicit_dag else "missing",
            (
                "every imported work span declares explicit dependencies"
                if explicit_dag
                else "complete explicit workflow dependencies were not established"
            ),
            next_action=(
                None
                if explicit_dag
                else "export runeconomy.depends_on for every work span"
            ),
        ),
        "static_request_replay": _capability(
            "available" if static_runnable else "missing",
            (
                "the source includes replay requests, evaluator, gates, backend, and capabilities"
                if static_runnable
                else "missing explicit replay context: " + ", ".join(missing_context)
            ),
            next_action=(
                None
                if static_runnable
                else "supply explicit experiment context without reconstructing private requests"
            ),
        ),
        "complete_agent_comparison": _capability(
            "available" if agent_runnable else "missing",
            (
                "baseline and candidate commands, dataset, and evaluator are explicit"
                if agent_runnable
                else "agent commands, dataset, or evaluator are not present"
            ),
            next_action=(
                None
                if agent_runnable
                else "initialize an agent-process workspace with explicit local commands and files"
            ),
        ),
        "public_verification_claim": _capability(
            "missing",
            "imported evidence can motivate an experiment but is not a paired gated verdict",
            next_action="run a frozen paired experiment and satisfy every declared quality and performance gate",
        ),
    }
    return {
        "assessment_version": "0.1",
        "capabilities": capabilities,
        "runnable_experiment": static_runnable or agent_runnable,
        "recommended_mode": (
            "static_request"
            if static_runnable
            else "agent_process"
            if agent_runnable
            else None
        ),
        "next_action": (
            "initialize a workspace from this evidence"
            if static_runnable or agent_runnable
            else capabilities["static_request_replay"]["next_action"]
        ),
    }


def assess_finding_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    findings = payload.get("findings", [])
    return {
        "assessment_version": "0.1",
        "capabilities": {
            "basic_diagnosis": _capability(
                "available" if findings else "missing",
                f"{len(findings)} source-reported finding(s) imported",
            ),
            "structural_prompt_attribution": _capability(
                "partial",
                "source conclusions and block labels are preserved when present",
            ),
            "exact_token_cache_analysis": _capability(
                "missing",
                "hashes and counts do not establish raw token sequences",
                next_action="join with an exact-token trace if token-level analysis is required",
            ),
            "critical_path_analysis": _capability(
                "missing",
                "the finding export contains no complete workflow DAG",
                next_action="supply a corresponding workflow trace with explicit dependencies",
            ),
            "static_request_replay": _capability(
                "missing",
                "the source does not contain replayable request bodies or an evaluator",
                next_action="supply replay requests, backend identity, gates, and an evaluator",
            ),
            "complete_agent_comparison": _capability(
                "missing",
                "the source does not contain baseline and candidate agent commands",
            ),
            "public_verification_claim": _capability(
                "missing",
                "an imported diagnosis is not a paired verification result",
            ),
        },
        "runnable_experiment": False,
        "recommended_mode": None,
        "next_action": "initialize an unresolved workspace and supply only evidence-backed fields",
    }


def _enrich_findings(
    analysis: dict[str, Any],
    context: Mapping[str, Any] | None,
    source_digest: str,
) -> None:
    if context is None:
        return
    template_id = context.get("experiment_template_id")
    for finding in analysis.get("findings", []):
        if not isinstance(finding, dict):
            continue
        if template_id is not None and finding.get("experiment_template_id") != template_id:
            continue
        parameters = finding.setdefault("experiment_parameters", {})
        if not isinstance(parameters, dict):
            continue
        for key, value in context.items():
            if key != "experiment_template_id":
                parameters[key] = deepcopy(value)
        parameters["source_evidence_sha256"] = source_digest
        if not _static_context_missing(context):
            finding["unsupported_reason"] = None


def import_evidence(
    path: Path,
    *,
    source_type: str = "auto",
    trace_id: str | None = None,
    workload: str = "unknown",
    backend: str = "unknown",
    model: str | None = None,
    block_size: int = 16,
) -> dict[str, Any]:
    """Import one supported source into a capability-graded evidence bundle."""

    if path.is_dir():
        if source_type not in {"auto", "openinference", "phoenix"}:
            raise TraceValidationError(
                "directory imports currently support OpenInference and Phoenix JSON files"
            )
        files = sorted(item for item in path.rglob("*.json") if item.is_file())
        if not files:
            raise TraceValidationError(f"{path}: no JSON evidence files found")
        runs = [
            import_evidence(
                item,
                source_type="auto",
                workload=workload,
                backend=backend,
                block_size=block_size,
            )
            for item in files
        ]
        digest_input = "\n".join(
            f"{item.relative_to(path)}\0{run['source']['sha256']}"
            for item, run in zip(files, runs, strict=True)
        )
        capabilities: dict[str, dict[str, Any]] = {}
        keys = runs[0]["capability_assessment"]["capabilities"].keys()
        for key in keys:
            values = [run["capability_assessment"]["capabilities"][key] for run in runs]
            statuses = {value["status"] for value in values}
            status = next(iter(statuses)) if len(statuses) == 1 else "partial"
            capabilities[key] = _capability(
                status,
                f"{sum(value['status'] == 'available' for value in values)} of {len(values)} run(s) available",
            )
        return {
            "artifact_type": "runeconomy.evidence_bundle",
            "schema_version": EVIDENCE_BUNDLE_VERSION,
            "source": {
                "kind": "trace_directory",
                "name": path.name,
                "sha256": "sha256:" + hashlib.sha256(digest_input.encode()).hexdigest(),
            },
            "runs": runs,
            "findings": [
                finding
                for run in runs
                for finding in run.get("analysis", {}).get("findings", [])
            ],
            "capability_assessment": {
                "assessment_version": "0.1",
                "capabilities": capabilities,
                "runnable_experiment": False,
                "recommended_mode": None,
                "next_action": "select one run or finding explicitly before initializing an experiment",
            },
        }
    digest = _digest_path(path)
    detected = source_type
    raw_json: dict[str, Any] | None = None
    if source_type in {"auto", "openinference", "phoenix"}:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            raw_json = loaded
            if source_type == "auto":
                detected = (
                    "phoenix"
                    if isinstance(loaded.get("spans"), list)
                    and "traceId" in loaded
                    and any(
                        isinstance(item, Mapping) and "span_kind" in item
                        for item in loaded["spans"]
                    )
                    else "openinference"
                )
    if detected == "auto":
        first = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
        try:
            row = json.loads(first)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"cannot detect evidence type for {path}") from exc
        if isinstance(row, dict) and "cause" in row:
            detected = "prefixwise"
        elif isinstance(row, dict) and (
            "capture" in row or "prompt_token_ids" in row
        ):
            detected = "vllm"
        else:
            raise TraceValidationError(
                "cannot detect evidence type; pass --type openinference, vllm, or prefixwise"
            )

    source = {"kind": detected, "name": path.name, "sha256": digest}
    if detected == "prefixwise":
        imported = import_ledger_jsonl(path, source_name="prefixwise")
        return {
            "artifact_type": "runeconomy.evidence_bundle",
            "schema_version": EVIDENCE_BUNDLE_VERSION,
            "source": source,
            "findings": imported["findings"],
            "limitations": imported.get("limitations", []),
            "capability_assessment": assess_finding_evidence(imported),
        }
    if detected == "vllm":
        if not model or not trace_id:
            raise TraceValidationError(
                "vLLM evidence requires --model and --trace-id"
            )
        trace_payload = import_vllm_jsonl(
            path,
            trace_id=trace_id,
            workload=workload,
            model=model,
            block_size=block_size,
        )
        context = None
    elif detected == "openinference":
        trace_payload = import_openinference(
            path,
            trace_id=trace_id,
            workload=workload,
            backend=backend,
            block_size=block_size,
        )
        context = _experiment_context(raw_json or {})
    elif detected == "phoenix":
        trace_payload = import_phoenix_json(
            path, workload=workload, backend=backend, block_size=block_size
        )
        context = None
    else:
        raise TraceValidationError(f"unsupported evidence type {detected!r}")

    trace = Trace.from_dict(trace_payload)
    analysis = analyze_trace(trace)
    _enrich_findings(analysis, context, digest)
    assessment = assess_trace_evidence(trace_payload, analysis, context)
    return {
        "artifact_type": "runeconomy.evidence_bundle",
        "schema_version": EVIDENCE_BUNDLE_VERSION,
        "source": source,
        "trace": trace_payload,
        "analysis": analysis,
        "capability_assessment": assessment,
    }


def load_evidence_bundle(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceValidationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "runeconomy.evidence_bundle":
        raise TraceValidationError(f"{path} is not a RunEconomy evidence bundle")
    if payload.get("schema_version") != EVIDENCE_BUNDLE_VERSION:
        raise TraceValidationError(
            f"unsupported evidence bundle version {payload.get('schema_version')!r}"
        )
    return payload
