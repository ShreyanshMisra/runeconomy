"""Turn structured findings into validated manifests or honest skeletons."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Mapping

from .experiments import ExperimentManifest, ExperimentValidationError


AUTOMATABLE_TEMPLATES = {
    "tools.scope_allowlist",
    "prompt.canonicalize_tools",
    "prompt.move_dynamic_metadata",
}


def _stable_legacy_id(
    trace_id: str, rule_id: str, affected_request_ids: list[str]
) -> str:
    identity = "\0".join((trace_id, rule_id, *affected_request_ids))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"finding-{digest}"


def select_finding(payload: Any, finding_id: str | None = None) -> dict[str, Any]:
    """Select one finding from either a finding object or an analysis report."""

    if not isinstance(payload, dict):
        raise ExperimentValidationError("finding source must be a JSON object")
    if isinstance(payload.get("findings"), list):
        trace_id = str(payload.get("trace_id", "unknown-trace"))
        findings: list[dict[str, Any]] = []
        for index, raw in enumerate(payload["findings"]):
            if not isinstance(raw, dict):
                raise ExperimentValidationError(
                    f"findings[{index}] must be a JSON object"
                )
            normalized = deepcopy(raw)
            if not isinstance(normalized.get("finding_id"), str):
                request_id = normalized.get("request_id")
                affected = (
                    [request_id] if isinstance(request_id, str) else []
                )
                normalized["affected_request_ids"] = affected
                normalized["finding_id"] = _stable_legacy_id(
                    trace_id, str(normalized.get("rule_id", "unknown")), affected
                )
            findings.append(normalized)
        if finding_id is not None:
            matches = [item for item in findings if item["finding_id"] == finding_id]
            if len(matches) != 1:
                available = ", ".join(item["finding_id"] for item in findings)
                raise ExperimentValidationError(
                    f"finding_id {finding_id!r} was not found; available: {available}"
                )
            return matches[0]
        if len(findings) != 1:
            available = ", ".join(item["finding_id"] for item in findings)
            raise ExperimentValidationError(
                "analysis report contains multiple findings; pass --finding-id with "
                f"one of: {available}"
            )
        return findings[0]

    finding = deepcopy(payload)
    if not isinstance(finding.get("rule_id"), str):
        raise ExperimentValidationError(
            "finding source is neither an analysis report nor a structured finding"
        )
    if not isinstance(finding.get("finding_id"), str):
        request_id = finding.get("request_id")
        affected = [request_id] if isinstance(request_id, str) else []
        finding["affected_request_ids"] = affected
        finding["finding_id"] = _stable_legacy_id(
            "standalone", finding["rule_id"], affected
        )
    if finding_id is not None and finding["finding_id"] != finding_id:
        raise ExperimentValidationError(
            f"finding object has id {finding['finding_id']!r}, not {finding_id!r}"
        )
    return finding


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, Mapping):
        return None
    if isinstance(tool.get("name"), str):
        return str(tool["name"])
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return str(function["name"])
    return None


def _missing_parameters(
    parameters: Mapping[str, Any], names: tuple[str, ...]
) -> list[dict[str, str]]:
    return [
        {
            "path": f"experiment_parameters.{name}",
            "reason": "required to generate and validate a runnable manifest",
        }
        for name in names
        if name not in parameters
    ]


def _manifest_draft(finding: Mapping[str, Any]) -> dict[str, Any]:
    parameters = finding.get("experiment_parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    return {
        "schema_version": "0.3",
        "experiment_id": f"from-{finding.get('finding_id', 'finding')}",
        "backend": parameters.get("backend"),
        "sampling": parameters.get("sampling"),
        "execution": parameters.get("execution"),
        "acceptance": parameters.get("acceptance"),
        "design": {
            "source_finding_id": finding.get("finding_id"),
            "source_rule_id": finding.get("rule_id"),
            "hypothesis": {
                "causal_mechanism": finding.get("causal_mechanism"),
                "prediction": {
                    "metric": finding.get("predicted_metric"),
                    "direction": finding.get("predicted_direction"),
                },
            },
        },
        "constants": {},
        "cases": [],
    }


def _skeleton(
    finding: Mapping[str, Any], unresolved: list[dict[str, str]]
) -> dict[str, Any]:
    reason = finding.get("unsupported_reason")
    if isinstance(reason, str) and reason:
        unresolved.insert(
            0,
            {
                "path": "finding.unsupported_reason",
                "reason": reason,
            },
        )
    return {
        "artifact_type": "runeconomy.experiment_skeleton",
        "skeleton_version": "0.1",
        "generation_status": "unresolved",
        "source_finding": {
            "finding_id": finding.get("finding_id"),
            "rule_id": finding.get("rule_id"),
            "experiment_template_id": finding.get("experiment_template_id"),
            "affected_request_ids": finding.get("affected_request_ids", []),
            "required_capabilities": finding.get("required_capabilities", []),
        },
        "unresolved_fields": unresolved,
        "manifest_draft": _manifest_draft(finding),
    }


def _variant_requests(
    template_id: str, parameters: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    baseline = deepcopy(parameters["baseline_request"])
    if not isinstance(baseline, dict):
        raise ExperimentValidationError(
            "experiment_parameters.baseline_request must be an object"
        )
    if template_id == "tools.scope_allowlist":
        tools = baseline.get("tools")
        names = parameters.get("candidate_tool_names")
        if not isinstance(tools, list) or not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            raise ExperimentValidationError(
                "tools.scope_allowlist requires baseline_request.tools and a string "
                "candidate_tool_names array"
            )
        tool_names = [_tool_name(tool) for tool in tools]
        if any(name is None for name in tool_names) or len(set(tool_names)) != len(
            tool_names
        ):
            raise ExperimentValidationError(
                "baseline_request.tools must have unique supported function names"
            )
        by_name = dict(zip(tool_names, tools, strict=True))
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ExperimentValidationError(
                f"candidate_tool_names are absent from baseline tools: {unknown}"
            )
        candidate = deepcopy(baseline)
        candidate["tools"] = [deepcopy(by_name[name]) for name in names]
        tool_choice = candidate.get("tool_choice")
        selected_name = None
        if isinstance(tool_choice, Mapping):
            if isinstance(tool_choice.get("name"), str):
                selected_name = tool_choice["name"]
            function = tool_choice.get("function")
            if isinstance(function, Mapping) and isinstance(
                function.get("name"), str
            ):
                selected_name = function["name"]
        if selected_name is not None and selected_name not in names:
            raise ExperimentValidationError(
                "candidate allowlist removes the function required by tool_choice"
            )
        if tool_choice == "required" and not names:
            raise ExperimentValidationError(
                "candidate allowlist cannot be empty when tool_choice is required"
            )
        return baseline, candidate, None, None
    if template_id == "prompt.canonicalize_tools":
        if not isinstance(baseline.get("tools"), list):
            raise ExperimentValidationError(
                "prompt.canonicalize_tools requires baseline_request.tools"
            )
        return (
            baseline,
            deepcopy(baseline),
            {"kind": "shuffle_tools"},
            {"kind": "canonicalize_tools"},
        )
    if template_id == "prompt.move_dynamic_metadata":
        candidate = deepcopy(parameters["candidate_request"])
        if not isinstance(candidate, dict):
            raise ExperimentValidationError(
                "experiment_parameters.candidate_request must be an object"
            )
        return baseline, candidate, None, None
    raise ExperimentValidationError(
        f"experiment template {template_id!r} cannot be generated automatically"
    )


def initialize_experiment(finding: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated manifest or an explicitly unresolved skeleton."""

    template_id = finding.get("experiment_template_id")
    parameters = finding.get("experiment_parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    unresolved: list[dict[str, str]] = []
    if not isinstance(template_id, str) or not template_id:
        unresolved.append(
            {
                "path": "finding.experiment_template_id",
                "reason": "no experiment template is associated with this finding",
            }
        )
        return _skeleton(finding, unresolved)
    if template_id not in AUTOMATABLE_TEMPLATES:
        unresolved.append(
            {
                "path": "finding.experiment_template_id",
                "reason": (
                    f"template {template_id!r} requires an application or serving "
                    "change that RunEconomy cannot generate safely"
                ),
            }
        )
        return _skeleton(finding, unresolved)

    common = (
        "backend",
        "sampling",
        "execution",
        "acceptance",
        "evaluator",
        "baseline_request",
        "confirmed_capabilities",
    )
    unresolved.extend(_missing_parameters(parameters, common))
    if template_id == "tools.scope_allowlist":
        unresolved.extend(_missing_parameters(parameters, ("candidate_tool_names",)))
    elif template_id == "prompt.move_dynamic_metadata":
        unresolved.extend(_missing_parameters(parameters, ("candidate_request",)))

    confirmed = parameters.get("confirmed_capabilities")
    required = finding.get("required_capabilities", [])
    if "confirmed_capabilities" in parameters and not isinstance(
        confirmed, (list, tuple)
    ):
        unresolved.append(
            {
                "path": "experiment_parameters.confirmed_capabilities",
                "reason": "must be an array of capability names",
            }
        )
    elif not isinstance(required, (list, tuple)):
        unresolved.append(
            {
                "path": "finding.required_capabilities",
                "reason": "must be an array of capability names",
            }
        )
    elif isinstance(confirmed, (list, tuple)):
        if any(not isinstance(item, str) for item in (*confirmed, *required)):
            raise ExperimentValidationError(
                "required_capabilities and confirmed_capabilities must contain strings"
            )
        missing_capabilities = sorted(set(required) - set(confirmed))
        if missing_capabilities:
            unresolved.append(
                {
                    "path": "experiment_parameters.confirmed_capabilities",
                    "reason": "missing required capabilities: "
                    + ", ".join(missing_capabilities),
                }
            )
    if unresolved:
        return _skeleton(finding, unresolved)

    baseline, candidate, baseline_transform, candidate_transform = _variant_requests(
        template_id, parameters
    )
    baseline_variant: dict[str, Any] = {"request": baseline}
    candidate_variant: dict[str, Any] = {"request": candidate}
    if baseline_transform is not None:
        baseline_variant["request_transform"] = baseline_transform
    if candidate_transform is not None:
        candidate_variant["request_transform"] = candidate_transform
    manifest = {
        "schema_version": "0.3",
        "experiment_id": str(
            parameters.get("experiment_id", f"from-{finding['finding_id']}")
        ),
        "backend": deepcopy(parameters["backend"]),
        "sampling": deepcopy(parameters["sampling"]),
        "execution": deepcopy(parameters["execution"]),
        "acceptance": deepcopy(parameters["acceptance"]),
        "design": {
            "source_finding": {
                "finding_id": finding.get("finding_id"),
                "rule_id": finding.get("rule_id"),
                "affected_request_ids": deepcopy(
                    finding.get("affected_request_ids", [])
                ),
                "evidence_level": finding.get("evidence_level"),
                "confidence": finding.get("confidence"),
            },
            "hypothesis": {
                "causal_mechanism": finding.get("causal_mechanism"),
                "prediction": {
                    "metric": finding.get("predicted_metric"),
                    "direction": finding.get("predicted_direction"),
                },
            },
            "required_capabilities": deepcopy(
                finding.get("required_capabilities", [])
            ),
            "confirmed_capabilities": deepcopy(
                parameters["confirmed_capabilities"]
            ),
        },
        "constants": deepcopy(parameters.get("constants", {})),
        "cases": [
            {
                "case_id": str(
                    parameters.get("case_id", finding.get("finding_id", "case-1"))
                ),
                "evaluator": deepcopy(parameters["evaluator"]),
                "variants": {
                    "baseline": baseline_variant,
                    "candidate": candidate_variant,
                },
            }
        ],
    }
    # This is the boundary between a runnable artifact and a skeleton. Never
    # label the artifact runnable unless the normal manifest parser accepts it.
    parsed = ExperimentManifest.from_dict(manifest)
    if parsed.acceptance.task_success_method != "paired_exact":
        raise ExperimentValidationError(
            "generated experiments require acceptance.task_success_method='paired_exact'"
        )
    expected_target = finding.get("optimization_target")
    if (
        isinstance(expected_target, str)
        and parsed.acceptance.optimization_target != expected_target
    ):
        raise ExperimentValidationError(
            "acceptance.optimization_target must match the source finding: "
            f"expected {expected_target!r}"
        )
    return manifest
