"""Doctor report schema 0.3 for adapter backends other than vLLM.

vLLM keeps doctor schema 0.2 unchanged. Schema 0.3 keeps every 0.2 top-level
capability key (so saved-report consumers such as ``experiment validate
--doctor-report`` and support bundles read it the same way) and adds
``backend_kind``, the protocol ``capability_table`` and graded ``identity``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..backends import Capability
from .base import ADAPTER_PROTOCOL_VERSION, CapabilityTable, ServingIdentity


DOCTOR_ADAPTER_SCHEMA_VERSION = "0.3"
LEGACY_CAPABILITY_KEYS = (
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


def mode(
    name: str,
    status: str,
    *,
    safe: bool,
    physical_cache_state: str,
    note: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "mode": name,
        "status": status,
        "supported": status == "supported",
        "safe": safe and status == "supported",
        "physical_cache_state": physical_cache_state,
        "note": note,
        "next_action": next_action,
    }


def _readiness(
    modes: Sequence[Mapping[str, Any]], next_actions: Sequence[str]
) -> dict[str, Any]:
    supported = [
        item["mode"] for item in modes if item["status"] == "supported" and item["safe"]
    ]
    if supported:
        status, summary = "ready", f"Safe modes available: {', '.join(supported)}."
    elif any(item["status"] == "unknown" for item in modes):
        status = "unknown"
        summary = "Compatibility could not be established from the available metadata."
    else:
        status = "blocked"
        summary = "No safe RunEconomy experiment mode is currently available."
    return {
        "status": status,
        "summary": summary,
        "supported_modes": supported,
        "blockers": [] if supported else ["No safe experiment mode was established."],
        "next_actions": list(next_actions),
    }


def build_report(
    *,
    table: CapabilityTable,
    identity: ServingIdentity,
    legacy: Mapping[str, Capability],
    modes: Sequence[Mapping[str, Any]],
    next_actions: Sequence[str],
    probes: Sequence[str],
) -> dict[str, Any]:
    missing = [key for key in LEGACY_CAPABILITY_KEYS if key not in legacy]
    if missing:
        raise ValueError(f"missing legacy capability keys: {missing}")
    report: dict[str, Any] = {
        "schema_version": DOCTOR_ADAPTER_SCHEMA_VERSION,
        "adapter_protocol_version": ADAPTER_PROTOCOL_VERSION,
        "backend": identity.backend_kind,
        "backend_kind": identity.backend_kind,
        "backend_version": identity.backend_version,
        "model_ids": list(identity.model_ids),
        **{
            key: {
                "status": legacy[key].status,
                "evidence": legacy[key].evidence,
                "detail": legacy[key].detail,
                "required": legacy[key].required,
                "next_action": legacy[key].next_action,
            }
            for key in LEGACY_CAPABILITY_KEYS
        },
        "capability_table": table.to_dict(),
        "identity": identity.to_dict(),
        "experiment_modes": [dict(item) for item in modes],
        "readiness": _readiness(modes, next_actions),
        "probe_safety": {
            "read_only": True,
            "generation_invoked": False,
            "cache_reset_invoked": False,
            "probes": list(probes),
        },
    }
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Human doctor output for a schema-0.3 adapter report."""

    readiness = report.get("readiness", {})
    table = report.get("capability_table", {})
    identity = report.get("identity", {})
    kind = str(report.get("backend_kind", "unknown"))
    models = report.get("model_ids") or []
    lines = [
        f"RunEconomy doctor · {str(readiness.get('status', 'unknown')).upper()}",
        str(readiness.get("summary", "")),
        "Target details: redacted",
        f"Backend kind: {kind} · version {report.get('backend_version') or 'unknown'}",
        f"Models: {', '.join(models) if models else 'unknown'}",
        f"Identity grade: {identity.get('grade', 'unknown')}",
        "",
        "Capability                  Status      Evidence",
        "--------------------------  ----------  ----------------------------------------",
    ]
    downgraded: list[tuple[str, str]] = []
    for item in table.get("capabilities", []):
        lines.append(
            f"{str(item.get('label', item.get('name'))):<26}  "
            f"{str(item.get('status')):<10}  {item.get('evidence', '')}"
        )
        if item.get("status") != "supported" and item.get("effect"):
            downgraded.append((str(item.get("label")), str(item["effect"])))
    undisclosed = identity.get("undisclosed_fields") or []
    lines.extend(["", "Identity"])
    if identity.get("grade") == "provider_declared":
        lines.append(
            "Provider-declared: the provider reports a model, not its tokenizer, "
            "template or serving stack. Declare acceptance.identity_policy with "
            "minimum_grade provider_declared; verdicts describe the observed service only."
        )
        if undisclosed:
            lines.append(f"Undisclosed: {', '.join(undisclosed)}")
    else:
        lines.append(
            "Operator-verified: pin model revision, tokenizer and template "
            "identity, and engine configuration in the experiment manifest."
        )
    if downgraded:
        lines.extend(["", "Downgraded capabilities"])
        lines.extend(f"  {label}: {effect}" for label, effect in downgraded)
    lines.extend(["", "Experiment modes:"])
    for item in report.get("experiment_modes", []):
        suffix = ""
        if item.get("status") == "supported":
            suffix = " (safe)" if item.get("safe") else " (manual/private only)"
        lines.append(f"  {item.get('mode')}: {item.get('status')}{suffix}")
        lines.append(f"    {item.get('note')}")
    actions = list(readiness.get("next_actions") or [])
    if actions:
        lines.extend(["", "What to do next"])
        lines.extend(f"  {index}. {action}" for index, action in enumerate(actions, 1))
    lines.extend(
        [
            "",
            "Safety: doctor read metadata only; it did not generate text, reset "
            "caches, or mutate the server.",
        ]
    )
    return "\n".join(lines)
