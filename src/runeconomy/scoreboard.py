"""Verification cards: one public, aggregate-only summary per decision record.

A card restates what a frozen decision record already says: the intervention,
the environment, the verdict, the decisive gate and the headline effects. It
never adds a number the record does not contain. Descriptive studies get cards
marked ``descriptive`` and never carry a verdict.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

CARD_SCHEMA_VERSION = "0.1"
PROOF_RESULT_TYPES = frozenset({"runeconomy.phase7_proof_decision"})
STUDY_RESULT_TYPES = frozenset({"runeconomy.phase7_study"})


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(record: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for gate in record.get("gates", []):
        if isinstance(gate, Mapping) and gate.get("name") == name:
            return gate
    return None


def _environment(record: Mapping[str, Any]) -> dict[str, Any]:
    env = record.get("environment") or {}
    servers = env.get("servers") or {}
    gpu = env.get("gpu")
    inventory = env.get("gpu_inventory") or []
    gpu_name = inventory[0].get("name") if inventory and isinstance(inventory[0], Mapping) else gpu
    return {
        "engine": "vllm" if env.get("vllm_version") else None,
        "engine_version": env.get("vllm_version"),
        "gpu": gpu_name,
        "served_models": sorted(
            str(server.get("served_model")) for server in servers.values() if isinstance(server, Mapping) and server.get("served_model")
        ),
        "quantization": {
            name: server.get("quantization") for name, server in sorted(servers.items()) if isinstance(server, Mapping)
        },
    }


def _latency(arm: Mapping[str, Any]) -> dict[str, Any]:
    latency = arm.get("successful_workflow_latency_ms") or {}
    return {key: latency.get(key) for key in ("mean", "p50", "p95")}


def proof_card(record: Mapping[str, Any], record_path: str, record_sha256: str) -> dict[str, Any]:
    decision = record["decision"]
    arms = record["arms"]
    paired = record.get("paired") or {}
    execution_load = (record.get("execution") or {}).get("load")
    load = (execution_load or {}).get("load") or {}
    cost_gate = _gate(record, "usd_per_1k_successful_workflows_superiority")
    p95_gate = _gate(record, "p95_workflow_latency_ms_superiority")
    latency = paired.get("successful_pair_latency_difference_ms") or {}
    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "id": f"proof-{record['proof']}",
        "kind": "proof",
        "title": f"Proof {record['proof']}",
        "intervention": record.get("intervention"),
        "hypothesis": record.get("hypothesis"),
        "preregistered": True,
        "verdict": decision["status"],
        "decisive_gate": decision.get("decisive_gate"),
        "reason": decision.get("decisive_explanation") or decision.get("reason"),
        "environment": _environment(record),
        "workload": {
            "pairs": paired.get("pairs"),
            "concurrency": load.get("concurrency") if execution_load else 1,
            "arrival_pattern": execution_load.get("arrival_pattern") if execution_load else "sequential",
        },
        "task_success": {name: arms[name].get("task_success_rate") for name in ("baseline", "candidate")},
        "successful_workflow_latency_ms": {name: _latency(arms[name]) for name in ("baseline", "candidate")},
        "paired_latency_difference_ms": {"mean": latency.get("mean"), "pairs": latency.get("n")},
        "p95_gate": None if p95_gate is None else {"passed": p95_gate.get("passed"), "required": p95_gate.get("required")},
        "usd_per_1k_successful_workflows": None
        if cost_gate is None
        else {
            "baseline": cost_gate["observed"].get("baseline"),
            "candidate": cost_gate["observed"].get("candidate"),
            "passed": cost_gate.get("passed"),
        },
        "claim_boundary": list(record.get("claim_boundary") or []),
        "record": {"path": record_path, "sha256": record_sha256},
    }


def study_card(record: Mapping[str, Any], record_path: str, record_sha256: str) -> dict[str, Any]:
    card: dict[str, Any] = {
        "schema_version": CARD_SCHEMA_VERSION,
        "id": f"study-{record['study']}",
        "kind": "study",
        "title": record.get("title"),
        "preregistered": False,
        "verdict": "descriptive",
        "evidence_grade": record.get("evidence_grade"),
        "record": {"path": record_path, "sha256": record_sha256},
    }
    if "levels" in record:
        card["levels"] = [
            {
                "concurrency": level.get("concurrency"),
                "baseline_p95_ms": (level["arms"]["baseline"].get("successful_workflow_latency_ms") or {}).get("p95"),
                "candidate_p95_ms": (level["arms"]["candidate"].get("successful_workflow_latency_ms") or {}).get("p95"),
                "paired_mean_difference_ms": (level.get("paired_successful_latency_difference_ms") or {}).get("mean"),
                "relative": (level.get("paired_successful_latency_difference_ms") or {}).get("relative_to_baseline"),
            }
            for level in record["levels"]
        ]
    if "arms" in record:
        card["design"] = record.get("design")
        card["arms"] = {
            name: {
                "mean_cached_tokens": arm.get("mean_cached_tokens"),
                "mean_input_tokens": arm.get("mean_input_tokens"),
                "server_ttft_p50_ms": (arm.get("server_ttft_ms") or {}).get("p50"),
                "task_success_rate": arm.get("task_success_rate"),
            }
            for name, arm in record["arms"].items()
        }
    return card


def build_cards(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    cards = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        digest = _sha256(path)
        result_type = record.get("result_type")
        if result_type in PROOF_RESULT_TYPES:
            cards.append(proof_card(record, relative, digest))
        elif result_type in STUDY_RESULT_TYPES:
            cards.append(study_card(record, relative, digest))
        else:
            raise ValueError(f"{relative}: unsupported result_type {result_type!r}")
    order = {"proof": 0, "study": 1}
    return sorted(cards, key=lambda card: (order[card["kind"]], card["id"]))


def _fmt(value: Any, suffix: str = "", digits: int = 0) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _prose(text: str) -> str:
    """Render record text in the site's prose style without changing its meaning."""

    return str(text).replace("; ", ", ")


def render_markdown(cards: list[Mapping[str, Any]], repository: str) -> str:
    lines = [
        "# Verified decision cards",
        "",
        "Each card restates one frozen RunEconomy decision record. Proof cards",
        "were preregistered and executed once. Study cards are descriptive and",
        "carry no verdict. Every number links back to its record and digest.",
        "Latency is measured on successful workflows only, so a regression card",
        "can show lower latency for the attempts that still succeeded.",
        "",
        "| Card | Intervention | Verdict | Task success (base → cand.) | p95 successful-workflow latency (ms) | $ per 1k successful |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for card in cards:
        if card["kind"] != "proof":
            continue
        success = card["task_success"]
        latency = card["successful_workflow_latency_ms"]
        cost = card.get("usd_per_1k_successful_workflows")
        cost_text = "not gated" if cost is None else f"{_fmt(cost['baseline'], digits=4)} → {_fmt(cost['candidate'], digits=4)}"
        lines.append(
            f"| [{card['title']}]({repository}/blob/main/{card['record']['path']}) | {card['intervention']} | "
            f"**{card['verdict']}** | {_fmt(success['baseline'] * 100, '%')} → {_fmt(success['candidate'] * 100, '%')} | "
            f"{_fmt(latency['baseline']['p95'])} → {_fmt(latency['candidate']['p95'])} | {cost_text} |"
        )
    for card in cards:
        if card["kind"] != "proof":
            continue
        env = card["environment"]
        lines += [
            "",
            f"## {card['title']}: {card['verdict']}",
            "",
            f"- **Intervention:** {card['intervention']}",
            f"- **Environment:** {env['engine']} {env['engine_version']} on {env['gpu']}"
            f"\n- **Workload:** {_fmt(card['workload']['pairs'])} pairs, {card['workload']['arrival_pattern']} at concurrency {_fmt(card['workload']['concurrency'])}",
            f"- **Decisive gate:** `{card['decisive_gate']}`. {_prose(card['reason'])}",
            f"- **Record:** [`{card['record']['path']}`]({repository}/blob/main/{card['record']['path']}) `{card['record']['sha256']}`",
            "",
            "Claim boundary:",
            "",
            *[f"- {_prose(item)}" for item in card["claim_boundary"]],
        ]
    studies = [card for card in cards if card["kind"] == "study"]
    if studies:
        lines += ["", "## Descriptive studies", ""]
        for card in studies:
            lines.append(
                f"- [{card['title']}]({repository}/blob/main/{card['record']['path']}): {_prose(card['evidence_grade'])}."
            )
    return "\n".join(lines) + "\n"
