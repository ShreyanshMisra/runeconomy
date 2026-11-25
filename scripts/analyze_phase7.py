"""Summarize a Phase 6/7 window archive and derive the held-out calibration record.

Every number here is computed from the attempt records in each
``result.json``. Nothing is copied from a decision's prose. The calibration
record sizes the held-out runs only from calibration evidence.

    uv run python scripts/analyze_phase7.py summarize artifacts/phase-7-calibration.tar.gz
    uv run python scripts/analyze_phase7.py calibration-record artifacts/phase-7-calibration.tar.gz \\
        --output artifacts/phase-7-calibration-record.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path
from typing import Any

from runeconomy.analysis import analyze_trace
from runeconomy.detectors import diagnose_result
from runeconomy.models import Trace


NON_COMPLETION_INFRASTRUCTURE = {"infrastructure_failure", "timeout"}
NON_COMPLETION_APPLICATION = {"application_failure", "model_output_failure"}


def _members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, mode="r:gz") as handle:
        return {
            member.name: handle.extractfile(member).read()  # type: ignore[union-attr]
            for member in handle.getmembers()
            if member.isfile()
        }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _arm(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for attempt in attempts:
        statuses[attempt["status"]] = statuses.get(attempt["status"], 0) + 1
    evaluated = [a for a in attempts if a.get("task_success") is not None]
    successful = [a for a in attempts if a.get("task_success") is True]
    latency = [float(a["workflow_latency_ms"]) for a in successful if a.get("workflow_latency_ms") is not None]
    costs = [a["economics"]["cost_usd"] for a in attempts if isinstance(a.get("economics"), dict)
             and a["economics"].get("cost_usd") is not None]

    def mean_metric(name: str) -> float | None:
        values = [a["metrics"][name] for a in attempts if isinstance(a.get("metrics"), dict)
                  and isinstance(a["metrics"].get(name), (int, float))]
        return statistics.fmean(values) if values else None

    return {
        "attempts": len(attempts),
        "status_counts": dict(sorted(statuses.items())),
        "task_success_rate": len(successful) / len(evaluated) if evaluated else None,
        "infrastructure_failure_rate": sum(statuses.get(s, 0) for s in NON_COMPLETION_INFRASTRUCTURE) / len(attempts),
        "application_failure_rate": sum(statuses.get(s, 0) for s in NON_COMPLETION_APPLICATION) / len(attempts),
        "successful_workflow_latency_ms": {
            "p50": _percentile(latency, 0.5), "p95": _percentile(latency, 0.95),
            "mean": statistics.fmean(latency) if latency else None,
        },
        "mean_input_tokens": mean_metric("input_tokens"),
        "mean_output_tokens": mean_metric("output_tokens"),
        "mean_cached_input_tokens": mean_metric("cached_input_tokens"),
        "mean_cost_usd_per_attempt": statistics.fmean(costs) if costs else None,
    }


def _paired(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, int], dict[str, bool | None]] = {}
    for attempt in attempts:
        cells.setdefault((attempt["task_id"], attempt["repetition"]), {})[attempt["variant"]] = attempt.get("task_success")
    complete = [cell for cell in cells.values() if cell.get("baseline") is not None and cell.get("candidate") is not None]
    discordant = sum(cell["baseline"] != cell["candidate"] for cell in complete)
    latency_pairs = []
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for attempt in attempts:
        if attempt.get("task_success") is True and attempt.get("workflow_latency_ms") is not None:
            by_key.setdefault((attempt["task_id"], attempt["repetition"]), {})[attempt["variant"]] = float(attempt["workflow_latency_ms"])
    for values in by_key.values():
        if "baseline" in values and "candidate" in values:
            latency_pairs.append(values["candidate"] - values["baseline"])
    return {
        "pairs": len(complete),
        "discordance_rate": discordant / len(complete) if complete else None,
        "success_difference": (
            (sum(bool(c["candidate"]) for c in complete) - sum(bool(c["baseline"]) for c in complete)) / len(complete)
            if complete else None
        ),
        "successful_pair_latency_difference_ms": {
            "mean": statistics.fmean(latency_pairs) if latency_pairs else None,
            "sd": statistics.stdev(latency_pairs) if len(latency_pairs) > 1 else None,
            "n": len(latency_pairs),
        },
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    attempts = result.get("attempts") or result.get("trials") or []
    population = [a for a in attempts if not a.get("warmup") and (a.get("load") or {}).get("role", "population") != "filler"]
    decision = result["decision"]
    return {
        "verdict": decision["status"],
        "reason": decision.get("reason"),
        "decisive_gate": decision.get("decisive_gate"),
        "required_checks": {
            check["name"]: check["passed"] for check in decision.get("checks", []) if check.get("required", True)
        },
        "baseline": _arm([a for a in population if a.get("variant") == "baseline"]),
        "candidate": _arm([a for a in population if a.get("variant") == "candidate"]),
        "paired": _paired(population) if "task_id" in (population[0] if population else {}) else None,
        "preregistration": result.get("preregistration"),
        "kv_pressure_findings": [f["summary"] for f in diagnose_result(result)["findings"]],
    }


def baseline_trace_rules(members: dict[str, bytes], experiment: str) -> dict[str, int]:
    """Count detector findings on the baseline arm's own normalized traces."""

    counts: dict[str, int] = {}
    prefix = f"results/{experiment}/artifacts/"
    for name, content in members.items():
        if not (name.startswith(prefix) and name.endswith("normalized-trace.json") and "-baseline-" in name):
            continue
        payload = json.loads(content)
        report = analyze_trace(Trace.from_dict(payload), dependency_evidence=payload.get("dependency_evidence"))
        for rule in {finding["rule_id"] for finding in report["findings"]}:
            counts[rule] = counts.get(rule, 0) + 1
    return counts


def summarize_archive(archive: Path) -> dict[str, Any]:
    members = _members(archive)
    record = json.loads(members["results/window-record.json"])
    summaries: dict[str, Any] = {}
    for name in sorted(members):
        parts = name.split("/")
        if len(parts) == 3 and parts[2] == "result.json":
            experiment = parts[1]
            summaries[experiment] = {
                **summarize_result(json.loads(members[name])),
                "result_sha256": "sha256:" + hashlib.sha256(members[name]).hexdigest(),
                "baseline_trace_findings": baseline_trace_rules(members, experiment),
            }
    return {
        "archive_sha256": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
        "window": record,
        "experiments": summaries,
    }


def calibration_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Planning inputs for the held-out designs, from calibration evidence only."""

    def planning(key: str) -> dict[str, Any]:
        item = summary["experiments"][key]
        baseline, paired = item["baseline"], item["paired"]
        infrastructure = max(item["baseline"]["infrastructure_failure_rate"], item["candidate"]["infrastructure_failure_rate"])
        application = max(item["baseline"]["application_failure_rate"], item["candidate"]["application_failure_rate"])
        cost = statistics.fmean([value for value in (item["baseline"]["mean_cost_usd_per_attempt"],
                                                     item["candidate"]["mean_cost_usd_per_attempt"]) if value is not None])
        return {
            "schema_version": "0.2",
            "target_power": 0.8,
            "success_assumptions": {
                "source": "calibration",
                "artifact_sha256": item["result_sha256"],
                # A perfect calibration arm is not a perfect true rate: cap at one
                # failure across both arms so the power plan is not degenerate.
                "expected_success_rate": round(
                    min(baseline["task_success_rate"],
                        1 - 1 / max(item["baseline"]["attempts"] + item["candidate"]["attempts"], 1)), 4
                ),
                # A calibration sample cannot show discordance below one pair in
                # its size; floor it so the plan is not optimistic.
                "expected_discordance_rate": round(max(paired["discordance_rate"], 1 / max(paired["pairs"], 1)), 4),
                "expected_difference": round(paired["success_difference"], 4),
            },
            "completion_model": {
                "source": "calibration",
                "artifact_sha256": item["result_sha256"],
                "infrastructure_failure_rate": round(max(infrastructure, 0.005), 4),
                # Zero observed failures is not a zero rate: floor at one failure
                # across both calibration arms.
                "application_failure_rate": round(
                    max(application, 1 / max(item["baseline"]["attempts"] + item["candidate"]["attempts"], 1)), 4
                ),
                "pass_probability": 0.95,
            },
            "cost_assumptions": {"source": "calibration", "usd_per_attempt": round(cost, 6), "fixed_usd": 0.0},
            "latency_assumptions": {
                "expected_difference_ms": round(paired["successful_pair_latency_difference_ms"]["mean"], 1),
                "paired_standard_deviation_ms": round(paired["successful_pair_latency_difference_ms"]["sd"], 1),
            },
        }

    return {
        "artifact_type": "runeconomy.phase7_calibration_record",
        "calibration_archive_sha256": summary["archive_sha256"],
        "proof3": {"planning": planning("p3-calibration"), "margin": 0.05},
        "proof4": {"planning": planning("p4-calibration")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("summarize", "calibration-record"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_archive(args.archive)
    payload = summary if args.command == "summarize" else calibration_record(summary)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
