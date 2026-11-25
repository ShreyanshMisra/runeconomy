"""Build the Proof 3/4 decision records, reports, and reproducibility bundle.

Every number is recomputed from the ``result.json`` files inside the
window archives; nothing is copied from prose. Outputs:

- ``proof-3-decision.json`` and ``proof-4-decision.json``: aggregate decision
  records with the frozen digests, every gate, and the claim boundary
- ``study-1.json`` and ``study-2.json``: descriptive study aggregates
- ``*-report.html``: the standard RunEconomy report for each result
- ``phase-7-reproducibility.tar.gz``: frozen workspaces, inputs, agent,
  repository snapshot, runner and scripts, result files, and the records
- ``SHA256SUMS`` for every output

    uv run python scripts/build_phase7_artifacts.py \\
        --heldout artifacts/phase-7-heldout.tar.gz --heldout-bundle artifacts/phase-7-bundles/heldout \\
        --study2 artifacts/phase-7-study2.tar.gz --study2-bundle artifacts/phase-7-bundles/study2 \\
        --calibration-record artifacts/phase-7-calibration-record.json \\
        --output artifacts/phase-7-publication
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_phase7 import summarize_result  # noqa: E402

from runeconomy.reports import render_experiment_html  # noqa: E402

RECORD_VERSION = "0.1"
SOURCE_FILES = (
    "scripts/build_phase7_workload.py",
    "scripts/prepare_phase7.py",
    "scripts/analyze_phase7.py",
    "scripts/build_phase7_artifacts.py",
    "deploy/modal_phase7.py",
    "examples/lab_agent/agent.py",
)
PROOFS = {
    "p3-heldout": {
        "proof": 3,
        "target": "verified",
        "hypothesis": (
            "Online FP8 lowers p95 successful-workflow latency and cost per successful workflow "
            "under 16-agent load without lowering task success."
        ),
        "intervention": "BF16 weights versus vLLM online fp8_per_tensor quantization (one GPU per variant)",
        "claim_boundary": [
            "One model (Qwen/Qwen2.5-14B-Instruct at a pinned revision), one GPU type (NVIDIA L40S), and vLLM 0.34.0.",
            "One synthetic multi-step repository-lookup agent on a fixed repository snapshot; not production traffic.",
            "16 closed-loop agents per variant, each variant on its own GPU (isolated serving topology).",
            "Cost uses the Modal L40S list price with fair-share attribution by in-flight time; it is not an invoice.",
            "Both arms succeeded on every task, so non-inferiority is shown against a 5-point margin, not equality of quality on harder work.",
        ],
    },
    "p4-heldout": {
        "proof": 4,
        "target": "regression",
        "hypothesis": (
            "Pruning the repository-search tool makes the workflow faster on every request benchmark, "
            "but it lowers task success below the declared floor and margin."
        ),
        "intervention": "Remove the repository-search tool and answer directly (same model and server)",
        "claim_boundary": [
            "One model (Qwen/Qwen2.5-14B-Instruct at a pinned revision) on one vLLM 0.34.0 BF16 server.",
            "One synthetic multi-step repository-lookup agent whose answers need repository evidence.",
            "The regression shows that RunEconomy blocks a change that request benchmarks favor; it is not a claim about tool pruning in general.",
        ],
    },
}


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, mode="r:gz") as handle:
        return {
            member.name: handle.extractfile(member).read()  # type: ignore[union-attr]
            for member in handle.getmembers()
            if member.isfile()
        }


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def _gate_rows(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for check in decision.get("checks", []):
        observed = check.get("observed")
        if check.get("name") == "trace_and_environment_identity" and isinstance(observed, Mapping):
            # Keep the verdict-relevant counts, not the full identity strings.
            observed = {
                variant: {
                    key: value.get(key)
                    for key in ("complete_dag_count", "complete_dag_rate", "environment_identity_complete")
                }
                for variant, value in observed.items()
                if isinstance(value, Mapping)
            }
        rows.append(_round({
            "name": check.get("name"),
            "required": check.get("required", True),
            "passed": check.get("passed"),
            "observed": observed,
            "threshold": check.get("threshold"),
        }))
    return rows


def _environment(result: Mapping[str, Any], window: Mapping[str, Any], members: Mapping[str, bytes]) -> dict[str, Any]:
    stage = window["stages"][0]
    servers = {
        name: {
            key: spec.get(key)
            for key in ("model", "served_model", "quantization", "max_model_len", "max_num_seqs",
                        "gpu_memory_utilization", "gpu")
        }
        for name, spec in stage["servers"].items()
    }
    return {
        "vllm_version": window.get("vllm_version"),
        "chat_template_digest": window.get("chat_template_digest"),
        "gpu": window.get("gpu"),
        "gpu_inventory": [{"index": item["index"], "name": item["name"]} for item in window.get("gpu_inventory", [])],
        "servers": servers,
        "runeconomy_version": result.get("provenance", {}).get("runeconomy_version"),
    }


def _preregistration(bundle: Path, workspace: str, result: Mapping[str, Any]) -> dict[str, Any]:
    record_path = bundle / workspace / "preregistration.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    locks = record["locks"]
    embedded = result.get("preregistration") or {}
    record_sha = _sha256_bytes(record_path.read_bytes())
    if embedded.get("record_sha256") != record_sha or embedded.get("status") != "intact":
        raise SystemExit(f"{workspace}: the result does not carry this intact freeze record")
    return {
        "record_sha256": record_sha,
        "frozen_at": record["frozen_at"],
        "status_at_execution": embedded["status"],
        "design_completeness": record["design_completeness"],
        "manifest_sha256": locks["manifest"]["sha256"],
        "gates_sha256": locks["gates"]["sha256"],
        "execution_sha256": locks["execution"]["sha256"],
        "dataset_sha256": locks["inputs"]["dataset"]["sha256"],
        "evaluator_sha256": locks["inputs"]["evaluator"]["sha256"],
        "arrival_schedule_sha256": (locks["execution"]["value"].get("load") or {}).get("schedule_sha256"),
        "additional_files": locks.get("additional_files", []),
        "gates": locks["gates"]["value"],
    }


def proof_record(
    key: str,
    result: Mapping[str, Any],
    result_bytes: bytes,
    archive: Path,
    window: Mapping[str, Any],
    bundle: Path,
    workspace: str,
    calibration: Mapping[str, Any],
    members: Mapping[str, bytes],
) -> dict[str, Any]:
    meta = PROOFS[key]
    summary = summarize_result(result)
    decision = result["decision"]
    outcome = decision["status"]
    planning = calibration[f"proof{meta['proof']}"]
    return {
        "schema_version": RECORD_VERSION,
        "result_type": "runeconomy.phase7_proof_decision",
        "proof": meta["proof"],
        "publication_status": "private_unpublished",
        "target_outcome": meta["target"],
        "outcome_matches_target": outcome == meta["target"],
        "hypothesis": meta["hypothesis"],
        "intervention": meta["intervention"],
        "decision": {
            "status": outcome,
            "reason": decision.get("reason"),
            "decisive_gate": (decision.get("decisive_gate") or {}).get("name"),
            "decisive_explanation": (decision.get("decisive_gate") or {}).get("explanation"),
            "rerun_permitted": False,
        },
        "preregistration": _preregistration(bundle, workspace, result),
        "calibration": {
            "archive_sha256": calibration["calibration_archive_sha256"],
            "planned_pairs": planning.get("pairs"),
            "assumptions": planning.get("planning"),
        },
        "execution": {
            "pairs": (summary.get("paired") or {}).get("pairs"),
            "attempts_per_variant": summary["baseline"]["attempts"],
            "load": result.get("design", {}).get("execution_load"),
            "cost_model": result.get("design", {}).get("cost_model"),
            "remote_window_seconds": round(window.get("remote_elapsed_seconds", 0.0), 1),
            "window_cost_usd_estimate": round(window.get("projected_cost_usd", 0.0), 2),
        },
        "environment": _environment(result, window, members),
        "gates": _gate_rows(decision),
        "arms": _round({"baseline": summary["baseline"], "candidate": summary["candidate"]}),
        "paired": _round(summary["paired"]),
        "effects": _round(result.get("summary", {}).get("effects", {})),
        "claim_boundary": meta["claim_boundary"],
        "provenance": {
            "private_raw_archive_sha256": _sha256_bytes(archive.read_bytes()),
            "result_sha256": _sha256_bytes(result_bytes),
            "contains_task_level_data": False,
        },
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def study1_record(result: Mapping[str, Any], result_bytes: bytes, archive: Path) -> dict[str, Any]:
    """Per-concurrency-level FP8-vs-BF16 effects from the sweep's attempts."""

    population = [
        attempt for attempt in result["attempts"]
        if not attempt.get("warmup") and (attempt.get("load") or {}).get("role", "population") != "filler"
    ]
    levels = []
    for level in sorted({attempt["load"]["level"] for attempt in population}):
        at_level = [attempt for attempt in population if attempt["load"]["level"] == level]
        arms: dict[str, Any] = {}
        for variant in ("baseline", "candidate"):
            attempts = [attempt for attempt in at_level if attempt["variant"] == variant]
            successes = [attempt for attempt in attempts if attempt.get("task_success") is True]
            latency = [float(attempt["workflow_latency_ms"]) for attempt in successes]
            costs = [attempt["economics"]["cost_usd"] for attempt in attempts
                     if isinstance(attempt.get("economics"), Mapping) and attempt["economics"].get("cost_usd") is not None]
            arms[variant] = {
                "attempts": len(attempts),
                "successes": len(successes),
                "successful_workflow_latency_ms": {
                    "p50": _percentile(latency, 0.5), "p95": _percentile(latency, 0.95),
                    "mean": statistics.fmean(latency) if latency else None,
                },
                "usd_per_1k_successful_workflows": (
                    1000 * sum(costs) / len(successes) if successes and len(costs) == len(attempts) else None
                ),
            }
        pairs: dict[tuple[str, int], dict[str, float]] = {}
        for attempt in at_level:
            if attempt.get("task_success") is True:
                pairs.setdefault((attempt["task_id"], attempt["repetition"]), {})[attempt["variant"]] = float(
                    attempt["workflow_latency_ms"]
                )
        complete = [cell for cell in pairs.values() if len(cell) == 2]
        differences = [cell["candidate"] - cell["baseline"] for cell in complete]
        baseline_mean = statistics.fmean(cell["baseline"] for cell in complete) if complete else None
        sd = statistics.stdev(differences) if len(differences) > 1 else None
        half_width = None if sd is None else 1.96 * sd / math.sqrt(len(differences))
        mean_difference = statistics.fmean(differences) if differences else None
        levels.append(_round({
            "concurrency": level,
            "arms": arms,
            "paired_successful_latency_difference_ms": {
                "mean": mean_difference,
                "approximate_95_ci": (
                    None if half_width is None or mean_difference is None
                    else [mean_difference - half_width, mean_difference + half_width]
                ),
                "relative_to_baseline": (
                    None if baseline_mean in (None, 0) or mean_difference is None else mean_difference / baseline_mean
                ),
                "pairs": len(differences),
            },
            "server_prefix_cache_hit_rate_in_block": {
                target: values.get("prefix_cache_hit_rate_in_block")
                for block in result.get("load_blocks", []) if block.get("level") == level
                for target, values in ((block.get("telemetry") or {}).get("server_metrics") or {}).get("targets", {}).items()
            },
        }, 3))
    return {
        "schema_version": RECORD_VERSION,
        "result_type": "runeconomy.phase7_study",
        "study": 1,
        "title": "FP8 vs BF16 on a multi-step agent under load",
        "publication_status": "private_unpublished",
        "evidence_grade": "descriptive (not preregistered; the gate verdict at the top level is reported, not claimed)",
        "gate_level_verdict": result["decision"]["status"],
        "levels": levels,
        "interval_note": "per-level intervals are normal approximations for description, not preregistered gates",
        "provenance": {
            "private_raw_archive_sha256": _sha256_bytes(archive.read_bytes()),
            "result_sha256": _sha256_bytes(result_bytes),
        },
    }


def study2_record(
    result: Mapping[str, Any], result_bytes: bytes, archive: Path, superseded: Mapping[str, Any] | None
) -> dict[str, Any]:
    trials = [trial for trial in result["trials"] if (trial.get("load") or {}).get("role") == "population"]
    arms: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        rows = [trial for trial in trials if trial.get("variant") == variant]
        cached = [trial["cached_tokens"] for trial in rows if trial.get("cached_tokens") is not None]
        server_ttft = [trial["server_ttft_ms"] for trial in rows if trial.get("server_ttft_ms") is not None]
        client_ttft = [trial["client_ttft_ms"] for trial in rows if trial.get("client_ttft_ms") is not None]
        inputs = [trial["input_tokens"] for trial in rows if trial.get("input_tokens") is not None]
        arms[variant] = _round({
            "requests": len(rows),
            "mean_input_tokens": statistics.fmean(inputs) if inputs else None,
            "mean_cached_tokens": statistics.fmean(cached) if cached else None,
            "cache_coverage": (sum(cached) / sum(inputs)) if cached and inputs and len(cached) == len(inputs) else None,
            "server_ttft_ms": {"p50": _percentile(server_ttft, 0.5), "p95": _percentile(server_ttft, 0.95),
                               "mean": statistics.fmean(server_ttft) if server_ttft else None},
            "client_ttft_ms": {"p50": _percentile(client_ttft, 0.5), "p95": _percentile(client_ttft, 0.95),
                               "mean": statistics.fmean(client_ttft) if client_ttft else None},
            "task_success_rate": (
                sum(1 for trial in rows if (trial.get("evaluation") or {}).get("success") is True) / len(rows)
                if rows else None
            ),
        }, 3)
    return {
        "schema_version": RECORD_VERSION,
        "result_type": "runeconomy.phase7_study",
        "study": 2,
        "title": "What a dynamic value at the start of a prompt prefix costs in vLLM prefix-cache reuse",
        "publication_status": "private_unpublished",
        "evidence_grade": "descriptive (not preregistered)",
        "design": (
            "48 routing requests share one ~1.8K-token system prompt. The baseline puts a per-request ID and "
            "timestamp first; the candidate moves it after the stable prefix. Same model, server, and sampling; "
            "8 closed-loop clients; 5 repetitions."
        ),
        "decision": {
            "status": result["decision"]["status"],
            "reason": result["decision"].get("reason"),
            "decisive_gate": (result["decision"].get("decisive_gate") or {}).get("name"),
        },
        "gates": _gate_rows(result["decision"]),
        "arms": arms,
        "effects": _round(result.get("summary", {}).get("effects", {}), 3),
        "superseded_execution": superseded,
        "provenance": {
            "private_raw_archive_sha256": _sha256_bytes(archive.read_bytes()),
            "result_sha256": _sha256_bytes(result_bytes),
        },
    }


def _report(result_bytes: bytes, output: Path, *, descriptive_study: bool = False) -> None:
    report = render_experiment_html(json.loads(result_bytes), descriptive_study=descriptive_study)
    output.write_text(report, encoding="utf-8")


def _add_tree(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store":
            archive.add(path, arcname=f"{arcname}/{path.relative_to(source).as_posix()}", recursive=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--heldout-bundle", type=Path, required=True)
    parser.add_argument("--study2", type=Path, required=True)
    parser.add_argument("--study2-bundle", type=Path, required=True)
    parser.add_argument("--calibration-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    heldout = _members(args.heldout)
    window = json.loads(heldout["results/window-record.json"])
    plan = json.loads((args.heldout_bundle / "plan.json").read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_record.read_text(encoding="utf-8"))
    outputs: dict[str, bytes] = {}

    for key, name in (("p3-heldout", "proof-3"), ("p4-heldout", "proof-4")):
        result_bytes = heldout[f"results/{key}/result.json"]
        result = json.loads(result_bytes)
        record = proof_record(key, result, result_bytes, args.heldout, window, args.heldout_bundle,
                              plan["experiments"][key]["workspace"], calibration, heldout)
        outputs[f"{name}-decision.json"] = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        _report(result_bytes, args.output / f"{name}-report.html")

    study1_bytes = heldout["results/study-1/result.json"]
    outputs["study-1.json"] = (json.dumps(
        study1_record(json.loads(study1_bytes), study1_bytes, args.heldout), indent=2, sort_keys=True) + "\n").encode()
    _report(study1_bytes, args.output / "study-1-report.html", descriptive_study=True)

    first_study2 = heldout["results/study-2/result.json"]
    superseded = {
        "result_sha256": _sha256_bytes(first_study2),
        "private_raw_archive_sha256": _sha256_bytes(args.heldout.read_bytes()),
        "status": json.loads(first_study2)["decision"]["status"],
        "why_superseded": (
            "Measurement defect, not an outcome: the runner launched vLLM without per-request usage details "
            "and metrics, so cached tokens and server TTFT were unmeasured, and 144 requests per variant was "
            "below the p95 gate's 200-observation minimum. The runner and planner were fixed before the rerun."
        ),
    }
    study2 = _members(args.study2)
    study2_bytes = study2["results/study-2/result.json"]
    outputs["study-2.json"] = (json.dumps(
        study2_record(json.loads(study2_bytes), study2_bytes, args.study2, superseded), indent=2, sort_keys=True,
    ) + "\n").encode()
    _report(study2_bytes, args.output / "study-2-report.html", descriptive_study=True)

    for name, content in outputs.items():
        (args.output / name).write_bytes(content)

    bundle_path = args.output / "phase-7-reproducibility.tar.gz"
    with tarfile.open(bundle_path, mode="w:gz") as archive:
        _add_tree(archive, args.heldout_bundle, "bundles/heldout")
        _add_tree(archive, args.study2_bundle, "bundles/study2")
        for relative in SOURCE_FILES:
            archive.add(ROOT / relative, arcname=f"source/{relative}", recursive=False)
        archive.add(args.calibration_record, arcname="calibration/phase-7-calibration-record.json", recursive=False)
        for key in ("p3-heldout", "p4-heldout", "study-1"):
            info = tarfile.TarInfo(f"results/{key}/result.json")
            info.size = len(heldout[f"results/{key}/result.json"])
            archive.addfile(info, io.BytesIO(heldout[f"results/{key}/result.json"]))
        info = tarfile.TarInfo("results/study-2/result.json")
        info.size = len(study2_bytes)
        archive.addfile(info, io.BytesIO(study2_bytes))
        for name, content in outputs.items():
            info = tarfile.TarInfo(f"records/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    sums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file()
    )
    (args.output / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(sums, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
