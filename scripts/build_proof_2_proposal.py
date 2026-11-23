"""Build an unfrozen Proof 2 contract proposal from private calibration evidence."""

from __future__ import annotations

import ast
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from runeconomy.planning import plan_experiment
from runeconomy.workspaces import create_agent_workspace


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "artifacts" / "proof-2-preparation"
CALIBRATION = ROOT / "artifacts" / "proof-2-calibration-extracted"
CALIBRATION_ARCHIVE = ROOT / "artifacts" / "proof-2-calibration.tar.gz"
OUTPUT = ROOT / "artifacts" / "proof-2-proposed-contract"
PROPOSED_TASKS = 180
TASKS_PER_FAMILY = 90


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _answers(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ANSWERS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict) and all(
                isinstance(key, str) and isinstance(answer, str)
                for key, answer in value.items()
            ):
                return value
    raise RuntimeError("trusted generated evaluator has no literal ANSWERS map")


def _evaluator_source(answers: dict[str, str]) -> str:
    encoded = json.dumps(answers, indent=2, sort_keys=True)
    return f'''"""Proposed, unfrozen deterministic evaluator for Proof 2."""
import json
import sys

ANSWERS = {encoded}

request = json.load(sys.stdin)
task_id = request.get("task_id")
if task_id not in ANSWERS:
    raise ValueError("unknown proposed Proof 2 task")
observed = str(request.get("output_text", "")).strip()
success = observed == ANSWERS[task_id]
json.dump({{"success": success, "score": float(success), "details": {{}}}}, sys.stdout)
'''


def _calibration_statistics() -> dict[str, Any]:
    run = CALIBRATION / "run"
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    attempts: dict[tuple[str, str], dict[str, Any]] = {}
    families: dict[str, str] = {}
    for attempt in result["attempts"]:
        task = json.loads(
            (run / attempt["artifacts"]["directory"] / "task.json").read_text(
                encoding="utf-8"
            )
        )
        attempts[(task["task_id"], attempt["variant"])] = attempt
        families[task["task_id"]] = task["family"]

    differences: list[float] = []
    baseline: list[float] = []
    candidate: list[float] = []
    family_differences: dict[str, list[float]] = {}
    for task_id, family in families.items():
        base = float(attempts[(task_id, "baseline")]["workflow_latency_ms"])
        cand = float(attempts[(task_id, "candidate")]["workflow_latency_ms"])
        baseline.append(base)
        candidate.append(cand)
        differences.append(cand - base)
        family_differences.setdefault(family, []).append(cand - base)

    return {
        "pair_count": len(differences),
        "baseline_successes": result["summary"]["variants"]["baseline"][
            "task_success_count"
        ],
        "candidate_successes": result["summary"]["variants"]["candidate"][
            "task_success_count"
        ],
        "baseline_mean_workflow_latency_ms": statistics.fmean(baseline),
        "candidate_mean_workflow_latency_ms": statistics.fmean(candidate),
        "candidate_minus_baseline_mean_ms": statistics.fmean(differences),
        "candidate_minus_baseline_median_ms": statistics.median(differences),
        "paired_standard_deviation_ms": statistics.stdev(differences),
        "candidate_minus_baseline_95_ci_ms": result["summary"]["effects"][
            "workflow_latency_ms"
        ]["confidence_interval"],
        "client_ttft_difference_ms": result["summary"]["effects"]["client_ttft_ms"][
            "candidate_minus_baseline_mean"
        ],
        "family": {
            family: {
                "pair_count": len(values),
                "candidate_minus_baseline_mean_ms": statistics.fmean(values),
                "paired_standard_deviation_ms": statistics.stdev(values),
            }
            for family, values in sorted(family_differences.items())
        },
    }


def build(output: Path = OUTPUT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite proposed contract: {output}")
    output.mkdir(parents=True, exist_ok=True)

    draft = json.loads(
        (PREPARATION / "draft-heldout-tasks.json").read_text(encoding="utf-8")
    )["tasks"]
    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for task in draft:
        family = task["family"]
        if family_counts[family] < TASKS_PER_FAMILY:
            selected.append(task)
            family_counts[family] += 1
    if len(selected) != PROPOSED_TASKS or set(family_counts.values()) != {
        TASKS_PER_FAMILY
    }:
        raise RuntimeError("draft pool cannot support the balanced proposed sample")

    all_answers = _answers(PREPARATION / "draft-heldout-evaluator.py")
    selected_answers = {
        task["task_id"]: all_answers[task["task_id"]] for task in selected
    }
    dataset = output / "tasks.json"
    evaluator = output / "evaluator.py"
    dataset.write_text(
        json.dumps({"tasks": selected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluator.write_text(_evaluator_source(selected_answers), encoding="utf-8")

    workspace = output / "workspace"
    create_agent_workspace(
        output=workspace,
        baseline_command="python /root/proof-2/agent.py",
        candidate_command="python /root/proof-2/agent.py",
        dataset_path=dataset,
        evaluator_path=evaluator,
        repetitions=1,
        order="randomized",
        seed=20261123,
        max_task_success_regression=0.05,
        min_task_success_rate=0.90,
        optimization_target="engine_change_latency",
        latency_semantics="superiority",
        max_latency_regression_ms=0.0,
    )
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["design"] = {
        "planning": {
            "schema_version": "0.1",
            "hypothesis": "Online FP8 reduces successful-workflow latency without unacceptable task-quality loss.",
            "intervention": "Change only the pinned vLLM backend from BF16 to online fp8_per_tensor.",
            "predicted_direction": "decrease",
            "claim": {
                "level": "workflow",
                "primary_metric": "workflow_latency_ms",
            },
            "calibration": {"artifact_sha256": _digest(CALIBRATION_ARCHIVE)},
            "heldout": {
                "artifact_sha256": _digest(dataset),
                "operator_declared_unseen": True,
            },
            "evaluator_frozen": False,
            "gates_frozen": False,
            "stopping_rule": {
                "kind": "fixed_sample",
                "favorable_result_dependent": False,
            },
            "claim_boundary": "One pinned Qwen2.5-14B repository agent workload on vLLM 0.34.0 and two L40S GPUs.",
            "intervention_variables": ["backend_identity"],
            "identity_change": True,
            "measurement_scope": {
                "tool_time": True,
                "concurrency": False,
                "routing": False,
                "production_behavior": False,
                "live_agent": True,
            },
            "latency_assumptions": {
                "expected_difference_ms": -50.0,
                "paired_standard_deviation_ms": _calibration_statistics()[
                    "paired_standard_deviation_ms"
                ],
            },
        }
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan = plan_experiment(workspace)
    (output / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "proposed_unfrozen",
        "heldout_executed": False,
        "proposed_task_count": len(selected),
        "family_counts": dict(sorted(family_counts.items())),
        "dataset_sha256": _digest(dataset),
        "evaluator_sha256": _digest(evaluator),
        "calibration_archive_sha256": _digest(CALIBRATION_ARCHIVE),
        "calibration": _calibration_statistics(),
        "planner": plan,
    }
    (output / "proposal-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = build()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
