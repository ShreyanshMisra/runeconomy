"""Rebuild the sanitized bundled demo from the ignored raw measurement.

This is a maintainer tool. It deliberately excludes trials, prompt token IDs,
request content, output digests, and generated text from the package artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from runeconomy.experiments import (
    ExperimentManifest,
    _build_decision,
    _build_summary,
    _build_workflow_summary,
)

ROOT = Path(__file__).parents[1]
DEFAULT_RAW = ROOT / "reports" / "repository-agent-tool-pruning-modal.json"
DEFAULT_MANIFEST = (
    ROOT
    / "examples"
    / "experiments"
    / "repository-agent-tool-pruning-recorded.json"
)
DEFAULT_OUTPUT = (
    ROOT / "src" / "runeconomy" / "data" / "repository-agent-tool-pruning.json"
)


def build(raw_path: Path, manifest_path: Path) -> dict:
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    manifest = ExperimentManifest.from_path(manifest_path)
    trials = copy.deepcopy(raw["trials"])
    for trial in trials:
        if (
            trial.get("client_ttft_ms") is not None
            and trial.get("server_ttft_ms") is not None
        ):
            trial["client_server_ttft_gap_ms"] = (
                trial["client_ttft_ms"] - trial["server_ttft_ms"]
            )

    summary = _build_summary(trials, manifest.execution.seed)
    summary["workflow"] = _build_workflow_summary(
        trials, manifest.execution.seed, manifest.cases
    )
    decision = _build_decision(summary["workflow"], manifest.acceptance)
    decision = {
        **decision,
        "status": "inconclusive",
        "reason": (
            "the latency and token effects are measured, but the historical "
            "workload did not use a held-out or predeclared tool allowlist and "
            "the paired success bound does not establish a zero-regression margin"
        ),
        "checks": [
            {
                "name": "evaluation_design_integrity",
                "passed": False,
                "required": True,
                "observed": {
                    "candidate_selection": "post_hoc_same_workload",
                    "held_out_evaluation": False,
                },
                "threshold": "predeclared candidate or separate calibration workload",
            },
            *decision["checks"],
        ],
    }

    provenance = copy.deepcopy(raw["provenance"])
    provenance["legacy_manifest_sha256"] = provenance.pop("manifest_sha256")
    provenance["reproducibility_manifest_sha256"] = (
        manifest.reproducibility_fingerprint()
    )
    provenance["recorded_manifest"] = (
        "examples/experiments/repository-agent-tool-pruning-recorded.json"
    )
    provenance["acceptance_review"] = {
        **asdict(manifest.acceptance),
        "review_timing": "post_measurement_methodology repair",
    }
    provenance["environment_identity"] = {
        "verified": False,
        "reason": (
            "the historical runner recorded declared pins but did not query and "
            "compare live identity before and after the run"
        ),
    }

    workflow = summary["workflow"]
    return {
        "result_schema_version": "0.3",
        "created_at": raw["created_at"],
        "experiment_id": raw["experiment_id"],
        "provenance": provenance,
        "summary": summary,
        "decision": decision,
        "measurement_context": {
            "workflow_step_count": 2,
            "measured_workflow_pair_count": 15,
            "measured_inference_call_count": 60,
            "warmup_workflow_pair_count": 2,
            "warmup_inference_call_count": 8,
            "baseline_tool_count": 12,
            "candidate_tool_count": 3,
            "candidate_tool_names": [
                "list_repository",
                "read_file",
                "search_repository",
            ],
            "changed_request_fields": ["tools"],
            "evaluator_kind": (
                "two deterministic regex evaluators; workflow success required "
                "both to pass"
            ),
            "candidate_selection": "post_hoc_same_workload",
            "held_out_evaluation": False,
            "primary_outcome": "total_latency_ms",
            "mechanism_metric": "server_ttft_ms",
            "client_timing_metric": "client_ttft_ms",
            "mean_client_server_ttft_gap_ms": {
                variant: workflow["variants"][variant][
                    "client_server_ttft_gap_ms"
                ]["mean"]
                for variant in ("baseline", "candidate")
            },
        },
        "sanitization": {
            "source": "measured Modal/vLLM experiment",
            "raw_result_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_trial_count": len(raw["trials"]),
            "raw_trials_included": False,
            "generated_output_included": False,
            "claim": (
                "Aggregate evidence only; raw trials remain local and generated "
                "output was never persisted."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.raw, args.manifest)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
