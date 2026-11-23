"""Build the aggregate-only public artifact for the frozen held-out proof.

The source comparison, workload, and raw attempt artifacts remain ignored. This
maintainer tool copies only an explicit allowlist of aggregate evidence and
refuses to build when any frozen implementation hash has changed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
DEFAULT_RESULT = ROOT / "artifacts/public-proof-heldout-live-extracted/run/result.json"
DEFAULT_METADATA = (
    ROOT / "artifacts/public-proof-heldout-live-extracted/heldout-metadata.json"
)
DEFAULT_DATASET = (
    ROOT / "artifacts/public-proof-heldout-live-extracted/pilot-tasks.json"
)
DEFAULT_OUTPUT = ROOT / "src/runeconomy/data/public-proof-rejection.json"
DEFAULT_CASE_STUDY = ROOT / "docs/public-proof.md"

FROZEN_FILES = {
    "agent": (
        ROOT / "examples/reliable_proof/agent.py",
        "e6f42c619f15be192512d234ee21b623a09e69402a62ab16516190c9ea348dcb",
    ),
    "driver": (
        ROOT / "deploy/modal_public_proof.py",
        "1a6a589353e5352116a58ee446493fbc83c486f4fc7634a49483ac81b474b1b1",
    ),
    "comparison": (
        ROOT / "src/runeconomy/comparison.py",
        "d8df549ebcb7d814aa1d70b02e2d9c482830e02d7812a40376d1c8161dd31e8f",
    ),
    "status_protocol": (
        ROOT / "src/runeconomy/attempt_status.py",
        "b3885985bfa88510d901600b28e3d6341427b2390b3dedb2c16ccd124e68e723",
    ),
}

# The public proof was executed with this exact comparison implementation.
# Later product changes must not reinterpret the consumed workload.  Read the
# historical Git blob instead of the mutable working-tree module when verifying
# the frozen implementation hash.
FROZEN_COMPARISON_REVISION = "4b04964352c4810a451df04487c7a688ea88702e"
FROZEN_COMPARISON_PATH = "src/runeconomy/comparison.py"

FORBIDDEN_KEYS = {
    "attempts",
    "artifacts",
    "base_url",
    "commands",
    "endpoint",
    "evaluator_stderr",
    "evaluator_stdout",
    "generated_output",
    "output_digest",
    "prompt",
    "question",
    "request",
    "response",
    "stderr",
    "stdout",
    "task_answer",
    "task_id",
    "trace",
}
FORBIDDEN_TEXT = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"(?:^|\s)/(?:Users|home|root|private|tmp)/"),
    re.compile(r"modal\.direct", re.IGNORECASE),
    re.compile(r"authorization\s*:", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._~-]+", re.IGNORECASE),
)


def _sha256(path: Path, *, prefixed: bool = False) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}" if prefixed else digest


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _driver_constants(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {
        "MODEL_NAME",
        "MODEL_REVISION",
        "CHAT_TEMPLATE_DIGEST",
        "VLLM_VERSION",
        "GPU",
        "BLOCK_SIZE",
        "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION",
        "MAX_NUM_SEQS",
    }
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"driver is missing frozen constants: {sorted(missing)}")
    return values


def _metric(effect: Mapping[str, Any]) -> dict[str, Any]:
    interval = effect.get("confidence_interval")
    paired_count = int(effect.get("paired_count", 0) or 0)
    return {
        "available": paired_count > 0
        and effect.get("candidate_minus_baseline_mean") is not None,
        "candidate_minus_baseline_mean": effect.get("candidate_minus_baseline_mean"),
        "confidence_interval": interval,
        "confidence_level": effect.get("confidence_level"),
        "method": effect.get("method"),
        "paired_count": paired_count,
    }


def _check(check: Mapping[str, Any]) -> dict[str, Any]:
    name = str(check["name"])
    observed = check.get("observed")
    if name == "trace_and_environment_identity":
        if not isinstance(observed, Mapping):
            raise ValueError("identity check has invalid observed evidence")
        observed = {
            variant: {
                key: value
                for key, value in observed[variant].items()
                if key
                in {
                    "complete_dag_count",
                    "complete_dag_rate",
                    "environment_identity_complete",
                    "environment_identity_stable",
                    "valid_trace_count",
                }
            }
            for variant in ("baseline", "candidate")
        }
    elif name == "task_success_noninferiority":
        if not isinstance(observed, Mapping):
            raise ValueError("paired success check has invalid observed evidence")
        observed = {
            key: observed.get(key)
            for key in (
                "candidate_minus_baseline_mean",
                "confidence_interval",
                "confidence_level",
                "method",
                "one_sided_lower_bound",
                "outcome_counts",
                "paired_count",
            )
        }
    elif name in {
        "successful_task_workflow_latency_improvement",
        "successful_task_workflow_latency_non_regression",
    }:
        if not isinstance(observed, Mapping):
            raise ValueError("latency check has invalid observed evidence")
        observed = _metric(observed)
    return {
        "name": name,
        "passed": bool(check.get("passed")),
        "threshold": check.get("threshold"),
        "observed": observed,
    }


def validate_public_artifact(payload: Mapping[str, Any]) -> None:
    """Reject fields or strings that would cross the public privacy boundary."""

    expected_top_level = {
        "artifact_kind",
        "artifact_version",
        "claim_boundary",
        "conditional_effects",
        "decision",
        "evidence_integrity",
        "experiment",
        "failure_categories",
        "frozen_hashes",
        "gates",
        "intervention",
        "limitations",
        "outcomes",
        "paired_success",
        "provenance",
        "request_position_effects",
        "sanitization",
    }
    if set(payload) != expected_top_level:
        raise ValueError(
            "public artifact top-level fields differ from the allowlist: "
            f"{sorted(set(payload) ^ expected_top_level)}"
        )

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_KEYS:
                    raise ValueError(f"forbidden public-artifact field: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            for pattern in FORBIDDEN_TEXT:
                if pattern.search(value):
                    raise ValueError(
                        "public artifact contains a forbidden endpoint, credential, "
                        "or local path"
                    )

    walk(payload)
    if payload.get("artifact_kind") != "runeconomy.public_proof":
        raise ValueError("unexpected public artifact kind")
    if payload.get("decision", {}).get("status") != "invalid":
        raise ValueError("the frozen public proof must retain its invalid verdict")


def build(result_path: Path, metadata_path: Path, dataset_path: Path) -> dict[str, Any]:
    result = _load_object(result_path)
    metadata = _load_object(metadata_path)
    dataset = _load_object(dataset_path)
    tasks = dataset.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise ValueError("dataset must contain a tasks array")

    frozen_hashes: dict[str, str] = {}
    for name, (path, expected) in FROZEN_FILES.items():
        if name == "comparison":
            try:
                source = subprocess.run(
                    [
                        "git",
                        "show",
                        f"{FROZEN_COMPARISON_REVISION}:{FROZEN_COMPARISON_PATH}",
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                ).stdout
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ValueError(
                    "cannot read the frozen comparison implementation from Git"
                ) from exc
            observed = hashlib.sha256(source).hexdigest()
        else:
            observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"frozen {name} hash changed: expected {expected}, observed {observed}"
            )
        frozen_hashes[name] = f"sha256:{observed}"

    if result.get("decision", {}).get("status") != "invalid":
        raise ValueError("private result no longer has the frozen invalid verdict")
    if result.get("design", {}).get("task_count") != 40 or len(tasks) != 40:
        raise ValueError("the frozen proof must contain exactly 40 tasks")

    driver = _driver_constants(FROZEN_FILES["driver"][0])
    if metadata.get("model") != driver["MODEL_NAME"]:
        raise ValueError("metadata model does not match the frozen driver")
    if metadata.get("model_revision") != driver["MODEL_REVISION"]:
        raise ValueError("metadata model revision does not match the frozen driver")
    if metadata.get("backend_version") != driver["VLLM_VERSION"]:
        raise ValueError("metadata backend version does not match the frozen driver")
    if metadata.get("chat_template_digest") != driver["CHAT_TEMPLATE_DIGEST"]:
        raise ValueError("metadata chat template does not match the frozen driver")

    attempts = result.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 80:
        raise ValueError("the frozen proof must retain 80 attempt records")
    workflow_structures: set[tuple[int, int]] = set()
    request_cache_by_attempt: dict[tuple[str, str], dict[int, int]] = {}
    for attempt in attempts:
        artifacts = attempt.get("artifacts", {})
        relative_directory = artifacts.get("directory")
        normalized_name = artifacts.get("normalized_trace")
        if not isinstance(relative_directory, str) or not isinstance(
            normalized_name, str
        ):
            raise ValueError("attempt is missing its normalized-trace reference")
        normalized = _load_object(
            result_path.parent / relative_directory / normalized_name
        )
        requests = normalized.get("requests")
        spans = normalized.get("spans")
        if not isinstance(requests, list) or not isinstance(spans, list):
            raise ValueError("normalized trace is missing requests or spans")
        inference_span_count = sum(span.get("kind") == "inference" for span in spans)
        if len(requests) != inference_span_count:
            raise ValueError("request and inference-span counts disagree")
        workflow_structures.add(
            (
                len(requests),
                sum(span.get("kind") == "tool" for span in spans),
            )
        )
        cache_by_sequence: dict[int, int] = {}
        for request in requests:
            sequence = request.get("sequence")
            cached = request.get("metrics", {}).get("cached_input_tokens")
            if not isinstance(sequence, int) or not isinstance(cached, int):
                raise ValueError(
                    "normalized request is missing its sequence or cached-token count"
                )
            cache_by_sequence[sequence] = cached
        if len(cache_by_sequence) != len(requests):
            raise ValueError("normalized request sequences are not unique")
        request_cache_by_attempt[(attempt["task_id"], attempt["variant"])] = (
            cache_by_sequence
        )
    if len(workflow_structures) != 1:
        raise ValueError("attempt workflow structure was not stable")
    inference_calls_per_attempt, tool_spans_per_attempt = next(
        iter(workflow_structures)
    )
    identities = {
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
        for attempt in attempts
        for identity in attempt.get("metrics", {}).get("environment_identity", [])
    }
    if len(identities) != 1:
        raise ValueError("environment identity is missing or unstable")
    identity_text = next(iter(identities))
    frozen_hashes.update(
        {
            "workload": result["design"]["dataset_sha256"],
            "evaluator": result["design"]["evaluator_sha256"],
            "experiment_design": result["design"]["fingerprint"],
            "private_result": _sha256(result_path, prefixed=True),
            "environment_identity": "sha256:"
            + hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
        }
    )

    summary = result["summary"]
    variants = summary["variants"]
    effects = summary["effects"]
    checks = [_check(item) for item in result["decision"]["checks"]]
    checks_by_name = {item["name"]: item for item in checks}
    required_check_names = (
        "trace_and_environment_identity",
        "baseline_minimum_task_success",
        "candidate_minimum_task_success",
        "task_success_noninferiority",
        "baseline_completion",
        "candidate_completion",
        "successful_task_workflow_latency_improvement",
        "successful_task_workflow_latency_non_regression",
    )
    if set(required_check_names) != checks_by_name.keys():
        raise ValueError("private result acceptance checks changed")

    family_counts = dict(sorted(Counter(str(task["family"]) for task in tasks).items()))
    outcomes = {}
    for variant in ("baseline", "candidate"):
        item = variants[variant]
        outcomes[variant] = {
            "attempt_count": item["attempt_count"],
            "completed_rate": item["completed_rate"],
            "task_success_count": item["task_success_count"],
            "task_success_rate": item["task_success_rate"],
            "task_success_95_ci": item["task_success_95_ci"],
        }

    effect_names = (
        "cached_input_tokens",
        "server_ttft_ms",
        "client_ttft_ms",
        "workflow_latency_ms",
        "critical_path_ms",
        "inference_span_ms",
        "tool_span_ms",
        "input_tokens",
        "output_tokens",
        "queue_ms",
        "prefill_ms",
        "decode_ms",
        "retrieval_span_ms",
        "cost_usd",
        "gpu_time_ms",
    )
    conditional_effects = {name: _metric(effects[name]) for name in effect_names}

    attempts_by_task_variant = {
        (attempt["task_id"], attempt["variant"]): attempt for attempt in attempts
    }
    both_successful_tasks = sorted(
        task_id
        for task_id in {attempt["task_id"] for attempt in attempts}
        if attempts_by_task_variant[(task_id, "baseline")]["task_success"] is True
        and attempts_by_task_variant[(task_id, "candidate")]["task_success"] is True
    )
    if len(both_successful_tasks) != effects["workflow_latency_ms"]["paired_count"]:
        raise ValueError("both-successful task count differs from the paired effects")
    position_labels = {0: "Planning call", 1: "Final answer call"}
    request_position_effects = []
    for sequence in range(inference_calls_per_attempt):
        baseline_values = [
            request_cache_by_attempt[(task_id, "baseline")][sequence]
            for task_id in both_successful_tasks
        ]
        candidate_values = [
            request_cache_by_attempt[(task_id, "candidate")][sequence]
            for task_id in both_successful_tasks
        ]
        differences = [
            candidate - baseline
            for baseline, candidate in zip(
                baseline_values, candidate_values, strict=True
            )
        ]
        request_position_effects.append(
            {
                "sequence": sequence,
                "label": position_labels.get(sequence, f"Agent call {sequence + 1}"),
                "baseline_mean": statistics.fmean(baseline_values),
                "candidate_mean": statistics.fmean(candidate_values),
                "candidate_minus_baseline_mean": statistics.fmean(differences),
                "paired_count": len(differences),
            }
        )
    position_total = sum(
        item["candidate_minus_baseline_mean"] for item in request_position_effects
    )
    if not math.isclose(
        position_total,
        conditional_effects["cached_input_tokens"]["candidate_minus_baseline_mean"],
        abs_tol=1e-9,
    ):
        raise ValueError("request-position effects do not sum to the aggregate effect")

    status_versions = sorted(
        {
            str(attempt.get("agent_status", {}).get("schema_version"))
            for attempt in attempts
        }
    )
    acceptance = result["design"]["acceptance"]
    paired_success = effects["task_success"]
    latency_effect = effects["workflow_latency_ms"]
    valid_trace_count = sum(
        variants[name]["trace_evidence"]["valid_trace_count"]
        for name in ("baseline", "candidate")
    )
    complete_dag_count = sum(
        variants[name]["trace_evidence"]["complete_dag_count"]
        for name in ("baseline", "candidate")
    )

    def percentage(value: Any) -> str:
        return f"{float(value) * 100:.2f}%"

    def percentage_points(value: Any) -> str:
        return f"{float(value) * 100:.2f} percentage points"

    artifact: dict[str, Any] = {
        "artifact_kind": "runeconomy.public_proof",
        "artifact_version": "0.3",
        "experiment": {
            "result_type": result["result_type"],
            "result_version": result["result_version"],
            "schema_version": result["result_version"],
            "purpose": metadata["purpose"],
            "task_count": len(tasks),
            "task_family_counts": family_counts,
            "repetitions": result["design"]["repetitions"],
            "order": result["design"]["order"],
            "seed": result["design"]["seed"],
            "primary_metric": result["design"]["acceptance"]["primary_metric"],
            "latency_population": summary["latency_population"],
            "consumed_held_out_workload": True,
            "workflow_structure": {
                "inference_calls_per_attempt": inference_calls_per_attempt,
                "real_tool_spans_per_attempt": tool_spans_per_attempt,
                "tool_kind": "bounded repository search over supplied candidate paths",
            },
        },
        "intervention": {
            "baseline": "dynamic run metadata before the static system instructions",
            "candidate": "the same dynamic run metadata after the static system instructions",
            "changed_property": "system-message block order",
            "held_constant": (
                "agent code, static instructions, dynamic metadata content, model, "
                "backend, tools, evaluator, tasks, and acceptance gates"
            ),
            "expected_mechanism": (
                "keep the static instruction prefix stable so the second inference "
                "call can reuse more cached prompt tokens"
            ),
        },
        "frozen_hashes": frozen_hashes,
        "provenance": {
            "model": driver["MODEL_NAME"],
            "model_revision": driver["MODEL_REVISION"],
            "tokenizer": driver["MODEL_NAME"],
            "tokenizer_revision": driver["MODEL_REVISION"],
            "chat_template_digest": driver["CHAT_TEMPLATE_DIGEST"],
            "backend": "vllm",
            "backend_version": driver["VLLM_VERSION"],
            "api_dialect": "chat_completions",
            "hardware": driver["GPU"],
            "driver_location": result["provenance"]["driver_location"],
            "engine_configuration": {
                "block_size": driver["BLOCK_SIZE"],
                "max_model_len": driver["MAX_MODEL_LEN"],
                "gpu_memory_utilization": driver["GPU_MEMORY_UTILIZATION"],
                "max_num_seqs": driver["MAX_NUM_SEQS"],
            },
        },
        "gates": {
            "predeclared": result["design"]["acceptance"],
            "checks": [checks_by_name[name] for name in required_check_names],
        },
        "outcomes": outcomes,
        "paired_success": {
            key: effects["task_success"].get(key)
            for key in (
                "candidate_minus_baseline_mean",
                "confidence_interval",
                "confidence_level",
                "method",
                "one_sided_lower_bound",
                "outcome_counts",
                "paired_count",
            )
        },
        "conditional_effects": conditional_effects,
        "request_position_effects": request_position_effects,
        "failure_categories": {
            "task_quality_failure": sum(
                item["status_counts"].get("task_failure", 0)
                for item in variants.values()
            ),
            "timeout": 0,
            "process_or_infrastructure_failure": 0,
            "evaluator_failure": 0,
            "invalid_trace": 0,
        },
        "evidence_integrity": {
            "attempt_count": len(attempts),
            "attempt_status_schema_versions": status_versions,
            "complete_dag_count": complete_dag_count,
            "valid_trace_count": valid_trace_count,
            "environment_identity_complete": all(
                variants[name]["trace_evidence"]["environment_identity_complete"]
                for name in ("baseline", "candidate")
            ),
            "environment_identity_stable": all(
                variants[name]["trace_evidence"]["environment_identity_stable"]
                for name in ("baseline", "candidate")
            ),
            "raw_attempts_included": False,
            "generated_content_included": False,
        },
        "decision": {
            "status": "invalid",
            "reason": result["decision"]["reason"],
            "failed_quality_gates": [
                name
                for name in (
                    "baseline_minimum_task_success",
                    "task_success_noninferiority",
                )
                if not checks_by_name[name]["passed"]
            ],
            "gating_order": [
                {
                    "stage": "evidence_integrity",
                    "failure_verdict": "invalid",
                    "observed_result": (
                        f"passed: {valid_trace_count}/{len(attempts)} valid traces and "
                        f"{complete_dag_count}/{len(attempts)} complete explicit DAGs"
                    ),
                    "decisive": False,
                },
                {
                    "stage": "baseline_completion_and_absolute_task_success",
                    "failure_verdict": "invalid",
                    "observed_result": (
                        "failed: "
                        f"{percentage(outcomes['baseline']['task_success_95_ci'][0])} "
                        "lower bound was below "
                        f"{percentage(acceptance['min_task_success_rate'])}"
                    ),
                    "decisive": True,
                },
                {
                    "stage": "candidate_completion_and_absolute_task_success",
                    "failure_verdict": "regression",
                    "observed_result": (
                        "passed: "
                        f"{percentage(outcomes['candidate']['task_success_95_ci'][0])} "
                        "lower bound exceeded "
                        f"{percentage(acceptance['min_task_success_rate'])}"
                    ),
                    "decisive": False,
                },
                {
                    "stage": "candidate_point_estimate_task_success_guardrail",
                    "failure_verdict": "regression",
                    "observed_result": (
                        "passed: candidate point estimate "
                        f"{percentage(outcomes['candidate']['task_success_rate'])} "
                        "exceeded baseline "
                        f"{percentage(outcomes['baseline']['task_success_rate'])}"
                    ),
                    "decisive": False,
                },
                {
                    "stage": "paired_task_success_noninferiority",
                    "failure_verdict": "inconclusive",
                    "observed_result": (
                        "failed independently: "
                        f"{percentage_points(paired_success['one_sided_lower_bound'])} "
                        "was below -"
                        f"{percentage_points(acceptance['max_task_success_regression'])}"
                    ),
                    "decisive": False,
                },
                {
                    "stage": "successful_pair_workflow_latency",
                    "failure_verdict": "regression_or_inconclusive",
                    "observed_result": (
                        "conditionally favorable at "
                        f"{float(latency_effect['candidate_minus_baseline_mean']):+.2f} ms "
                        "but ineligible to verify the change"
                    ),
                    "decisive": False,
                },
            ],
            "decisive_stage": "baseline_completion_and_absolute_task_success",
        },
        "limitations": [
            "One pinned model revision, vLLM version, A10 hardware class, and repository-analysis workload were evaluated.",
            "Execution was sequential and colocated. Production concurrency and multi-replica routing were not tested.",
            (
                "Continuous effects use only the "
                f"{effects['workflow_latency_ms']['paired_count']} task pairs where both variants succeeded."
            ),
            "Workflow latency includes process startup and agent execution but excludes evaluator time.",
            "Prefill, decode, retrieval, cost, and GPU-time effects were unavailable and remain unknown rather than zero.",
            f"The {len(tasks)}-task held-out workload is consumed and cannot be reused as fresh held-out evidence.",
            "External readers can audit the aggregate arithmetic, gate logic, provenance hashes, and claim boundary, but cannot independently recompute the experiment without the private raw tasks and traces.",
        ],
        "claim_boundary": {
            "supported": (
                "RunEconomy rejected an apparently faster, more cache-efficient "
                "candidate because the frozen successful-task quality requirements "
                "were not established."
            ),
            "not_supported": [
                "a verified performance improvement",
                "universal model behavior",
                "production-concurrency performance",
                "superiority over another tool",
            ],
        },
        "sanitization": {
            "generator": "scripts/build_public_proof_artifact.py",
            "aggregate_only": True,
            "private_evidence_required_at_build_time": True,
            "private_evidence_distributed": False,
        },
    }
    validate_public_artifact(artifact)
    return artifact


def render_case_study(payload: Mapping[str, Any]) -> str:
    """Render the public proof directly from the sanitized aggregate artifact."""

    validate_public_artifact(payload)
    from runeconomy.reports import render_public_proof_markdown

    return render_public_proof_markdown(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-study-output", type=Path, default=DEFAULT_CASE_STUDY)
    args = parser.parse_args()
    payload = build(args.result, args.metadata, args.dataset)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.case_study_output.write_text(render_case_study(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
