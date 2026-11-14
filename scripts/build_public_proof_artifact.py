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
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).parents[1]
DEFAULT_RESULT = (
    ROOT / "artifacts/public-proof-heldout-live-extracted/run/result.json"
)
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
        "available": paired_count > 0 and effect.get("candidate_minus_baseline_mean") is not None,
        "candidate_minus_baseline_mean": effect.get(
            "candidate_minus_baseline_mean"
        ),
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
        "artifact_version": "0.2",
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
            "Execution was sequential and colocated; production concurrency and multi-replica routing were not tested.",
            "Continuous effects use only the "
            f"{effects['workflow_latency_ms']['paired_count']} task pairs where both variants succeeded.",
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
    outcomes = payload["outcomes"]
    effects = payload["conditional_effects"]
    paired = payload["paired_success"]
    experiment = payload["experiment"]
    intervention = payload["intervention"]
    provenance = payload["provenance"]
    acceptance = payload["gates"]["predeclared"]

    def percent(value: Any, digits: int = 2) -> str:
        return f"{float(value) * 100:.{digits}f}%"

    def percentage_points(value: Any, digits: int = 2) -> str:
        return f"{float(value) * 100:.{digits}f} percentage points"

    def effect(name: str, unit: str) -> str:
        item = effects[name]
        interval = item["confidence_interval"]
        return (
            f"{float(item['candidate_minus_baseline_mean']):+.2f} {unit}; "
            f"95% interval {float(interval[0]):+.2f} to "
            f"{float(interval[1]):+.2f}; {item['paired_count']} pairs"
        )

    family_text = ", ".join(
        f"{count} {name.replace('_', '-')}"
        for name, count in experiment["task_family_counts"].items()
    )
    limitations = "\n".join(f"- {item}" for item in payload["limitations"])
    hashes = "\n".join(
        f"- {name.replace('_', ' ').title()}: `{value}`"
        for name, value in sorted(payload["frozen_hashes"].items())
    )
    gate_lines = "\n".join(
        f"{index}. `{item['stage']}` — {item['observed_result']}"
        f"{' **(decisive)**' if item['decisive'] else ''}."
        for index, item in enumerate(payload["decision"]["gating_order"], start=1)
    )
    paired_counts = paired["outcome_counts"]
    return f"""# Public proof: RunEconomy rejected an apparently faster candidate

This is the canonical RunEconomy proof for the frozen
{experiment['task_count']}-task held-out run.
The verdict is **`invalid`**, not verified. Every number below is generated
from the checked-in aggregate artifact; private tasks, answers, model output,
stderr, endpoints, and raw traces are not published.

## What changed

- Baseline: {intervention['baseline']}.
- Candidate: {intervention['candidate']}.
- Only changed property: {intervention['changed_property']}.
- Held constant: {intervention['held_constant']}.
- Expected mechanism: {intervention['expected_mechanism']}.

## What was measured

The frozen workload contained {experiment['task_count']} tasks ({family_text}),
with one baseline and one candidate attempt per task. Each attempt made two
inference calls around one real bounded repository-search tool. Order was
seeded and randomized (`{experiment['seed']}`). The predeclared product outcome
was end-to-end workflow latency, paired only where both variants succeeded.

Environment: `{provenance['model']}` revision
`{provenance['model_revision']}`, vLLM `{provenance['backend_version']}`, one
{provenance['hardware']}, Chat Completions, colocated driver.

## Task quality

| Variant | Successes | Point rate | Exact 95% interval | Absolute gate |
|---|---:|---:|---:|---:|
| Baseline | {outcomes['baseline']['task_success_count']}/{outcomes['baseline']['attempt_count']} | {percent(outcomes['baseline']['task_success_rate'])} | {percent(outcomes['baseline']['task_success_95_ci'][0])}–{percent(outcomes['baseline']['task_success_95_ci'][1])} | lower bound ≥ 80% |
| Candidate | {outcomes['candidate']['task_success_count']}/{outcomes['candidate']['attempt_count']} | {percent(outcomes['candidate']['task_success_rate'])} | {percent(outcomes['candidate']['task_success_95_ci'][0])}–{percent(outcomes['candidate']['task_success_95_ci'][1])} | lower bound ≥ 80% |

Paired success was
{percentage_points(paired['candidate_minus_baseline_mean'])}; its one-sided
lower bound was {percentage_points(paired['one_sided_lower_bound'])}, missing
the predeclared
−{percentage_points(acceptance['max_task_success_regression'])}
non-inferiority margin. Outcomes were {paired_counts['both_success']}
both-success, {paired_counts['baseline_only']} baseline-only,
{paired_counts['candidate_only']} candidate-only, and
{paired_counts['both_failure']} both-failure.

## Decision gates, in order

{gate_lines}

The baseline validity gate was decisive. RunEconomy therefore returned
**`invalid`** before the favorable latency result could authorize a verified
performance claim. The paired non-inferiority gate also failed independently.

## Conditional mechanism and latency observations

Among the {effects['workflow_latency_ms']['paired_count']} pairs where both
variants succeeded:

- Cached input tokens: {effect('cached_input_tokens', 'tokens')}.
- Server TTFT: {effect('server_ttft_ms', 'ms')}.
- Client TTFT: {effect('client_ttft_ms', 'ms')}.
- Workflow latency: {effect('workflow_latency_ms', 'ms')}.
- Critical-path latency: {effect('critical_path_ms', 'ms')}.

These are measured conditional effects, not verified improvements. Failed tasks
remain in the quality analysis and cannot make the candidate appear faster by
disappearing from the verdict.

## What RunEconomy demonstrated

RunEconomy connected an application-level prompt-layout change to cache,
inference, workflow, and task-success evidence; then declined to verify an
attractive latency result because the frozen quality requirements were not
established. That rejection—not a speedup claim—is the product proof.

## What this does not prove

{limitations}

It also does not establish universal model behavior, production-concurrency
performance, superiority over another tool, or a verified performance win.

## Audit trail

The machine-readable source is
[`src/runeconomy/data/public-proof-rejection.json`](../src/runeconomy/data/public-proof-rejection.json).
It can be rebuilt only when the ignored private evidence is present:

```bash
uv run python scripts/build_public_proof_artifact.py
```

Frozen hashes:

{hashes}

The public artifact contains aggregate evidence only. Its sanitizer fails
closed on unexpected top-level fields, endpoints, credentials, local paths,
and raw-evidence keys.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case-study-output", type=Path, default=DEFAULT_CASE_STUDY
    )
    args = parser.parse_args()
    payload = build(args.result, args.metadata, args.dataset)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.case_study_output.write_text(
        render_case_study(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
