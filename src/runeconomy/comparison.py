"""Closed-loop comparison of real baseline and candidate agent processes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shlex
import signal
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .attempt_status import AttemptStatus, AttemptStatusError
from .completion_policy import (
    CompletionPolicy,
    CompletionPolicyError,
    common_mode_failure_pairs,
    completion_policy_record,
)
from .critical_path import calculate_critical_path
from .decision_artifacts import enrich_decision_artifact
from .evaluators import (
    EvaluatorError,
    EvaluatorInfrastructureError,
    evaluate_output,
    validate_evaluator_spec,
)
from .evidence_quality import aggregate_evidence_quality
from .identity import (
    IdentityPolicy,
    IdentityPolicyError,
    assess_request_identity,
    identity_assurance_record,
    parse_identity_policy,
    variant_identity_summary,
)
from .execution import (
    CONCURRENT_ARRIVAL_PATTERNS,
    ROLE_FILLER,
    SEQUENTIAL,
    ArrivalSchedule,
    JsonlArtifactStore,
    LoadRunner,
    LoadSpec,
    LoadSpecError,
    ScheduledTrial,
    build_arrival_schedule,
    verify_schedule_pin,
)
from .metric_gates import (
    MetricGateError,
    Observation,
    economics_of,
    evaluate_metric_gates,
    load_integrity_checks,
    merge_load_and_metric_checks,
    parse_metric_gates,
    tpot_ms,
)
from .experiments import (
    ExperimentValidationError,
    _clopper_pearson_interval,
    _task_success_noninferior,
    _task_success_statistics,
)
from .paired_inference import PAIRED_EXACT, PAIRED_METHODS
from .importers import import_openinference
from .models import Trace, TraceValidationError


COMPARE_RESULT_VERSION = "0.2"
COMPARE_LOAD_RESULT_VERSION = "0.3"
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest_path(path: Path) -> str:
    try:
        return _digest_bytes(path.read_bytes())
    except OSError as exc:
        raise ExperimentValidationError(f"cannot read {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExperimentValidationError(f"cannot read dataset {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExperimentValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list) or not tasks:
        raise ExperimentValidationError(
            "dataset must be a non-empty array or an object with a tasks array"
        )
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ExperimentValidationError(f"dataset task {index} must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ExperimentValidationError(
                f"dataset task {index}.task_id must be a non-empty string"
            )
        if task_id in ids:
            raise ExperimentValidationError(f"duplicate task_id {task_id!r}")
        ids.add(task_id)
        normalized.append(dict(task))
    return normalized


def _command(value: str, label: str) -> list[str]:
    try:
        argv = shlex.split(value)
    except ValueError as exc:
        raise ExperimentValidationError(f"{label} command is invalid: {exc}") from exc
    if not argv:
        raise ExperimentValidationError(f"{label} command cannot be empty")
    return argv


def _command_identity(argv: list[str], workdir: Path) -> dict[str, Any]:
    referenced: dict[str, str] = {}
    executable = shutil.which(argv[0])
    candidates: list[tuple[str, Path]] = []
    if executable:
        candidates.append(("executable", Path(executable)))
    candidates.extend(
        (f"arg-{index}", Path(item))
        for index, item in enumerate(argv[1:], 1)
        if not item.startswith("-")
    )
    for label, candidate in candidates:
        resolved = candidate if candidate.is_absolute() else workdir / candidate
        if resolved.is_file():
            referenced[label] = _digest_path(resolved)
    return {
        "executable": Path(argv[0]).name,
        "argv_sha256": _digest_bytes(
            json.dumps(argv, separators=(",", ":")).encode()
        ),
        "referenced_file_sha256": dict(sorted(referenced.items())),
    }


def _workdir_identity(workdir: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=workdir,
            capture_output=True,
            timeout=30,
            check=False,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=workdir,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"kind": "unknown"}
    if (
        commit.returncode != 0
        or tracked_diff.returncode != 0
        or untracked.returncode != 0
    ):
        return {"kind": "non_git_directory"}
    untracked_digests: dict[str, str] = {}
    for encoded_path in untracked.stdout.split(b"\0"):
        if not encoded_path:
            continue
        relative_path = encoded_path.decode("utf-8", errors="surrogateescape")
        path = workdir / relative_path
        try:
            untracked_digests[relative_path] = _digest_bytes(path.read_bytes())
        except OSError:
            untracked_digests[relative_path] = "unavailable"
    return {
        "kind": "git",
        "commit": commit.stdout.strip(),
        "dirty": bool(tracked_diff.stdout or untracked_digests),
        "tracked_diff_sha256": _digest_bytes(tracked_diff.stdout),
        "untracked_file_count": len(untracked_digests),
        "untracked_files_sha256": _digest_bytes(
            json.dumps(
                untracked_digests, sort_keys=True, separators=(",", ":")
            ).encode()
        ),
    }


def _execute_agent(
    argv: list[str], *, cwd: Path, environment: dict[str, str], timeout_seconds: float
) -> tuple[int | None, str, str, bool]:
    """Execute an agent and terminate its process group on timeout."""

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode, stdout, stderr, False
    except KeyboardInterrupt:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.communicate()
        raise
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        return None, stdout, stderr, True


def _load_evaluator(path: Path) -> tuple[str, Any]:
    if path.suffix.lower() == ".json":
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ExperimentValidationError(f"cannot read evaluator {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ExperimentValidationError(
                f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
            ) from exc
        try:
            return "spec", validate_evaluator_spec(spec)
        except EvaluatorError as exc:
            raise ExperimentValidationError(str(exc)) from exc
    if path.suffix.lower() != ".py" or not path.is_file():
        raise ExperimentValidationError(
            "--evaluator must be a JSON evaluator spec or a Python evaluator script"
        )
    return "script", path.resolve()


def _evaluate(
    evaluator: tuple[str, Any],
    output_text: str,
    context: dict[str, Any],
    *,
    timeout_seconds: float,
    artifacts_dir: Path,
) -> dict[str, Any]:
    kind, value = evaluator
    if kind == "spec":
        try:
            return evaluate_output(value, output_text, context).to_dict()
        except EvaluatorInfrastructureError:
            raise
        except EvaluatorError as exc:
            raise EvaluatorInfrastructureError(str(exc)) from exc
    request = {"output_text": output_text, **context}
    try:
        completed = subprocess.run(
            [sys.executable, str(value)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvaluatorInfrastructureError(
            f"evaluator timed out after {timeout_seconds:g} seconds"
        ) from exc
    except OSError as exc:
        raise EvaluatorInfrastructureError(f"cannot run evaluator: {exc}") from exc
    (artifacts_dir / "evaluator-stdout.txt").write_text(
        completed.stdout, encoding="utf-8"
    )
    (artifacts_dir / "evaluator-stderr.txt").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise EvaluatorInfrastructureError(
            f"evaluator exited {completed.returncode}; see evaluator stderr artifact"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EvaluatorInfrastructureError(
            "evaluator stdout must be one JSON boolean or result object"
        ) from exc
    if isinstance(result, bool):
        return {
            "success": result,
            "score": float(result),
            "evaluator_kind": "script",
            "details": {},
        }
    if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
        raise EvaluatorInfrastructureError(
            "evaluator result must be a boolean or object with boolean success"
        )
    score = result.get("score", float(result["success"]))
    details = result.get("details", {})
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isinstance(details, dict)
    ):
        raise EvaluatorInfrastructureError(
            "evaluator result score must be numeric and details must be an object"
        )
    return {
        "success": result["success"],
        "score": float(score),
        "evaluator_kind": "script",
        "details": details,
    }


def _identities_stable(signatures: set[str]) -> bool:
    if not signatures:
        return False
    by_role: dict[Any, set[str]] = {}
    for signature in signatures:
        role = json.loads(signature).get("step_role")
        by_role.setdefault(role, set()).add(signature)
    return all(len(members) == 1 for members in by_role.values())


def _sum_metric(trace: Trace, name: str) -> float | int | None:
    values = [
        getattr(request.metrics, name)
        for request in trace.requests
        if getattr(request.metrics, name) is not None
    ]
    return sum(values) if values else None


def _trace_metrics(payload: Mapping[str, Any], trace: Trace) -> dict[str, Any]:
    critical = None
    if payload.get("dependency_evidence") == "complete_explicit":
        critical_path = calculate_critical_path(trace.spans)
        critical = asdict(critical_path) if critical_path is not None else None
        if critical is not None:
            critical["span_ids"] = [
                _digest_bytes(value.encode()) for value in critical["span_ids"]
            ]
            critical["inference_request_ids"] = [
                _digest_bytes(value.encode())
                for value in critical["inference_request_ids"]
            ]
    span_totals: dict[str, float] = {}
    for span in trace.spans:
        span_totals[span.kind] = span_totals.get(span.kind, 0.0) + span.duration_ms
    starts = [span.start_ms for span in trace.spans]
    ends = [span.end_ms for span in trace.spans]
    identity_fields = (
        "backend",
        "backend_version",
        "model",
        "model_revision",
        "tokenizer",
        "tokenizer_revision",
        "chat_template_digest",
    )
    payload_requests = payload.get("requests")
    payload_requests = payload_requests if isinstance(payload_requests, list) else []
    identities = []
    for index, request in enumerate(trace.requests):
        identity = {name: getattr(request, name) for name in identity_fields}
        raw = payload_requests[index] if index < len(payload_requests) else None
        role = raw.get("step_role") if isinstance(raw, Mapping) else None
        if isinstance(role, str) and role:
            identity["step_role"] = role
        identities.append(identity)
    identity_complete = all(
        value not in {None, "", "unknown"}
        for identity in identities
        for value in identity.values()
    )
    request_metrics = [request.metrics for request in trace.requests]
    return {
        "trace_id": _digest_bytes(trace.trace_id.encode()),
        "dependency_evidence": payload.get("dependency_evidence", "absent"),
        "trace_wall_span_ms": max(ends) - min(starts) if starts else None,
        "critical_path_ms": critical["duration_ms"] if critical else None,
        "critical_path": critical,
        "inference_span_ms": span_totals.get("inference"),
        "tool_span_ms": span_totals.get("tool"),
        "retrieval_span_ms": span_totals.get("retriever"),
        "input_tokens": _sum_metric(trace, "input_tokens"),
        "output_tokens": _sum_metric(trace, "output_tokens"),
        "cached_input_tokens": _sum_metric(trace, "cached_input_tokens"),
        "queue_ms": _sum_metric(trace, "queue_ms"),
        "prefill_ms": _sum_metric(trace, "prefill_ms"),
        "decode_ms": _sum_metric(trace, "decode_ms"),
        "cost_usd": _sum_metric(trace, "cost_usd"),
        "gpu_time_ms": _sum_metric(trace, "gpu_time_ms"),
        "server_ttft_ms": _sum_metric(trace, "ttft_ms"),
        "client_ttft_ms": _sum_metric(trace, "client_ttft_ms"),
        "request_latency_ms": _sum_metric(trace, "total_latency_ms"),
        "request_count": len(trace.requests),
        # Request-level values for request-level metric gates (TTFT, TPOT).
        "request_client_ttft_ms": [
            metrics.client_ttft_ms
            for metrics in request_metrics
            if metrics.client_ttft_ms is not None
        ],
        "request_server_ttft_ms": [
            metrics.ttft_ms for metrics in request_metrics if metrics.ttft_ms is not None
        ],
        "request_tpot_ms": [
            value
            for value in (
                tpot_ms(metrics.total_latency_ms, metrics.client_ttft_ms, metrics.output_tokens)
                for metrics in request_metrics
            )
            if value is not None
        ],
        "span_count": len(trace.spans),
        "environment_identity": identities,
        "environment_identity_complete": identity_complete,
        "evidence_quality": payload.get("evidence_quality"),
        **_identity_declarations(payload),
    }


_IDENTITY_DECLARATION_KEYS = (
    "identity_grade",
    "identity_undisclosed_fields",
    "provider_fingerprint_sha256",
)


def _identity_declarations(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Per-request graded-identity declarations, only when a trace has them.

    Legacy traces declare none, so their attempt metrics are unchanged.
    """

    requests = [
        request for request in payload.get("requests", []) if isinstance(request, Mapping)
    ]
    if not any(key in request for request in requests for key in _IDENTITY_DECLARATION_KEYS):
        return {}
    return {
        "identity_declarations": [
            {
                "grade": request.get("identity_grade"),
                "undisclosed_fields": list(request.get("identity_undisclosed_fields") or []),
                "provider_fingerprint_sha256": request.get("provider_fingerprint_sha256"),
            }
            for request in requests
        ]
    }


def _graded_identity(
    traced: list[dict[str, Any]], policy: IdentityPolicy
) -> dict[str, Any]:
    """Assess one variant's identity under a declared identity policy."""

    assessments: list[dict[str, Any]] = []
    fingerprints: list[str | None] = []
    for item in traced:
        metrics = item["metrics"]
        identities = metrics.get("environment_identity", [])
        declarations = metrics.get("identity_declarations") or [{}] * len(identities)
        for identity, declaration in zip(identities, declarations):
            assessments.append(
                assess_request_identity(
                    identity,
                    grade=declaration.get("grade"),
                    undisclosed=declaration.get("undisclosed_fields"),
                )
            )
            fingerprints.append(declaration.get("provider_fingerprint_sha256"))
    return variant_identity_summary(policy, assessments, fingerprints)


def _schedule(
    tasks: list[dict[str, Any]], repetitions: int, order: str, seed: int
) -> list[tuple[int, dict[str, Any], str]]:
    rng = random.Random(seed)
    result: list[tuple[int, dict[str, Any], str]] = []
    for repetition in range(repetitions):
        task_order = list(tasks)
        if order == "randomized":
            rng.shuffle(task_order)
        for task_index, task in enumerate(task_order):
            variants = ["baseline", "candidate"]
            if order == "randomized":
                rng.shuffle(variants)
            elif (repetition + task_index) % 2:
                variants.reverse()
            result.extend((repetition, task, variant) for variant in variants)
    return result


def _load_attempt_checkpoint(
    path: Path, schedule: list[tuple[int, dict[str, Any], str]]
) -> list[dict[str, Any]]:
    """Load and validate a durable prefix of the deterministic agent schedule."""

    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExperimentValidationError(
            "cannot read agent-process attempt checkpoint"
        ) from exc
    attempts: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            attempt = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentValidationError(
                f"attempt checkpoint record {index + 1} is invalid JSON"
            ) from exc
        if not isinstance(attempt, dict):
            raise ExperimentValidationError(
                f"attempt checkpoint record {index + 1} is not an object"
            )
        if index >= len(schedule):
            raise ExperimentValidationError(
                "attempt checkpoint contains more records than the frozen schedule"
            )
        repetition, task, variant = schedule[index]
        expected = {
            "sequence": index + 1,
            "task_id": _digest_bytes(str(task["task_id"]).encode()),
            "variant": variant,
            "repetition": repetition,
        }
        if any(attempt.get(key) != value for key, value in expected.items()):
            raise ExperimentValidationError(
                f"attempt checkpoint record {index + 1} does not match the frozen schedule"
            )
        attempts.append(attempt)
    return attempts


def _run_attempt(
    *,
    argv: list[str],
    task: dict[str, Any],
    variant: str,
    repetition: int,
    sequence: int,
    seed: int,
    workdir: Path,
    artifacts_dir: Path,
    evaluator: tuple[str, Any],
    timeout_seconds: float,
    evaluator_timeout_seconds: float,
    attempt_prefix: str | None = None,
    extra_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    task_id_digest = _digest_bytes(str(task["task_id"]).encode())
    task_suffix = task_id_digest.removeprefix("sha256:")[:12]
    prefix = f"{sequence:05d}" if attempt_prefix is None else attempt_prefix
    attempt_id = f"{prefix}-task-{task_suffix}-{variant}-r{repetition:03d}"
    attempt_dir = artifacts_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    task_path = attempt_dir / "task.json"
    output_path = attempt_dir / "output.txt"
    stdout_path = attempt_dir / "stdout.txt"
    stderr_path = attempt_dir / "stderr.txt"
    trace_path = attempt_dir / "trace.json"
    status_path = attempt_dir / "status.json"
    normalized_trace_path = attempt_dir / "normalized-trace.json"
    _write_json(task_path, task)
    environment = os.environ.copy()
    environment.update(
        {
            "RUNECONOMY_TASK_PATH": str(task_path.resolve()),
            "RUNECONOMY_TASK_ID": str(task["task_id"]),
            "RUNECONOMY_VARIANT": variant,
            "RUNECONOMY_REPETITION": str(repetition),
            "RUNECONOMY_SEED": str(seed),
            "RUNECONOMY_ATTEMPT_DIR": str(attempt_dir.resolve()),
            "RUNECONOMY_OUTPUT_PATH": str(output_path.resolve()),
            "RUNECONOMY_TRACE_PATH": str(trace_path.resolve()),
            "RUNECONOMY_STATUS_PATH": str(status_path.resolve()),
        }
    )
    if extra_environment:
        environment.update(extra_environment)
    started = time.perf_counter_ns()
    process_returncode: int | None = None
    status = "infrastructure_failure"
    failure_reason: str | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    execution_error = False
    try:
        process_returncode, stdout, stderr, timed_out = _execute_agent(
            argv,
            cwd=workdir,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        if timed_out:
            status = "timeout"
            failure_reason = f"agent exceeded {timeout_seconds:g} seconds"
    except OSError as exc:
        execution_error = True
        failure_reason = f"cannot execute agent: {exc}"
    workflow_latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    output_text = (
        output_path.read_text(encoding="utf-8", errors="replace")
        if output_path.is_file()
        else stdout
    )
    if not output_path.exists():
        output_path.write_text(output_text, encoding="utf-8")

    agent_status: AttemptStatus | None = None
    agent_status_digest: str | None = None
    agent_status_error: str | None = None
    if status_path.is_file():
        agent_status_digest = _digest_path(status_path)
        try:
            agent_status = AttemptStatus.from_path(status_path)
        except AttemptStatusError as exc:
            agent_status_error = str(exc)

    if not timed_out and not execution_error:
        if agent_status_error is not None:
            status = "invalid_status"
            failure_reason = agent_status_error
        elif agent_status is not None:
            if process_returncode != 0 and agent_status.status == "completed":
                status = "invalid_status"
                failure_reason = (
                    "agent declared completed but exited with code "
                    f"{process_returncode}"
                )
            else:
                status = agent_status.status
                if status != "completed":
                    failure_reason = f"agent declared {status}"
        elif process_returncode != 0:
            status = "infrastructure_failure"
            failure_reason = f"agent exited with code {process_returncode}"
        else:
            # Backward-compatible path for agents that predate status schema 0.1.
            status = "completed"

    trace_metrics: dict[str, Any] = {}
    trace_digest: str | None = None
    evaluator_result: dict[str, Any] | None = None
    if trace_path.is_file():
        trace_digest = _digest_path(trace_path)
        try:
            normalized = import_openinference(
                trace_path,
                workload=str(task["task_id"]),
            )
            recorded_task_id = normalized.get("task_id")
            if (
                recorded_task_id is not None
                and recorded_task_id != str(task["task_id"])
            ):
                raise TraceValidationError(
                    "trace runeconomy.task_id does not match the scheduled task"
                )
            trace = Trace.from_dict(normalized)
            _write_json(normalized_trace_path, normalized)
            trace_metrics = _trace_metrics(normalized, trace)
        except TraceValidationError as exc:
            if status == "completed":
                status = "invalid_trace"
                failure_reason = str(exc)
    elif status == "completed":
        status = "invalid_trace"
        failure_reason = "agent did not write RUNECONOMY_TRACE_PATH"

    if status == "completed":
        if process_returncode != 0:
            status = "invalid_status"
            failure_reason = "completed status requires a zero process exit code"
        elif not trace_metrics:
            status = "invalid_trace"
            failure_reason = "agent trace did not produce usable evidence"
        else:
            context = {
                "task": task,
                "task_id": task["task_id"],
                "variant": variant,
                "repetition": repetition,
                "attempt_id": attempt_id,
                # Repo-test and trajectory evaluators read the attempt's own
                # working copy and trace; both stay variant-local.
                "attempt_dir": str(attempt_dir.resolve()),
                "trace_path": str(trace_path.resolve()),
            }
            try:
                evaluator_result = _evaluate(
                    evaluator,
                    output_text,
                    context,
                    timeout_seconds=evaluator_timeout_seconds,
                    artifacts_dir=attempt_dir,
                )
            except EvaluatorInfrastructureError as exc:
                status = "evaluator_infrastructure_error"
                failure_reason = str(exc)
            else:
                _write_json(attempt_dir / "evaluation.json", evaluator_result)
                status = (
                    "completed" if evaluator_result["success"] else "task_failure"
                )

    declared_task_failure = status in {
        "task_failure",
        "model_output_failure",
        "application_failure",
    }

    failure_detail = failure_reason
    failure_artifact: str | None = None
    failure_details_digest: str | None = None
    if failure_detail is not None:
        failure_payload: dict[str, Any] = {
            "status": status,
            "detail": failure_detail,
        }
        if agent_status is not None:
            failure_payload["agent_status"] = agent_status.to_dict()
        failure_path = attempt_dir / "failure.json"
        _write_json(failure_path, failure_payload)
        failure_artifact = failure_path.name
        failure_details_digest = _digest_bytes(
            json.dumps(
                failure_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        )
    public_failure_reasons = {
        "timeout": "agent exceeded the configured timeout",
        "infrastructure_failure": "agent process failed",
        "model_output_failure": "agent could not interpret model output",
        "application_failure": "agent application failed",
        "invalid_status": "agent status evidence was missing or contradictory",
        "invalid_trace": "trace was missing or failed validation",
        "evaluator_infrastructure_error": "evaluator failed",
    }

    return {
        "attempt_id": attempt_id,
        "task_id": task_id_digest,
        "variant": variant,
        "repetition": repetition,
        "sequence": sequence,
        "status": status,
        "failure_reason": public_failure_reasons.get(status),
        "failure_details_digest": failure_details_digest,
        "process_returncode": process_returncode,
        "task_success": (
            evaluator_result["success"]
            if evaluator_result is not None
            else False
            if declared_task_failure
            else None
        ),
        "agent_status": (
            {
                **agent_status.to_dict(),
                "digest": agent_status_digest,
            }
            if agent_status is not None
            else None
        ),
        "evaluation": (
            {
                "success": evaluator_result["success"],
                "score": evaluator_result["score"],
                "evaluator_kind": evaluator_result["evaluator_kind"],
                "details_digest": _digest_bytes(
                    json.dumps(
                        evaluator_result.get("details", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ),
            }
            if evaluator_result is not None
            else None
        ),
        "workflow_latency_ms": workflow_latency_ms,
        "output_digest": _digest_bytes(output_text.encode("utf-8")),
        "trace_digest": trace_digest,
        "metrics": trace_metrics,
        "artifacts": {
            "directory": str(attempt_dir.relative_to(artifacts_dir)),
            "task": "task.json",
            "output": "output.txt",
            "stdout": "stdout.txt",
            "stderr": "stderr.txt",
            "trace": "trace.json" if trace_path.exists() else None,
            "status": "status.json" if status_path.exists() else None,
            "normalized_trace": (
                "normalized-trace.json" if normalized_trace_path.exists() else None
            ),
            "evaluator_stdout": (
                "evaluator-stdout.txt"
                if (attempt_dir / "evaluator-stdout.txt").exists()
                else None
            ),
            "evaluator_stderr": (
                "evaluator-stderr.txt"
                if (attempt_dir / "evaluator-stderr.txt").exists()
                else None
            ),
            "evaluation": (
                "evaluation.json"
                if (attempt_dir / "evaluation.json").exists()
                else None
            ),
            "failure": failure_artifact,
        },
    }


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _paired_t_interval(differences: list[float]) -> dict[str, Any]:
    count = len(differences)
    mean = statistics.fmean(differences) if differences else None
    if count < 2:
        interval = None
    else:
        standard_error = statistics.stdev(differences) / math.sqrt(count)
        # 2.042 is the df=30 critical value and is conservative for larger df.
        critical = _T_CRITICAL_95.get(count - 1, 2.042)
        interval = [mean - critical * standard_error, mean + critical * standard_error]
    return {
        "paired_count": count,
        "candidate_minus_baseline_mean": mean,
        "confidence_interval": interval,
        "confidence_level": 0.95,
        "method": "paired Student-t interval",
    }


def _summarize(
    attempts: list[dict[str, Any]],
    *,
    task_success_method: str = PAIRED_EXACT,
    max_task_success_regression: float = 0.0,
    completion_policy: CompletionPolicy | None = None,
    identity_policy: IdentityPolicy | None = None,
) -> dict[str, Any]:
    common_mode_pairs = (
        common_mode_failure_pairs(attempts) if completion_policy is not None else []
    )
    reclassified_keys = (
        {(pair["task_id"], pair["repetition"]) for pair in common_mode_pairs}
        if completion_policy is not None and completion_policy.reclassifies_common_mode
        else set()
    )
    variants: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        subset = [item for item in attempts if item["variant"] == variant]
        statuses: dict[str, int] = {}
        for item in subset:
            statuses[item["status"]] = statuses.get(item["status"], 0) + 1
        evaluated = [item for item in subset if item["task_success"] is not None]
        successes = sum(bool(item["task_success"]) for item in evaluated)
        successful = [item for item in evaluated if item["task_success"]]
        traced = [item for item in subset if item["metrics"]]
        identity_signatures_by_task: dict[str, set[str]] = {}
        for item in traced:
            task_signatures = identity_signatures_by_task.setdefault(
                item["task_id"], set()
            )
            for identity in item["metrics"].get("environment_identity", []):
                task_signatures.add(
                    json.dumps(
                        identity,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        identity_signatures = {
            signature
            for signatures in identity_signatures_by_task.values()
            for signature in signatures
        }
        quality_records = [
            item["metrics"]["evidence_quality"]
            for item in traced
            if isinstance(item["metrics"].get("evidence_quality"), Mapping)
        ]
        variants[variant] = {
            "attempt_count": len(subset),
            "status_counts": dict(sorted(statuses.items())),
            "completed_rate": (
                sum(item["status"] in {"completed", "task_failure"} for item in subset)
                / len(subset)
                if subset
                else None
            ),
            "evaluated_count": len(evaluated),
            "task_success_count": successes,
            "task_success_rate": successes / len(evaluated) if evaluated else None,
            "task_success_95_ci": _clopper_pearson_interval(successes, len(evaluated)),
            "trace_evidence": {
                "valid_trace_count": len(traced),
                "complete_dag_count": sum(
                    item["metrics"].get("dependency_evidence")
                    == "complete_explicit"
                    for item in traced
                ),
                "complete_dag_rate": (
                    sum(
                        item["metrics"].get("dependency_evidence")
                        == "complete_explicit"
                        for item in traced
                    )
                    / len(subset)
                    if subset
                    else None
                ),
                "environment_identity_complete": bool(
                    traced
                    and all(
                        item["metrics"].get("environment_identity_complete")
                        for item in traced
                    )
                ),
                # Stable means one identity per declared step role. Traces
                # that declare no runeconomy.step_role keep the original rule:
                # exactly one identity for the whole variant. A workflow that
                # routes steps to different models must declare the roles, so
                # deliberate routing is never confused with model drift.
                "environment_identity_stable": _identities_stable(identity_signatures),
                "environment_identity_signatures": sorted(identity_signatures),
                "environment_identity_signatures_by_task": {
                    task_id: sorted(signatures)
                    for task_id, signatures in sorted(
                        identity_signatures_by_task.items()
                    )
                },
                "evidence_quality": aggregate_evidence_quality(
                    quality_records,
                    source_type=f"{variant}_agent_trace",
                ),
            },
            "successful_task_metrics": {
                "workflow_latency_ms": _metric_summary(
                    [float(item["workflow_latency_ms"]) for item in successful]
                ),
                **{
                    name: _metric_summary(
                        [
                            float(item["metrics"][name])
                            for item in successful
                            if item["metrics"].get(name) is not None
                        ]
                    )
                    for name in (
                        "critical_path_ms",
                        "input_tokens",
                        "output_tokens",
                        "cached_input_tokens",
                        "queue_ms",
                        "prefill_ms",
                        "decode_ms",
                        "cost_usd",
                        "gpu_time_ms",
                        "server_ttft_ms",
                        "client_ttft_ms",
                        "inference_span_ms",
                        "tool_span_ms",
                        "retrieval_span_ms",
                    )
                },
            },
        }
        if completion_policy is not None:
            completed = sum(
                item["status"] in {"completed", "task_failure"} for item in subset
            )
            reclassified = sum(
                (item["task_id"], item["repetition"]) in reclassified_keys
                for item in subset
            )
            variants[variant]["completion_gate"] = {
                "policy": completion_policy.to_dict(),
                "attempt_count": len(subset),
                "completed_count": completed,
                "common_mode_task_outcome_count": reclassified,
                "counted_complete": completed + reclassified,
                "rate": (completed + reclassified) / len(subset) if subset else None,
            }

    by_key = {
        (item["task_id"], item["repetition"], item["variant"]): item
        for item in attempts
    }
    task_pairs: list[tuple[bool, bool]] = []
    metric_differences: dict[str, list[float]] = {
        name: []
        for name in (
            "workflow_latency_ms",
            "critical_path_ms",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "queue_ms",
            "prefill_ms",
            "decode_ms",
            "cost_usd",
            "gpu_time_ms",
            "server_ttft_ms",
            "client_ttft_ms",
            "inference_span_ms",
            "tool_span_ms",
            "retrieval_span_ms",
        )
    }
    pair_keys = sorted({(item["task_id"], item["repetition"]) for item in attempts})
    for task_id, repetition in pair_keys:
        baseline = by_key.get((task_id, repetition, "baseline"))
        candidate = by_key.get((task_id, repetition, "candidate"))
        if baseline is None or candidate is None:
            continue
        if baseline["task_success"] is not None and candidate["task_success"] is not None:
            task_pairs.append(
                (bool(baseline["task_success"]), bool(candidate["task_success"]))
            )
        if not baseline["task_success"] or not candidate["task_success"]:
            continue
        for name in metric_differences:
            left = (
                baseline["workflow_latency_ms"]
                if name == "workflow_latency_ms"
                else baseline["metrics"].get(name)
            )
            right = (
                candidate["workflow_latency_ms"]
                if name == "workflow_latency_ms"
                else candidate["metrics"].get(name)
            )
            if left is not None and right is not None:
                metric_differences[name].append(float(right) - float(left))
    effects = {
        name: _paired_t_interval(values)
        for name, values in metric_differences.items()
    }
    effects["task_success"] = _task_success_statistics(
        task_pairs,
        method=task_success_method,
        margin=max_task_success_regression,
    )
    quality_records = [
        item["metrics"]["evidence_quality"]
        for item in attempts
        if isinstance(item.get("metrics"), Mapping)
        and isinstance(item["metrics"].get("evidence_quality"), Mapping)
    ]
    summary: dict[str, Any] = {
        "variants": variants,
        "effects": effects,
        "evidence_quality": aggregate_evidence_quality(
            quality_records, source_type="agent_process_experiment"
        ),
        "latency_population": (
            "task-level pairs where both baseline and candidate succeeded"
        ),
    }
    if completion_policy is not None:
        summary["completion_policy"] = completion_policy_record(
            completion_policy, common_mode_pairs
        )
    if identity_policy is not None:
        graded: dict[str, Any] = {}
        for variant, values in variants.items():
            traced = [
                item
                for item in attempts
                if item["variant"] == variant and item["metrics"]
            ]
            graded[variant] = _graded_identity(traced, identity_policy)
            # Under a declared policy, completeness means complete graded
            # identity at or above the preregistered minimum grade.
            values["trace_evidence"]["environment_identity_complete"] = bool(
                traced and graded[variant]["valid"]
            )
            values["trace_evidence"]["identity_grade"] = graded[variant]["grade"]
        summary["identity_assurance"] = identity_assurance_record(
            identity_policy, graded
        )
    return summary


def _decision(
    summary: Mapping[str, Any],
    *,
    max_task_success_regression: float,
    min_task_success_rate: float,
    min_completed_rate: float,
    optimization_target: str = "successful_workflow_latency",
    latency_semantics: str = "superiority",
    max_latency_regression_ms: float = 0.0,
    task_success_method: str = PAIRED_EXACT,
) -> dict[str, Any]:
    baseline = summary["variants"]["baseline"]
    candidate = summary["variants"]["candidate"]
    success = summary["effects"]["task_success"]
    latency = summary["effects"]["workflow_latency_ms"]
    baseline_ci = baseline["task_success_95_ci"]
    candidate_ci = candidate["task_success_95_ci"]
    absolute_baseline = bool(baseline_ci and baseline_ci[0] >= min_task_success_rate)
    absolute_candidate = bool(candidate_ci and candidate_ci[0] >= min_task_success_rate)
    noninferior = _task_success_noninferior(
        success, method=task_success_method, margin=max_task_success_regression
    )

    def completion_rate(variant: Mapping[str, Any]) -> Any:
        gate = variant.get("completion_gate")
        return gate.get("rate") if isinstance(gate, Mapping) else variant["completed_rate"]

    baseline_completion_rate = completion_rate(baseline)
    candidate_completion_rate = completion_rate(candidate)
    baseline_complete = bool(
        baseline_completion_rate is not None
        and baseline_completion_rate >= min_completed_rate
    )
    candidate_complete = bool(
        candidate_completion_rate is not None
        and candidate_completion_rate >= min_completed_rate
    )
    latency_ci = latency["confidence_interval"]
    latency_limit = 0.0 if latency_semantics == "superiority" else max_latency_regression_ms
    latency_gate_passed = bool(latency_ci and latency_ci[1] < latency_limit)
    latency_regression = bool(latency_ci and latency_ci[0] > latency_limit)
    baseline_trace_evidence = baseline["trace_evidence"]
    candidate_trace_evidence = candidate["trace_evidence"]
    trace_evidence_valid = all(
        evidence["complete_dag_rate"] == 1.0
        and evidence["environment_identity_complete"]
        and evidence["environment_identity_stable"]
        for evidence in (baseline_trace_evidence, candidate_trace_evidence)
    )
    baseline_signatures = baseline_trace_evidence.get(
        "environment_identity_signatures", []
    )
    candidate_signatures = candidate_trace_evidence.get(
        "environment_identity_signatures", []
    )
    backend_identities_distinct = bool(
        len(baseline_signatures) == 1
        and len(candidate_signatures) == 1
        and baseline_signatures[0] != candidate_signatures[0]
    )
    distinct_identity_required = optimization_target == "engine_change_latency"
    checks = [
        {
            "name": "intentional_backend_identity_difference",
            "passed": backend_identities_distinct,
            "required": distinct_identity_required,
            "observed": {
                "baseline_identity_count": len(baseline_signatures),
                "candidate_identity_count": len(candidate_signatures),
                "distinct": backend_identities_distinct,
            },
            "threshold": (
                "one complete stable identity per variant, and baseline differs from candidate"
            ),
        },
        {
            "name": "trace_and_environment_identity",
            "passed": trace_evidence_valid,
            "observed": {
                "baseline": baseline_trace_evidence,
                "candidate": candidate_trace_evidence,
            },
            "threshold": (
                "every attempt has an explicit complete DAG and each variant has "
                "complete, stable environment identity"
            ),
        },
        {
            "name": "baseline_minimum_task_success",
            "passed": absolute_baseline,
            "observed": baseline_ci,
            "threshold": min_task_success_rate,
        },
        {
            "name": "candidate_minimum_task_success",
            "passed": absolute_candidate,
            "observed": candidate_ci,
            "threshold": min_task_success_rate,
        },
        {
            "name": "task_success_noninferiority",
            "passed": noninferior,
            "observed": success,
            "threshold": -max_task_success_regression,
            **(
                {"method": task_success_method}
                if task_success_method != PAIRED_EXACT
                else {}
            ),
        },
        {
            "name": "baseline_completion",
            "passed": baseline_complete,
            "observed": baseline_completion_rate,
            "threshold": min_completed_rate,
            **(
                {"completion_gate": baseline["completion_gate"]}
                if isinstance(baseline.get("completion_gate"), Mapping)
                else {}
            ),
        },
        {
            "name": "candidate_completion",
            "passed": candidate_complete,
            "observed": candidate_completion_rate,
            "threshold": min_completed_rate,
            **(
                {"completion_gate": candidate["completion_gate"]}
                if isinstance(candidate.get("completion_gate"), Mapping)
                else {}
            ),
        },
        {
            "name": (
                "successful_task_workflow_latency_improvement"
                if latency_semantics == "superiority"
                else "successful_task_workflow_latency_noninferiority"
            ),
            "passed": latency_gate_passed,
            "observed": latency,
            "threshold": (
                "upper bound of paired 95% Student-t interval < 0 ms"
                if latency_semantics == "superiority"
                else "upper bound of paired 95% Student-t interval < "
                f"{max_latency_regression_ms:g} ms"
            ),
        },
        {
            "name": "successful_task_workflow_latency_non_regression",
            "passed": not latency_regression,
            "required": False,
            "observed": latency,
            "threshold": f"lower bound of paired 95% Student-t interval <= {latency_limit:g} ms",
        },
    ]
    baseline_eval = baseline["evaluated_count"] > 0
    candidate_eval = candidate["evaluated_count"] > 0
    invalid_evidence_statuses = {
        "invalid_status",
        "invalid_trace",
        "evaluator_infrastructure_error",
    }
    baseline_invalid_evidence = any(
        baseline["status_counts"].get(name, 0) for name in invalid_evidence_statuses
    )
    candidate_invalid_evidence = any(
        candidate["status_counts"].get(name, 0) for name in invalid_evidence_statuses
    )
    baseline_execution_failure = any(
        baseline["status_counts"].get(name, 0)
        for name in {"infrastructure_failure", "timeout"}
    )
    if baseline_invalid_evidence or candidate_invalid_evidence:
        status = "invalid"
        reason = "one or more attempts had invalid status, trace, or evaluator evidence"
    elif not trace_evidence_valid:
        status = "invalid"
        reason = "trace DAG or environment identity evidence was incomplete or unstable"
    elif distinct_identity_required and not backend_identities_distinct:
        status = "invalid"
        reason = (
            "the engine-change target requires one stable backend identity per "
            "variant and an intentional difference between variants"
        )
    elif baseline_execution_failure:
        status = "invalid"
        reason = "baseline infrastructure or timeout failures invalidated the comparison"
    elif not baseline_eval or not candidate_eval:
        status = "invalid"
        reason = "task success was not evaluable for both variants"
    elif not baseline_complete or not absolute_baseline:
        status = "invalid"
        reason = "the baseline did not establish a valid, sufficiently successful workload"
    elif not candidate_complete or not absolute_candidate:
        status = "regression"
        reason = "the candidate failed completion or absolute task-success guardrails"
    elif (
        candidate["task_success_rate"]
        < baseline["task_success_rate"] - max_task_success_regression
    ):
        status = "regression"
        reason = "candidate task success exceeded the allowed point-estimate regression"
    elif not noninferior:
        status = "inconclusive"
        reason = "the paired success bound did not establish non-inferiority"
    elif latency_gate_passed:
        status = "verified"
        reason = (
            "task success was non-inferior and successful-task latency superiority was established"
            if latency_semantics == "superiority"
            else "task success and successful-task latency non-inferiority were established"
        )
    elif latency_regression:
        status = "regression"
        reason = (
            "task success passed but successful tasks were slower"
            if latency_semantics == "superiority"
            else "task success passed but successful-task latency exceeded the declared margin"
        )
    else:
        status = "inconclusive"
        reason = "quality passed but successful-task latency improvement was not established"
    return enrich_decision_artifact(
        {"status": status, "reason": reason, "checks": checks}, summary
    )


class _AgentTrialExecutor:
    """Run one agent-process attempt per scheduled trial (thread-safe)."""

    def __init__(
        self,
        *,
        tasks: list[dict[str, Any]],
        commands: Mapping[str, list[str]],
        seed: int,
        workdir: Path,
        artifacts_dir: Path,
        evaluator: tuple[str, Any],
        timeout_seconds: float,
        evaluator_timeout_seconds: float,
    ) -> None:
        self.tasks = tasks
        self.commands = commands
        self.seed = seed
        self.workdir = workdir
        self.artifacts_dir = artifacts_dir
        self.evaluator = evaluator
        self.timeout_seconds = timeout_seconds
        self.evaluator_timeout_seconds = evaluator_timeout_seconds

    def execute(
        self, trial: ScheduledTrial, *, role: str, filler_index: int | None
    ) -> dict[str, Any]:
        prefix = (
            f"filler-{filler_index:05d}"
            if role == ROLE_FILLER
            else f"{trial.index + 1:05d}"
        )
        return _run_attempt(
            argv=self.commands[trial.variant],
            task=self.tasks[trial.unit],
            variant=trial.variant,
            repetition=trial.repetition,
            sequence=trial.index + 1,
            seed=self.seed,
            workdir=self.workdir,
            artifacts_dir=self.artifacts_dir,
            evaluator=self.evaluator,
            timeout_seconds=self.timeout_seconds,
            evaluator_timeout_seconds=self.evaluator_timeout_seconds,
            attempt_prefix=prefix,
            extra_environment={
                "RUNECONOMY_LOAD_ROLE": role,
                "RUNECONOMY_LOAD_STAGE": str(trial.stage),
                "RUNECONOMY_LOAD_LEVEL": f"{trial.level:g}",
                "RUNECONOMY_LOAD_BLOCK": str(trial.block),
            },
        )


def _agent_observations(attempts: list[dict[str, Any]]) -> list[Observation]:
    observations: list[Observation] = []
    for attempt in attempts:
        metrics = attempt.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        cost, gpu = economics_of(attempt)
        measured = attempt.get("status") in {"completed", "task_failure"}
        observations.append(
            Observation(
                cluster=str(attempt["task_id"]),
                repetition=int(attempt["repetition"]),
                variant=str(attempt["variant"]),
                success=(
                    None
                    if attempt.get("task_success") is None
                    else bool(attempt["task_success"])
                ),
                workflow_latency_ms=(
                    float(attempt["workflow_latency_ms"]) if measured else None
                ),
                requests={
                    "client_ttft_ms": tuple(metrics.get("request_client_ttft_ms", ())),
                    "server_ttft_ms": tuple(metrics.get("request_server_ttft_ms", ())),
                    "tpot_ms": tuple(metrics.get("request_tpot_ms", ())),
                },
                cost_usd=cost,
                gpu_time_ms=gpu,
            )
        )
    return observations


def agent_arrival_schedule(
    load: LoadSpec, tasks: list[dict[str, Any]], *, repetitions: int, seed: int, order: str
) -> ArrivalSchedule:
    """Frozen schedule over task-ID digests, so it can be shared without task text."""

    try:
        return build_arrival_schedule(
            load,
            [_digest_bytes(str(task["task_id"]).encode()) for task in tasks],
            repetitions=repetitions,
            seed=seed,
            order=order,
        )
    except LoadSpecError as exc:
        raise ExperimentValidationError(str(exc)) from exc


def run_comparison(
    *,
    baseline_command: str,
    candidate_command: str,
    dataset_path: Path,
    evaluator_path: Path,
    output_path: Path,
    artifacts_dir: Path,
    workdir: Path,
    repetitions: int = 1,
    order: str = "randomized",
    seed: int = 0,
    timeout_seconds: float = 300.0,
    evaluator_timeout_seconds: float = 30.0,
    max_task_success_regression: float = 0.05,
    min_task_success_rate: float = 0.8,
    min_completed_rate: float = 1.0,
    optimization_target: str = "successful_workflow_latency",
    latency_semantics: str = "superiority",
    max_latency_regression_ms: float = 0.0,
    resume: bool = False,
    task_success_method: str = PAIRED_EXACT,
    completion_policy: Mapping[str, Any] | None = None,
    identity_policy: Mapping[str, Any] | None = None,
    arrival_pattern: str = SEQUENTIAL,
    load: Mapping[str, Any] | None = None,
    metric_gates: list[Mapping[str, Any]] | None = None,
    observers: Any = (),
    attempt_economics: Any = None,
) -> dict[str, Any]:
    """Execute and compare two real agent commands on paired tasks.

    ``task_success_method`` and ``completion_policy`` are preregistered
    acceptance semantics. Their defaults reproduce the original contract, and
    they enter the frozen design only when a non-default value is declared.
    ``arrival_pattern``/``load`` and ``metric_gates`` follow
    ``docs/load-pairing-and-tail-gates.md`` and likewise enter the design only
    when declared. ``observers`` and ``attempt_economics`` are the telemetry
    and cost seams; neither changes the frozen design.
    """

    if repetitions <= 0:
        raise ExperimentValidationError("repetitions must be positive")
    if order not in {"randomized", "alternating"}:
        raise ExperimentValidationError("order must be randomized or alternating")
    if timeout_seconds <= 0 or evaluator_timeout_seconds <= 0:
        raise ExperimentValidationError("timeouts must be positive")
    if optimization_target not in {
        "successful_workflow_latency",
        "engine_change_latency",
    }:
        raise ExperimentValidationError(
            "optimization_target must be successful_workflow_latency or engine_change_latency"
        )
    if latency_semantics not in {"superiority", "noninferiority"}:
        raise ExperimentValidationError(
            "latency_semantics must be superiority or noninferiority"
        )
    if (
        isinstance(max_latency_regression_ms, bool)
        or not isinstance(max_latency_regression_ms, (int, float))
        or max_latency_regression_ms < 0
    ):
        raise ExperimentValidationError(
            "max_latency_regression_ms must be non-negative"
        )
    for name, value in (
        ("max_task_success_regression", max_task_success_regression),
        ("min_task_success_rate", min_task_success_rate),
        ("min_completed_rate", min_completed_rate),
    ):
        if value < 0 or value > 1:
            raise ExperimentValidationError(f"{name} must be between 0 and 1")
    if task_success_method not in PAIRED_METHODS:
        raise ExperimentValidationError(
            "task_success_method must be paired_exact or paired_score_exact"
        )
    try:
        parsed_policy = (
            None
            if completion_policy is None
            else CompletionPolicy.from_value(completion_policy)
        )
    except CompletionPolicyError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    try:
        parsed_identity_policy = parse_identity_policy(identity_policy)
    except IdentityPolicyError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    parsed_load: LoadSpec | None = None
    if arrival_pattern != SEQUENTIAL:
        if arrival_pattern not in CONCURRENT_ARRIVAL_PATTERNS:
            raise ExperimentValidationError(
                "arrival_pattern must be sequential, closed_loop, constant_rps, or poisson"
            )
        try:
            parsed_load = LoadSpec.from_value(arrival_pattern, load)
        except LoadSpecError as exc:
            raise ExperimentValidationError(str(exc)) from exc
        if resume:
            raise ExperimentValidationError(
                "resume is not supported for concurrent arrival patterns; a "
                "partial block cannot be replayed under the same load"
            )
    elif load is not None:
        raise ExperimentValidationError("load requires a concurrent arrival_pattern")
    try:
        parsed_gates = parse_metric_gates(
            None if metric_gates is None else list(metric_gates)
        )
    except MetricGateError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    workdir = workdir.resolve()
    if not workdir.is_dir():
        raise ExperimentValidationError(f"workdir is not a directory: {workdir}")
    workdir_identity = _workdir_identity(workdir)
    tasks = _load_dataset(dataset_path)
    evaluator = _load_evaluator(evaluator_path)
    commands = {
        "baseline": _command(baseline_command, "baseline"),
        "candidate": _command(candidate_command, "candidate"),
    }
    design = {
        "baseline_command_digest": _digest_bytes(baseline_command.encode()),
        "candidate_command_digest": _digest_bytes(candidate_command.encode()),
        "commands": {
            variant: _command_identity(argv, workdir)
            for variant, argv in commands.items()
        },
        "dataset_sha256": _digest_path(dataset_path),
        "evaluator_sha256": _digest_path(evaluator_path),
        "task_count": len(tasks),
        "repetitions": repetitions,
        "order": order,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "evaluator_timeout_seconds": evaluator_timeout_seconds,
        "workdir_identity": workdir_identity,
        "acceptance": {
            "primary_metric": "workflow_latency_ms on paired successful tasks",
            "optimization_target": optimization_target,
            "latency_semantics": latency_semantics,
            "max_latency_regression_ms": float(max_latency_regression_ms),
            "max_task_success_regression": max_task_success_regression,
            "min_task_success_rate": min_task_success_rate,
            "min_completed_rate": min_completed_rate,
        },
    }
    # Non-default semantics are added only when declared so existing frozen
    # designs keep their fingerprint and resume checks.
    if task_success_method != PAIRED_EXACT:
        design["acceptance"]["task_success_method"] = task_success_method
    if parsed_policy is not None:
        design["acceptance"]["completion_policy"] = parsed_policy.to_dict()
    if parsed_identity_policy is not None:
        design["acceptance"]["identity_policy"] = parsed_identity_policy.to_dict()
    if parsed_gates:
        design["acceptance"]["metric_gates"] = [gate.to_dict() for gate in parsed_gates]
    cost_model_record = getattr(attempt_economics, "to_dict", None)
    if callable(cost_model_record):
        design["cost_model"] = {
            "model": cost_model_record(),
            "sha256": getattr(attempt_economics, "sha256", None),
        }
    arrival_schedule: ArrivalSchedule | None = None
    if parsed_load is not None:
        arrival_schedule = agent_arrival_schedule(
            parsed_load, tasks, repetitions=repetitions, seed=seed, order=order
        )
        try:
            verify_schedule_pin(arrival_schedule)
        except LoadSpecError as exc:
            raise ExperimentValidationError(str(exc)) from exc
        design["execution_load"] = {
            "arrival_pattern": arrival_pattern,
            "load": parsed_load.to_dict(),
            "arrival_schedule_sha256": arrival_schedule.digest(),
        }
    design["fingerprint"] = _digest_bytes(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode()
    )
    if parsed_load is not None:
        assert arrival_schedule is not None
        return _run_comparison_under_load(
            design=design,
            load=parsed_load,
            schedule=arrival_schedule,
            tasks=tasks,
            commands=commands,
            seed=seed,
            workdir=workdir,
            workdir_identity=workdir_identity,
            artifacts_dir=artifacts_dir,
            output_path=output_path,
            evaluator=evaluator,
            timeout_seconds=timeout_seconds,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
            observers=tuple(observers),
            attempt_economics=attempt_economics,
            decision_arguments={
                "max_task_success_regression": max_task_success_regression,
                "min_task_success_rate": min_task_success_rate,
                "min_completed_rate": min_completed_rate,
                "optimization_target": optimization_target,
                "latency_semantics": latency_semantics,
                "max_latency_regression_ms": float(max_latency_regression_ms),
                "task_success_method": task_success_method,
            },
            completion_policy=parsed_policy,
            metric_gates=parsed_gates,
            identity_policy=parsed_identity_policy,
        )
    schedule = _schedule(tasks, repetitions, order, seed)
    if artifacts_dir.exists():
        if not resume:
            raise ExperimentValidationError(
                f"artifacts directory already exists: {artifacts_dir}"
            )
        design_path = artifacts_dir / "design.json"
        try:
            stored_design = json.loads(design_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentValidationError(
                "resume requires a readable design.json checkpoint"
            ) from exc
        comparable_stored = dict(stored_design)
        comparable_current = dict(design)
        for value in (comparable_stored, comparable_current):
            value.pop("fingerprint", None)
            value.pop("workdir_identity", None)
        if comparable_stored != comparable_current:
            raise ExperimentValidationError(
                "resume design does not match commands, data, evaluator, schedule, or gates"
            )
        design = stored_design
        attempts = _load_attempt_checkpoint(
            artifacts_dir / "attempts.jsonl", schedule
        )
    else:
        if resume:
            raise ExperimentValidationError(
                f"resume artifacts directory does not exist: {artifacts_dir}"
            )
        artifacts_dir.mkdir(parents=True)
        _write_json(artifacts_dir / "design.json", design)
        attempts = []
    for sequence, (repetition, task, variant) in enumerate(schedule, 1):
        if sequence <= len(attempts):
            continue
        attempt = _run_attempt(
            argv=commands[variant],
            task=task,
            variant=variant,
            repetition=repetition,
            sequence=sequence,
            seed=seed,
            workdir=workdir,
            artifacts_dir=artifacts_dir,
            evaluator=evaluator,
            timeout_seconds=timeout_seconds,
            evaluator_timeout_seconds=evaluator_timeout_seconds,
        )
        attempts.append(attempt)
        with (artifacts_dir / "attempts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    if attempt_economics is not None:
        for attempt in attempts:
            attempt["economics"] = dict(attempt_economics(attempt))
    summary = _summarize(
        attempts,
        task_success_method=task_success_method,
        max_task_success_regression=max_task_success_regression,
        completion_policy=parsed_policy,
        identity_policy=parsed_identity_policy,
    )
    decision = _decision(
        summary,
        max_task_success_regression=max_task_success_regression,
        min_task_success_rate=min_task_success_rate,
        min_completed_rate=min_completed_rate,
        optimization_target=optimization_target,
        latency_semantics=latency_semantics,
        max_latency_regression_ms=float(max_latency_regression_ms),
        task_success_method=task_success_method,
    )
    if parsed_gates:
        decision = enrich_decision_artifact(
            merge_load_and_metric_checks(
                decision,
                integrity_checks=[],
                metric_checks=evaluate_metric_gates(
                    parsed_gates, _agent_observations(attempts), seed=seed
                ),
            ),
            summary,
        )
    result = {
        "result_type": "runeconomy.closed_loop_comparison",
        "result_version": COMPARE_RESULT_VERSION,
        "design": design,
        "summary": summary,
        "evidence_quality": summary["evidence_quality"],
        "decision": decision,
        "attempts": attempts,
        "provenance": {
            "runeconomy_version": __import__("runeconomy").__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "workdir_identity": workdir_identity,
            "artifacts_directory": artifacts_dir.name,
            "result_path": output_path.name,
        },
        "limitations": [
            "Workflow latency includes process startup and agent execution but excludes evaluator time.",
            "Continuous effects use only task pairs where both variants succeeded.",
            "Critical-path latency is available only when every imported work span declares runeconomy.depends_on.",
            "Raw outputs and traces remain in the local artifacts directory and may contain sensitive data.",
        ],
    }
    _write_json(artifacts_dir / "result.json", result)
    return result


def _run_comparison_under_load(
    *,
    design: dict[str, Any],
    load: LoadSpec,
    schedule: ArrivalSchedule,
    tasks: list[dict[str, Any]],
    commands: Mapping[str, list[str]],
    seed: int,
    workdir: Path,
    workdir_identity: Mapping[str, Any],
    artifacts_dir: Path,
    output_path: Path,
    evaluator: tuple[str, Any],
    timeout_seconds: float,
    evaluator_timeout_seconds: float,
    observers: tuple[Any, ...],
    attempt_economics: Any,
    decision_arguments: Mapping[str, Any],
    completion_policy: CompletionPolicy | None,
    metric_gates: tuple[Any, ...],
    identity_policy: IdentityPolicy | None = None,
) -> dict[str, Any]:
    if artifacts_dir.exists():
        raise ExperimentValidationError(
            f"artifacts directory already exists: {artifacts_dir}"
        )
    artifacts_dir.mkdir(parents=True)
    _write_json(artifacts_dir / "design.json", design)
    _write_json(artifacts_dir / "arrival-schedule.json", schedule.to_dict())
    executor = _AgentTrialExecutor(
        tasks=tasks,
        commands=commands,
        seed=seed,
        workdir=workdir,
        artifacts_dir=artifacts_dir,
        evaluator=evaluator,
        timeout_seconds=timeout_seconds,
        evaluator_timeout_seconds=evaluator_timeout_seconds,
    )
    run = LoadRunner(
        schedule,
        executor,
        store=JsonlArtifactStore(artifacts_dir / "attempts.jsonl"),
        observers=observers,
    ).run()
    attempts = run.records
    if attempt_economics is not None:
        for attempt in attempts:
            attempt["economics"] = dict(attempt_economics(attempt))
    gate_stage = load.gate_stage
    summary_arguments = {
        "task_success_method": decision_arguments["task_success_method"],
        "max_task_success_regression": decision_arguments["max_task_success_regression"],
        "completion_policy": completion_policy,
        "identity_policy": identity_policy,
    }
    gated = [attempt for attempt in attempts if attempt["load"]["stage"] == gate_stage]
    summary = _summarize(gated, **summary_arguments)
    stages: list[dict[str, Any]] = []
    for stage, level in enumerate(load.levels):
        if stage == gate_stage:
            continue
        stage_attempts = [
            attempt for attempt in attempts if attempt["load"]["stage"] == stage
        ]
        stages.append(
            {
                "stage": stage,
                load.level_name: level,
                "gated": False,
                "descriptive_only": True,
                "summary": _summarize(stage_attempts, **summary_arguments),
                "metric_gates": evaluate_metric_gates(
                    metric_gates, _agent_observations(stage_attempts), seed=seed
                ),
            }
        )
    summary["load"] = {
        "arrival_schedule": schedule.summary(),
        "gate_stage": gate_stage,
        load.level_name: load.gate_level,
        "population": "gate-stage attempts only; other stages are descriptive",
        "stages": stages,
        "filler_attempts": len(run.fillers),
    }
    decision = _decision(summary, **decision_arguments)
    decision = enrich_decision_artifact(
        merge_load_and_metric_checks(
            decision,
            integrity_checks=load_integrity_checks(
                gated,
                open_loop=load.open_loop,
                gate_level=load.gate_level,
                max_dispatch_lag_ms=load.max_dispatch_lag_ms,
            ),
            metric_checks=evaluate_metric_gates(
                metric_gates, _agent_observations(gated), seed=seed
            ),
        ),
        summary,
    )
    _write_json(
        artifacts_dir / "load-fillers.json",
        {"filler_attempts": run.fillers, "blocks": run.blocks},
    )
    result = {
        "result_type": "runeconomy.closed_loop_comparison",
        "result_version": COMPARE_LOAD_RESULT_VERSION,
        "design": design,
        "summary": summary,
        "evidence_quality": summary["evidence_quality"],
        "decision": decision,
        "attempts": attempts,
        "load_blocks": run.blocks,
        "provenance": {
            "runeconomy_version": __import__("runeconomy").__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "workdir_identity": workdir_identity,
            "artifacts_directory": artifacts_dir.name,
            "result_path": output_path.name,
            "arrival_schedule": "arrival-schedule.json",
            "arrival_schedule_sha256": schedule.digest(),
        },
        "limitations": [
            "Workflow latency includes process startup and agent execution but excludes evaluator time.",
            "Continuous effects use only task pairs where both variants succeeded.",
            "Critical-path latency is available only when every imported work span declares runeconomy.depends_on.",
            "Raw outputs and traces remain in the local artifacts directory and may contain sensitive data.",
            f"Load claim boundary: {load.claim_boundary()}; only the preregistered gate level is gated.",
            "Interval methods assume independent units given the design; transient server stalls shared by concurrent trials can make intervals optimistic.",
        ],
    }
    _write_json(artifacts_dir / "result.json", result)
    return result
