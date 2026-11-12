"""Versioned, reproducible A/B experiment manifests and runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from .cache import longest_common_prefix
from .evaluators import EvaluatorError, evaluate_output, validate_evaluator_spec
from .vllm import CapturedResponse


class ExperimentValidationError(ValueError):
    pass


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentValidationError(f"{location} must be an object")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentValidationError(f"{location} must be a non-empty string")
    return value


def _positive_int(value: Any, location: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ExperimentValidationError(f"{location} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class BackendConfig:
    base_url: str
    api_key_env: str | None
    model: str
    model_revision: str
    tokenizer_revision: str
    chat_template_digest: str
    backend_version: str
    hardware: str
    block_size: int
    engine_config: dict[str, Any]


@dataclass(frozen=True)
class ExecutionConfig:
    repetitions: int
    warmup_repetitions: int
    order: str
    seed: int
    cache_treatment: str
    arrival_pattern: str
    reset_external: bool
    require_evaluation: bool
    cases_form_workflow: bool


@dataclass(frozen=True)
class AcceptanceConfig:
    primary_metric: str
    optimization_target: str
    max_task_success_regression: float
    min_completed_rate: float
    min_cached_token_gain: float
    min_input_token_reduction: float
    require_latency_improvement: bool


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    evaluator: dict[str, Any] | None
    baseline_transform: dict[str, Any] | None
    candidate_transform: dict[str, Any] | None


@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: str
    experiment_id: str
    backend: BackendConfig
    sampling: dict[str, Any]
    execution: ExecutionConfig
    acceptance: AcceptanceConfig
    constants: dict[str, Any]
    cases: tuple[ExperimentCase, ...]

    @classmethod
    def from_dict(cls, payload: Any) -> "ExperimentManifest":
        root = _object(payload, "manifest")
        version = _string(root.get("schema_version"), "schema_version")
        if version != "0.1":
            raise ExperimentValidationError(
                f"unsupported schema_version {version!r}; expected '0.1'"
            )
        experiment_id = _string(root.get("experiment_id"), "experiment_id")

        backend_data = _object(root.get("backend"), "backend")
        if backend_data.get("kind") != "vllm":
            raise ExperimentValidationError("backend.kind must be 'vllm'")
        backend = BackendConfig(
            base_url=_string(backend_data.get("base_url"), "backend.base_url"),
            api_key_env=(
                None
                if backend_data.get("api_key_env") is None
                else _string(backend_data.get("api_key_env"), "backend.api_key_env")
            ),
            model=_string(backend_data.get("model"), "backend.model"),
            model_revision=_string(
                backend_data.get("model_revision"), "backend.model_revision"
            ),
            tokenizer_revision=_string(
                backend_data.get("tokenizer_revision"), "backend.tokenizer_revision"
            ),
            chat_template_digest=_string(
                backend_data.get("chat_template_digest"),
                "backend.chat_template_digest",
            ),
            backend_version=_string(
                backend_data.get("backend_version"), "backend.backend_version"
            ),
            hardware=_string(backend_data.get("hardware"), "backend.hardware"),
            block_size=_positive_int(
                backend_data.get("block_size", 16), "backend.block_size"
            ),
            engine_config=_object(
                backend_data.get("engine_config"), "backend.engine_config"
            ),
        )
        if not backend.engine_config:
            raise ExperimentValidationError(
                "backend.engine_config must pin the serving configuration"
            )
        if backend.engine_config.get("enable_prefix_caching") is not True:
            raise ExperimentValidationError(
                "backend.engine_config.enable_prefix_caching must be true"
            )

        execution_data = _object(root.get("execution"), "execution")
        order = execution_data.get("order", "alternating")
        if order not in {"alternating", "randomized"}:
            raise ExperimentValidationError(
                "execution.order must be 'alternating' or 'randomized'"
            )
        cache_treatment = execution_data.get("cache_treatment", "warm")
        if cache_treatment not in {"warm", "cold"}:
            raise ExperimentValidationError(
                "execution.cache_treatment must be 'warm' or 'cold'"
            )
        arrival_pattern = execution_data.get("arrival_pattern", "sequential")
        if arrival_pattern != "sequential":
            raise ExperimentValidationError(
                "execution.arrival_pattern currently supports only 'sequential'"
            )
        seed = execution_data.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ExperimentValidationError("execution.seed must be an integer")
        reset_external = execution_data.get("reset_external", False)
        if not isinstance(reset_external, bool):
            raise ExperimentValidationError("execution.reset_external must be boolean")
        require_evaluation = execution_data.get("require_evaluation", False)
        if not isinstance(require_evaluation, bool):
            raise ExperimentValidationError(
                "execution.require_evaluation must be boolean"
            )
        cases_form_workflow = execution_data.get("cases_form_workflow", False)
        if not isinstance(cases_form_workflow, bool):
            raise ExperimentValidationError(
                "execution.cases_form_workflow must be boolean"
            )
        execution = ExecutionConfig(
            repetitions=_positive_int(
                execution_data.get("repetitions"), "execution.repetitions"
            ),
            warmup_repetitions=_positive_int(
                execution_data.get("warmup_repetitions", 0),
                "execution.warmup_repetitions",
                allow_zero=True,
            ),
            order=order,
            seed=seed,
            cache_treatment=cache_treatment,
            arrival_pattern=arrival_pattern,
            reset_external=reset_external,
            require_evaluation=require_evaluation,
            cases_form_workflow=cases_form_workflow,
        )

        acceptance_data = _object(root.get("acceptance", {}), "acceptance")
        primary_metric = acceptance_data.get("primary_metric", "total_latency_ms")
        if primary_metric not in {
            "client_ttft_ms",
            "server_ttft_ms",
            "total_latency_ms",
        }:
            raise ExperimentValidationError(
                "acceptance.primary_metric must be client_ttft_ms, "
                "server_ttft_ms, or total_latency_ms"
            )
        optimization_target = acceptance_data.get("optimization_target", "cache_reuse")
        if optimization_target not in {"cache_reuse", "input_tokens"}:
            raise ExperimentValidationError(
                "acceptance.optimization_target must be cache_reuse or input_tokens"
            )

        def probability(name: str, default: float) -> float:
            value = acceptance_data.get(name, default)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= float(value) <= 1
            ):
                raise ExperimentValidationError(
                    f"acceptance.{name} must be between 0 and 1"
                )
            return float(value)

        min_cached_token_gain = acceptance_data.get("min_cached_token_gain", 0)
        if (
            not isinstance(min_cached_token_gain, (int, float))
            or isinstance(min_cached_token_gain, bool)
            or min_cached_token_gain < 0
        ):
            raise ExperimentValidationError(
                "acceptance.min_cached_token_gain must be non-negative"
            )
        min_input_token_reduction = acceptance_data.get("min_input_token_reduction", 0)
        if (
            not isinstance(min_input_token_reduction, (int, float))
            or isinstance(min_input_token_reduction, bool)
            or min_input_token_reduction < 0
        ):
            raise ExperimentValidationError(
                "acceptance.min_input_token_reduction must be non-negative"
            )
        require_latency = acceptance_data.get("require_latency_improvement", True)
        if not isinstance(require_latency, bool):
            raise ExperimentValidationError(
                "acceptance.require_latency_improvement must be boolean"
            )
        acceptance = AcceptanceConfig(
            primary_metric=primary_metric,
            optimization_target=optimization_target,
            max_task_success_regression=probability("max_task_success_regression", 0.0),
            min_completed_rate=probability("min_completed_rate", 1.0),
            min_cached_token_gain=float(min_cached_token_gain),
            min_input_token_reduction=float(min_input_token_reduction),
            require_latency_improvement=require_latency,
        )

        constants_data = _object(root.get("constants", {}), "constants")
        constants: dict[str, Any] = {}
        for name, value in constants_data.items():
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[A-Z][A-Z0-9_]*", name) is None
            ):
                raise ExperimentValidationError(
                    "constant names must match [A-Z][A-Z0-9_]*"
                )
            constants[name] = deepcopy(value)

        sampling = _object(root.get("sampling"), "sampling")
        if not sampling:
            raise ExperimentValidationError("sampling must pin at least one setting")
        unsafe_sampling_keys = {
            "input",
            "instructions",
            "model",
            "stream",
            "tools",
            "previous_response_id",
        }.intersection(sampling)
        if unsafe_sampling_keys:
            raise ExperimentValidationError(
                "sampling contains non-sampling fields: "
                + ", ".join(sorted(unsafe_sampling_keys))
            )

        cases_data = root.get("cases")
        if not isinstance(cases_data, list) or not cases_data:
            raise ExperimentValidationError("cases must be a non-empty array")
        cases: list[ExperimentCase] = []
        seen_ids: set[str] = set()
        forbidden = {"stream", "previous_response_id", "background"}
        for index, raw_case in enumerate(cases_data):
            case_data = _object(raw_case, f"cases[{index}]")
            case_id = _string(case_data.get("case_id"), f"cases[{index}].case_id")
            if case_id in seen_ids:
                raise ExperimentValidationError(f"duplicate case_id {case_id!r}")
            seen_ids.add(case_id)
            variants = _object(case_data.get("variants"), f"cases[{index}].variants")
            if set(variants) != {"baseline", "candidate"}:
                raise ExperimentValidationError(
                    f"cases[{index}].variants must contain exactly baseline and candidate"
                )
            requests: dict[str, dict[str, Any]] = {}
            transforms: dict[str, dict[str, Any] | None] = {}
            evaluator_data = case_data.get("evaluator")
            evaluator: dict[str, Any] | None = None
            if evaluator_data is not None:
                try:
                    evaluator = validate_evaluator_spec(
                        evaluator_data, f"cases[{index}].evaluator"
                    )
                except EvaluatorError as exc:
                    raise ExperimentValidationError(str(exc)) from exc
            elif execution.require_evaluation:
                raise ExperimentValidationError(
                    f"cases[{index}].evaluator is required by execution.require_evaluation"
                )
            for variant in ("baseline", "candidate"):
                variant_data = _object(
                    variants[variant], f"cases[{index}].variants.{variant}"
                )
                request = _object(
                    variant_data.get("request"),
                    f"cases[{index}].variants.{variant}.request",
                )
                present = forbidden.intersection(request)
                if present:
                    names = ", ".join(sorted(present))
                    raise ExperimentValidationError(
                        f"{case_id}/{variant} request contains runner-controlled or "
                        f"stateful fields: {names}"
                    )
                overlap = set(request).intersection(sampling)
                if overlap:
                    raise ExperimentValidationError(
                        f"{case_id}/{variant} overrides pinned sampling fields: "
                        + ", ".join(sorted(overlap))
                    )
                request_model = request.get("model")
                if request_model is not None and request_model != backend.model:
                    raise ExperimentValidationError(
                        f"{case_id}/{variant} model differs from backend.model"
                    )
                requests[variant] = dict(request)
                transform_value = variant_data.get("request_transform")
                if transform_value is None:
                    transforms[variant] = None
                else:
                    transform = _object(
                        transform_value,
                        f"cases[{index}].variants.{variant}.request_transform",
                    )
                    if transform.get("kind") not in {
                        "shuffle_tools",
                        "canonicalize_tools",
                    }:
                        raise ExperimentValidationError(
                            f"{case_id}/{variant} request_transform.kind must be "
                            "shuffle_tools or canonicalize_tools"
                        )
                    tools = request.get("tools")
                    is_fixture = isinstance(tools, dict) and set(tools) == {"$fixture"}
                    if not isinstance(tools, list) and not is_fixture:
                        raise ExperimentValidationError(
                            f"{case_id}/{variant} tool transform requires request.tools "
                            "to be an array or $fixture reference"
                        )
                    transforms[variant] = dict(transform)
            cases.append(
                ExperimentCase(
                    case_id,
                    requests["baseline"],
                    requests["candidate"],
                    evaluator,
                    transforms["baseline"],
                    transforms["candidate"],
                )
            )

        manifest = cls(
            version,
            experiment_id,
            backend,
            dict(sampling),
            execution,
            acceptance,
            constants,
            tuple(cases),
        )
        _validate_manifest_placeholders(manifest)
        return manifest

    def fingerprint(self) -> str:
        canonical = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def uses_executable_evaluators(self) -> bool:
        """Return whether a manifest can execute local Python or commands."""

        def executable(spec: Mapping[str, Any] | None) -> bool:
            if spec is None:
                return False
            if spec.get("kind") in {"python", "command"}:
                return True
            return spec.get("kind") == "all" and any(
                executable(child)
                for child in spec.get("evaluators", [])
                if isinstance(child, dict)
            )

        return any(executable(case.evaluator) for case in self.cases)

    @classmethod
    def from_path(cls, path: Path) -> "ExperimentManifest":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise ExperimentValidationError(f"cannot read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ExperimentValidationError(
                f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
            ) from exc
        root = _object(payload, "manifest")
        fixtures = root.pop("text_fixtures", [])
        if not isinstance(fixtures, list):
            raise ExperimentValidationError("text_fixtures must be an array")
        constants = _object(root.setdefault("constants", {}), "constants")
        for index, fixture_value in enumerate(fixtures):
            fixture = _object(fixture_value, f"text_fixtures[{index}]")
            name = _string(fixture.get("name"), f"text_fixtures[{index}].name")
            if name in constants:
                raise ExperimentValidationError(
                    f"duplicate constant or fixture name {name!r}"
                )
            relative_path = _string(fixture.get("path"), f"text_fixtures[{index}].path")
            expected_digest = _string(
                fixture.get("sha256"), f"text_fixtures[{index}].sha256"
            )
            fixture_path = (path.parent / relative_path).resolve()
            try:
                text = fixture_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ExperimentValidationError(
                    f"cannot read fixture {fixture_path}: {exc}"
                ) from exc
            observed_digest = (
                "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            )
            if observed_digest != expected_digest:
                raise ExperimentValidationError(
                    f"fixture {name!r} digest mismatch: expected {expected_digest}, "
                    f"observed {observed_digest}"
                )
            constants[name] = text
        json_fixtures = root.pop("json_fixtures", [])
        if not isinstance(json_fixtures, list):
            raise ExperimentValidationError("json_fixtures must be an array")
        for index, fixture_value in enumerate(json_fixtures):
            fixture = _object(fixture_value, f"json_fixtures[{index}]")
            name = _string(fixture.get("name"), f"json_fixtures[{index}].name")
            if name in constants:
                raise ExperimentValidationError(
                    f"duplicate constant or fixture name {name!r}"
                )
            relative_path = _string(fixture.get("path"), f"json_fixtures[{index}].path")
            expected_digest = _string(
                fixture.get("sha256"), f"json_fixtures[{index}].sha256"
            )
            fixture_path = (path.parent / relative_path).resolve()
            try:
                raw = fixture_path.read_bytes()
                document = json.loads(raw)
            except OSError as exc:
                raise ExperimentValidationError(
                    f"cannot read fixture {fixture_path}: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ExperimentValidationError(
                    f"JSON fixture {fixture_path} is invalid: {exc.msg}"
                ) from exc
            observed_digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            if observed_digest != expected_digest:
                raise ExperimentValidationError(
                    f"fixture {name!r} digest mismatch: expected {expected_digest}, "
                    f"observed {observed_digest}"
                )
            constants[name] = document
        return cls.from_dict(root)


class CaptureClient(Protocol):
    def capture(self, request_body: Mapping[str, Any]) -> CapturedResponse: ...

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None: ...


def _materialize_trial(value: Any, replacements: Mapping[str, Any]) -> Any:
    """Replace deterministic trial and fixture placeholders recursively."""

    if isinstance(value, str):
        result = value
        for name, replacement in replacements.items():
            if isinstance(replacement, str):
                result = result.replace("{{" + name + "}}", replacement)
        unresolved = re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", result)
        if unresolved:
            raise ExperimentValidationError(
                f"unresolved request placeholder {unresolved.group(0)}"
            )
        return result
    if isinstance(value, list):
        return [_materialize_trial(item, replacements) for item in value]
    if isinstance(value, dict):
        if set(value) == {"$fixture"}:
            name = value["$fixture"]
            if not isinstance(name, str) or name not in replacements:
                raise ExperimentValidationError(
                    f"unknown structured fixture reference {name!r}"
                )
            return deepcopy(replacements[name])
        return {
            key: _materialize_trial(item, replacements) for key, item in value.items()
        }
    return value


def _validate_manifest_placeholders(manifest: ExperimentManifest) -> None:
    replacements = dict(manifest.constants)
    replacements["AID_TRIAL_ID"] = "validation-trial"
    for case in manifest.cases:
        for variant, request in (
            ("baseline", case.baseline),
            ("candidate", case.candidate),
        ):
            try:
                _materialize_trial(request, replacements)
            except ExperimentValidationError as exc:
                raise ExperimentValidationError(
                    f"{case.case_id}/{variant}: {exc}"
                ) from exc


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    name = tool.get("name")
    if isinstance(name, str):
        return name
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return ""


def _apply_request_transform(
    request: dict[str, Any], transform: Mapping[str, Any] | None, seed: int
) -> dict[str, Any]:
    result = deepcopy(request)
    if transform is None:
        return result
    tools = result.get("tools")
    if not isinstance(tools, list) or len(tools) < 2:
        raise ExperimentValidationError(
            f"{transform.get('kind')} requires at least two materialized tools"
        )
    kind = transform["kind"]
    if kind == "shuffle_tools":
        random.Random(seed).shuffle(tools)
    elif kind == "canonicalize_tools":
        tools.sort(
            key=lambda tool: (
                _tool_name(tool),
                json.dumps(tool, sort_keys=True, separators=(",", ":")),
            )
        )
    else:  # Protected by manifest validation.
        raise ExperimentValidationError(f"unsupported request transform {kind!r}")
    return result


def _predicted_eligible_tokens(
    token_ids: tuple[int, ...], prior_prompts: list[tuple[int, ...]], block_size: int
) -> int:
    if not token_ids or not prior_prompts:
        return 0
    shared = max(longest_common_prefix(token_ids, prior) for prior in prior_prompts)
    bounded = min(shared, len(token_ids) - 1)
    return 0 if bounded <= 0 else (bounded // block_size) * block_size


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
    }


def _bootstrap_interval(
    values: list[float], *, seed: int, samples: int = 2000
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    ]
    return [
        _percentile(means, 0.025) or 0.0,
        _percentile(means, 0.975) or 0.0,
    ]


def _schedule(
    manifest: ExperimentManifest, repetitions: int, *, warmup: bool
) -> list[tuple[int, ExperimentCase, str, bool]]:
    rng = random.Random(manifest.execution.seed + (1 if warmup else 0))
    schedule: list[tuple[int, ExperimentCase, str, bool]] = []
    for repetition in range(repetitions):
        for case in manifest.cases:
            variants = ["baseline", "candidate"]
            if manifest.execution.order == "randomized":
                rng.shuffle(variants)
            elif repetition % 2:
                variants.reverse()
            schedule.extend((repetition, case, variant, warmup) for variant in variants)
    return schedule


def _build_summary(trials: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    measured = [trial for trial in trials if not trial["warmup"]]
    metric_names = (
        "client_ttft_ms",
        "total_latency_ms",
        "server_ttft_ms",
        "server_queue_ms",
        "generation_time_ms",
        "cached_tokens",
        "input_tokens",
        "output_tokens",
        "cache_coverage",
        "predicted_eligible_cached_tokens",
        "task_success",
    )
    variants: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        subset = [trial for trial in measured if trial["variant"] == variant]
        variants[variant] = {
            metric: _metric_summary(
                [
                    float(trial[metric])
                    for trial in subset
                    if trial.get(metric) is not None
                ]
            )
            for metric in metric_names
        }
        status_counts: dict[str, int] = {}
        for trial in subset:
            status = trial.get("status") or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
        variants[variant]["outcomes"] = {
            "status_counts": dict(sorted(status_counts.items())),
            "completed_rate": (
                None if not subset else status_counts.get("completed", 0) / len(subset)
            ),
            "evaluated_count": sum(
                trial.get("task_success") is not None for trial in subset
            ),
            "task_success_rate": (
                None
                if not any(trial.get("task_success") is not None for trial in subset)
                else statistics.fmean(
                    float(trial["task_success"])
                    for trial in subset
                    if trial.get("task_success") is not None
                )
            ),
        }

    by_pair = {
        (trial["case_id"], trial["repetition"], trial["variant"]): trial
        for trial in measured
    }
    effects: dict[str, Any] = {}
    for metric in metric_names:
        differences: list[float] = []
        for case in {trial["case_id"] for trial in measured}:
            for repetition in {trial["repetition"] for trial in measured}:
                baseline = by_pair.get((case, repetition, "baseline"))
                candidate = by_pair.get((case, repetition, "candidate"))
                if (
                    baseline is not None
                    and candidate is not None
                    and baseline.get(metric) is not None
                    and candidate.get(metric) is not None
                ):
                    differences.append(
                        float(candidate[metric]) - float(baseline[metric])
                    )
        baseline_mean = variants["baseline"][metric]["mean"]
        mean_difference = statistics.fmean(differences) if differences else None
        effects[metric] = {
            "paired_count": len(differences),
            "candidate_minus_baseline_mean": mean_difference,
            "relative_change": (
                None
                if mean_difference is None or baseline_mean in (None, 0)
                else mean_difference / baseline_mean
            ),
            "bootstrap_95_ci": _bootstrap_interval(
                differences, seed=seed + sum(ord(char) for char in metric)
            ),
        }
    digest_pairs = 0
    digest_matches = 0
    for case in {trial["case_id"] for trial in measured}:
        for repetition in {trial["repetition"] for trial in measured}:
            baseline = by_pair.get((case, repetition, "baseline"))
            candidate = by_pair.get((case, repetition, "candidate"))
            if (
                baseline is not None
                and candidate is not None
                and baseline.get("output_digest") is not None
                and candidate.get("output_digest") is not None
            ):
                digest_pairs += 1
                digest_matches += (
                    baseline["output_digest"] == candidate["output_digest"]
                )
    predicted_baseline = variants["baseline"]["predicted_eligible_cached_tokens"][
        "mean"
    ]
    predicted_candidate = variants["candidate"]["predicted_eligible_cached_tokens"][
        "mean"
    ]
    actual_delta = effects["cached_tokens"]["candidate_minus_baseline_mean"]
    predicted_delta = (
        None
        if predicted_baseline is None or predicted_candidate is None
        else predicted_candidate - predicted_baseline
    )
    return {
        "variants": variants,
        "effects": effects,
        "prediction_vs_observation": {
            "predicted_candidate_minus_baseline_cached_tokens": predicted_delta,
            "observed_candidate_minus_baseline_cached_tokens": actual_delta,
            "realization_ratio": (
                None
                if predicted_delta in (None, 0) or actual_delta is None
                else actual_delta / predicted_delta
            ),
            "claim": "prediction is a full-block static upper bound under the observed request order",
        },
        "paired_output_digest": {
            "pair_count": digest_pairs,
            "exact_match_rate": (
                None if not digest_pairs else digest_matches / digest_pairs
            ),
            "claim": "exact-byte-equivalent output only; not semantic correctness",
        },
    }


def _build_workflow_summary(
    trials: list[dict[str, Any]], seed: int, expected_case_count: int
) -> dict[str, Any]:
    """Aggregate sequential cases into one critical-path workflow trial."""

    measured = [trial for trial in trials if not trial["warmup"]]
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for trial in measured:
        groups.setdefault((trial["repetition"], trial["variant"]), []).append(trial)

    additive_metrics = (
        "client_ttft_ms",
        "total_latency_ms",
        "server_ttft_ms",
        "server_queue_ms",
        "generation_time_ms",
        "cached_tokens",
        "input_tokens",
        "output_tokens",
        "predicted_eligible_cached_tokens",
    )
    aggregate_trials: list[dict[str, Any]] = []
    for (repetition, variant), steps in sorted(groups.items()):
        if len(steps) != expected_case_count:
            raise ExperimentValidationError(
                f"workflow repetition {repetition}/{variant} has {len(steps)} "
                f"steps; expected {expected_case_count}"
            )
        aggregate: dict[str, Any] = {
            "case_id": "workflow",
            "repetition": repetition,
            "variant": variant,
            "warmup": False,
            "status": (
                "completed"
                if all(step.get("status") == "completed" for step in steps)
                else "incomplete"
            ),
            "task_success": (
                None
                if any(step.get("task_success") is None for step in steps)
                else all(bool(step["task_success"]) for step in steps)
            ),
            "output_digest": hashlib.sha256(
                "|".join(str(step.get("output_digest")) for step in steps).encode(
                    "utf-8"
                )
            ).hexdigest(),
        }
        for metric in additive_metrics:
            values = [step.get(metric) for step in steps]
            aggregate[metric] = (
                None if any(value is None for value in values) else sum(values)
            )
        input_tokens = aggregate["input_tokens"]
        cached_tokens = aggregate["cached_tokens"]
        aggregate["cache_coverage"] = (
            None
            if input_tokens in (None, 0) or cached_tokens is None
            else cached_tokens / input_tokens
        )
        aggregate_trials.append(aggregate)

    summary = _build_summary(aggregate_trials, seed)
    summary["claim"] = (
        "cases are treated as sequential critical-path inference steps; "
        "reported workflow latency is their sum and excludes tool runtime"
    )
    summary["step_count"] = expected_case_count
    return summary


def _build_decision(
    summary: Mapping[str, Any], acceptance: AcceptanceConfig
) -> dict[str, Any]:
    baseline_outcomes = summary["variants"]["baseline"]["outcomes"]
    candidate_outcomes = summary["variants"]["candidate"]["outcomes"]
    baseline_success = baseline_outcomes["task_success_rate"]
    candidate_success = candidate_outcomes["task_success_rate"]
    baseline_completed = baseline_outcomes["completed_rate"]
    candidate_completed = candidate_outcomes["completed_rate"]
    cached_effect = summary["effects"]["cached_tokens"]
    input_effect = summary["effects"]["input_tokens"]
    latency_effect = summary["effects"][acceptance.primary_metric]
    prediction = summary["prediction_vs_observation"]

    quality_available = baseline_success is not None and candidate_success is not None
    quality_noninferior = (
        quality_available
        and candidate_success
        >= baseline_success - acceptance.max_task_success_regression
    )
    completion_ok = (
        baseline_completed is not None
        and candidate_completed is not None
        and baseline_completed >= acceptance.min_completed_rate
        and candidate_completed >= acceptance.min_completed_rate
    )
    cache_interval = cached_effect["bootstrap_95_ci"]
    cache_ok = (
        cache_interval is not None
        and cache_interval[0] >= acceptance.min_cached_token_gain
    )
    prediction_aligned = (
        prediction["predicted_candidate_minus_baseline_cached_tokens"] is not None
        and prediction["predicted_candidate_minus_baseline_cached_tokens"] > 0
        and prediction["observed_candidate_minus_baseline_cached_tokens"] is not None
        and prediction["observed_candidate_minus_baseline_cached_tokens"] > 0
    )
    input_interval = input_effect["bootstrap_95_ci"]
    input_reduction_ok = (
        input_interval is not None
        and input_interval[1] <= -acceptance.min_input_token_reduction
    )
    latency_interval = latency_effect["bootstrap_95_ci"]
    latency_ok = latency_interval is not None and latency_interval[1] < 0
    required_latency_ok = latency_ok or not acceptance.require_latency_improvement

    cache_required = acceptance.optimization_target == "cache_reuse"
    input_required = acceptance.optimization_target == "input_tokens"
    optimization_ok = (
        cache_ok and prediction_aligned if cache_required else input_reduction_ok
    )

    checks = [
        {
            "name": "task_success_noninferior",
            "passed": quality_noninferior,
            "observed": {
                "baseline": baseline_success,
                "candidate": candidate_success,
            },
            "threshold": {"max_regression": acceptance.max_task_success_regression},
        },
        {
            "name": "completion_rate",
            "passed": completion_ok,
            "observed": {
                "baseline": baseline_completed,
                "candidate": candidate_completed,
            },
            "threshold": acceptance.min_completed_rate,
        },
        {
            "name": "cached_token_gain",
            "passed": cache_ok,
            "required": cache_required,
            "observed": cached_effect,
            "threshold": acceptance.min_cached_token_gain,
        },
        {
            "name": "prediction_direction_matches_observation",
            "passed": prediction_aligned,
            "required": cache_required,
            "observed": prediction,
            "threshold": "both deltas > 0",
        },
        {
            "name": "input_token_reduction",
            "passed": input_reduction_ok,
            "required": input_required,
            "observed": input_effect,
            "threshold": acceptance.min_input_token_reduction,
        },
        {
            "name": f"{acceptance.primary_metric}_improvement",
            "passed": latency_ok,
            "required": acceptance.require_latency_improvement,
            "observed": latency_effect,
            "threshold": "upper bound of 95% CI < 0",
        },
    ]
    if not quality_available:
        status = "invalid"
        reason = "task success was not evaluated for both variants"
    elif not quality_noninferior or not completion_ok:
        status = "regression"
        reason = "candidate failed a task-quality or completion guardrail"
    elif optimization_ok and required_latency_ok:
        status = "verified"
        reason = (
            "all required quality, optimization-target, and performance checks passed"
        )
    else:
        status = "inconclusive"
        reason = "quality passed, but one or more required performance checks did not"
    return {"status": status, "reason": reason, "checks": checks}


def _prepared_request(
    manifest: ExperimentManifest,
    repetition: int,
    case: ExperimentCase,
    variant: str,
    warmup: bool,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    trial_id = f"{'warmup' if warmup else 'measured'}-{case.case_id}-{repetition:04d}"
    replacements = dict(manifest.constants)
    replacements["AID_TRIAL_ID"] = trial_id
    request = _materialize_trial(
        case.baseline if variant == "baseline" else case.candidate,
        replacements,
    )
    transform = (
        case.baseline_transform if variant == "baseline" else case.candidate_transform
    )
    transform_seed_bytes = hashlib.sha256(
        f"{manifest.execution.seed}:{trial_id}:{variant}".encode("utf-8")
    ).digest()[:8]
    request = _apply_request_transform(
        request, transform, int.from_bytes(transform_seed_bytes, "big")
    )
    request.update(manifest.sampling)
    request.setdefault("model", manifest.backend.model)
    return trial_id, request, transform


def _checkpoint_header(manifest: ExperimentManifest) -> dict[str, Any]:
    return {
        "record_type": "header",
        "checkpoint_schema_version": "0.1",
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.fingerprint(),
    }


def _write_checkpoint_record(path: Path, record: Mapping[str, Any], mode: str) -> None:
    try:
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ExperimentValidationError(
            f"cannot write experiment checkpoint {path}: {exc}"
        ) from exc


def _initialize_checkpoint(path: Path, manifest: ExperimentManifest) -> None:
    if path.exists():
        raise ExperimentValidationError(
            f"checkpoint {path} already exists; use resume or choose another path"
        )
    _write_checkpoint_record(path, _checkpoint_header(manifest), "x")


def _load_checkpoint(
    path: Path,
    manifest: ExperimentManifest,
    schedule: list[tuple[int, ExperimentCase, str, bool]],
) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExperimentValidationError(
            f"cannot read checkpoint {path}: {exc}"
        ) from exc
    if not lines:
        raise ExperimentValidationError(f"checkpoint {path} is empty")
    records: list[Any] = []
    for line_number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ExperimentValidationError(
                f"{path}:{line_number}:{exc.colno}: invalid checkpoint JSON: {exc.msg}"
            ) from exc
    expected_header = _checkpoint_header(manifest)
    if records[0] != expected_header:
        raise ExperimentValidationError(
            f"checkpoint {path} does not match the resolved experiment manifest"
        )
    trials: list[dict[str, Any]] = []
    for index, record in enumerate(records[1:]):
        if not isinstance(record, dict) or record.get("record_type") != "trial":
            raise ExperimentValidationError(
                f"checkpoint {path} record {index + 2} is not a trial record"
            )
        trial = record.get("trial")
        if not isinstance(trial, dict):
            raise ExperimentValidationError(
                f"checkpoint {path} record {index + 2} has no trial object"
            )
        if index >= len(schedule):
            raise ExperimentValidationError(
                f"checkpoint {path} contains more trials than the manifest schedule"
            )
        repetition, case, variant, warmup = schedule[index]
        expected_trial_id = (
            f"{'warmup' if warmup else 'measured'}-{case.case_id}-{repetition:04d}"
        )
        expected = {
            "sequence": index,
            "case_id": case.case_id,
            "variant": variant,
            "repetition": repetition,
            "trial_id": expected_trial_id,
            "warmup": warmup,
        }
        if any(trial.get(key) != value for key, value in expected.items()):
            raise ExperimentValidationError(
                f"checkpoint {path} trial {index} does not match the deterministic schedule"
            )
        trials.append(trial)
    return trials


def run_experiment(
    manifest: ExperimentManifest,
    client: CaptureClient,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    execution = manifest.execution
    schedule = _schedule(
        manifest, execution.warmup_repetitions, warmup=True
    ) + _schedule(manifest, execution.repetitions, warmup=False)
    if resume and checkpoint_path is None:
        raise ExperimentValidationError("resume requires a checkpoint path")
    if checkpoint_path is not None:
        trials = _load_checkpoint(checkpoint_path, manifest, schedule) if resume else []
        if not resume:
            _initialize_checkpoint(checkpoint_path, manifest)
    else:
        trials = []
    prior_prompts: list[tuple[int, ...]] = []
    if execution.cache_treatment == "warm":
        client.reset_prefix_cache(reset_external=execution.reset_external)

    for sequence, (repetition, case, variant, warmup) in enumerate(schedule):
        if execution.cache_treatment == "cold":
            client.reset_prefix_cache(reset_external=execution.reset_external)
            prior_prompts.clear()
        trial_id, request, transform = _prepared_request(
            manifest, repetition, case, variant, warmup
        )
        if sequence < len(trials):
            if execution.cache_treatment == "warm":
                replayed = client.capture(request)
                stored_tokens = tuple(trials[sequence].get("prompt_token_ids", []))
                if replayed.prompt_token_ids != stored_tokens:
                    raise ExperimentValidationError(
                        "cannot resume warm-cache experiment because replayed prompt "
                        f"tokens changed at sequence {sequence}"
                    )
                prior_prompts.append(replayed.prompt_token_ids)
            continue
        captured = client.capture(request)
        trial = captured.to_dict()
        trial["predicted_eligible_cached_tokens"] = _predicted_eligible_tokens(
            captured.prompt_token_ids,
            prior_prompts,
            manifest.backend.block_size,
        )
        prior_prompts.append(captured.prompt_token_ids)
        evaluation = None
        evaluation_error = None
        task_success = None
        if case.evaluator is not None:
            context = {
                "case_id": case.case_id,
                "variant": variant,
                "repetition": repetition,
                "trial_id": trial_id,
                "warmup": warmup,
                "response_status": captured.status,
            }
            try:
                evaluated = evaluate_output(
                    case.evaluator, captured.output_text, context
                )
            except EvaluatorError as exc:
                evaluation_error = {
                    "kind": "evaluator_infrastructure_error",
                    "message": str(exc),
                }
            else:
                evaluation = evaluated.to_dict()
                task_success = evaluated.success and captured.status == "completed"
        trial.update(
            {
                "sequence": sequence,
                "case_id": case.case_id,
                "variant": variant,
                "repetition": repetition,
                "trial_id": trial_id,
                "warmup": warmup,
                "evaluation": evaluation,
                "evaluation_error": evaluation_error,
                "task_success": task_success,
                "request_transform": (
                    None if transform is None else transform.get("kind")
                ),
            }
        )
        trials.append(trial)
        if checkpoint_path is not None:
            _write_checkpoint_record(
                checkpoint_path, {"record_type": "trial", "trial": trial}, "a"
            )

    summary = _build_summary(trials, execution.seed)
    if execution.cases_form_workflow:
        summary["workflow"] = _build_workflow_summary(
            trials, execution.seed, len(manifest.cases)
        )
    decision_basis = summary.get("workflow", summary)
    evaluator_errors = [
        {
            "sequence": trial["sequence"],
            "case_id": trial["case_id"],
            "variant": trial["variant"],
            **trial["evaluation_error"],
        }
        for trial in trials
        if trial.get("evaluation_error") is not None
    ]
    decision = _build_decision(decision_basis, manifest.acceptance)
    if evaluator_errors:
        decision = {
            **decision,
            "status": "invalid",
            "reason": (
                f"{len(evaluator_errors)} trial(s) had evaluator infrastructure "
                "errors, so task quality is incomplete"
            ),
            "checks": [
                {
                    "name": "evaluator_integrity",
                    "passed": False,
                    "required": True,
                    "observed": {"error_count": len(evaluator_errors)},
                    "threshold": 0,
                },
                *decision["checks"],
            ],
        }
    return {
        "result_schema_version": "0.2",
        "experiment_id": manifest.experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "backend_kind": "vllm",
            "model": manifest.backend.model,
            "model_revision": manifest.backend.model_revision,
            "tokenizer_revision": manifest.backend.tokenizer_revision,
            "chat_template_digest": manifest.backend.chat_template_digest,
            "backend_version": manifest.backend.backend_version,
            "hardware": manifest.backend.hardware,
            "block_size": manifest.backend.block_size,
            "engine_config": manifest.backend.engine_config,
            "sampling": manifest.sampling,
            "cache_treatment": execution.cache_treatment,
            "arrival_pattern": execution.arrival_pattern,
            "cases_form_workflow": execution.cases_form_workflow,
            "order": execution.order,
            "seed": execution.seed,
            "acceptance": asdict(manifest.acceptance),
            "manifest_sha256": manifest.fingerprint(),
        },
        "trials": trials,
        "evaluator_errors": evaluator_errors,
        "summary": summary,
        "decision": decision,
    }
