"""Versioned, reproducible A/B experiment manifests and runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import secrets
import statistics
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from .binomial import (
    binomial_cdf as _binomial_cdf,  # noqa: F401 - historical private export
    binomial_lower_bound as _one_sided_binomial_lower,
    binomial_upper_bound as _one_sided_binomial_upper,
    clopper_pearson_interval as _clopper_pearson_interval,
    inverse_binomial_cdf as _inverse_binomial_cdf,  # noqa: F401 - compatibility
)
from .cache import longest_common_prefix
from .decision_artifacts import enrich_decision_artifact
from .adapters import BACKEND_KINDS, PROFILES as BACKEND_PROFILES
from .identity import (
    IdentityPolicyError,
    assess_request_identity,
    identity_assurance_record,
    is_placeholder as is_placeholder_identity,
    parse_identity_policy,
    variant_identity_summary,
)
from .evaluators import EvaluatorError, evaluate_output, validate_evaluator_spec
from .execution import (
    CONCURRENT_ARRIVAL_PATTERNS,
    ROLE_FILLER,
    ROLE_WARMUP,
    SEQUENTIAL,
    SHARED,
    ArrivalSchedule,
    JsonlArtifactStore,
    LoadRunner,
    LoadSpec,
    LoadSpecError,
    ScheduledTrial,
    build_arrival_schedule,
    verify_schedule_pin,
)
from .cost_model import CostModelError, parse_cost_model
from .metric_gates import (
    MetricGate,
    MetricGateError,
    Observation,
    economics_of,
    evaluate_metric_gates,
    load_integrity_checks,
    merge_load_and_metric_checks,
    parse_metric_gates,
    tpot_ms,
)
from .paired_inference import (
    PAIRED_EXACT,
    PAIRED_SCORE_EXACT,
    POINT_ESTIMATE,
    outcome_counts,
    paired_score_exact_statistics,
)
from .vllm import CapturedResponse


# Manifests written before 0.2.0a3 used the second name; both resolve to the
# same per-trial value so existing manifests keep working.
_TRIAL_ID_PLACEHOLDERS = ("RUNECONOMY_TRIAL_ID", "AID_TRIAL_ID")


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
    api_dialect: str = "responses"
    # Adapter backends (manifest schema 0.3+). Defaults keep every vLLM
    # manifest, fingerprint and result exactly as before.
    kind: str = "vllm"
    provider: str | None = None
    undisclosed_identity: tuple[str, ...] = ()


# Static identity fields an openai_compat manifest may declare undisclosed.
STATIC_UNDISCLOSABLE_FIELDS = frozenset(
    {
        "model_revision",
        "tokenizer_revision",
        "chat_template_digest",
        "backend_version",
        "hardware",
        "engine_config",
    }
)


def _parse_backend(backend_data: Mapping[str, Any], version: str) -> BackendConfig:
    kind = backend_data.get("kind")
    if kind not in BACKEND_KINDS:
        raise ExperimentValidationError(
            "backend.kind must be 'vllm', 'sglang', or 'openai_compat'"
        )
    if kind == "vllm":
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
            api_dialect=backend_data.get("api_dialect", "responses"),
        )
        if backend.api_dialect not in {"responses", "chat_completions"}:
            raise ExperimentValidationError(
                "backend.api_dialect must be 'responses' or 'chat_completions'"
            )
        if not backend.engine_config:
            raise ExperimentValidationError(
                "backend.engine_config must pin the serving configuration"
            )
        if backend.engine_config.get("enable_prefix_caching") is not True:
            raise ExperimentValidationError(
                "backend.engine_config.enable_prefix_caching must be true"
            )
        return backend

    # Adapter kinds are new in this release, so no recorded manifest uses them.
    if version not in {"0.3", "0.4"}:
        raise ExperimentValidationError(
            f"backend.kind {kind!r} requires schema_version '0.3' or later"
        )
    profile = BACKEND_PROFILES[kind]
    raw_undisclosed = backend_data.get("undisclosed_identity", [])
    if kind != "openai_compat" and raw_undisclosed:
        raise ExperimentValidationError(
            f"backend.undisclosed_identity is allowed only for openai_compat; "
            f"{profile.label} identity is operator-verified and must be pinned"
        )
    if not isinstance(raw_undisclosed, list) or not all(
        isinstance(item, str) for item in raw_undisclosed
    ):
        raise ExperimentValidationError(
            "backend.undisclosed_identity must be a list of field names"
        )
    unsupported = sorted(set(raw_undisclosed) - STATIC_UNDISCLOSABLE_FIELDS)
    if unsupported:
        raise ExperimentValidationError(
            "backend.undisclosed_identity names fields that must always be declared: "
            + ", ".join(unsupported)
        )
    undisclosed = tuple(sorted(set(raw_undisclosed)))

    def identity_field(name: str) -> str | None:
        value = backend_data.get(name)
        if name in undisclosed:
            if value is not None:
                raise ExperimentValidationError(
                    f"backend.{name} is listed as undisclosed and must be null or absent"
                )
            return None
        return _string(value, f"backend.{name}")

    provider = backend_data.get("provider")
    if kind == "openai_compat":
        provider = _string(provider, "backend.provider")
    elif provider is not None:
        raise ExperimentValidationError("backend.provider is allowed only for openai_compat")
    if "engine_config" in undisclosed:
        if backend_data.get("engine_config") is not None:
            raise ExperimentValidationError(
                "backend.engine_config is listed as undisclosed and must be null or absent"
            )
        engine_config: dict[str, Any] = {}
    else:
        engine_config = _object(backend_data.get("engine_config"), "backend.engine_config")
        if not engine_config:
            raise ExperimentValidationError(
                "backend.engine_config must pin the serving configuration"
            )
    if kind == "sglang" and engine_config.get("disable_radix_cache") is not False:
        raise ExperimentValidationError(
            "backend.engine_config.disable_radix_cache must be pinned to false for SGLang"
        )
    api_dialect = backend_data.get("api_dialect", "chat_completions")
    if api_dialect not in profile.api_dialects:
        raise ExperimentValidationError(
            f"backend.api_dialect must be 'chat_completions' for {kind}"
        )
    return BackendConfig(
        base_url=_string(backend_data.get("base_url"), "backend.base_url"),
        api_key_env=(
            None
            if backend_data.get("api_key_env") is None
            else _string(backend_data.get("api_key_env"), "backend.api_key_env")
        ),
        model=_string(backend_data.get("model"), "backend.model"),
        model_revision=identity_field("model_revision"),  # type: ignore[arg-type]
        tokenizer_revision=identity_field("tokenizer_revision"),  # type: ignore[arg-type]
        chat_template_digest=identity_field("chat_template_digest"),  # type: ignore[arg-type]
        backend_version=identity_field("backend_version"),  # type: ignore[arg-type]
        hardware=identity_field("hardware"),  # type: ignore[arg-type]
        block_size=_positive_int(backend_data.get("block_size", 16), "backend.block_size"),
        engine_config=engine_config,
        api_dialect=api_dialect,
        kind=kind,
        provider=provider,
        undisclosed_identity=undisclosed,
    )


def _validate_backend_execution(
    backend: BackendConfig, execution: "ExecutionConfig"
) -> None:
    """Reject execution modes the backend kind cannot provide."""

    if backend.kind == "vllm":
        return
    profile = BACKEND_PROFILES[backend.kind]
    if execution.cache_isolation not in profile.cache_isolation_modes:
        raise ExperimentValidationError(
            f"execution.cache_isolation {execution.cache_isolation!r} is unavailable "
            f"on {backend.kind}; supported: "
            + ", ".join(sorted(profile.cache_isolation_modes))
        )
    if execution.reset_external and not profile.supports_reset_external:
        raise ExperimentValidationError(
            f"execution.reset_external is unavailable on {backend.kind}"
        )


def _parse_identity_policy_field(
    acceptance_data: Mapping[str, Any], version: str, backend: BackendConfig
) -> dict[str, Any] | None:
    raw = acceptance_data.get("identity_policy")
    if raw is None:
        if backend.kind == "openai_compat":
            raise ExperimentValidationError(
                "openai_compat identity is provider-declared: declare "
                "acceptance.identity_policy with minimum_grade provider_declared"
            )
        return None
    if version not in {"0.3", "0.4"}:
        raise ExperimentValidationError(
            "acceptance.identity_policy requires schema_version '0.3' or later"
        )
    try:
        policy = parse_identity_policy(raw)
    except IdentityPolicyError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    assert policy is not None
    if not policy.accepts(BACKEND_PROFILES[backend.kind].identity_grade):
        raise ExperimentValidationError(
            f"{backend.kind} identity is provider-declared, so minimum_grade "
            "operator_verified can never be met; declare provider_declared"
        )
    for name in (
        "model",
        "model_revision",
        "tokenizer_revision",
        "chat_template_digest",
        "backend_version",
        "hardware",
    ):
        if is_placeholder_identity(getattr(backend, name)):
            raise ExperimentValidationError(
                f"backend.{name} is a placeholder; under an identity policy pin a "
                "concrete value or (openai_compat only) list it in "
                "backend.undisclosed_identity"
            )
    return policy.to_dict()


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
    cache_isolation: str = "shared"
    load: LoadSpec | None = None


@dataclass(frozen=True)
class AcceptanceConfig:
    primary_metric: str
    optimization_target: str
    max_task_success_regression: float
    min_completed_rate: float
    min_cached_token_gain: float
    min_input_token_reduction: float
    require_latency_improvement: bool
    min_task_success_rate: float = 0.0
    task_success_method: str = "paired_exact"
    require_environment_identity: bool = False


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    evaluator: dict[str, Any] | None
    baseline_transform: dict[str, Any] | None
    candidate_transform: dict[str, Any] | None
    workflow_id: str | None = None


def _parse_manifest_cost_model(root: Mapping[str, Any], version: str) -> dict[str, Any] | None:
    if root.get("cost_model") is None:
        return None
    if version != "0.4":
        raise ExperimentValidationError("cost_model requires schema_version '0.4'")
    try:
        model = parse_cost_model(root["cost_model"])
    except CostModelError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    return None if model is None else model.to_dict()


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
    design: dict[str, Any] = field(default_factory=dict)
    # Parsed from acceptance.metric_gates (schema 0.4). Held here rather than
    # on AcceptanceConfig so asdict(acceptance) keeps its historical shape.
    metric_gates: tuple[MetricGate, ...] = ()
    # Parsed from acceptance.identity_policy (schema 0.3+), held here for the
    # same reason: legacy acceptance payloads keep their exact shape.
    identity_policy: dict[str, Any] | None = None
    # Optional versioned cost model (schema 0.4). It prices attempts for the
    # unit-economics metric gates and is part of the fingerprint when set.
    cost_model: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> "ExperimentManifest":
        root = _object(payload, "manifest")
        version = _string(root.get("schema_version"), "schema_version")
        if version not in {"0.1", "0.2", "0.3", "0.4"}:
            raise ExperimentValidationError(
                f"unsupported schema_version {version!r}; expected '0.1', '0.2', "
                "'0.3', or '0.4'"
            )
        experiment_id = _string(root.get("experiment_id"), "experiment_id")

        backend_data = _object(root.get("backend"), "backend")
        backend = _parse_backend(backend_data, version)

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
        arrival_pattern = execution_data.get("arrival_pattern", SEQUENTIAL)
        load: LoadSpec | None = None
        if arrival_pattern != SEQUENTIAL:
            if arrival_pattern not in CONCURRENT_ARRIVAL_PATTERNS:
                raise ExperimentValidationError(
                    "execution.arrival_pattern must be sequential, closed_loop, "
                    "constant_rps, or poisson"
                )
            if version != "0.4":
                raise ExperimentValidationError(
                    f"execution.arrival_pattern {arrival_pattern!r} requires "
                    "schema_version '0.4'"
                )
            try:
                # Static replay targets one backend.base_url, so both variants
                # always share a serving stack.
                load = LoadSpec.from_value(
                    arrival_pattern,
                    execution_data.get("load"),
                    forced_topology=SHARED,
                )
            except LoadSpecError as exc:
                raise ExperimentValidationError(str(exc)) from exc
        elif "load" in execution_data:
            raise ExperimentValidationError(
                "execution.load requires a concurrent execution.arrival_pattern"
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
        cache_isolation = execution_data.get(
            "cache_isolation",
            "physical_reset" if cache_treatment == "cold" else "shared",
        )
        if cache_isolation not in {"shared", "salt_isolated", "physical_reset"}:
            raise ExperimentValidationError(
                "execution.cache_isolation must be shared, salt_isolated, or "
                "physical_reset"
            )
        if cache_treatment == "warm" and cache_isolation != "shared":
            raise ExperimentValidationError(
                "warm cache treatment requires execution.cache_isolation='shared'"
            )
        if cache_treatment == "cold" and cache_isolation == "shared":
            raise ExperimentValidationError(
                "cold cache treatment requires salt_isolated or physical_reset isolation"
            )
        if load is not None and cache_isolation == "physical_reset":
            raise ExperimentValidationError(
                "physical_reset cache isolation cannot run under concurrent load; "
                "use salt_isolated cold isolation or a warm cache treatment"
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
            cache_isolation=cache_isolation,
            load=load,
        )
        _validate_backend_execution(backend, execution)

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
        require_environment_identity = acceptance_data.get(
            "require_environment_identity", False
        )
        if not isinstance(require_environment_identity, bool):
            raise ExperimentValidationError(
                "acceptance.require_environment_identity must be boolean"
            )
        acceptance = AcceptanceConfig(
            primary_metric=primary_metric,
            optimization_target=optimization_target,
            max_task_success_regression=probability("max_task_success_regression", 0.0),
            min_task_success_rate=probability(
                "min_task_success_rate", 0.80 if version == "0.3" else 0.0
            ),
            task_success_method=str(
                acceptance_data.get("task_success_method", "paired_exact")
            ),
            min_completed_rate=probability("min_completed_rate", 1.0),
            min_cached_token_gain=float(min_cached_token_gain),
            min_input_token_reduction=float(min_input_token_reduction),
            require_latency_improvement=require_latency,
            require_environment_identity=require_environment_identity,
        )
        identity_policy = _parse_identity_policy_field(acceptance_data, version, backend)
        metric_gates: tuple[MetricGate, ...] = ()
        if "metric_gates" in acceptance_data:
            if version != "0.4":
                raise ExperimentValidationError(
                    "acceptance.metric_gates requires schema_version '0.4'"
                )
            try:
                metric_gates = parse_metric_gates(acceptance_data["metric_gates"])
            except MetricGateError as exc:
                raise ExperimentValidationError(str(exc)) from exc
        if acceptance.task_success_method not in {
            PAIRED_EXACT,
            PAIRED_SCORE_EXACT,
            POINT_ESTIMATE,
        }:
            raise ExperimentValidationError(
                "acceptance.task_success_method must be paired_exact, "
                "paired_score_exact, or point_estimate"
            )
        if version == "0.1" and acceptance.task_success_method == PAIRED_SCORE_EXACT:
            # Schema 0.1 fingerprints omit the method, so a non-default method
            # would not be bound to the recorded manifest hash.
            raise ExperimentValidationError(
                "acceptance.task_success_method paired_score_exact requires "
                "schema_version '0.2' or later"
            )
        if "completion_policy" in acceptance_data:
            raise ExperimentValidationError(
                "acceptance.completion_policy is supported only for agent-process "
                "experiments, whose attempts carry failure provenance"
            )

        design = _object(root.get("design", {}), "design")
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
            "cache_salt",
            "input",
            "instructions",
            "messages",
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
        forbidden = {"stream", "previous_response_id", "background", "cache_salt"}
        for index, raw_case in enumerate(cases_data):
            case_data = _object(raw_case, f"cases[{index}]")
            case_id = _string(case_data.get("case_id"), f"cases[{index}].case_id")
            workflow_id = (
                None
                if case_data.get("workflow_id") is None
                else _string(
                    case_data.get("workflow_id"), f"cases[{index}].workflow_id"
                )
            )
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
                    case_id=case_id,
                    baseline=requests["baseline"],
                    candidate=requests["candidate"],
                    evaluator=evaluator,
                    baseline_transform=transforms["baseline"],
                    candidate_transform=transforms["candidate"],
                    workflow_id=workflow_id,
                )
            )

        manifest = cls(
            schema_version=version,
            experiment_id=experiment_id,
            backend=backend,
            sampling=dict(sampling),
            execution=execution,
            acceptance=acceptance,
            constants=constants,
            cases=tuple(cases),
            design=deepcopy(design),
            metric_gates=metric_gates,
            identity_policy=identity_policy,
            cost_model=_parse_manifest_cost_model(root, version),
        )
        _validate_manifest_placeholders(manifest)
        workflow_ids = [case.workflow_id for case in manifest.cases]
        if any(workflow_id is not None for workflow_id in workflow_ids):
            if not manifest.execution.cases_form_workflow:
                raise ExperimentValidationError(
                    "case workflow_id requires execution.cases_form_workflow=true"
                )
            if any(workflow_id is None for workflow_id in workflow_ids):
                raise ExperimentValidationError(
                    "every case must define workflow_id when workflow grouping is used"
                )
        return manifest

    def to_resolved_dict(self) -> dict[str, Any]:
        """Return a fixture-resolved payload in the public manifest shape."""

        cases: list[dict[str, Any]] = []
        for case in self.cases:
            variants: dict[str, Any] = {}
            for variant, request, transform in (
                ("baseline", case.baseline, case.baseline_transform),
                ("candidate", case.candidate, case.candidate_transform),
            ):
                variant_payload: dict[str, Any] = {"request": deepcopy(request)}
                if transform is not None:
                    variant_payload["request_transform"] = deepcopy(transform)
                variants[variant] = variant_payload
            case_payload: dict[str, Any] = {
                "case_id": case.case_id,
                "variants": variants,
            }
            if case.workflow_id is not None:
                case_payload["workflow_id"] = case.workflow_id
            if case.evaluator is not None:
                case_payload["evaluator"] = deepcopy(case.evaluator)
            cases.append(case_payload)

        backend = asdict(self.backend)
        _normalize_backend_fields(backend)
        execution = asdict(self.execution)
        if backend.get("api_dialect") == "responses":
            backend.pop("api_dialect")
        legacy_isolation = (
            "physical_reset"
            if execution.get("cache_treatment") == "cold"
            else "shared"
        )
        if execution.get("cache_isolation") == legacy_isolation:
            execution.pop("cache_isolation")
        _normalize_load_fields(
            execution, self.execution.load, include_pin=True
        )
        acceptance = asdict(self.acceptance)
        _normalize_metric_gates(acceptance, self.metric_gates)
        if self.identity_policy is not None:
            acceptance["identity_policy"] = dict(self.identity_policy)
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "backend": {"kind": self.backend.kind, **backend},
            "sampling": deepcopy(self.sampling),
            "execution": execution,
            "acceptance": acceptance,
            "design": deepcopy(self.design),
            "constants": deepcopy(self.constants),
            "cases": cases,
            **({"cost_model": deepcopy(self.cost_model)} if self.cost_model is not None else {}),
        }

    def _fingerprint_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        # Preserve hashes for manifests recorded before workflow grouping was
        # introduced. An absent optional field is not a semantic change.
        for case in payload["cases"]:
            if case.get("workflow_id") is None:
                case.pop("workflow_id", None)
        if not payload.get("design"):
            payload.pop("design", None)
        if payload["acceptance"].get("min_task_success_rate") == 0.0:
            payload["acceptance"].pop("min_task_success_rate", None)
        if self.schema_version == "0.1":
            payload["acceptance"].pop("task_success_method", None)
            payload["acceptance"].pop("require_environment_identity", None)
        # Defaults introduced after the recorded 0.1-0.3 manifests are omitted
        # so parsing an old manifest does not alter its historical fingerprint.
        if payload["backend"].get("api_dialect") == "responses":
            payload["backend"].pop("api_dialect", None)
        _normalize_backend_fields(payload["backend"], keep_kind=True)
        identity_policy = payload.pop("identity_policy", None)
        if identity_policy is not None:
            payload["acceptance"]["identity_policy"] = identity_policy
        legacy_isolation = (
            "physical_reset"
            if payload["execution"].get("cache_treatment") == "cold"
            else "shared"
        )
        if payload["execution"].get("cache_isolation") == legacy_isolation:
            payload["execution"].pop("cache_isolation", None)
        # Load and metric-gate fields exist only in schema 0.4; omitting them
        # when absent keeps every earlier fingerprint unchanged.
        _normalize_load_fields(payload["execution"], self.execution.load, include_pin=True)
        payload.pop("metric_gates", None)
        if payload.get("cost_model") is None:
            payload.pop("cost_model", None)
        _normalize_metric_gates(payload["acceptance"], self.metric_gates)
        return payload

    def fingerprint(self) -> str:
        payload = self._fingerprint_payload()
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def reproducibility_fingerprint(self) -> str:
        """Hash experiment semantics without endpoint or credential locations.

        ``base_url`` and ``api_key_env`` affect transport, not the workload. They
        are deliberately excluded so a sanitized public manifest can be checked
        against a private measured manifest without publishing its endpoint.
        """

        payload = self._fingerprint_payload()
        payload["backend"].pop("base_url", None)
        payload["backend"].pop("api_key_env", None)
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def uses_executable_evaluators(self) -> bool:
        """Return whether a manifest can execute local Python or commands."""

        from .evaluators import is_executable_evaluator

        return any(is_executable_evaluator(case.evaluator) for case in self.cases)

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


def _normalize_backend_fields(
    backend: dict[str, Any], *, keep_kind: bool = False
) -> None:
    """Drop adapter fields at their vLLM defaults so old payloads are unchanged."""

    kind = backend.get("kind", "vllm")
    if kind == "vllm" or not keep_kind:
        backend.pop("kind", None)
    if backend.get("provider") is None:
        backend.pop("provider", None)
    if not backend.get("undisclosed_identity"):
        backend.pop("undisclosed_identity", None)
    else:
        backend["undisclosed_identity"] = list(backend["undisclosed_identity"])
        if "engine_config" in backend["undisclosed_identity"]:
            backend["engine_config"] = None


def _normalize_load_fields(
    execution: dict[str, Any], load: LoadSpec | None, *, include_pin: bool
) -> None:
    if load is None:
        execution.pop("load", None)
    else:
        execution["load"] = load.to_dict(include_pin=include_pin)


def _normalize_metric_gates(
    acceptance: dict[str, Any], gates: tuple[MetricGate, ...]
) -> None:
    if gates:
        acceptance["metric_gates"] = [gate.to_dict() for gate in gates]
    else:
        acceptance.pop("metric_gates", None)


class CaptureClient(Protocol):
    def capture(self, request_body: Mapping[str, Any]) -> CapturedResponse: ...

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None: ...


def _expected_environment_identity(manifest: ExperimentManifest) -> dict[str, Any]:
    return {
        "model": manifest.backend.model,
        "model_revision": manifest.backend.model_revision,
        "tokenizer_revision": manifest.backend.tokenizer_revision,
        "chat_template_digest": manifest.backend.chat_template_digest,
        "backend_version": manifest.backend.backend_version,
        "engine_config": manifest.backend.engine_config,
    }


def _can_reset(client: CaptureClient) -> bool:
    """Adapters without a cache reset declare ``supports_cache_reset = False``.

    Their warm runs start from an unknown cache state, which the adapter's
    capability record states in the result.
    """

    return getattr(client, "supports_cache_reset", True) is not False


_STATIC_IDENTITY_FIELDS = (
    "backend",
    "backend_version",
    "model",
    "model_revision",
    "tokenizer_revision",
    "chat_template_digest",
)


def _static_identity_assurance(
    manifest: "ExperimentManifest", trials: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Grade the static backend identity under a declared identity policy.

    Both variants replay against one backend. Each measured trial's identity
    is the pinned manifest identity, with the model the provider says served
    it when the backend reports one. A served model that differs from the
    manifest, or changes during the run, invalidates identity.
    """

    policy = parse_identity_policy(manifest.identity_policy)
    assert policy is not None
    backend = manifest.backend
    grade = BACKEND_PROFILES[backend.kind].identity_grade
    pinned = {
        "backend": backend.provider or backend.kind,
        "backend_version": backend.backend_version,
        "model": backend.model,
        "model_revision": backend.model_revision,
        "tokenizer_revision": backend.tokenizer_revision,
        "chat_template_digest": backend.chat_template_digest,
    }
    undisclosed = [name for name in backend.undisclosed_identity if name in pinned]
    variants: dict[str, Any] = {}
    for variant in ("baseline", "candidate"):
        measured = [
            trial for trial in trials if trial["variant"] == variant and not trial["warmup"]
        ]
        identities = [
            {**pinned, "model": trial.get("response_model") or backend.model}
            for trial in measured
        ]
        summary = variant_identity_summary(
            policy,
            [
                assess_request_identity(
                    identity,
                    grade=grade,
                    undisclosed=undisclosed,
                    fields=_STATIC_IDENTITY_FIELDS,
                )
                for identity in identities
            ],
            [trial.get("provider_fingerprint_sha256") for trial in measured],
        )
        served = sorted({identity["model"] for identity in identities})
        summary["served_models"] = served
        summary["served_model_matches_manifest"] = served == [backend.model]
        summary["valid"] = bool(summary["valid"] and summary["served_model_matches_manifest"])
        variants[variant] = summary
    record = identity_assurance_record(policy, variants)
    passed = all(item["valid"] for item in variants.values())
    check = {
        "name": "serving_identity_grade",
        "passed": passed,
        "required": True,
        "observed": {
            name: {
                "grade": item["grade"],
                "valid": item["valid"],
                "served_models": item["served_models"],
                "undisclosed_fields": item["undisclosed_fields"],
            }
            for name, item in variants.items()
        },
        "threshold": (
            f"complete graded identity at or above {policy.minimum_grade}, and the "
            "served model equals the pinned manifest model throughout"
        ),
    }
    return record, check


def _read_environment_identity(client: CaptureClient) -> dict[str, Any] | None:
    provider = getattr(client, "environment_identity", None)
    if not callable(provider):
        return None
    identity = provider()
    if identity is None:
        return None
    if not isinstance(identity, Mapping):
        raise ExperimentValidationError(
            "environment identity provider returned a non-object value"
        )
    return dict(identity)


def _environment_identity_check(
    expected: Mapping[str, Any],
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fields = tuple(expected)
    missing_before = [field for field in fields if before is None or field not in before]
    missing_after = [field for field in fields if after is None or field not in after]
    expected_mismatches = {
        field: {"expected": expected[field], "observed": before[field]}
        for field in fields
        if before is not None and field in before and before[field] != expected[field]
    }
    drift = {
        field: {"before": before[field], "after": after[field]}
        for field in fields
        if before is not None
        and after is not None
        and field in before
        and field in after
        and before[field] != after[field]
    }
    verified = not missing_before and not missing_after and not expected_mismatches and not drift
    return {
        "verified": verified,
        "expected": dict(expected),
        "before": None if before is None else dict(before),
        "after": None if after is None else dict(after),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "expected_mismatches": expected_mismatches,
        "drift": drift,
    }


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


def _trial_replacements(
    manifest: ExperimentManifest, trial_id: str
) -> dict[str, Any]:
    replacements = dict(manifest.constants)
    for name in _TRIAL_ID_PLACEHOLDERS:
        replacements[name] = trial_id
    return replacements


def _validate_manifest_placeholders(manifest: ExperimentManifest) -> None:
    replacements = _trial_replacements(manifest, "validation-trial")
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


def _predicted_eligible_or_unknown(
    token_ids: tuple[int, ...] | None,
    prior_prompts: list[tuple[int, ...] | None],
    block_size: int,
) -> int | None:
    """Unknown prompt tokens (an OpenAI-compatible API) predict unknown, not 0."""

    if token_ids is None:
        return None
    return _predicted_eligible_tokens(
        token_ids, [prior for prior in prior_prompts if prior is not None], block_size
    )


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


def _paired_success_statistics(
    pairs: list[tuple[bool, bool]], *, confidence: float = 0.95
) -> dict[str, Any]:
    """Bound a paired success-rate difference without a degenerate bootstrap.

    The difference is p(candidate-only) - p(baseline-only). Exact binomial
    bounds are computed for both marginal discordance probabilities and joined
    with Bonferroni coverage. This is conservative but valid for small samples,
    including the all-agreement case.
    """

    count = len(pairs)
    both_success = sum(baseline and candidate for baseline, candidate in pairs)
    both_failure = sum(not baseline and not candidate for baseline, candidate in pairs)
    candidate_only = sum(not baseline and candidate for baseline, candidate in pairs)
    baseline_only = sum(baseline and not candidate for baseline, candidate in pairs)
    if not count:
        return {
            "paired_count": 0,
            "candidate_minus_baseline_mean": None,
            "confidence_interval": None,
            "one_sided_lower_bound": None,
            "outcome_counts": {
                "both_success": 0,
                "both_failure": 0,
                "candidate_only": 0,
                "baseline_only": 0,
            },
            "method": "exact marginal binomial bounds with Bonferroni coverage",
        }

    alpha = 1.0 - confidence
    marginal_confidence = 1.0 - alpha / 2.0
    candidate_only_interval = _clopper_pearson_interval(
        candidate_only, count, confidence=marginal_confidence
    )
    baseline_only_interval = _clopper_pearson_interval(
        baseline_only, count, confidence=marginal_confidence
    )
    assert candidate_only_interval is not None
    assert baseline_only_interval is not None
    simultaneous_interval = [
        candidate_only_interval[0] - baseline_only_interval[1],
        candidate_only_interval[1] - baseline_only_interval[0],
    ]
    one_sided_lower = _one_sided_binomial_lower(
        candidate_only, count, alpha=alpha / 2.0
    ) - _one_sided_binomial_upper(
        baseline_only, count, alpha=alpha / 2.0
    )
    return {
        "paired_count": count,
        "candidate_minus_baseline_mean": (candidate_only - baseline_only) / count,
        "confidence_interval": simultaneous_interval,
        "confidence_level": confidence,
        "one_sided_lower_bound": one_sided_lower,
        "outcome_counts": {
            "both_success": both_success,
            "both_failure": both_failure,
            "candidate_only": candidate_only,
            "baseline_only": baseline_only,
        },
        "method": "exact marginal binomial bounds with Bonferroni coverage",
    }


def _task_success_statistics(
    pairs: list[tuple[bool, bool]],
    *,
    method: str = PAIRED_EXACT,
    margin: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return paired success statistics under the preregistered method.

    Contracts that omit ``task_success_method`` keep the original
    ``paired_exact`` statistics byte for byte. ``paired_score_exact`` adds an
    exact p-value at the preregistered margin and an inverted lower bound.
    """

    if method == PAIRED_SCORE_EXACT:
        return paired_score_exact_statistics(
            outcome_counts(pairs), margin=margin, confidence=confidence
        )
    return _paired_success_statistics(pairs, confidence=confidence)


def _task_success_noninferior(
    effect: Mapping[str, Any], *, method: str, margin: float
) -> bool:
    """Apply the preregistered paired gate to one task-success effect."""

    if method == POINT_ESTIMATE:
        return True
    if method == PAIRED_SCORE_EXACT:
        return bool(
            effect.get("method_id") == PAIRED_SCORE_EXACT
            and effect.get("noninferiority_margin") == margin
            and effect.get("rejects_inferiority") is True
        )
    lower = effect.get("one_sided_lower_bound")
    return lower is not None and lower >= -margin


def _schedule(
    manifest: ExperimentManifest, repetitions: int, *, warmup: bool
) -> list[tuple[int, ExperimentCase, str, bool]]:
    rng = random.Random(manifest.execution.seed + (1 if warmup else 0))
    schedule: list[tuple[int, ExperimentCase, str, bool]] = []
    grouped_workflows = (
        manifest.execution.cases_form_workflow
        and all(case.workflow_id is not None for case in manifest.cases)
    )
    for repetition in range(repetitions):
        if grouped_workflows:
            workflow_order = list(
                dict.fromkeys(case.workflow_id for case in manifest.cases)
            )
            for workflow_id in workflow_order:
                variants = ["baseline", "candidate"]
                if manifest.execution.order == "randomized":
                    rng.shuffle(variants)
                elif repetition % 2:
                    variants.reverse()
                workflow_cases = [
                    case for case in manifest.cases if case.workflow_id == workflow_id
                ]
                for variant in variants:
                    schedule.extend(
                        (repetition, case, variant, warmup)
                        for case in workflow_cases
                    )
            continue
        for case in manifest.cases:
            variants = ["baseline", "candidate"]
            if manifest.execution.order == "randomized":
                rng.shuffle(variants)
            elif repetition % 2:
                variants.reverse()
            schedule.extend((repetition, case, variant, warmup) for variant in variants)
    return schedule


def _build_summary(
    trials: list[dict[str, Any]],
    seed: int,
    *,
    task_success_method: str = PAIRED_EXACT,
    max_task_success_regression: float = 0.0,
) -> dict[str, Any]:
    measured = [trial for trial in trials if not trial["warmup"]]
    metric_names = (
        "client_ttft_ms",
        "total_latency_ms",
        "server_ttft_ms",
        "client_server_ttft_gap_ms",
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
        evaluated_successes = [
            bool(trial["task_success"])
            for trial in subset
            if trial.get("task_success") is not None
        ]
        success_count = sum(evaluated_successes)
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
            "task_success_count": success_count,
            "task_success_rate": (
                None
                if not evaluated_successes
                else success_count / len(evaluated_successes)
            ),
            "task_success_95_ci": _clopper_pearson_interval(
                success_count, len(evaluated_successes)
            ),
            "task_success_interval_method": "Clopper-Pearson exact binomial",
        }

    by_pair = {
        (trial["case_id"], trial["repetition"], trial["variant"]): trial
        for trial in measured
    }
    effects: dict[str, Any] = {}
    for metric in metric_names:
        differences: list[float] = []
        for case in sorted({trial["case_id"] for trial in measured}):
            for repetition in sorted({trial["repetition"] for trial in measured}):
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
    success_pairs: list[tuple[bool, bool]] = []
    for case in sorted({trial["case_id"] for trial in measured}):
        for repetition in sorted({trial["repetition"] for trial in measured}):
            baseline = by_pair.get((case, repetition, "baseline"))
            candidate = by_pair.get((case, repetition, "candidate"))
            if (
                baseline is not None
                and candidate is not None
                and baseline.get("task_success") is not None
                and candidate.get("task_success") is not None
            ):
                success_pairs.append(
                    (bool(baseline["task_success"]), bool(candidate["task_success"]))
                )
    effects["task_success"].update(
        _task_success_statistics(
            success_pairs,
            method=task_success_method,
            margin=max_task_success_regression,
        )
    )
    digest_pairs = 0
    digest_matches = 0
    for case in sorted({trial["case_id"] for trial in measured}):
        for repetition in sorted({trial["repetition"] for trial in measured}):
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
    trials: list[dict[str, Any]],
    seed: int,
    cases: tuple[ExperimentCase, ...],
    *,
    task_success_method: str = PAIRED_EXACT,
    max_task_success_regression: float = 0.0,
) -> dict[str, Any]:
    """Aggregate sequential cases into critical-path workflow trials."""

    measured = [trial for trial in trials if not trial["warmup"]]
    explicit_groups = all(case.workflow_id is not None for case in cases)
    expected_counts: dict[str, int] = {}
    for case in cases:
        workflow_id = case.workflow_id if explicit_groups else "workflow"
        assert workflow_id is not None
        expected_counts[workflow_id] = expected_counts.get(workflow_id, 0) + 1

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for trial in measured:
        workflow_id = (
            str(trial["workflow_id"]) if explicit_groups else "workflow"
        )
        groups.setdefault(
            (workflow_id, trial["repetition"], trial["variant"]), []
        ).append(trial)

    additive_metrics = (
        "client_ttft_ms",
        "total_latency_ms",
        "server_ttft_ms",
        "client_server_ttft_gap_ms",
        "server_queue_ms",
        "generation_time_ms",
        "cached_tokens",
        "input_tokens",
        "output_tokens",
        "predicted_eligible_cached_tokens",
    )
    aggregate_trials: list[dict[str, Any]] = []
    for (workflow_id, repetition, variant), steps in sorted(groups.items()):
        expected_case_count = expected_counts[workflow_id]
        if len(steps) != expected_case_count:
            raise ExperimentValidationError(
                f"workflow {workflow_id}/{repetition}/{variant} has {len(steps)} "
                f"steps; expected {expected_case_count}"
            )
        aggregate: dict[str, Any] = {
            "case_id": workflow_id,
            "workflow_id": workflow_id,
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

    summary = _build_summary(
        aggregate_trials,
        seed,
        task_success_method=task_success_method,
        max_task_success_regression=max_task_success_regression,
    )
    summary["claim"] = (
        "cases are treated as sequential critical-path inference steps; "
        "reported workflow latency is their sum and excludes tool runtime"
    )
    step_counts = sorted(expected_counts.values())
    summary["workflow_count"] = len(expected_counts)
    summary["step_count"] = (
        step_counts[0]
        if step_counts and all(count == step_counts[0] for count in step_counts)
        else None
    )
    summary["step_counts"] = dict(sorted(expected_counts.items()))
    return summary


def _assess_predeclared_prediction(
    summary: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any] | None:
    hypothesis = design.get("hypothesis")
    if not isinstance(hypothesis, Mapping):
        return None
    prediction = hypothesis.get("prediction")
    if not isinstance(prediction, Mapping):
        return None
    metric = prediction.get("metric")
    direction = prediction.get("direction")
    if not isinstance(metric, str) or direction not in {
        "increase",
        "decrease",
        "no_change",
    }:
        return {
            "status": "invalid_prediction",
            "metric": metric,
            "direction": direction,
            "causal_mechanism": hypothesis.get("causal_mechanism"),
            "reason": "prediction requires a known metric and direction",
        }
    effects = summary.get("effects")
    effect = effects.get(metric) if isinstance(effects, Mapping) else None
    if not isinstance(effect, Mapping):
        return {
            "status": "unavailable",
            "metric": metric,
            "direction": direction,
            "causal_mechanism": hypothesis.get("causal_mechanism"),
            "reason": "the predicted metric was not measured",
        }
    mean = effect.get("candidate_minus_baseline_mean")
    interval = effect.get("bootstrap_95_ci")
    status = "not_established"
    if (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) for value in interval)
    ):
        lower, upper = float(interval[0]), float(interval[1])
        if direction == "decrease":
            status = "supported" if upper < 0 else "contradicted" if lower > 0 else status
        elif direction == "increase":
            status = "supported" if lower > 0 else "contradicted" if upper < 0 else status
        else:
            status = "compatible" if lower <= 0 <= upper else "contradicted"
    return {
        "status": status,
        "metric": metric,
        "direction": direction,
        "causal_mechanism": hypothesis.get("causal_mechanism"),
        "candidate_minus_baseline_mean": mean,
        "confidence_interval": interval,
        "assessment_rule": (
            "direction is supported only when the paired 95% interval lies wholly "
            "in the predicted direction"
        ),
    }


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
    baseline_success_interval = baseline_outcomes.get("task_success_95_ci")
    candidate_success_interval = candidate_outcomes.get("task_success_95_ci")
    baseline_absolute_quality_ok = bool(
        baseline_success_interval is not None
        and baseline_success_interval[0] >= acceptance.min_task_success_rate
    )
    candidate_absolute_quality_ok = bool(
        candidate_success_interval is not None
        and candidate_success_interval[0] >= acceptance.min_task_success_rate
    )
    success_effect = summary["effects"]["task_success"]
    point_quality_noninferior = (
        quality_available
        and candidate_success
        >= baseline_success - acceptance.max_task_success_regression
    )
    quality_noninferior = bool(
        point_quality_noninferior
        and _task_success_noninferior(
            success_effect,
            method=acceptance.task_success_method,
            margin=acceptance.max_task_success_regression,
        )
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
            "name": "minimum_task_success_rate",
            "passed": baseline_absolute_quality_ok and candidate_absolute_quality_ok,
            "observed": {
                "baseline": baseline_success,
                "candidate": candidate_success,
                "baseline_95_ci": baseline_success_interval,
                "candidate_95_ci": candidate_success_interval,
            },
            "threshold": {
                "minimum_95_ci_lower_bound": acceptance.min_task_success_rate,
                "method": "Clopper-Pearson exact binomial",
            },
        },
        {
            "name": "task_success_noninferior",
            "passed": quality_noninferior,
            "observed": {
                "baseline": baseline_success,
                "candidate": candidate_success,
                "baseline_95_ci": baseline_outcomes.get("task_success_95_ci"),
                "candidate_95_ci": candidate_outcomes.get("task_success_95_ci"),
                "paired_difference": success_effect,
            },
            "threshold": {
                "max_regression": acceptance.max_task_success_regression,
                "method": acceptance.task_success_method,
            },
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
    elif not baseline_absolute_quality_ok:
        status = "invalid"
        reason = (
            "the baseline did not establish the configured minimum absolute "
            "task-success rate, so this workload cannot verify an optimization"
        )
    elif not candidate_absolute_quality_ok:
        status = "regression"
        reason = "the candidate failed the minimum absolute task-success rate"
    elif not point_quality_noninferior or not completion_ok:
        status = "regression"
        reason = "candidate failed a task-quality or completion guardrail"
    elif not quality_noninferior:
        status = "inconclusive"
        reason = (
            "observed task success did not regress, but the paired confidence "
            "bound did not establish the configured non-inferiority margin"
        )
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
    *,
    trial_id: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if trial_id is None:
        trial_id = f"{'warmup' if warmup else 'measured'}-{case.case_id}-{repetition:04d}"
    replacements = _trial_replacements(manifest, trial_id)
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


def _evaluate_trial(
    case: ExperimentCase,
    captured: CapturedResponse,
    *,
    variant: str,
    repetition: int,
    trial_id: str,
    warmup: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool | None]:
    if case.evaluator is None:
        return None, None, None
    context = {
        "case_id": case.case_id,
        "variant": variant,
        "repetition": repetition,
        "trial_id": trial_id,
        "warmup": warmup,
        "response_status": captured.status,
    }
    try:
        evaluated = evaluate_output(case.evaluator, captured.output_text, context)
    except EvaluatorError as exc:
        return None, {"kind": "evaluator_infrastructure_error", "message": str(exc)}, None
    return (
        evaluated.to_dict(),
        None,
        evaluated.success and captured.status == "completed",
    )


def _static_units(manifest: ExperimentManifest) -> list[list[ExperimentCase]]:
    """Units dispatched under load: cases, or whole ordered workflows."""

    if not manifest.execution.cases_form_workflow:
        return [[case] for case in manifest.cases]
    if all(case.workflow_id is not None for case in manifest.cases):
        order = list(dict.fromkeys(case.workflow_id for case in manifest.cases))
        return [
            [case for case in manifest.cases if case.workflow_id == workflow_id]
            for workflow_id in order
        ]
    return [list(manifest.cases)]


def static_unit_ids(manifest: ExperimentManifest) -> list[str]:
    units = _static_units(manifest)
    if not manifest.execution.cases_form_workflow:
        return [unit[0].case_id for unit in units]
    return [str(unit[0].workflow_id or "workflow") for unit in units]


def static_arrival_schedule(manifest: ExperimentManifest) -> ArrivalSchedule:
    """Build the frozen arrival schedule of a concurrent static manifest."""

    load = manifest.execution.load
    if load is None:
        raise ExperimentValidationError("the manifest declares sequential arrivals")
    try:
        return build_arrival_schedule(
            load,
            static_unit_ids(manifest),
            repetitions=manifest.execution.repetitions,
            seed=manifest.execution.seed,
            order=manifest.execution.order,
            warmup_repetitions=manifest.execution.warmup_repetitions,
        )
    except LoadSpecError as exc:
        raise ExperimentValidationError(str(exc)) from exc


class _StaticTrialExecutor:
    """Replay one static unit (a case or an ordered workflow) per call."""

    def __init__(self, manifest: ExperimentManifest, client: CaptureClient) -> None:
        self.manifest = manifest
        self.client = client
        self.units = _static_units(manifest)
        self._prior_prompts: list[tuple[int, ...]] = []
        self._lock = threading.Lock()

    def execute(
        self, trial: ScheduledTrial, *, role: str, filler_index: int | None
    ) -> dict[str, Any]:
        execution = self.manifest.execution
        warmup = role == ROLE_WARMUP
        unit_trials: list[dict[str, Any]] = []
        for case in self.units[trial.unit]:
            if role == ROLE_FILLER:
                trial_id = f"filler-{filler_index:06d}-{case.case_id}"
            else:
                prefix = "warmup" if warmup else "measured"
                trial_id = f"{prefix}-s{trial.stage}-{case.case_id}-{trial.repetition:04d}"
            trial_id, request, transform = _prepared_request(
                self.manifest,
                trial.repetition,
                case,
                trial.variant,
                warmup,
                trial_id=trial_id,
            )
            salted = execution.cache_isolation == "salt_isolated"
            if salted:
                request["cache_salt"] = secrets.token_urlsafe(32)
            captured = self.client.capture(request)
            if role == ROLE_FILLER:
                unit_trials.append(
                    {
                        "case_id": case.case_id,
                        "variant": trial.variant,
                        "status": captured.status,
                        "total_latency_ms": captured.total_latency_ms,
                    }
                )
                continue
            record = captured.to_dict()
            with self._lock:
                # Prompts of trials that finished before this one: a full-block
                # static upper bound under the observed concurrent order.
                record["predicted_eligible_cached_tokens"] = (
                    0
                    if salted
                    else _predicted_eligible_or_unknown(
                        captured.prompt_token_ids,
                        self._prior_prompts,
                        self.manifest.backend.block_size,
                    )
                )
                if not salted:
                    self._prior_prompts.append(captured.prompt_token_ids)
            evaluation, evaluation_error, task_success = _evaluate_trial(
                case,
                captured,
                variant=trial.variant,
                repetition=trial.repetition,
                trial_id=trial_id,
                warmup=warmup,
            )
            record.update(
                {
                    "case_id": case.case_id,
                    "workflow_id": case.workflow_id,
                    "variant": trial.variant,
                    "repetition": trial.repetition,
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
            unit_trials.append(record)
        return {
            "variant": trial.variant,
            "unit": trial.unit,
            "repetition": trial.repetition,
            "unit_trials": unit_trials,
        }


def _static_observations(
    trials: list[dict[str, Any]], *, workflow: bool
) -> list[Observation]:
    """Reduce measured static trials to metric-gate observations."""

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for trial in trials:
        cluster = (
            str(trial.get("workflow_id") or "workflow")
            if workflow
            else str(trial["case_id"])
        )
        groups.setdefault((cluster, trial["repetition"], trial["variant"]), []).append(trial)
    observations: list[Observation] = []
    for (cluster, repetition, variant), steps in sorted(groups.items()):
        latencies = [step.get("total_latency_ms") for step in steps]
        successes = [step.get("task_success") for step in steps]
        costs = [economics_of(step) for step in steps]

        def total(values: list[Any]) -> float | None:
            return None if any(value is None for value in values) else float(sum(values))

        observations.append(
            Observation(
                cluster=cluster,
                repetition=repetition,
                variant=variant,
                success=(
                    None
                    if any(value is None for value in successes)
                    else all(bool(value) for value in successes)
                ),
                workflow_latency_ms=total(latencies),
                requests={
                    **{
                        name: tuple(
                            float(step[name])
                            for step in steps
                            if step.get(name) is not None
                        )
                        for name in ("client_ttft_ms", "server_ttft_ms")
                    },
                    "tpot_ms": tuple(
                        value
                        for value in (
                            tpot_ms(
                                step.get("total_latency_ms"),
                                step.get("client_ttft_ms"),
                                step.get("output_tokens"),
                            )
                            for step in steps
                        )
                        if value is not None
                    ),
                },
                cost_usd=total([cost for cost, _ in costs]),
                gpu_time_ms=total([gpu for _, gpu in costs]),
            )
        )
    return observations


def _run_static_under_load(
    manifest: ExperimentManifest,
    client: CaptureClient,
    *,
    checkpoint_path: Path | None,
    observers: tuple[Any, ...],
) -> tuple[ArrivalSchedule, list[dict[str, Any]], list[dict[str, Any]], Any]:
    schedule = static_arrival_schedule(manifest)
    try:
        verify_schedule_pin(schedule)
    except LoadSpecError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    store = None
    if checkpoint_path is not None:
        if checkpoint_path.exists():
            raise ExperimentValidationError(
                f"checkpoint {checkpoint_path} already exists; choose another path"
            )
        store = JsonlArtifactStore(
            checkpoint_path,
            {
                **_checkpoint_header(manifest),
                "checkpoint_schema_version": "0.2",
                "arrival_schedule_sha256": schedule.digest(),
            },
        )
    if manifest.execution.cache_treatment == "warm" and _can_reset(client):
        client.reset_prefix_cache(reset_external=manifest.execution.reset_external)
    runner = LoadRunner(
        schedule, _StaticTrialExecutor(manifest, client), store=store, observers=observers
    )
    result = runner.run()
    trials: list[dict[str, Any]] = []
    for unit_record in result.records:
        for step in unit_record["unit_trials"]:
            step["sequence"] = len(trials)
            step["load"] = dict(unit_record["load"])
            if "server_telemetry" in unit_record:
                step["server_telemetry"] = unit_record["server_telemetry"]
            trials.append(step)
    return schedule, trials, result.records, result


def run_experiment(
    manifest: ExperimentManifest,
    client: CaptureClient,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    observers: tuple[Any, ...] | list[Any] = (),
    attempt_economics: Any = None,
) -> dict[str, Any]:
    """Run a frozen static experiment.

    ``observers`` receive load-block and trial callbacks (see
    :class:`runeconomy.execution.LoadObserver`). ``attempt_economics`` maps a
    measured trial record to ``{"cost_usd", "gpu_time_ms", ...}``; the result
    is stored on the record as ``economics`` before any decision is made.
    """

    if manifest.execution.load is not None:
        if resume:
            raise ExperimentValidationError(
                "resume is not supported for concurrent arrival patterns; a "
                "partial block cannot be replayed under the same load"
            )
        identity_before = _read_environment_identity(client)
        schedule, trials, unit_records, load_result = _run_static_under_load(
            manifest,
            client,
            checkpoint_path=checkpoint_path,
            observers=tuple(observers),
        )
        return _finish_experiment(
            manifest,
            client,
            trials,
            identity_before=identity_before,
            attempt_economics=attempt_economics,
            schedule=schedule,
            unit_records=unit_records,
            load_result=load_result,
        )
    return _run_sequential_experiment(
        manifest,
        client,
        checkpoint_path=checkpoint_path,
        resume=resume,
        attempt_economics=attempt_economics,
    )


def _run_sequential_experiment(
    manifest: ExperimentManifest,
    client: CaptureClient,
    *,
    checkpoint_path: Path | None,
    resume: bool,
    attempt_economics: Any,
) -> dict[str, Any]:
    execution = manifest.execution
    if resume and not BACKEND_PROFILES[manifest.backend.kind].exact_prompt_tokens:
        raise ExperimentValidationError(
            f"resume is not supported on {manifest.backend.kind}: replayed prompts "
            "cannot be verified without exact prompt tokens"
        )
    identity_before = _read_environment_identity(client)
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
    prior_prompts: list[tuple[int, ...] | None] = []
    if execution.cache_treatment == "warm" and _can_reset(client):
        client.reset_prefix_cache(reset_external=execution.reset_external)

    for sequence, (repetition, case, variant, warmup) in enumerate(schedule):
        if execution.cache_treatment == "cold" and execution.cache_isolation == "physical_reset":
            client.reset_prefix_cache(reset_external=execution.reset_external)
            prior_prompts.clear()
        trial_id, request, transform = _prepared_request(
            manifest, repetition, case, variant, warmup
        )
        if execution.cache_isolation == "salt_isolated":
            request["cache_salt"] = secrets.token_urlsafe(32)
            prior_prompts.clear()
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
        trial["predicted_eligible_cached_tokens"] = _predicted_eligible_or_unknown(
            captured.prompt_token_ids,
            prior_prompts,
            manifest.backend.block_size,
        )
        prior_prompts.append(captured.prompt_token_ids)
        evaluation, evaluation_error, task_success = _evaluate_trial(
            case,
            captured,
            variant=variant,
            repetition=repetition,
            trial_id=trial_id,
            warmup=warmup,
        )
        trial.update(
            {
                "sequence": sequence,
                "case_id": case.case_id,
                "workflow_id": case.workflow_id,
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

    return _finish_experiment(
        manifest,
        client,
        trials,
        identity_before=identity_before,
        attempt_economics=attempt_economics,
    )


def _summaries(
    manifest: ExperimentManifest, trials: list[dict[str, Any]]
) -> dict[str, Any]:
    execution = manifest.execution
    method_arguments = {
        "task_success_method": manifest.acceptance.task_success_method,
        "max_task_success_regression": (
            manifest.acceptance.max_task_success_regression
        ),
    }
    summary = _build_summary(trials, execution.seed, **method_arguments)
    if execution.cases_form_workflow:
        summary["workflow"] = _build_workflow_summary(
            trials, execution.seed, manifest.cases, **method_arguments
        )
    return summary


def _finish_experiment(
    manifest: ExperimentManifest,
    client: CaptureClient,
    trials: list[dict[str, Any]],
    *,
    identity_before: Mapping[str, Any] | None,
    attempt_economics: Any,
    schedule: ArrivalSchedule | None = None,
    unit_records: list[dict[str, Any]] | None = None,
    load_result: Any = None,
) -> dict[str, Any]:
    execution = manifest.execution
    expected_identity = _expected_environment_identity(manifest)
    if attempt_economics is not None:
        for trial in trials:
            if not trial["warmup"]:
                trial["economics"] = dict(attempt_economics(trial))
    load = execution.load
    stage_summaries: list[dict[str, Any]] = []
    if load is not None:
        gate_stage = load.gate_stage
        gated_trials = [
            trial for trial in trials if trial["load"]["stage"] == gate_stage
        ]
        for stage, level in enumerate(load.levels):
            if stage == gate_stage:
                continue
            stage_trials = [trial for trial in trials if trial["load"]["stage"] == stage]
            stage_summary = _summaries(manifest, stage_trials)
            stage_basis = stage_summary.get("workflow", stage_summary)
            stage_summaries.append(
                {
                    "stage": stage,
                    load.level_name: level,
                    "gated": False,
                    "summary": stage_summary,
                    "metric_gates": evaluate_metric_gates(
                        manifest.metric_gates,
                        _static_observations(
                            [t for t in stage_trials if not t["warmup"]],
                            workflow=execution.cases_form_workflow,
                        ),
                        seed=execution.seed,
                    ),
                    "descriptive_only": True,
                    "basis_effects": stage_basis.get("effects"),
                }
            )
    else:
        gated_trials = trials
    summary = _summaries(manifest, gated_trials)
    decision_basis = summary.get("workflow", summary)
    prediction_assessment = _assess_predeclared_prediction(
        decision_basis, manifest.design
    )
    if prediction_assessment is not None:
        summary["predeclared_prediction"] = prediction_assessment
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
    if load is not None or manifest.metric_gates:
        metric_checks = evaluate_metric_gates(
            manifest.metric_gates,
            _static_observations(
                [trial for trial in gated_trials if not trial["warmup"]],
                workflow=execution.cases_form_workflow,
            ),
            seed=execution.seed,
        )
        integrity_checks = (
            []
            if load is None or unit_records is None
            else load_integrity_checks(
                [
                    record
                    for record in unit_records
                    if record["load"]["stage"] == load.gate_stage
                    and record["load"]["role"] != ROLE_WARMUP
                ],
                open_loop=load.open_loop,
                gate_level=load.gate_level,
                max_dispatch_lag_ms=load.max_dispatch_lag_ms,
            )
        )
        decision = merge_load_and_metric_checks(
            decision, integrity_checks=integrity_checks, metric_checks=metric_checks
        )
        if load is not None:
            summary["load"] = {
                "arrival_schedule": schedule.summary() if schedule else None,
                "gate_stage": load.gate_stage,
                load.level_name: load.gate_level,
                "population": "gate-stage measured trials only; other stages are descriptive",
                "stages": stage_summaries,
                "filler_trials": len(load_result.fillers) if load_result else 0,
            }
    identity_after = _read_environment_identity(client)
    identity_check = _environment_identity_check(
        expected_identity, identity_before, identity_after
    )
    if manifest.acceptance.require_environment_identity:
        decision = {
            **decision,
            **(
                {}
                if identity_check["verified"]
                else {
                    "status": "invalid",
                    "reason": (
                        "the live environment identity was missing, mismatched, or "
                        "changed during the experiment"
                    ),
                }
            ),
            "checks": [
                {
                    "name": "environment_identity",
                    "passed": identity_check["verified"],
                    "required": True,
                    "observed": identity_check,
                    "threshold": "all pinned fields observed before and after without drift",
                },
                *decision["checks"],
            ],
        }
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
    if manifest.identity_policy is not None:
        assurance, identity_grade_check = _static_identity_assurance(manifest, trials)
        decision_basis["identity_assurance"] = assurance
        decision = {
            **decision,
            **(
                {}
                if identity_grade_check["passed"]
                else {
                    "status": "invalid",
                    "reason": (
                        "graded serving identity was incomplete, below the declared "
                        "minimum grade, or the served model differed from the manifest"
                    ),
                }
            ),
            "checks": [identity_grade_check, *decision["checks"]],
        }
    decision = enrich_decision_artifact(decision, decision_basis)
    acceptance_record = asdict(manifest.acceptance)
    _normalize_metric_gates(acceptance_record, manifest.metric_gates)
    if manifest.identity_policy is not None:
        acceptance_record["identity_policy"] = dict(manifest.identity_policy)
    result = {
        "result_schema_version": (
            "0.4"
            if load is not None or manifest.metric_gates
            else "0.3"
        ),
        "experiment_id": manifest.experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "backend_kind": manifest.backend.kind,
            "api_dialect": manifest.backend.api_dialect,
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
            "cache_isolation": execution.cache_isolation,
            "arrival_pattern": execution.arrival_pattern,
            "cases_form_workflow": execution.cases_form_workflow,
            "order": execution.order,
            "seed": execution.seed,
            "acceptance": acceptance_record,
            "design": deepcopy(manifest.design),
            "manifest_sha256": manifest.fingerprint(),
            "reproducibility_manifest_sha256": (
                manifest.reproducibility_fingerprint()
            ),
            "environment_identity": identity_check,
        },
        "trials": trials,
        "evaluator_errors": evaluator_errors,
        "summary": summary,
        "decision": decision,
    }
    if manifest.cost_model is not None:
        model = parse_cost_model(manifest.cost_model)
        result["provenance"]["cost_model"] = {"model": model.to_dict(), "sha256": model.sha256}
    if manifest.backend.kind != "vllm":
        result["provenance"]["provider"] = manifest.backend.provider
        result["provenance"]["undisclosed_identity"] = list(
            manifest.backend.undisclosed_identity
        )
    capability_record = getattr(client, "capability_record", None)
    serving_capabilities = capability_record() if callable(capability_record) else None
    if serving_capabilities is not None:
        result["serving_capabilities"] = serving_capabilities
    if load is not None and schedule is not None:
        result["provenance"]["load"] = load.to_dict()
        result["provenance"]["arrival_schedule_sha256"] = schedule.digest()
        result["arrival_schedule"] = schedule.to_dict()
        result["load_blocks"] = load_result.blocks if load_result else []
        result["load_fillers"] = load_result.fillers if load_result else []
    return result
