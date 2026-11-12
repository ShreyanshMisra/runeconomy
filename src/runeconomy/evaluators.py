"""Deterministic, local task-success evaluators for experiment outputs."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import jsonschema


class EvaluatorError(ValueError):
    pass


class EvaluatorInfrastructureError(EvaluatorError):
    """Raised when an evaluator cannot run, rather than when a task fails."""


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    evaluator_kind: str
    score: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluatorError(f"{location} must be an object")
    return value


def validate_evaluator_spec(spec: Any, location: str = "evaluator") -> dict[str, Any]:
    value = _require_object(spec, location)
    kind = value.get("kind")
    if kind == "exact":
        if not isinstance(value.get("expected"), str):
            raise EvaluatorError(f"{location}.expected must be a string")
    elif kind == "regex":
        pattern = value.get("pattern")
        if not isinstance(pattern, str):
            raise EvaluatorError(f"{location}.pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise EvaluatorError(f"{location}.pattern is invalid: {exc}") from exc
        flags = value.get("flags", [])
        allowed_flags = {"IGNORECASE", "MULTILINE", "DOTALL"}
        if not isinstance(flags, list) or any(
            flag not in allowed_flags for flag in flags
        ):
            raise EvaluatorError(
                f"{location}.flags may contain IGNORECASE, MULTILINE, or DOTALL"
            )
    elif kind == "json_schema":
        schema = _require_object(value.get("schema"), f"{location}.schema")
        try:
            jsonschema.validators.validator_for(schema).check_schema(schema)
        except jsonschema.exceptions.SchemaError as exc:
            raise EvaluatorError(
                f"{location}.schema is invalid: {exc.message}"
            ) from exc
    elif kind == "python":
        target = value.get("callable")
        if not isinstance(target, str) or ":" not in target:
            raise EvaluatorError(
                f"{location}.callable must use the form 'module:function'"
            )
    elif kind == "command":
        argv = value.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise EvaluatorError(f"{location}.argv must be a non-empty string array")
        timeout = value.get("timeout_seconds", 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise EvaluatorError(f"{location}.timeout_seconds must be positive")
    elif kind == "all":
        evaluators = value.get("evaluators")
        if not isinstance(evaluators, list) or not evaluators:
            raise EvaluatorError(f"{location}.evaluators must be a non-empty array")
        for index, child in enumerate(evaluators):
            validate_evaluator_spec(child, f"{location}.evaluators[{index}]")
    else:
        raise EvaluatorError(
            f"{location}.kind must be exact, regex, json_schema, python, command, or all"
        )
    return dict(value)


def _normalize_text(text: str, spec: Mapping[str, Any]) -> str:
    result = text.strip() if spec.get("strip", True) else text
    return result if spec.get("case_sensitive", True) else result.casefold()


def _coerce_python_result(result: Any) -> tuple[bool, float, dict[str, Any]]:
    if isinstance(result, bool):
        return result, float(result), {}
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        score = float(result)
        return score > 0, score, {}
    if isinstance(result, dict) and isinstance(result.get("success"), bool):
        score_value = result.get("score", float(result["success"]))
        if not isinstance(score_value, (int, float)):
            raise EvaluatorError("python evaluator result.score must be numeric")
        details = result.get("details", {})
        if not isinstance(details, dict):
            raise EvaluatorError("python evaluator result.details must be an object")
        return result["success"], float(score_value), details
    raise EvaluatorError(
        "python evaluator must return bool, number, or an object with boolean success"
    )


def evaluate_output(
    spec: Mapping[str, Any], output_text: str | None, context: Mapping[str, Any]
) -> EvaluationResult:
    validated = validate_evaluator_spec(dict(spec))
    kind = validated["kind"]
    if output_text is None:
        return EvaluationResult(False, kind, 0.0, {"reason": "no_output_text"})

    if kind == "exact":
        actual = _normalize_text(output_text, validated)
        expected = _normalize_text(validated["expected"], validated)
        success = actual == expected
        return EvaluationResult(success, kind, float(success), {})

    if kind == "regex":
        flags = 0
        for flag_name in validated.get("flags", []):
            flags |= getattr(re, flag_name)
        pattern = re.compile(validated["pattern"], flags)
        match = (
            pattern.fullmatch(output_text)
            if validated.get("full_match", True)
            else pattern.search(output_text)
        )
        success = match is not None
        return EvaluationResult(success, kind, float(success), {})

    if kind == "json_schema":
        try:
            document = json.loads(output_text)
        except json.JSONDecodeError as exc:
            return EvaluationResult(
                False,
                kind,
                0.0,
                {"reason": "invalid_json", "line": exc.lineno, "column": exc.colno},
            )
        try:
            jsonschema.validate(document, validated["schema"])
        except jsonschema.ValidationError as exc:
            return EvaluationResult(
                False,
                kind,
                0.0,
                {"reason": "schema_mismatch", "path": list(exc.absolute_path)},
            )
        return EvaluationResult(True, kind, 1.0, {})

    if kind == "python":
        module_name, function_name = validated["callable"].split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ValueError) as exc:
            raise EvaluatorInfrastructureError(
                f"cannot import python evaluator module {module_name!r}: {exc}"
            ) from exc
        function = getattr(module, function_name, None)
        if not callable(function):
            raise EvaluatorInfrastructureError(
                f"python evaluator {validated['callable']!r} is not callable"
            )
        try:
            raw_result = function(output_text, dict(context))
        except Exception as exc:
            raise EvaluatorInfrastructureError(
                f"python evaluator {validated['callable']!r} raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        success, score, details = _coerce_python_result(raw_result)
        return EvaluationResult(success, kind, score, details)

    if kind == "command":
        timeout_seconds = float(validated.get("timeout_seconds", 30))
        try:
            completed = subprocess.run(
                validated["argv"],
                input=output_text,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluatorInfrastructureError(
                f"command evaluator timed out after {timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise EvaluatorInfrastructureError(
                f"cannot run command evaluator {validated['argv'][0]!r}: {exc}"
            ) from exc
        success = completed.returncode == 0
        details: dict[str, Any] = {"returncode": completed.returncode}
        if completed.stderr:
            details["stderr"] = completed.stderr[-2000:]
        return EvaluationResult(success, kind, float(success), details)

    child_results = [
        evaluate_output(child, output_text, context)
        for child in validated["evaluators"]
    ]
    success = all(result.success for result in child_results)
    score = min(result.score for result in child_results)
    return EvaluationResult(
        success,
        kind,
        score,
        {"children": [result.to_dict() for result in child_results]},
    )
