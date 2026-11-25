"""Extended evaluator kinds: repo tests, tool trajectories, LLM judges and
adapters that run Promptfoo assertions, Braintrust scorers and Inspect AI
scorers unchanged.

Every kind maps onto RunEconomy's per-attempt contract: ``success`` (bool),
``score`` (float) and ``details`` (object). An evaluator that cannot reach a
judgment raises :class:`EvaluatorInfrastructureError`. That makes the attempt
unevaluable, which counts against completion. It is never a silent task
failure or success.

``pytest``
    Runs the repository's tests in the attempt's working copy. Exit 0 is
    success and exit 1 is task failure. Any other pytest exit (interrupted,
    internal error, usage error, no tests collected) is an infrastructure
    error.
``tool_trajectory``
    Grades the sequence of tool calls, read from the attempt's TOOL spans or
    from JSON output, against an expected trajectory.
``llm_judge``
    A rubric or reference-based judge on an OpenAI-compatible endpoint with a
    pinned judge model. It refuses to run unless a calibration record (see
    :func:`calibrate_judge`) is present, digest-matched, made with the same
    judge and rubric, and meets the declared human-agreement threshold. Every
    judgment records the judge identity the endpoint reported.
``promptfoo``
    Promptfoo ``assert`` entries, as written in a Promptfoo config (JSON
    form), including ``file://`` Python assertions with ``get_assert``.
``braintrust``
    A Braintrust/autoevals scorer function called as
    ``scorer(input=..., output=..., expected=...)``. It passes when its score
    meets ``threshold``.
``inspect_ai``
    An Inspect AI scorer factory (for example ``inspect_ai.scorer:includes``),
    called with the attempt's output in a single-sample ``TaskState`` and the
    task's target. Its ``Score.value`` maps to a number through ``value_map``
    (Inspect's CORRECT/INCORRECT/PARTIAL/NOANSWER by default) and passes when
    it meets ``threshold``.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


EXTENDED_KINDS = ("pytest", "tool_trajectory", "llm_judge", "promptfoo", "braintrust", "inspect_ai")
INSPECT_VALUE_MAP = {"C": 1.0, "I": 0.0, "P": 0.5, "N": 0.0}
TRAJECTORY_MATCHES = ("exact", "in_order", "unordered", "superset")
JUDGE_MODES = ("rubric", "reference")
JUDGE_CALIBRATION_VERSION = "0.1"
PROMPTFOO_TYPES = (
    "equals",
    "contains",
    "icontains",
    "contains-any",
    "contains-all",
    "icontains-any",
    "icontains-all",
    "regex",
    "starts-with",
    "is-json",
    "contains-json",
    "python",
)


def _errors() -> tuple[type[Exception], type[Exception]]:
    from .evaluators import EvaluatorError, EvaluatorInfrastructureError

    return EvaluatorError, EvaluatorInfrastructureError


def _invalid(message: str) -> Exception:
    return _errors()[0](message)


def _infrastructure(message: str) -> Exception:
    return _errors()[1](message)


def _string_list(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise _invalid(f"{location} must be an array of non-empty strings")
    return list(value)


def _positive(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise _invalid(f"{location} must be a positive number")
    return float(value)


def _unit(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise _invalid(f"{location} must be between 0 and 1")
    return float(value)


def is_executable(spec: Mapping[str, Any]) -> bool:
    """Whether a spec runs local code or contacts a network judge."""

    kind = spec.get("kind")
    if kind in {"pytest", "braintrust", "llm_judge", "inspect_ai"}:
        return True
    if kind == "promptfoo":
        return any(
            isinstance(item, Mapping) and _base_type(str(item.get("type", ""))) == "python"
            for item in spec.get("assert", [])
        )
    return False


# ---------------------------------------------------------------- validation


def validate_extended(value: dict[str, Any], location: str) -> dict[str, Any]:
    kind = value["kind"]
    if kind == "pytest":
        if "args" in value:
            _string_list(value["args"], f"{location}.args")
        if "workdir" in value and (not isinstance(value["workdir"], str) or not value["workdir"]):
            raise _invalid(f"{location}.workdir must be a non-empty string")
        if "tests_field" in value and not isinstance(value["tests_field"], str):
            raise _invalid(f"{location}.tests_field must be a string")
        _positive(value.get("timeout_seconds", 600), f"{location}.timeout_seconds")
    elif kind == "tool_trajectory":
        if ("expected" in value) == ("expected_field" in value):
            raise _invalid(f"{location} requires exactly one of expected or expected_field")
        if "expected" in value:
            _string_list(value["expected"], f"{location}.expected")
        if value.get("match", "exact") not in TRAJECTORY_MATCHES:
            raise _invalid(f"{location}.match must be one of {', '.join(TRAJECTORY_MATCHES)}")
        if value.get("source", "trace") not in {"trace", "output_json"}:
            raise _invalid(f"{location}.source must be trace or output_json")
        if "forbidden" in value:
            _string_list(value["forbidden"], f"{location}.forbidden")
        if "max_calls" in value and (
            isinstance(value["max_calls"], bool) or not isinstance(value["max_calls"], int) or value["max_calls"] < 0
        ):
            raise _invalid(f"{location}.max_calls must be a non-negative integer")
    elif kind == "llm_judge":
        if value.get("mode", "rubric") not in JUDGE_MODES:
            raise _invalid(f"{location}.mode must be rubric or reference (pairwise judgments are not per-attempt pass/fail)")
        if not isinstance(value.get("rubric"), str) or not value["rubric"].strip():
            raise _invalid(f"{location}.rubric must be a non-empty string")
        judge = value.get("judge")
        if not isinstance(judge, Mapping):
            raise _invalid(f"{location}.judge must be an object")
        for field in ("model", "base_url_env"):
            if not isinstance(judge.get(field), str) or not judge[field]:
                raise _invalid(f"{location}.judge.{field} must be a non-empty string")
        _unit(value.get("pass_threshold", 0.5), f"{location}.pass_threshold")
        calibration = value.get("calibration")
        if not isinstance(calibration, Mapping) or not isinstance(calibration.get("path"), str):
            raise _invalid(
                f"{location}.calibration.path is required: an uncalibrated judge cannot gate task success"
            )
        if not str(calibration.get("sha256", "")).startswith("sha256:"):
            raise _invalid(f"{location}.calibration.sha256 must pin the calibration record")
        _unit(calibration.get("min_kappa", 0.6), f"{location}.calibration.min_kappa")
    elif kind == "promptfoo":
        assertions = value.get("assert")
        if not isinstance(assertions, list) or not assertions:
            raise _invalid(f"{location}.assert must be a non-empty array of Promptfoo assertions")
        for index, item in enumerate(assertions):
            if not isinstance(item, Mapping) or not isinstance(item.get("type"), str):
                raise _invalid(f"{location}.assert[{index}].type is required")
            base = _base_type(item["type"])
            if base not in PROMPTFOO_TYPES:
                raise _invalid(
                    f"{location}.assert[{index}].type {item['type']!r} does not map onto a deterministic "
                    f"per-attempt pass/fail; supported: {', '.join(PROMPTFOO_TYPES)} and their not- forms"
                )
            if base == "regex":
                try:
                    re.compile(str(item.get("value", "")))
                except re.error as exc:
                    raise _invalid(f"{location}.assert[{index}].value is not a valid regex: {exc}") from exc
    elif kind == "braintrust":
        scorer = value.get("scorer")
        if not isinstance(scorer, str) or ":" not in scorer:
            raise _invalid(f"{location}.scorer must use the form 'module:function'")
        _unit(value.get("threshold", 0.5), f"{location}.threshold")
    elif kind == "inspect_ai":
        scorer = value.get("scorer")
        if not isinstance(scorer, str) or ":" not in scorer:
            raise _invalid(f"{location}.scorer must use the form 'module:factory'")
        if "scorer_args" in value and not isinstance(value["scorer_args"], Mapping):
            raise _invalid(f"{location}.scorer_args must be an object")
        for field in ("input_field", "target_field", "value_key"):
            if field in value and (not isinstance(value[field], str) or not value[field]):
                raise _invalid(f"{location}.{field} must be a non-empty string")
        value_map = value.get("value_map", INSPECT_VALUE_MAP)
        if not isinstance(value_map, Mapping) or not value_map:
            raise _invalid(f"{location}.value_map must be a non-empty object")
        for key, mapped in value_map.items():
            _unit(mapped, f"{location}.value_map.{key}")
        _unit(value.get("threshold", 1.0), f"{location}.threshold")
    return dict(value)


# ---------------------------------------------------------------- evaluation


def evaluate_extended(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    kind = spec["kind"]
    handler: Callable[..., tuple[bool, float, dict[str, Any]]] = {
        "pytest": _pytest,
        "tool_trajectory": _trajectory,
        "llm_judge": _judge,
        "promptfoo": _promptfoo,
        "braintrust": _braintrust,
        "inspect_ai": _inspect_ai,
    }[kind]
    return handler(spec, output_text, context)


def _task(context: Mapping[str, Any]) -> Mapping[str, Any]:
    task = context.get("task")
    return task if isinstance(task, Mapping) else {}


def _pytest(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    attempt_dir = context.get("attempt_dir")
    if not isinstance(attempt_dir, str):
        raise _infrastructure("the pytest evaluator needs an agent-process attempt directory")
    workdir = Path(attempt_dir) / spec.get("workdir", ".")
    if not workdir.is_dir():
        raise _infrastructure("the pytest working directory does not exist in the attempt")
    selected: list[str] = []
    if spec.get("tests_field"):
        value = _task(context).get(spec["tests_field"])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise _infrastructure(f"task.{spec['tests_field']} must list pytest node IDs")
        selected = list(value)
    with tempfile.TemporaryDirectory(prefix="runeconomy-pytest-") as scratch:
        report = Path(scratch) / "junit.xml"
        command = [
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={report}",
            *spec.get("args", ["-q"]), *selected,
        ]
        try:
            completed = subprocess.run(
                command, cwd=workdir, capture_output=True, text=True,
                timeout=float(spec.get("timeout_seconds", 600)), check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise _infrastructure("pytest evaluator timed out") from exc
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        if report.exists():
            # The report is written by pytest into a scratch directory this
            # function owns, under the same trust as the tests it runs.
            root = ElementTree.parse(report).getroot()
            suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
            for suite in suites:
                for key in counts:
                    counts[key] += int(suite.get(key, 0) or 0)
    if completed.returncode not in (0, 1):
        raise _infrastructure(
            f"pytest exited {completed.returncode} (interrupted, internal error, usage error or no tests collected)"
        )
    executed = counts["tests"] - counts["skipped"]
    passed = executed - counts["failures"] - counts["errors"]
    score = passed / executed if executed > 0 else 0.0
    return completed.returncode == 0 and executed > 0, score, {"pytest_exit_code": completed.returncode, **counts}


def _trace_tools(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _infrastructure("cannot read the attempt trace for tool calls") from exc
    spans: list[tuple[int, str]] = []
    for resource in payload.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                attributes = {
                    item.get("key"): next(iter((item.get("value") or {}).values()), None)
                    for item in span.get("attributes", [])
                    if isinstance(item, Mapping)
                }
                if str(attributes.get("openinference.span.kind", "")).upper() != "TOOL":
                    continue
                name = attributes.get("tool.name") or span.get("name")
                spans.append((int(span.get("startTimeUnixNano", 0)), str(name)))
    return [name for _, name in sorted(spans)]


def _output_tools(output_text: str) -> list[str]:
    try:
        document = json.loads(output_text)
    except json.JSONDecodeError:
        return []
    calls = document.get("tool_calls", []) if isinstance(document, Mapping) else document
    names: list[str] = []
    for call in calls if isinstance(calls, list) else []:
        if isinstance(call, str):
            names.append(call)
        elif isinstance(call, Mapping):
            function = call.get("function")
            name = call.get("name") or (function.get("name") if isinstance(function, Mapping) else None)
            if isinstance(name, str):
                names.append(name)
    return names


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    iterator = iter(actual)
    return all(any(item == candidate for candidate in iterator) for item in expected)


def _trajectory(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    if spec.get("expected_field"):
        expected = _task(context).get(spec["expected_field"])
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise _infrastructure(f"task.{spec['expected_field']} must list expected tool names")
    else:
        expected = list(spec["expected"])
    if spec.get("source", "trace") == "trace":
        trace_path = context.get("trace_path")
        if not isinstance(trace_path, str):
            raise _infrastructure("tool_trajectory source=trace needs the attempt trace")
        actual = _trace_tools(Path(trace_path))
    else:
        actual = _output_tools(output_text)
    match = spec.get("match", "exact")
    matched = {
        "exact": actual == expected,
        "in_order": _is_subsequence(expected, actual),
        "unordered": sorted(actual) == sorted(expected),
        "superset": set(expected) <= set(actual),
    }[match]
    forbidden = sorted(set(spec.get("forbidden", [])) & set(actual))
    within = "max_calls" not in spec or len(actual) <= spec["max_calls"]
    success = matched and not forbidden and within
    covered = len(set(expected) & set(actual)) / len(set(expected)) if expected else 1.0
    return success, (covered if not success else 1.0), {
        "match": match,
        "expected": expected,
        "actual": actual,
        "forbidden_called": forbidden,
        "within_max_calls": within,
    }


# ------------------------------------------------------------- LLM judge


def _rubric_sha256(spec: Mapping[str, Any]) -> str:
    payload = {"mode": spec.get("mode", "rubric"), "rubric": spec["rubric"]}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _judge_messages(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> list[dict[str, str]]:
    task = _task(context)
    parts = [f"Rubric:\n{spec['rubric']}"]
    if task.get("input") is not None or task.get("prompt") is not None:
        parts.append(f"Task:\n{task.get('input', task.get('prompt'))}")
    if spec.get("mode") == "reference":
        reference = task.get(spec.get("reference_field", "expected"))
        if reference is None:
            raise _infrastructure("reference judging needs the task's reference answer")
        parts.append(f"Reference answer:\n{reference}")
    parts.append(f"Response to grade:\n{output_text}")
    return [
        {
            "role": "system",
            "content": (
                "You grade one response against a rubric. Reply with only a JSON object: "
                '{"score": <number from 0 to 1>, "reason": "<one sentence>"}.'
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def judge_once(
    spec: Mapping[str, Any],
    output_text: str,
    context: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    environment = os.environ if environment is None else environment
    judge = spec["judge"]
    base = environment.get(judge["base_url_env"])
    if not base:
        raise _infrastructure(f"judge endpoint variable {judge['base_url_env']!r} is not set")
    key = None
    if judge.get("api_key_env"):
        key = environment.get(judge["api_key_env"])
        if not key:
            raise _infrastructure(f"judge API key variable {judge['api_key_env']!r} is not set")
    body = {
        "model": judge["model"],
        "messages": _judge_messages(spec, output_text, context),
        "temperature": 0,
        "max_tokens": int(spec.get("max_output_tokens", 256)),
    }
    if "seed" in judge:
        body["seed"] = judge["seed"]
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if key:
        request.add_unredirected_header("Authorization", f"Bearer {key}")
    try:
        with opener(request, timeout=float(spec.get("timeout_seconds", 60))) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise _infrastructure("the judge endpoint did not return a usable response") from exc
    reported = str(payload.get("model", ""))
    if reported and not (reported == judge["model"] or reported.startswith(judge["model"])):
        raise _infrastructure("the judge endpoint served a different model than the pinned judge")
    try:
        content = payload["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        verdict = json.loads(match.group(0) if match else content)
        score = float(verdict["score"])
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise _infrastructure("the judge reply was not a JSON score") from exc
    if not 0 <= score <= 1:
        raise _infrastructure("the judge score was outside 0..1")
    fingerprint = payload.get("system_fingerprint")
    return {
        "score": score,
        "reason": str(verdict.get("reason", ""))[:500],
        "judge_model_reported": reported or None,
        "judge_fingerprint_sha256": (
            None if not fingerprint else "sha256:" + hashlib.sha256(str(fingerprint).encode()).hexdigest()
        ),
    }


def _calibration(spec: Mapping[str, Any]) -> dict[str, Any]:
    calibration = spec["calibration"]
    path = Path(calibration["path"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _infrastructure("the judge calibration record is missing") from exc
    if "sha256:" + hashlib.sha256(raw).hexdigest() != calibration["sha256"]:
        raise _infrastructure("the judge calibration record does not match its pinned digest")
    record = json.loads(raw)
    if record.get("judge", {}).get("model") != spec["judge"]["model"]:
        raise _infrastructure("the calibration was made with a different judge model")
    if record.get("rubric_sha256") != _rubric_sha256(spec):
        raise _infrastructure("the calibration was made with a different rubric")
    kappa = record.get("cohen_kappa")
    if not isinstance(kappa, (int, float)) or kappa < float(calibration.get("min_kappa", 0.6)):
        raise _infrastructure("the judge's recorded human agreement is below the declared minimum")
    return record


def _judge(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    record = _calibration(spec)
    verdict = judge_once(spec, output_text, context)
    threshold = float(spec.get("pass_threshold", 0.5))
    return verdict["score"] >= threshold, verdict["score"], {
        **verdict,
        "pass_threshold": threshold,
        "calibration_sha256": spec["calibration"]["sha256"],
        "calibration_cohen_kappa": record["cohen_kappa"],
    }


def cohen_kappa(human: list[bool], judge: list[bool]) -> float | None:
    n = len(human)
    if n == 0 or n != len(judge):
        return None
    observed = sum(a == b for a, b in zip(human, judge)) / n
    p_human, p_judge = sum(human) / n, sum(judge) / n
    expected = p_human * p_judge + (1 - p_human) * (1 - p_judge)
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def calibrate_judge(
    spec: Mapping[str, Any],
    labels: list[Mapping[str, Any]],
    *,
    judge: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure judge-versus-human agreement on labeled examples.

    Each label is ``{"output_text": str, "task": {...}, "human_pass": bool}``.
    The returned record is what ``llm_judge.calibration`` pins by digest.
    """

    from .evaluators import validate_evaluator_spec

    spec = validate_evaluator_spec(dict(spec))
    if spec["kind"] != "llm_judge":
        raise _invalid("calibration applies to llm_judge specs")
    if len(labels) < 20:
        raise _invalid("calibration needs at least 20 human-labeled examples")
    run = judge or (lambda text, context: judge_once(spec, text, context))
    threshold = float(spec.get("pass_threshold", 0.5))
    human: list[bool] = []
    judged: list[bool] = []
    for index, label in enumerate(labels):
        if not isinstance(label.get("output_text"), str) or not isinstance(label.get("human_pass"), bool):
            raise _invalid(f"labels[{index}] needs output_text and boolean human_pass")
        verdict = run(label["output_text"], {"task": label.get("task", {})})
        human.append(label["human_pass"])
        judged.append(verdict["score"] >= threshold)
    confusion = {
        "both_pass": sum(h and j for h, j in zip(human, judged)),
        "both_fail": sum((not h) and (not j) for h, j in zip(human, judged)),
        "human_pass_judge_fail": sum(h and not j for h, j in zip(human, judged)),
        "human_fail_judge_pass": sum((not h) and j for h, j in zip(human, judged)),
    }
    labels_digest = hashlib.sha256(json.dumps(list(labels), sort_keys=True, default=str).encode()).hexdigest()
    return {
        "artifact_type": "runeconomy.judge_calibration",
        "schema_version": JUDGE_CALIBRATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "judge": {key: spec["judge"][key] for key in ("model", "provider", "model_revision") if key in spec["judge"]},
        "mode": spec.get("mode", "rubric"),
        "rubric_sha256": _rubric_sha256(spec),
        "pass_threshold": threshold,
        "n": len(labels),
        "labels_sha256": "sha256:" + labels_digest,
        "raw_agreement": sum(h == j for h, j in zip(human, judged)) / len(labels),
        "cohen_kappa": cohen_kappa(human, judged),
        "confusion": confusion,
    }


# -------------------------------------------------------------- Promptfoo


def _base_type(kind: str) -> str:
    return kind[4:] if kind.startswith("not-") else kind


def _render(value: Any, variables: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}",
            lambda match: str(variables.get(match.group(1), match.group(0))),
            value,
        )
    if isinstance(value, list):
        return [_render(item, variables) for item in value]
    return value


def _python_assertion(value: Any, output: str, context: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("file://"):
        reference = value[len("file://") :]
        path_text, _, function_name = reference.partition(":")
        path = Path(path_text)
        loaded = importlib.util.spec_from_file_location(f"runeconomy_promptfoo_{path.stem}", path)
        if loaded is None or loaded.loader is None:
            raise _infrastructure("cannot load the Promptfoo python assertion file")
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
        function = getattr(module, function_name or "get_assert", None)
        if not callable(function):
            raise _infrastructure("the Promptfoo python assertion defines no get_assert")
        return function(output, context)
    # Inline Promptfoo python assertions are expressions over `output` and
    # `context`, exactly as Promptfoo evaluates them. Specs containing python
    # assertions are executable evaluators (see is_executable), so they run
    # only with --allow-exec-evaluators, the same trust as `python` and
    # `command` evaluators. Never accept them from untrusted manifests.
    return eval(str(value), {"__builtins__": __builtins__}, {"output": output, "context": context})  # noqa: S307


def _promptfoo_result(raw: Any, threshold: float | None) -> tuple[bool, float, str]:
    if isinstance(raw, bool):
        return raw, float(raw), ""
    if isinstance(raw, (int, float)):
        # Promptfoo: a numeric result passes when it meets the assertion's
        # threshold, or is positive when no threshold is declared.
        passed = raw >= float(threshold) if threshold is not None else raw > 0
        return passed, float(raw), ""
    if isinstance(raw, Mapping):
        passed = raw.get("pass")
        score = raw.get("score", float(bool(passed)))
        if not isinstance(passed, bool):
            raise _infrastructure("a Promptfoo GradingResult needs a boolean pass")
        return passed, float(score), str(raw.get("reason", ""))[:300]
    raise _infrastructure("a Promptfoo assertion returned an unsupported value")


def _promptfoo_check(item: Mapping[str, Any], output: str, variables: Mapping[str, Any], context: dict[str, Any]) -> tuple[bool, float, str]:
    kind = item["type"]
    base = _base_type(kind)
    value = _render(item.get("value"), variables)
    threshold = item.get("threshold")
    if base == "python":
        try:
            passed, score, reason = _promptfoo_result(_python_assertion(value, output, context), threshold)
        except Exception as exc:
            if isinstance(exc, _errors()[0]):
                raise
            raise _infrastructure(f"Promptfoo python assertion raised {type(exc).__name__}") from exc
    else:
        lowered = output.lower()
        values = value if isinstance(value, list) else [value]
        if base == "equals":
            passed = output == str(value)
        elif base == "contains":
            passed = str(value) in output
        elif base == "icontains":
            passed = str(value).lower() in lowered
        elif base == "contains-any":
            passed = any(str(item) in output for item in values)
        elif base == "contains-all":
            passed = all(str(item) in output for item in values)
        elif base == "icontains-any":
            passed = any(str(item).lower() in lowered for item in values)
        elif base == "icontains-all":
            passed = all(str(item).lower() in lowered for item in values)
        elif base == "regex":
            passed = re.search(str(value), output) is not None
        elif base == "starts-with":
            passed = output.startswith(str(value))
        elif base == "is-json":
            try:
                document = json.loads(output)
                passed = True
                if isinstance(value, Mapping):
                    import jsonschema

                    jsonschema.validate(document, value)
            except (json.JSONDecodeError, Exception):  # noqa: BLE001 - schema mismatch fails the assertion
                passed = False
        else:  # contains-json
            passed = any(_parses(candidate) for candidate in re.findall(r"\{.*\}|\[.*\]", output, re.DOTALL))
        score, reason = float(passed), ""
    if kind.startswith("not-"):
        passed, score = not passed, 1.0 - score
    return passed, score, reason


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _promptfoo(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    task = _task(context)
    variables = task.get(spec["vars_field"]) if spec.get("vars_field") else task
    variables = variables if isinstance(variables, Mapping) else {}
    promptfoo_context = {"vars": dict(variables), "test": {"vars": dict(variables)}, "prompt": task.get("prompt")}
    results = []
    weighted = total = 0.0
    for item in spec["assert"]:
        passed, score, reason = _promptfoo_check(item, output_text, variables, promptfoo_context)
        weight = float(item.get("weight", 1.0))
        weighted += weight * score
        total += weight
        results.append({"type": item["type"], "pass": passed, "score": score, **({"reason": reason} if reason else {})})
    success = all(result["pass"] for result in results)
    return success, (weighted / total if total else float(success)), {"assertions": results}


# -------------------------------------------------------------- Braintrust


def _braintrust(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    module_name, function_name = spec["scorer"].split(":", 1)
    try:
        scorer = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise _infrastructure(f"cannot import Braintrust scorer {spec['scorer']!r}") from exc
    task = _task(context)
    arguments = {
        "input": task.get(spec.get("input_field", "input")),
        "output": output_text,
        "expected": task.get(spec.get("expected_field", "expected")),
    }
    try:
        result = scorer(**arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(result)  # type: ignore[arg-type]
    except Exception as exc:
        raise _infrastructure(f"Braintrust scorer raised {type(exc).__name__}") from exc
    if isinstance(result, list):
        scores = [_score_value(item) for item in result]
        score = None if any(value is None for value in scores) else min(scores)  # type: ignore[type-var]
        name = "+".join(str(getattr(item, "name", "score")) for item in result)
    else:
        score = _score_value(result)
        declared = result.get("name") if isinstance(result, Mapping) else getattr(result, "name", None)
        name = str(declared or function_name)
    if score is None:
        raise _infrastructure("the Braintrust scorer abstained (score None); the attempt is unevaluable")
    threshold = float(spec.get("threshold", 0.5))
    return score >= threshold, float(score), {"scorer": name, "threshold": threshold}


def _score_value(result: Any) -> float | None:
    if isinstance(result, bool):
        return float(result)
    if isinstance(result, (int, float)):
        return float(result)
    value = result.get("score") if isinstance(result, Mapping) else getattr(result, "score", None)
    return None if value is None else float(value)


# -------------------------------------------------------------- Inspect AI


def _inspect_ai(spec: Mapping[str, Any], output_text: str, context: Mapping[str, Any]) -> tuple[bool, float, dict[str, Any]]:
    try:
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName, ModelOutput
        from inspect_ai.scorer import Target
        from inspect_ai.solver import TaskState
    except ImportError as exc:
        raise _infrastructure("the inspect_ai evaluator needs Inspect AI: pip install inspect-ai") from exc
    module_name, factory_name = spec["scorer"].split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
    except (ImportError, AttributeError) as exc:
        raise _infrastructure(f"cannot import Inspect AI scorer {spec['scorer']!r}") from exc
    task = _task(context)
    prompt = task.get(spec.get("input_field", "input"))
    raw_target = task.get(spec.get("target_field", "target"))
    if raw_target is None:
        raise _infrastructure(f"task field {spec.get('target_field', 'target')!r} is missing; the attempt is unevaluable")
    target = [str(item) for item in raw_target] if isinstance(raw_target, list) else str(raw_target)
    model = "runeconomy/attempt"
    try:
        scorer = factory(**dict(spec.get("scorer_args", {})))
        state = TaskState(
            model=ModelName(model),
            sample_id=str(context.get("task_id") or task.get("task_id") or "attempt"),
            epoch=1,
            input="" if prompt is None else str(prompt),
            messages=[ChatMessageUser(content="" if prompt is None else str(prompt)), ChatMessageAssistant(content=output_text)],
            target=Target(target),
            output=ModelOutput.from_content(model, output_text),
            completed=True,
            metadata=dict(task.get("metadata", {})) if isinstance(task.get("metadata"), Mapping) else {},
        )
        result = scorer(state, Target(target))
        if inspect.isawaitable(result):
            result = asyncio.run(result)  # type: ignore[arg-type]
    except Exception as exc:
        raise _infrastructure(f"Inspect AI scorer raised {type(exc).__name__}") from exc
    if result is None:
        raise _infrastructure("the Inspect AI scorer returned no score; the attempt is unevaluable")
    value = getattr(result, "value", result)
    if isinstance(value, Mapping):
        key = spec.get("value_key")
        if key is None or key not in value:
            raise _infrastructure("the Inspect AI score is a dict; declare value_key to select one entry")
        value = value[key]
    value_map = {str(k): float(v) for k, v in dict(spec.get("value_map", INSPECT_VALUE_MAP)).items()}
    if isinstance(value, bool):
        score = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        score = float(value)
    elif isinstance(value, str) and value in value_map:
        score = value_map[value]
    else:
        raise _infrastructure(f"the Inspect AI score value {value!r} does not map to a number; extend value_map")
    threshold = float(spec.get("threshold", 1.0))
    details: dict[str, Any] = {"scorer": spec["scorer"], "value": value, "threshold": threshold}
    answer = getattr(result, "answer", None)
    if answer is not None:
        details["answer"] = str(answer)[:200]
    return score >= threshold, score, details
