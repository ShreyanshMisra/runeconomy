from __future__ import annotations

import hashlib
import json
import sys
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from runeconomy.comparison import run_comparison
from runeconomy.evaluator_kinds import calibrate_judge, cohen_kappa
from runeconomy.evaluators import (
    EvaluatorError,
    EvaluatorInfrastructureError,
    evaluate_output,
    is_executable_evaluator,
    validate_evaluator_spec,
)


FIXTURES = Path(__file__).parent / "fixtures" / "evaluators"


def _trace(path: Path, tools: list[str]) -> None:
    spans = [
        {
            "name": name,
            "startTimeUnixNano": str(1000 + index),
            "attributes": [{"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}}],
        }
        for index, name in enumerate(tools)
    ]
    spans.append({"name": "llm", "startTimeUnixNano": "1", "attributes": [
        {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}]})
    path.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}), encoding="utf-8")


class PytestEvaluatorTests(unittest.TestCase):
    def _repo(self, root: Path, body: str) -> None:
        (root / "repo" / "tests").mkdir(parents=True)
        (root / "repo" / "tests" / "test_x.py").write_text(textwrap.dedent(body), encoding="utf-8")

    def test_passing_failing_and_unusable_suites(self) -> None:
        spec = {"kind": "pytest", "workdir": "repo"}
        with TemporaryDirectory() as raw:
            root = Path(raw)
            self._repo(root, "def test_a():\n    assert True\n\ndef test_b():\n    assert 1 == 1\n")
            result = evaluate_output(spec, "", {"attempt_dir": str(root)})
            self.assertTrue(result.success)
            self.assertEqual(result.details["tests"], 2)
        with TemporaryDirectory() as raw:
            root = Path(raw)
            self._repo(root, "def test_a():\n    assert True\n\ndef test_b():\n    assert False\n")
            result = evaluate_output(spec, "", {"attempt_dir": str(root)})
            self.assertFalse(result.success)
            self.assertEqual(result.score, 0.5)
        with TemporaryDirectory() as raw:
            root = Path(raw)
            self._repo(root, "x = 1\n")
            with self.assertRaises(EvaluatorInfrastructureError):
                evaluate_output(spec, "", {"attempt_dir": str(root)})
        with self.assertRaises(EvaluatorInfrastructureError):
            evaluate_output(spec, "", {})
        self.assertTrue(is_executable_evaluator(spec))


class TrajectoryEvaluatorTests(unittest.TestCase):
    def test_trace_matching_modes_and_forbidden_tools(self) -> None:
        with TemporaryDirectory() as raw:
            trace = Path(raw) / "trace.json"
            _trace(trace, ["search", "read", "search", "answer"])
            context = {"trace_path": str(trace), "task": {"tools": ["search", "answer"]}}
            cases = [
                ({"expected": ["search", "read", "search", "answer"], "match": "exact"}, True),
                ({"expected": ["search", "answer"], "match": "in_order"}, True),
                ({"expected": ["answer", "search"], "match": "in_order"}, False),
                ({"expected_field": "tools", "match": "superset"}, True),
                ({"expected": ["search", "answer"], "match": "superset", "forbidden": ["read"]}, False),
                ({"expected": ["search", "answer"], "match": "superset", "max_calls": 3}, False),
            ]
            for extra, expected in cases:
                with self.subTest(extra=extra):
                    spec = {"kind": "tool_trajectory", **extra}
                    self.assertEqual(evaluate_output(spec, "", context).success, expected)
        self.assertFalse(is_executable_evaluator({"kind": "tool_trajectory", "expected": []}))

    def test_output_json_source_and_validation(self) -> None:
        spec = {"kind": "tool_trajectory", "source": "output_json", "expected": ["lookup"], "match": "exact"}
        output = json.dumps({"tool_calls": [{"function": {"name": "lookup"}}]})
        self.assertTrue(evaluate_output(spec, output, {}).success)
        for bad in (
            {"kind": "tool_trajectory"},
            {"kind": "tool_trajectory", "expected": ["a"], "match": "fuzzy"},
        ):
            with self.assertRaises(EvaluatorError):
                validate_evaluator_spec(bad)


class _Judge(BaseHTTPRequestHandler):
    model = "judge-model-2026-01"

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][1]["content"]
        score = 0.9 if "Response to grade:\ngood" in text else 0.1
        payload = {
            "model": type(self).model,
            "system_fingerprint": "fp_x",
            "choices": [{"message": {"content": json.dumps({"score": score, "reason": "graded"})}}],
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        pass


class LLMJudgeTests(unittest.TestCase):
    def setUp(self) -> None:
        _Judge.model = "judge-model-2026-01"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Judge)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.environment = {"JUDGE_URL": f"http://127.0.0.1:{self.server.server_port}/v1"}

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _spec(self, calibration: Path, digest: str, min_kappa: float = 0.6) -> dict[str, Any]:
        return {
            "kind": "llm_judge",
            "rubric": "Score 1 when the answer is correct and complete.",
            "judge": {"model": "judge-model-2026-01", "base_url_env": "JUDGE_URL", "provider": "test"},
            "pass_threshold": 0.5,
            "calibration": {"path": str(calibration), "sha256": digest, "min_kappa": min_kappa},
        }

    def test_calibrated_judge_grades_and_records_identity(self) -> None:
        with TemporaryDirectory() as raw, patch.dict("os.environ", self.environment):
            path = Path(raw) / "calibration.json"
            spec = self._spec(path, "sha256:" + "0" * 64)
            labels = [{"output_text": "good" if i % 2 else "bad", "human_pass": bool(i % 2)} for i in range(20)]
            labels[0]["human_pass"] = True  # one human/judge disagreement
            record = calibrate_judge(spec, labels)
            self.assertEqual(record["n"], 20)
            self.assertAlmostEqual(record["raw_agreement"], 0.95)
            self.assertGreater(record["cohen_kappa"], 0.85)
            raw_record = json.dumps(record).encode()
            path.write_bytes(raw_record)
            spec = self._spec(path, "sha256:" + hashlib.sha256(raw_record).hexdigest())
            good = evaluate_output(spec, "good", {"task": {"input": "q"}})
            bad = evaluate_output(spec, "bad", {"task": {"input": "q"}})
            self.assertTrue(good.success)
            self.assertFalse(bad.success)
            self.assertEqual(good.details["judge_model_reported"], "judge-model-2026-01")
            self.assertTrue(good.details["judge_fingerprint_sha256"].startswith("sha256:"))
            self.assertNotIn("fp_x", json.dumps(good.details))

            strict = self._spec(path, spec["calibration"]["sha256"], min_kappa=0.99)
            with self.assertRaises(EvaluatorInfrastructureError):
                evaluate_output(strict, "good", {})
            with self.assertRaises(EvaluatorInfrastructureError):
                evaluate_output(self._spec(path, "sha256:" + "1" * 64), "good", {})
            _Judge.model = "a-different-model"
            with self.assertRaises(EvaluatorInfrastructureError):
                evaluate_output(spec, "good", {})

    def test_uncalibrated_or_pairwise_judges_are_rejected(self) -> None:
        with self.assertRaises(EvaluatorError):
            validate_evaluator_spec({"kind": "llm_judge", "rubric": "r", "judge": {"model": "m", "base_url_env": "X"}})
        with self.assertRaises(EvaluatorError):
            validate_evaluator_spec(
                {"kind": "llm_judge", "mode": "pairwise", "rubric": "r", "judge": {"model": "m", "base_url_env": "X"},
                 "calibration": {"path": "c", "sha256": "sha256:x"}}
            )
        self.assertIsNone(cohen_kappa([True, True], [True, True]))
        self.assertEqual(cohen_kappa([True, False], [True, False]), 1.0)


class ExternalFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(FIXTURES))

    def tearDown(self) -> None:
        sys.path.remove(str(FIXTURES))

    def test_promptfoo_assertions_run_unchanged(self) -> None:
        spec = {
            "kind": "promptfoo",
            "assert": [
                {"type": "icontains", "value": "{{expected}}"},
                {"type": "not-contains", "value": "error"},
                {"type": "regex", "value": "^[A-Za-z_]+$"},
                {"type": "python", "value": f"file://{FIXTURES / 'promptfoo_assert.py'}"},
                {"type": "python", "value": "len(output) < 40"},
            ],
        }
        context = {"task": {"expected": "rollback_release"}}
        self.assertTrue(evaluate_output(spec, "rollback_release", context).success)
        failed = evaluate_output(spec, "rollback_other", context)
        self.assertFalse(failed.success)
        self.assertEqual([item["pass"] for item in failed.details["assertions"]], [False, True, True, False, True])
        self.assertTrue(is_executable_evaluator(spec))
        self.assertFalse(is_executable_evaluator({"kind": "promptfoo", "assert": [{"type": "equals", "value": "x"}]}))
        json_spec = {"kind": "promptfoo", "assert": [{"type": "is-json", "value": {"type": "object", "required": ["a"]}}]}
        self.assertTrue(evaluate_output(json_spec, '{"a": 1}', {}).success)
        self.assertFalse(evaluate_output(json_spec, '{"b": 1}', {}).success)
        with self.assertRaises(EvaluatorError):
            validate_evaluator_spec({"kind": "promptfoo", "assert": [{"type": "llm-rubric", "value": "x"}]})

    def test_braintrust_scorers_run_unchanged(self) -> None:
        context = {"task": {"input": "q", "expected": "rollback"}}
        exact = evaluate_output({"kind": "braintrust", "scorer": "braintrust_scorers:ExactMatch"}, "rollback", context)
        self.assertTrue(exact.success)
        self.assertEqual(exact.details["scorer"], "ExactMatch")
        self.assertTrue(evaluate_output({"kind": "braintrust", "scorer": "braintrust_scorers:AsyncPrefix"}, "ro", context).success)
        number = evaluate_output({"kind": "braintrust", "scorer": "braintrust_scorers:plain_number", "threshold": 0.8}, "x", context)
        self.assertFalse(number.success)
        with self.assertRaises(EvaluatorInfrastructureError):
            evaluate_output({"kind": "braintrust", "scorer": "braintrust_scorers:Abstains"}, "x", context)

    @unittest.skipUnless(__import__("importlib").util.find_spec("autoevals"), "autoevals is not installed")
    def test_real_autoevals_scorer(self) -> None:  # pragma: no cover - optional dependency
        result = evaluate_output({"kind": "braintrust", "scorer": "autoevals:ExactMatch"}, "a", {"task": {"expected": "a"}})
        self.assertTrue(result.success)

    def test_inspect_ai_spec_validation(self) -> None:
        validate_evaluator_spec({"kind": "inspect_ai", "scorer": "inspect_ai.scorer:includes"})
        self.assertTrue(is_executable_evaluator({"kind": "inspect_ai", "scorer": "inspect_ai.scorer:includes"}))
        for bad in (
            {"kind": "inspect_ai", "scorer": "includes"},
            {"kind": "inspect_ai", "scorer": "m:f", "scorer_args": []},
            {"kind": "inspect_ai", "scorer": "m:f", "threshold": 2},
            {"kind": "inspect_ai", "scorer": "m:f", "value_map": {"C": 3}},
            {"kind": "inspect_ai", "scorer": "m:f", "target_field": ""},
        ):
            with self.assertRaises(EvaluatorError, msg=str(bad)):
                validate_evaluator_spec(bad)

    def test_inspect_ai_missing_package_is_unevaluable(self) -> None:
        with patch.dict(sys.modules, {"inspect_ai": None, "inspect_ai.model": None}):
            with self.assertRaises(EvaluatorInfrastructureError):
                evaluate_output({"kind": "inspect_ai", "scorer": "inspect_ai.scorer:includes"}, "x", {"task": {"target": "x"}})

    @unittest.skipUnless(__import__("importlib").util.find_spec("inspect_ai"), "inspect_ai is not installed")
    def test_inspect_ai_scorers_run_unchanged(self) -> None:  # pragma: no cover - optional dependency
        context = {"task": {"input": "Which function rolls back?", "target": "rollback_release"}}
        builtin = {"kind": "inspect_ai", "scorer": "inspect_ai.scorer:includes"}
        passed = evaluate_output(builtin, "It is rollback_release.", context)
        self.assertTrue(passed.success)
        self.assertEqual(passed.details["value"], "C")
        self.assertFalse(evaluate_output(builtin, "deploy_release", context).success)
        self.assertTrue(evaluate_output({"kind": "inspect_ai", "scorer": "inspect_ai.scorer:match"}, "rollback_release", context).success)
        listed = {"task": {"target": ["a", "rollback_release"]}}
        self.assertTrue(evaluate_output({"kind": "inspect_ai", "scorer": "inspect_scorers:graded"}, "rollback_release", listed).success)
        partial = {"kind": "inspect_ai", "scorer": "inspect_scorers:graded"}
        self.assertFalse(evaluate_output(partial, "maybe", listed).success)
        lowered = evaluate_output({**partial, "threshold": 0.5}, "maybe", listed)
        self.assertTrue(lowered.success)
        self.assertEqual(lowered.score, 0.5)
        keyed = {"kind": "inspect_ai", "scorer": "inspect_scorers:by_key"}
        with self.assertRaises(EvaluatorInfrastructureError):
            evaluate_output(keyed, "rollback_release", context)
        self.assertTrue(evaluate_output({**keyed, "value_key": "exact"}, "rollback_release", context).success)
        with self.assertRaises(EvaluatorInfrastructureError):
            evaluate_output({"kind": "inspect_ai", "scorer": "inspect_scorers:explodes"}, "x", context)
        with self.assertRaises(EvaluatorInfrastructureError):
            evaluate_output(builtin, "x", {"task": {"input": "q"}})
        with self.assertRaises(EvaluatorInfrastructureError):
            evaluate_output({"kind": "inspect_ai", "scorer": "inspect_scorers:missing"}, "x", context)

    def test_external_evaluators_gate_an_agent_process_run(self) -> None:
        from tests.test_comparison import AGENT

        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
            (root / "tasks.json").write_text(
                json.dumps({"tasks": [
                    {"task_id": "a", "expected": "alpha"},
                    {"task_id": "b", "expected": "beta", "candidate_behavior": "task_failure"},
                ]}),
                encoding="utf-8",
            )
            for name, spec in (
                ("promptfoo.json", {"kind": "promptfoo", "assert": [
                    {"type": "python", "value": f"file://{FIXTURES / 'promptfoo_assert.py'}"}]}),
                ("braintrust.json", {"kind": "braintrust", "scorer": "braintrust_scorers:ExactMatch"}),
            ):
                (root / name).write_text(json.dumps(spec), encoding="utf-8")
                with self.subTest(format=name):
                    result = run_comparison(
                        baseline_command=f"{sys.executable} {root / 'agent.py'}",
                        candidate_command=f"{sys.executable} {root / 'agent.py'}",
                        dataset_path=root / "tasks.json",
                        evaluator_path=root / name,
                        output_path=root / f"{name}.result.json",
                        artifacts_dir=root / f"{name}-artifacts",
                        workdir=root,
                        order="alternating",
                        min_task_success_rate=0.0,
                    )
                    # Task IDs are stored as digests; compare outcomes per variant.
                    outcomes: dict[str, list[bool]] = {}
                    for item in result["attempts"]:
                        outcomes.setdefault(item["variant"], []).append(item["task_success"])
                    self.assertEqual(
                        {variant: sorted(values) for variant, values in outcomes.items()},
                        {"baseline": [True, True], "candidate": [False, True]},
                    )
                    self.assertTrue(all(item["status"] in {"completed", "task_failure"} for item in result["attempts"]))


if __name__ == "__main__":
    unittest.main()
