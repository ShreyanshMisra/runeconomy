from __future__ import annotations

import sys
import unittest

from runeconomy.evaluators import (
    EvaluatorError,
    EvaluatorInfrastructureError,
    evaluate_output,
    validate_evaluator_spec,
)


def local_rubric(output: str, context: dict) -> dict:
    success = output.strip() == context["expected"]
    return {"success": success, "score": 1.0 if success else 0.0}


class EvaluatorTests(unittest.TestCase):
    def test_exact_and_regex(self) -> None:
        exact = evaluate_output(
            {"kind": "exact", "expected": "READY", "case_sensitive": False},
            " ready\n",
            {},
        )
        regex = evaluate_output(
            {
                "kind": "regex",
                "pattern": r"\s*check_deployments[.!]?\s*",
                "flags": ["IGNORECASE"],
            },
            "CHECK_DEPLOYMENTS.",
            {},
        )
        self.assertTrue(exact.success)
        self.assertTrue(regex.success)

    def test_json_schema(self) -> None:
        spec = {
            "kind": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"action": {"const": "rollback"}},
                "required": ["action"],
                "additionalProperties": False,
            },
        }
        self.assertTrue(evaluate_output(spec, '{"action":"rollback"}', {}).success)
        failed = evaluate_output(spec, '{"action":"wait"}', {})
        self.assertFalse(failed.success)
        self.assertEqual(failed.details["reason"], "schema_mismatch")

    def test_python_and_command(self) -> None:
        python_result = evaluate_output(
            {
                "kind": "python",
                "callable": "tests.test_evaluators:local_rubric",
            },
            "PASS",
            {"expected": "PASS"},
        )
        command_result = evaluate_output(
            {
                "kind": "command",
                "argv": [
                    sys.executable,
                    "-c",
                    "import sys; raise SystemExit(sys.stdin.read().strip() != 'PASS')",
                ],
            },
            "PASS",
            {},
        )
        self.assertTrue(python_result.success)
        self.assertTrue(command_result.success)

    def test_all_and_missing_output(self) -> None:
        result = evaluate_output(
            {
                "kind": "all",
                "evaluators": [
                    {"kind": "exact", "expected": "PASS"},
                    {"kind": "regex", "pattern": "PASS"},
                ],
            },
            None,
            {},
        )
        self.assertFalse(result.success)
        self.assertEqual(result.details["reason"], "no_output_text")

    def test_invalid_spec_is_rejected(self) -> None:
        with self.assertRaises(EvaluatorError):
            validate_evaluator_spec({"kind": "regex", "pattern": "["})

    def test_missing_python_module_is_an_infrastructure_error(self) -> None:
        with self.assertRaisesRegex(EvaluatorInfrastructureError, "cannot import"):
            evaluate_output(
                {"kind": "python", "callable": "not_a_real_module:check"},
                "PASS",
                {},
            )

    def test_missing_command_and_timeout_are_infrastructure_errors(self) -> None:
        with self.assertRaisesRegex(EvaluatorInfrastructureError, "cannot run"):
            evaluate_output(
                {"kind": "command", "argv": ["not-a-real-runeconomy-command"]},
                "PASS",
                {},
            )
        with self.assertRaisesRegex(EvaluatorInfrastructureError, "timed out"):
            evaluate_output(
                {
                    "kind": "command",
                    "argv": [sys.executable, "-c", "import time; time.sleep(1)"],
                    "timeout_seconds": 0.01,
                },
                "PASS",
                {},
            )


if __name__ == "__main__":
    unittest.main()
