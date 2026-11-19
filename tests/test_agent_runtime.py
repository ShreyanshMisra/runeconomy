from __future__ import annotations

import json
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.agent import (
    Attempt,
    AttemptEnvironmentError,
    AttemptLifecycleError,
)
from runeconomy.attempt_status import (
    AttemptStatus,
    AttemptStatusError,
    write_attempt_status,
)
from runeconomy.comparison import run_comparison


SENSITIVE = "https://private.invalid/?token=do-not-print"


def _environment(directory: Path) -> dict[str, str]:
    task = directory / "task.json"
    task.write_text('{"task_id":"task-1"}', encoding="utf-8")
    return {
        "RUNECONOMY_TASK_PATH": str(task),
        "RUNECONOMY_TASK_ID": "task-1",
        "RUNECONOMY_VARIANT": "baseline",
        "RUNECONOMY_REPETITION": "2",
        "RUNECONOMY_SEED": "17",
        "RUNECONOMY_ATTEMPT_DIR": str(directory),
        "RUNECONOMY_OUTPUT_PATH": str(directory / "output.txt"),
        "RUNECONOMY_TRACE_PATH": str(directory / "trace.json"),
        "RUNECONOMY_STATUS_PATH": str(directory / "status.json"),
    }


class AgentRuntimeTests(unittest.TestCase):
    def test_reads_typed_environment_and_completes_once(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment = _environment(directory)
            with Attempt.from_environment(environment) as attempt:
                self.assertEqual(attempt.task_path, directory / "task.json")
                self.assertEqual(attempt.output_path, directory / "output.txt")
                self.assertEqual(attempt.trace_path, directory / "trace.json")
                self.assertEqual(attempt.attempt_dir, directory)
                self.assertEqual(attempt.task_id, "task-1")
                self.assertEqual(attempt.variant, "baseline")
                self.assertEqual(attempt.repetition, 2)
                self.assertEqual(attempt.seed, 17)
                attempt.completed()
                with self.assertRaises(AttemptLifecycleError):
                    attempt.completed()
            status = AttemptStatus.from_path(directory / "status.json")
            self.assertEqual(status.status, "completed")

    def test_every_supported_failure_method_uses_fixed_component(self) -> None:
        methods = {
            "task_failure": ("task_failure", "agent"),
            "model_output_failure": ("model_output_failure", "model_output"),
            "application_failure": ("application_failure", "application"),
            "infrastructure_failure": (
                "infrastructure_failure",
                "infrastructure",
            ),
        }
        for method, (expected_status, component) in methods.items():
            with self.subTest(method=method), TemporaryDirectory() as raw_directory:
                directory = Path(raw_directory)
                attempt = Attempt.from_environment(_environment(directory))
                getattr(attempt, method)(stage="planning", code="declared_failure")
                status = AttemptStatus.from_path(directory / "status.json")
                self.assertEqual(status.status, expected_status)
                assert status.provenance is not None
                self.assertEqual(status.provenance.component, component)

    def test_missing_malformed_and_outside_environment_values_fail(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            base = _environment(directory)
            cases = []
            missing = dict(base)
            missing.pop("RUNECONOMY_TASK_PATH")
            cases.append(missing)
            bad_integer = dict(base, RUNECONOMY_REPETITION="two")
            cases.append(bad_integer)
            bad_variant = dict(base, RUNECONOMY_VARIANT="control")
            cases.append(bad_variant)
            outside = dict(
                base, RUNECONOMY_STATUS_PATH=str(directory.parent / "status.json")
            )
            cases.append(outside)
            relative = dict(base, RUNECONOMY_TRACE_PATH="trace.json")
            cases.append(relative)
            collision = dict(
                base, RUNECONOMY_STATUS_PATH=base["RUNECONOMY_OUTPUT_PATH"]
            )
            cases.append(collision)
            for environment in cases:
                with self.subTest(environment=environment):
                    with self.assertRaises(AttemptEnvironmentError):
                        Attempt.from_environment(environment)

            negative_seed = dict(base, RUNECONOMY_SEED="-17")
            self.assertEqual(
                Attempt.from_environment(negative_seed).seed,
                -17,
            )

    def test_existing_status_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment = _environment(directory)
            status_path = directory / "status.json"
            status_path.write_text("unrelated", encoding="utf-8")
            with self.assertRaises(AttemptEnvironmentError):
                Attempt.from_environment(environment)
            self.assertEqual(status_path.read_text(encoding="utf-8"), "unrelated")

    def test_helper_refuses_a_status_created_after_initialization(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            attempt = Attempt.from_environment(_environment(directory))
            attempt.status_path.write_text("unrelated", encoding="utf-8")
            with self.assertRaises(AttemptLifecycleError):
                attempt.completed()
            self.assertEqual(
                attempt.status_path.read_text(encoding="utf-8"), "unrelated"
            )
            self.assertFalse(list(directory.glob(".*.record")))

    def test_unhandled_exception_records_conservative_status_and_reraises(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            with self.assertRaisesRegex(RuntimeError, "private failure"):
                with Attempt.from_environment(_environment(directory)):
                    raise RuntimeError(f"private failure {SENSITIVE}")
            serialized = (directory / "status.json").read_text(encoding="utf-8")
            self.assertNotIn("private failure", serialized)
            self.assertNotIn(SENSITIVE, serialized)
            status = AttemptStatus.from_path(directory / "status.json")
            self.assertEqual(status.status, "application_failure")
            assert status.provenance is not None
            self.assertEqual(status.provenance.code, "unhandled_exception")

    def test_clean_exit_without_terminal_status_records_failure_and_raises(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            with self.assertRaises(AttemptLifecycleError):
                with Attempt.from_environment(_environment(directory)):
                    pass
            status = AttemptStatus.from_path(directory / "status.json")
            self.assertEqual(status.status, "application_failure")
            assert status.provenance is not None
            self.assertEqual(
                status.provenance.code, "terminal_status_not_declared"
            )

    def test_free_form_provenance_is_rejected_without_writing(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            attempt = Attempt.from_environment(_environment(directory))
            with self.assertRaises(AttemptStatusError):
                attempt.application_failure(
                    stage="execution", code=f"exception {SENSITIVE}"
                )
            self.assertFalse((directory / "status.json").exists())

    def test_runner_accepts_helper_and_timeout_still_takes_precedence(self) -> None:
        agent_source = '''
import json
import os
import time
from pathlib import Path
from runeconomy.agent import Attempt

with Attempt.from_environment() as attempt:
    task = json.loads(attempt.task_path.read_text())
    if task.get("sleep"):
        attempt.application_failure(stage="execution", code="declared_before_timeout")
        time.sleep(1)
    attempt.output_path.write_text(task["expected"], encoding="utf-8")
    spans = [{
        "traceId": "trace-id", "spanId": "step-1",
        "startTimeUnixNano": "1000000000", "endTimeUnixNano": "1010000000",
        "attributes": [
            {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
            {"key": "llm.model_name", "value": {"stringValue": "fixture-model"}},
            {"key": "llm.model_revision", "value": {"stringValue": "revision"}},
            {"key": "llm.provider", "value": {"stringValue": "vllm"}},
            {"key": "llm.backend_version", "value": {"stringValue": "0.1"}},
            {"key": "runeconomy.tokenizer", "value": {"stringValue": "tokenizer"}},
            {"key": "runeconomy.tokenizer_revision", "value": {"stringValue": "revision"}},
            {"key": "runeconomy.chat_template_digest", "value": {"stringValue": "sha256:template"}},
            {"key": "runeconomy.depends_on", "value": {"stringValue": "[]"}}
        ]
    }]
    attempt.trace_path.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}))
    attempt.completed()
'''
        evaluator_source = '''
import json, sys
request = json.load(sys.stdin)
json.dump({"success": request["output_text"].strip() == request["task"]["expected"]}, sys.stdout)
'''
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            agent = directory / "agent.py"
            evaluator = directory / "evaluator.py"
            dataset = directory / "tasks.json"
            agent.write_text(textwrap.dedent(agent_source), encoding="utf-8")
            evaluator.write_text(textwrap.dedent(evaluator_source), encoding="utf-8")
            dataset.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"task_id": "normal", "expected": "ok"},
                            {"task_id": "timeout", "expected": "ok", "sleep": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = run_comparison(
                baseline_command=f"{sys.executable} {agent}",
                candidate_command=f"{sys.executable} {agent}",
                dataset_path=dataset,
                evaluator_path=evaluator,
                output_path=directory / "result.json",
                artifacts_dir=directory / "artifacts",
                workdir=directory,
                repetitions=1,
                order="alternating",
                timeout_seconds=0.2,
                min_task_success_rate=0.0,
            )
            statuses = [attempt["status"] for attempt in result["attempts"]]
            self.assertEqual(statuses.count("completed"), 2)
            self.assertEqual(statuses.count("timeout"), 2)
            self.assertNotIn(SENSITIVE, json.dumps(result))

    def test_direct_writer_and_legacy_agent_contracts_remain_available(self) -> None:
        with TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "status.json"
            write_attempt_status(path, "completed")
            self.assertEqual(AttemptStatus.from_path(path).status, "completed")
        self.assertTrue(callable(write_attempt_status))


if __name__ == "__main__":
    unittest.main()
