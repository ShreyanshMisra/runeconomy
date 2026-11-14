from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.comparison import (
    _decision,
    _summarize,
    _workdir_identity,
    run_comparison,
)


AGENT = r'''
import json
import os
import sys
import time
from pathlib import Path

from runeconomy.attempt_status import write_attempt_status

task = json.loads(Path(os.environ["RUNECONOMY_TASK_PATH"]).read_text())
variant = os.environ["RUNECONOMY_VARIANT"]
attempt = Path(os.environ["RUNECONOMY_ATTEMPT_DIR"])
behavior = task.get("candidate_behavior") if variant == "candidate" else None
if behavior == "infrastructure_failure":
    raise SystemExit(7)
if behavior == "timeout":
    time.sleep(1)

# A real stateful two-step workflow: step 2 reads step 1's persisted output.
state = attempt / "step-1-state.txt"
state.write_text(task["expected"], encoding="utf-8")
step_2_input = state.read_text(encoding="utf-8")
output = "wrong" if behavior == "task_failure" else step_2_input
Path(os.environ["RUNECONOMY_OUTPUT_PATH"]).write_text(output, encoding="utf-8")
if behavior == "invalid_trace":
    raise SystemExit(0)

def attr(key, value):
    return {"key": key, "value": {"stringValue": value}}

trace_id = "trace-" + os.environ["RUNECONOMY_TASK_ID"] + "-" + variant
model_name = "fixture/drifted" if behavior == "identity_drift" else "fixture/model"
spans = [
    {
        "traceId": trace_id,
        "spanId": "step-1",
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "1010000000",
        "attributes": [
            attr("openinference.span.kind", "LLM"),
            attr("llm.model_name", model_name),
            attr("llm.model_revision", "model-revision"),
            attr("llm.provider", "vllm"),
            attr("llm.backend_version", "0.1"),
            attr("runeconomy.tokenizer", "fixture/tokenizer"),
            attr("runeconomy.tokenizer_revision", "tokenizer-revision"),
            attr("runeconomy.chat_template_digest", "sha256:template"),
            attr("runeconomy.depends_on", "[]"),
            {"key": "llm.token_count.prompt", "value": {"intValue": 10}},
        ],
    },
    {
        "traceId": trace_id,
        "spanId": "step-2",
        "startTimeUnixNano": "1010000000",
        "endTimeUnixNano": "1030000000",
        "attributes": [
            attr("openinference.span.kind", "LLM"),
            attr("llm.model_name", model_name),
            attr("llm.model_revision", "model-revision"),
            attr("llm.provider", "vllm"),
            attr("llm.backend_version", "0.1"),
            attr("runeconomy.tokenizer", "fixture/tokenizer"),
            attr("runeconomy.tokenizer_revision", "tokenizer-revision"),
            attr("runeconomy.chat_template_digest", "sha256:template"),
            attr("runeconomy.depends_on", '["step-1"]'),
            {"key": "llm.token_count.prompt", "value": {"intValue": 20}},
        ],
    },
]
Path(os.environ["RUNECONOMY_TRACE_PATH"]).write_text(
    json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}),
    encoding="utf-8",
)
if behavior == "model_output_failure":
    write_attempt_status(
        os.environ["RUNECONOMY_STATUS_PATH"],
        "model_output_failure",
        component="model_output",
        stage="planning",
        code="invalid_structured_output",
    )
    raise SystemExit(9)
if behavior == "application_failure":
    write_attempt_status(
        os.environ["RUNECONOMY_STATUS_PATH"],
        "application_failure",
        component="application",
        stage="tool_dispatch",
        code="tool_contract_failed",
    )
    raise SystemExit(8)
if behavior == "invalid_status":
    Path(os.environ["RUNECONOMY_STATUS_PATH"]).write_text(
        json.dumps({"schema_version": "0.1", "status": "completed", "secret": output}),
        encoding="utf-8",
    )
    raise SystemExit(6)
write_attempt_status(os.environ["RUNECONOMY_STATUS_PATH"], "completed")
'''


EVALUATOR = r'''
import json
import sys

request = json.load(sys.stdin)
success = request["output_text"].strip() == request["task"]["expected"]
json.dump({"success": success, "score": float(success), "details": {}}, sys.stdout)
'''


class ComparisonTests(unittest.TestCase):
    def _files(self, directory: Path, tasks: list[dict]) -> tuple[Path, Path, Path]:
        agent = directory / "agent.py"
        evaluator = directory / "evaluator.py"
        dataset = directory / "tasks.json"
        agent.write_text(textwrap.dedent(AGENT), encoding="utf-8")
        evaluator.write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
        dataset.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
        return agent, evaluator, dataset

    def test_real_process_comparison_preserves_variant_local_state(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            agent, evaluator, dataset = self._files(
                directory,
                [
                    {"task_id": "one", "expected": "alpha"},
                    {"task_id": "../../two", "expected": "beta"},
                ],
            )
            artifacts = directory / "artifacts"
            result = run_comparison(
                baseline_command=f"{sys.executable} {agent}",
                candidate_command=f"{sys.executable} {agent}",
                dataset_path=dataset,
                evaluator_path=evaluator,
                output_path=directory / "result.json",
                artifacts_dir=artifacts,
                workdir=directory,
                repetitions=1,
                order="alternating",
                seed=17,
                min_task_success_rate=0.0,
            )

            self.assertEqual(len(result["attempts"]), 4)
            self.assertTrue(
                all(item["status"] == "completed" for item in result["attempts"])
            )
            self.assertTrue(
                all(
                    item["metrics"]["dependency_evidence"] == "complete_explicit"
                    for item in result["attempts"]
                )
            )
            self.assertTrue(
                all(item["metrics"]["critical_path_ms"] == 30.0 for item in result["attempts"])
            )
            for attempt in result["attempts"]:
                attempt_directory = (
                    artifacts / attempt["artifacts"]["directory"]
                ).resolve()
                self.assertTrue(attempt_directory.is_relative_to(artifacts.resolve()))
                state = attempt_directory / "step-1-state.txt"
                self.assertTrue(state.is_file())
            self.assertEqual(
                result["summary"]["effects"]["workflow_latency_ms"]["paired_count"],
                2,
            )
            serialized = json.dumps(result)
            self.assertNotIn(raw_directory, serialized)
            self.assertNotIn("../../two", serialized)
            self.assertEqual(result["provenance"]["artifacts_directory"], "artifacts")
            self.assertEqual(result["provenance"]["result_path"], "result.json")
            self.assertIn(result["decision"]["status"], {"inconclusive", "verified"})

    def test_failure_classes_are_not_pooled_into_latency(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            agent, evaluator, dataset = self._files(
                directory,
                [
                    {
                        "task_id": "task-failure",
                        "expected": "ok",
                        "candidate_behavior": "task_failure",
                    },
                    {
                        "task_id": "bad-trace",
                        "expected": "ok",
                        "candidate_behavior": "invalid_trace",
                    },
                    {
                        "task_id": "process-error",
                        "expected": "ok",
                        "candidate_behavior": "infrastructure_failure",
                    },
                    {
                        "task_id": "slow",
                        "expected": "ok",
                        "candidate_behavior": "timeout",
                    },
                ],
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
                timeout_seconds=0.1,
                min_task_success_rate=0.0,
            )

            candidate = result["summary"]["variants"]["candidate"]
            self.assertEqual(
                candidate["status_counts"],
                {
                    "infrastructure_failure": 1,
                    "invalid_trace": 1,
                    "task_failure": 1,
                    "timeout": 1,
                },
            )
            self.assertEqual(
                result["summary"]["effects"]["workflow_latency_ms"]["paired_count"],
                0,
            )
            self.assertEqual(result["decision"]["status"], "invalid")
            for attempt in result["attempts"]:
                if attempt["failure_reason"] is not None:
                    self.assertNotIn(raw_directory, attempt["failure_reason"])
                    self.assertIsNotNone(attempt["failure_details_digest"])
                    self.assertIsNotNone(attempt["artifacts"]["failure"])

    def test_status_protocol_classifies_failures_and_preserves_trace(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            agent, evaluator, dataset = self._files(
                directory,
                [
                    {
                        "task_id": "model-output",
                        "expected": "ok",
                        "candidate_behavior": "model_output_failure",
                    },
                    {
                        "task_id": "application",
                        "expected": "ok",
                        "candidate_behavior": "application_failure",
                    },
                    {
                        "task_id": "malformed-status",
                        "expected": "do-not-publish",
                        "candidate_behavior": "invalid_status",
                    },
                ],
            )
            result = run_comparison(
                baseline_command=f"{sys.executable} {agent}",
                candidate_command=f"{sys.executable} {agent}",
                dataset_path=dataset,
                evaluator_path=evaluator,
                output_path=directory / "result.json",
                artifacts_dir=directory / "artifacts",
                workdir=directory,
                order="alternating",
                min_task_success_rate=0.0,
            )

            candidates = {
                item["status"]: item
                for item in result["attempts"]
                if item["variant"] == "candidate"
            }
            self.assertEqual(
                set(candidates),
                {"model_output_failure", "application_failure", "invalid_status"},
            )
            for name in ("model_output_failure", "application_failure"):
                attempt = candidates[name]
                self.assertFalse(attempt["task_success"])
                self.assertEqual(
                    attempt["metrics"]["dependency_evidence"],
                    "complete_explicit",
                )
                self.assertIsNotNone(attempt["trace_digest"])
                self.assertIsNotNone(attempt["agent_status"])
            serialized = json.dumps(result)
            self.assertNotIn("do-not-publish", serialized)
            self.assertNotIn("secret", serialized)
            self.assertEqual(result["decision"]["status"], "invalid")

    def test_partial_trace_request_count_is_not_identity_drift(self) -> None:
        identity = {
            "backend": "vllm",
            "backend_version": "0.33.0",
            "model": "fixture/model",
            "model_revision": "model-revision",
            "tokenizer": "fixture/tokenizer",
            "tokenizer_revision": "tokenizer-revision",
            "chat_template_digest": "sha256:template",
        }

        def attempt(variant: str, repetition: int, request_count: int) -> dict:
            return {
                "variant": variant,
                "task_id": "sha256:task",
                "repetition": repetition,
                "status": "completed",
                "task_success": True,
                "workflow_latency_ms": 10.0,
                "metrics": {
                    "dependency_evidence": "complete_explicit",
                    "environment_identity": [identity] * request_count,
                    "environment_identity_complete": True,
                },
            }

        summary = _summarize(
            [
                attempt("baseline", 0, 1),
                attempt("candidate", 0, 1),
                attempt("baseline", 1, 2),
                attempt("candidate", 1, 2),
            ]
        )
        for variant in ("baseline", "candidate"):
            evidence = summary["variants"][variant]["trace_evidence"]
            self.assertTrue(evidence["environment_identity_stable"])
            self.assertEqual(len(evidence["environment_identity_signatures"]), 1)

    def test_clear_successful_workflow_slowdown_is_a_regression(self) -> None:
        trace_evidence = {
            "complete_dag_rate": 1.0,
            "environment_identity_complete": True,
            "environment_identity_stable": True,
        }
        variant = {
            "task_success_95_ci": [0.9, 1.0],
            "task_success_rate": 1.0,
            "completed_rate": 1.0,
            "evaluated_count": 40,
            "status_counts": {"completed": 40},
            "trace_evidence": trace_evidence,
        }
        summary = {
            "variants": {"baseline": variant, "candidate": variant},
            "effects": {
                "task_success": {
                    "one_sided_lower_bound": -0.05,
                    "paired_count": 40,
                },
                "workflow_latency_ms": {
                    "paired_count": 40,
                    "candidate_minus_baseline_mean": 15.0,
                    "confidence_interval": [5.0, 25.0],
                },
            },
        }
        decision = _decision(
            summary,
            max_task_success_regression=0.1,
            min_task_success_rate=0.8,
            min_completed_rate=1.0,
        )

        self.assertEqual(decision["status"], "regression")
        self.assertIn("successful tasks were slower", decision["reason"])
        guardrail = next(
            check
            for check in decision["checks"]
            if check["name"]
            == "successful_task_workflow_latency_non_regression"
        )
        self.assertFalse(guardrail["passed"])

    def test_identity_drift_within_a_variant_invalidates_comparison(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            agent, evaluator, dataset = self._files(
                directory,
                [
                    {"task_id": "stable", "expected": "ok"},
                    {
                        "task_id": "drifted",
                        "expected": "ok",
                        "candidate_behavior": "identity_drift",
                    },
                ],
            )
            result = run_comparison(
                baseline_command=f"{sys.executable} {agent}",
                candidate_command=f"{sys.executable} {agent}",
                dataset_path=dataset,
                evaluator_path=evaluator,
                output_path=directory / "result.json",
                artifacts_dir=directory / "artifacts",
                workdir=directory,
                min_task_success_rate=0.0,
            )

            self.assertEqual(result["decision"]["status"], "invalid")
            candidate = result["summary"]["variants"]["candidate"]
            self.assertFalse(
                candidate["trace_evidence"]["environment_identity_stable"]
            )

    def test_workdir_identity_hashes_changed_content_not_only_filenames(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            subprocess.run(["git", "init", "-q"], cwd=directory, check=True)
            source = directory / "agent.py"
            source.write_text("original\n", encoding="utf-8")
            subprocess.run(["git", "add", "agent.py"], cwd=directory, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=RunEconomy Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                cwd=directory,
                check=True,
            )

            source.write_text("candidate-a\n", encoding="utf-8")
            first = _workdir_identity(directory)
            source.write_text("candidate-b\n", encoding="utf-8")
            second = _workdir_identity(directory)

            self.assertTrue(first["dirty"])
            self.assertNotEqual(
                first["tracked_diff_sha256"], second["tracked_diff_sha256"]
            )

if __name__ == "__main__":
    unittest.main()
