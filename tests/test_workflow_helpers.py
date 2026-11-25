from __future__ import annotations

import json
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import runeconomy
from runeconomy.comparison import run_comparison
from runeconomy.importers import import_openinference
from runeconomy.workflow import WorkflowTraceError, WorkflowTracker, depends_on

from tests.test_comparison import EVALUATOR


AGENT = r'''
import json, os, time
from pathlib import Path

import runeconomy
from runeconomy.agent import Attempt

with Attempt.from_environment() as attempt:
    task = json.loads(attempt.task_path.read_text())

    @runeconomy.track_workflow
    def solve():
        with runeconomy.step("plan") as plan:
            time.sleep(0.01)
            plan.record_llm(model="fixture/model", input_tokens=10, output_tokens=2, provider="vllm",
                            model_revision="rev", backend_version="0.1")
        with runeconomy.step("search", kind="TOOL", depends_on=[plan]) as search:
            time.sleep(0.01)
        with runeconomy.step("answer", depends_on=[plan, search]) as answer:
            answer.record_llm(model="fixture/model", input_tokens=30, output_tokens=3, cached_tokens=8,
                              provider="vllm", model_revision="rev", backend_version="0.1")
        return task["expected"]

    attempt.output_path.write_text(solve(), encoding="utf-8")
    attempt.completed()
'''


class WorkflowTrackerTests(unittest.TestCase):
    def test_trace_has_explicit_edges_and_imports_as_complete_dag(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "trace.json"
            with runeconomy.track_workflow(trace_path=path, task_id="task-1"):
                with runeconomy.step("plan") as plan:
                    plan.record_llm(model="m", input_tokens=5, output_tokens=1)
                with runeconomy.step("tool", kind="TOOL", depends_on=[plan]) as tool:
                    pass
                with runeconomy.step("answer", depends_on=depends_on(plan, tool)) as answer:
                    answer.record_llm(model="m", input_tokens=9, output_tokens=2)
            trace = import_openinference(path)
        spans = {item["span_id"]: item for item in trace["spans"]}
        self.assertEqual(trace["dependency_evidence"], "complete_explicit")
        self.assertEqual(spans[answer.id]["depends_on"], [plan.id, tool.id])
        self.assertEqual(spans[plan.id]["depends_on"], [])
        self.assertEqual(len(trace["requests"]), 2)

    def test_edges_must_name_started_steps_and_kinds_are_checked(self) -> None:
        tracker = WorkflowTracker(environment={})
        with tracker:
            with self.assertRaises(WorkflowTraceError):
                tracker.step("x", depends_on=["not-a-step"])
            with self.assertRaises(WorkflowTraceError):
                tracker.step("x", kind="DATABASE")
            with tracker.step("tool", kind="TOOL") as tool:
                with self.assertRaises(WorkflowTraceError):
                    tool.record_llm(model="m")
        with self.assertRaises(WorkflowTraceError):
            runeconomy.step("outside")

    def test_parallel_steps_in_worker_threads_join_the_active_workflow(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        with TemporaryDirectory() as raw:
            path = Path(raw) / "trace.json"
            with runeconomy.track_workflow(trace_path=path, task_id="t"):
                with runeconomy.step("plan") as plan:
                    plan.record_llm(model="m", input_tokens=1, output_tokens=1)

                def lookup(name: str) -> str:
                    with runeconomy.step(name, kind="TOOL", depends_on=[plan]) as item:
                        return item.id

                with ThreadPoolExecutor(max_workers=2) as pool:
                    ids = list(pool.map(lookup, ["a", "b"]))
                with runeconomy.step("answer", depends_on=ids) as answer:
                    answer.record_llm(model="m", input_tokens=2, output_tokens=1)
            trace = import_openinference(path)
        self.assertEqual(trace["dependency_evidence"], "complete_explicit")
        self.assertEqual(len(trace["spans"]), 4)

    def test_failed_workflow_still_writes_partial_trace(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "trace.json"
            with self.assertRaises(RuntimeError):
                with runeconomy.track_workflow(trace_path=path):
                    with runeconomy.step("plan"):
                        raise RuntimeError("boom")
            payload = json.loads(path.read_text())
        span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        self.assertEqual(span["status"]["message"], "RuntimeError")

    def test_agent_using_helpers_runs_with_explicit_critical_path(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
            (root / "evaluator.py").write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
            (root / "tasks.json").write_text(json.dumps({"tasks": [{"task_id": "a", "expected": "x"}]}))
            result = run_comparison(
                baseline_command=f"{sys.executable} {root / 'agent.py'}",
                candidate_command=f"{sys.executable} {root / 'agent.py'}",
                dataset_path=root / "tasks.json",
                evaluator_path=root / "evaluator.py",
                output_path=root / "result.json",
                artifacts_dir=root / "artifacts",
                workdir=root,
                order="alternating",
                min_task_success_rate=0.0,
            )
        for attempt in result["attempts"]:
            self.assertEqual(attempt["status"], "completed")
            self.assertTrue(attempt["task_success"])
            metrics = attempt["metrics"]
            self.assertEqual(metrics["dependency_evidence"], "complete_explicit")
            self.assertEqual(metrics["input_tokens"], 40)
            self.assertEqual(metrics["cached_input_tokens"], 8)
            self.assertIsNotNone(metrics["critical_path_ms"])


if __name__ == "__main__":
    unittest.main()


ROUTED_AGENT = r'''
import json, os, sys
import runeconomy
from runeconomy.agent import Attempt

roles = sys.argv[1] == "roles"
drift = sys.argv[2] == "drift"
with Attempt.from_environment() as attempt:
    task = json.loads(attempt.task_path.read_text())
    small = "fixture/small-drifted" if drift and task["task_id"] == "b" else "fixture/small"
    with runeconomy.track_workflow():
        with runeconomy.step("plan") as plan:
            plan.record_llm(model=small, input_tokens=5, output_tokens=2, provider="vllm",
                            model_revision="r", backend_version="0.1", tokenizer=small,
                            tokenizer_revision="r", chat_template_digest="sha256:t",
                            step_role="planner" if roles else None)
        with runeconomy.step("answer", depends_on=[plan]) as answer:
            answer.record_llm(model="fixture/large", input_tokens=9, output_tokens=3, provider="vllm",
                              model_revision="r", backend_version="0.1", tokenizer="fixture/large",
                              tokenizer_revision="r", chat_template_digest="sha256:t",
                              step_role="answerer" if roles else None)
    attempt.output_path.write_text(task["expected"])
    attempt.completed()
'''


class StepRoleIdentityTests(unittest.TestCase):
    def _stable(self, root: Path, *flags: str) -> bool:
        agent = root / "routed.py"
        agent.write_text(textwrap.dedent(ROUTED_AGENT), encoding="utf-8")
        (root / "evaluator.py").write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
        (root / "tasks.json").write_text(json.dumps({"tasks": [
            {"task_id": "a", "expected": "x"}, {"task_id": "b", "expected": "y"}]}))
        command = f"{sys.executable} {agent} {' '.join(flags)}"
        name = "-".join(flags)
        result = run_comparison(
            baseline_command=command, candidate_command=command,
            dataset_path=root / "tasks.json", evaluator_path=root / "evaluator.py",
            output_path=root / f"{name}.json", artifacts_dir=root / f"{name}-artifacts",
            workdir=root, order="alternating", min_task_success_rate=0.0,
        )
        check = next(c for c in result["decision"]["checks"] if c["name"] == "trace_and_environment_identity")
        return bool(check["passed"])

    def test_declared_roles_make_routing_stable_and_drift_still_fails(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertTrue(self._stable(root, "roles", "steady"))
            self.assertFalse(self._stable(root, "noroles", "steady"))
            self.assertFalse(self._stable(root, "roles", "drift"))
