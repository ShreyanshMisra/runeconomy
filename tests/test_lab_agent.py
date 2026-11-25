"""The lab agent end to end against a fake vLLM (no GPU)."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy.analysis import analyze_trace
from runeconomy.comparison import run_comparison
from runeconomy.models import Trace

from tests import fake_vllm

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "examples" / "lab_agent" / "agent.py"
SOURCE = '''
def load_registry(path):
    payload = read_json(path)
    entries = payload.get("registered_entries_for_lookup")
    return entries


class ReportBuilder:
    def render_summary(self, rows):
        return format_rows(rows, style="compact-summary-table")
'''
TASKS = [
    ("registered_entries_for_lookup", "string literal", "load_registry"),
    ('format_rows(rows, style="compact-summary-table")', "call expression", "render_summary"),
    ("compact-summary-table", "string literal", "render_summary"),
]


def _workspace(root: Path) -> tuple[Path, Path]:
    (root / "repo" / "pkg").mkdir(parents=True)
    (root / "repo" / "pkg" / "module.py").write_text(SOURCE + "\n" * 5 + "# padding line with several more descriptive words to fill the excerpt window as real source lines with code and comments would do\n" * 60, encoding="utf-8")
    for name in ("a.py", "b.py", "c.py", "d.py"):
        (root / "repo" / "pkg" / name).write_text("def unrelated():\n    return 1\n", encoding="utf-8")
    tasks, answers = [], {}
    for index, (phrase, kind, answer) in enumerate(TASKS):
        task_id = f"lab-{index}"
        tasks.append({
            "task_id": task_id,
            "question": f"Which Python function or class encloses the exact {kind} {phrase!r}?",
            "candidate_paths": ["pkg/a.py", "pkg/module.py", "pkg/b.py", "pkg/c.py", "pkg/d.py"],
            "answer_choices": ["load_registry", "render_summary", "ReportBuilder", "unrelated"],
        })
        answers[task_id] = answer
    (root / "tasks.json").write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    (root / "evaluator.py").write_text(
        "import json, sys\nANSWERS = " + json.dumps(answers)
        + "\nr = json.load(sys.stdin)\nok = r['output_text'].strip() == ANSWERS[r['task_id']]\n"
        "json.dump({'success': ok, 'score': float(ok), 'details': {}}, sys.stdout)\n",
        encoding="utf-8",
    )
    return root / "tasks.json", root / "evaluator.py"


class LabAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_vllm.FakeVLLM.delay_seconds = 0.06
        self.server = fake_vllm.serve()
        url = f"http://127.0.0.1:{self.server.server_port}"
        self.environment = {
            "RUNECONOMY_BASELINE_URL": url,
            "RUNECONOMY_CANDIDATE_URL": url,
            "RUNECONOMY_BASELINE_MODEL": "runeconomy-qwen2.5-14b-bf16",
            "RUNECONOMY_CANDIDATE_MODEL": "runeconomy-qwen2.5-14b-bf16",
            "RUNECONOMY_SMALL_URL": url,
            "RUNECONOMY_SMALL_MODEL": "runeconomy-qwen2.5-1.5b",
        }

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        fake_vllm.FakeVLLM.delay_seconds = 0.0

    def _compare(self, root: Path, baseline: str, candidate: str) -> dict:
        dataset, evaluator = _workspace(root)
        with patch.dict(os.environ, {**self.environment, "RUNECONOMY_REPO_ROOT": str(root / "repo")}):
            return run_comparison(
                baseline_command=f"{sys.executable} {AGENT} {baseline}",
                candidate_command=f"{sys.executable} {AGENT} {candidate}",
                dataset_path=dataset,
                evaluator_path=evaluator,
                output_path=root / "result.json",
                artifacts_dir=root / "artifacts",
                workdir=root,
                order="alternating",
                min_task_success_rate=0.0,
                max_task_success_regression=1.0,
            )

    def _baseline_rules(self, root: Path, result: dict) -> set[str]:
        rules: set[str] = set()
        for attempt in result["attempts"]:
            if attempt["variant"] != "baseline":
                continue
            payload = json.loads((root / "artifacts" / attempt["artifacts"]["directory"] / "normalized-trace.json").read_text())
            report = analyze_trace(Trace.from_dict(payload), dependency_evidence=payload.get("dependency_evidence"))
            rules |= {finding["rule_id"] for finding in report["findings"]}
        return rules

    def _assert_clean(self, result: dict) -> None:
        statuses = {attempt["status"] for attempt in result["attempts"]}
        self.assertLessEqual(statuses, {"completed", "task_failure"}, result["attempts"][0].get("failure_reason"))
        check = next(c for c in result["decision"]["checks"] if c["name"] == "trace_and_environment_identity")
        self.assertTrue(check["passed"], check["observed"])

    def test_default_workflow_is_proof_2_shaped_and_correct(self) -> None:
        with TemporaryDirectory() as raw:
            result = self._compare(Path(raw), "", "")
        self._assert_clean(result)
        self.assertTrue(all(attempt["task_success"] for attempt in result["attempts"]))
        self.assertEqual(result["attempts"][0]["metrics"]["dependency_evidence"], "complete_explicit")

    def test_tool_pruning_candidate_loses_task_success(self) -> None:
        with TemporaryDirectory() as raw:
            result = self._compare(Path(raw), "", "--search none")
        self._assert_clean(result)
        candidate = [a["task_success"] for a in result["attempts"] if a["variant"] == "candidate"]
        self.assertLess(sum(candidate), len(candidate))

    def test_carryover_baseline_is_detected_and_candidate_compacts(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            result = self._compare(root, "--lookups sequential --evidence carry --segments",
                                   "--lookups sequential --evidence compact --segments")
            self._assert_clean(result)
            self.assertIn("context.tool_output_carryover", self._baseline_rules(root, result))
        tokens = {variant: sum(a["metrics"]["input_tokens"] for a in result["attempts"] if a["variant"] == variant)
                  for variant in ("baseline", "candidate")}
        self.assertLess(tokens["candidate"], tokens["baseline"])

    def test_serialized_lookups_are_detected(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            result = self._compare(root, "--lookups sequential", "--lookups parallel")
            self._assert_clean(result)
            self.assertIn("workflow.serialized_independent_steps", self._baseline_rules(root, result))

    def test_verbose_plan_is_detected_as_inflation(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            result = self._compare(root, "--plan verbose", "")
            self._assert_clean(result)
            self.assertIn("output.structured_step_inflation", self._baseline_rules(root, result))

    def test_overprovisioned_planner_is_detected_and_routing_is_stable(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            result = self._compare(root, "--explain", "--explain --plan-model-env RUNECONOMY_SMALL")
            self._assert_clean(result)
            self.assertIn("models.step_overprovisioning", self._baseline_rules(root, result))


if __name__ == "__main__":
    unittest.main()


class JsonExtractionTests(unittest.TestCase):
    def test_string_values_with_braces_and_trailing_reasoning(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("lab_agent", AGENT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text = 'Reasoning about {braces}.\n{"query": "deepcopy(plan.get(\\"x\\") or {})"}'
        self.assertEqual(module._last_json_object(text, stage="t")["query"], 'deepcopy(plan.get("x") or {})')
        with self.assertRaises(module.ModelOutputFailure):
            module._last_json_object("no json here", stage="t")
