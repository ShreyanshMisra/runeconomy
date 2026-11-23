from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Proof2AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = _load("proof_2_agent", ROOT / "examples" / "proof_2" / "agent.py")

    def test_question_parser_supports_both_families(self) -> None:
        self.assertEqual(
            self.agent._search_phrase(
                "Which Python function or class encloses the exact string literal 'alpha beta gamma'?"
            ),
            "alpha beta gamma",
        )
        self.assertEqual(
            self.agent._search_phrase(
                'Which Python function or class encloses the exact call expression "client.capture(payload)"?'
            ),
            "client.capture(payload)",
        )

    def test_backend_profiles_change_identity_not_source_revision(self) -> None:
        baseline = self.agent._identity_attributes("baseline")
        candidate = self.agent._identity_attributes("candidate")
        self.assertNotEqual(baseline["llm.model_name"], candidate["llm.model_name"])
        self.assertEqual(
            baseline["llm.model_revision"], candidate["llm.model_revision"]
        )
        self.assertEqual(
            baseline["runeconomy.chat_template_digest"],
            candidate["runeconomy.chat_template_digest"],
        )


class Proof2WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load(
            "proof_2_builder", ROOT / "scripts" / "build_proof_2_workload.py"
        )

    def test_builder_creates_disjoint_stratified_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proof-2"
            record = self.builder.build(output)
            self.assertEqual(record["status"], "draft_heldout_not_frozen")
            self.assertEqual(record["splits"]["pilot"]["task_count"], 5)
            self.assertEqual(record["splits"]["calibration"]["task_count"], 40)
            self.assertEqual(
                record["splits"]["draft_heldout_pool"]["task_count"], 240
            )
            task_ids: list[set[str]] = []
            for name in ("pilot", "calibration", "draft-heldout"):
                payload = json.loads(
                    (output / f"{name}-tasks.json").read_text(encoding="utf-8")
                )
                ids = {task["task_id"] for task in payload["tasks"]}
                families = {task["family"] for task in payload["tasks"]}
                self.assertEqual(
                    families, {"call_expression", "string_literal"}
                )
                task_ids.append(ids)
            self.assertFalse(task_ids[0] & task_ids[1])
            self.assertFalse(task_ids[0] & task_ids[2])
            self.assertFalse(task_ids[1] & task_ids[2])

    def test_generated_evaluator_accepts_only_exact_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proof-2"
            self.builder.build(output)
            tasks = json.loads(
                (output / "pilot-tasks.json").read_text(encoding="utf-8")
            )["tasks"]
            evaluator = output / "pilot-evaluator.py"
            answers_source = evaluator.read_text(encoding="utf-8")
            namespace: dict[str, object] = {}
            prefix = answers_source.split("\n\nrequest =", 1)[0]
            exec(prefix, namespace)
            answers = namespace["ANSWERS"]
            assert isinstance(answers, dict)
            task_id = tasks[0]["task_id"]
            completed = subprocess.run(
                [sys.executable, str(evaluator)],
                input=json.dumps(
                    {"task_id": task_id, "output_text": answers[task_id]}
                ),
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertTrue(json.loads(completed.stdout)["success"])


class Proof2ModalTests(unittest.TestCase):
    def test_runner_is_calibration_only_and_pins_one_intervention(self) -> None:
        source = (ROOT / "deploy" / "modal_proof_2.py").read_text(encoding="utf-8")
        self.assertIn('GPU = "L40S:2"', source)
        self.assertIn('"candidate": "fp8_per_tensor"', source)
        self.assertIn('"baseline": None', source)
        self.assertIn('"VLLM_STRICT_QUANT_SCHEME": "1"', source)
        self.assertIn("expected_task_count not in {5, 40}", source)
        self.assertNotIn("def heldout(", source)
        self.assertIn('optimization_target="engine_change_latency"', source)


if __name__ == "__main__":
    unittest.main()
