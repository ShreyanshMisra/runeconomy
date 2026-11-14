from __future__ import annotations

import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "deploy" / "modal_public_proof.py"


class ModalPublicProofTests(unittest.TestCase):
    def test_pilot_is_colocated_bounded_and_private(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn('GPU = "A10"', source)
        self.assertIn("PILOT_TIMEOUT_SECONDS = 12 * 60", source)
        self.assertIn('"--host",\n        "127.0.0.1"', source)
        self.assertIn("same Modal container as vLLM", source)
        self.assertNotIn("@app.server", source)
        self.assertNotIn("unauthenticated=True", source)

    def test_runner_allows_only_approved_calibration_sizes(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("expected_task_count not in {5, 20, 40}", source)
        self.assertIn("len(tasks) != expected_task_count", source)
        self.assertIn("pilot-tasks.json", source)
        self.assertIn("pilot-evaluator.py", source)
        self.assertIn("calibration-tasks.json", source)
        self.assertIn("calibration-evaluator.py", source)
        self.assertIn("heldout-tasks.json", source)
        self.assertIn("heldout-evaluator.py", source)
        self.assertIn("run_pilot.remote(task_payload, evaluator_source, 40)", source)

    def test_full_calibration_uses_predeclared_quality_gates(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("0.1 if expected_task_count in {20, 40} else 1.0", source)
        self.assertIn("0.8 if expected_task_count in {20, 40} else 0.0", source)
        self.assertIn("run_pilot.remote(task_payload, evaluator_source, 20)", source)

    def test_pilot_pins_identity_and_packages_required_sources(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            'MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"',
            source,
        )
        self.assertIn('VLLM_VERSION = "0.33.0"', source)
        self.assertIn("_chat_template_digest() != CHAT_TEMPLATE_DIGEST", source)
        self.assertIn('.add_local_python_source("runeconomy")', source)
        self.assertIn('"examples/reliable_proof/agent.py"', source)
        self.assertIn('"src/runeconomy"', source)


if __name__ == "__main__":
    unittest.main()
