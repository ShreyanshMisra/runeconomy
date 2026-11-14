from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

from runeconomy.experiments import ExperimentManifest, _schedule
from scripts.build_heldout_manifest import build_manifest
from scripts.review_absolute_success import review_result

ROOT = Path(__file__).parents[1]


def _load_benchmark_deployment():
    path = ROOT / "deploy" / "modal_benchmark.py"
    spec = importlib.util.spec_from_file_location("modal_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Modal benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveExperimentDesignTests(unittest.TestCase):
    def test_posthoc_review_preserves_original_decision(self) -> None:
        raw = {
            "summary": {
                "workflow": {
                    "variants": {
                        "baseline": {
                            "outcomes": {
                                "task_success_rate": 0.30,
                                "task_success_95_ci": [0.16, 0.47],
                            }
                        },
                        "candidate": {
                            "outcomes": {
                                "task_success_rate": 0.40,
                                "task_success_95_ci": [0.25, 0.57],
                            }
                        },
                    }
                }
            },
            "decision": {
                "status": "verified",
                "reason": "predeclared gates passed",
                "checks": [],
            },
            "provenance": {},
        }

        reviewed = review_result(raw, 0.80)

        self.assertEqual(raw["decision"]["status"], "verified")
        self.assertEqual(reviewed["decision"]["status"], "invalid")
        self.assertFalse(reviewed["decision"]["checks"][0]["predeclared"])
        self.assertEqual(
            reviewed["provenance"]["methodology_review"]
            ["original_predeclared_decision"]["status"],
            "verified",
        )

    def test_checked_in_manifest_is_generated_from_frozen_design(self) -> None:
        path = (
            ROOT
            / "examples"
            / "experiments"
            / "repository-agent-tool-pruning-heldout.json"
        )
        checked_in = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(checked_in, build_manifest())

        manifest = ExperimentManifest.from_path(path)
        workflow_ids = {case.workflow_id for case in manifest.cases}
        self.assertEqual(len(workflow_ids), 40)
        self.assertEqual(len(manifest.cases), 80)
        self.assertTrue(manifest.acceptance.require_environment_identity)
        self.assertEqual(manifest.acceptance.max_task_success_regression, 0.10)
        self.assertEqual(len(_schedule(manifest, 1, warmup=False)), 160)
        symbols = {
            case.evaluator.get("expected")
            for case in manifest.cases
            if case.case_id.endswith("-inspect") and case.evaluator is not None
        }
        self.assertNotIn("RequestAnalysis", symbols)
        self.assertNotIn("Finding", symbols)
        self.assertIn("_percentile", symbols)
        self.assertIn("_metric_summary", symbols)

    def test_calibration_digest_and_selected_fixture_are_pinned(self) -> None:
        calibration_path = (
            ROOT
            / "examples"
            / "calibration"
            / "repository-readonly-tool-selection.json"
        )
        manifest_path = (
            ROOT
            / "examples"
            / "experiments"
            / "repository-agent-tool-pruning-heldout.json"
        )
        calibration_bytes = calibration_path.read_bytes()
        calibration = json.loads(calibration_bytes)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection = manifest["design"]["candidate_selection"]
        self.assertFalse(calibration["evaluation_data_used"])
        self.assertEqual(selection["selected_tools"], calibration["selected_tools"])
        self.assertEqual(
            selection["calibration_sha256"],
            "sha256:" + hashlib.sha256(calibration_bytes).hexdigest(),
        )

        scoped = json.loads(
            (ROOT / "examples" / "fixtures" / "coding-tools-scoped.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [tool["name"] for tool in scoped], calibration["selected_tools"]
        )

    def test_collocated_runner_is_single_gpu_and_cost_bounded(self) -> None:
        deployment = _load_benchmark_deployment()
        self.assertEqual(deployment.GPU, "A10")
        self.assertEqual(deployment.GPU_TIMEOUT_SECONDS, 70 * 60)
        self.assertEqual(deployment.MODEL_NAME, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(deployment.HUGGINGFACE_HUB_VERSION, "1.31.0")
        self.assertIn("--revision", deployment.vllm_command())
        source = (ROOT / "deploy" / "modal_benchmark.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("@app.server", source)
        self.assertNotIn("min_containers", source)
        self.assertIn("127.0.0.1", source)

        manifest_path = (
            ROOT
            / "examples"
            / "experiments"
            / "repository-agent-tool-pruning-heldout.json"
        )
        payload = deployment._resolved_payload(manifest_path, workflow_limit=2)
        transported = ExperimentManifest.from_dict(payload)
        self.assertEqual(len(transported.cases), 4)
        self.assertEqual(
            len({case.workflow_id for case in transported.cases}), 2
        )


if __name__ == "__main__":
    unittest.main()
