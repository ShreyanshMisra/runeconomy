from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from runeconomy.analysis import analyze_trace
from runeconomy.detectors import DETECTOR_TEMPLATES, diagnose_result, model_parameters_billions
from runeconomy.experiment_init import initialize_experiment
from runeconomy.models import Trace


FIXTURES = Path(__file__).parent / "fixtures" / "detectors"
ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _rules(payload: dict) -> dict[str, dict]:
    report = analyze_trace(Trace.from_dict(payload), dependency_evidence=payload.get("dependency_evidence"))
    return {finding["rule_id"]: finding for finding in report["findings"]}


class TraceDetectorTests(unittest.TestCase):
    def test_tool_output_carryover(self) -> None:
        finding = _rules(_load("tool-output-carryover.json"))["context.tool_output_carryover"]
        metrics = finding["supporting_metrics"]
        self.assertEqual(metrics["total_carried_tokens"], 1200)
        self.assertEqual(metrics["uncached_carried_tokens"], 603)
        self.assertEqual(finding["experiment_template_id"], "context.compact_tool_output")
        self.assertEqual(finding["predicted_direction"], "decrease")

    def test_carryover_needs_exact_tokens(self) -> None:
        payload = _load("tool-output-carryover.json")
        for request in payload["requests"]:
            request["prompt_token_ids"] = None
            request["prompt_token_count"] = request["metrics"]["input_tokens"]
            request["token_evidence_quality"] = "count_only"
            for segment in request["segments"]:
                segment.update({"start_token": None, "end_token": None, "attribution_quality": "structural"})
        self.assertNotIn("context.tool_output_carryover", _rules(payload))

    def test_serialized_independent_steps(self) -> None:
        finding = _rules(_load("serialized-independent-steps.json"))["workflow.serialized_independent_steps"]
        metrics = finding["supporting_metrics"]
        self.assertEqual(metrics["critical_path_ms"], 800.0)
        self.assertEqual(metrics["serialization_gap_ms"], 400.0)
        self.assertEqual(metrics["serialized_independent_pairs"], [["lookup-a", "lookup-b"]])
        self.assertEqual(finding["estimated_reducible_ms"], 400.0)

    def test_serialization_needs_complete_explicit_dependencies(self) -> None:
        for evidence in (None, "partial_explicit", "derived_sequential_capture_order"):
            payload = _load("serialized-independent-steps.json")
            payload["dependency_evidence"] = evidence
            self.assertNotIn("workflow.serialized_independent_steps", _rules(payload))

    def test_step_overprovisioning(self) -> None:
        finding = _rules(_load("step-overprovisioning.json"))["models.step_overprovisioning"]
        self.assertEqual(finding["affected_request_ids"], ["route", "format"])
        self.assertEqual(finding["confidence"], "low")
        self.assertEqual(finding["evidence_level"], "heuristic")

    def test_overprovisioning_is_silent_without_size_or_contrast(self) -> None:
        payload = _load("step-overprovisioning.json")
        for request in payload["requests"]:
            request["model"] = "private-model"
        self.assertNotIn("models.step_overprovisioning", _rules(payload))
        payload = _load("step-overprovisioning.json")
        payload["requests"] = [payload["requests"][0]]
        payload["spans"] = payload["spans"][:1]
        self.assertNotIn("models.step_overprovisioning", _rules(payload))
        self.assertEqual(model_parameters_billions("Qwen/Qwen2.5-1.5B-Instruct"), 1.5)
        self.assertEqual(model_parameters_billions("meta-llama/Llama-3.1-70B"), 70.0)
        self.assertIsNone(model_parameters_billions("gpt-4o-mini"))

    def test_structured_step_inflation(self) -> None:
        finding = _rules(_load("structured-step-inflation.json"))["output.structured_step_inflation"]
        self.assertEqual(finding["affected_request_ids"], ["plan-call"])
        self.assertEqual(finding["supporting_metrics"]["total_decode_ms"], 8600.0)
        payload = _load("structured-step-inflation.json")
        for request in payload["requests"]:
            request["invoked_tools"] = []
        self.assertNotIn("output.structured_step_inflation", _rules(payload))


class ResultDetectorTests(unittest.TestCase):
    def test_kv_pressure_from_load_telemetry(self) -> None:
        report = diagnose_result(_load("kv-pressure-result.json"))
        (finding,) = report["findings"]
        self.assertEqual(finding["rule_id"], "serving.kv_pressure")
        self.assertEqual(finding["supporting_metrics"]["preempting_targets"], ["candidate"])
        self.assertEqual(finding["confidence"], "high")
        self.assertEqual(report["summary"]["finding_count"], 1)

    def test_kv_pressure_is_silent_without_measured_telemetry(self) -> None:
        result = _load("kv-pressure-result.json")
        quiet = copy.deepcopy(result)
        quiet["load_blocks"][0]["telemetry"]["server_metrics"]["targets"] = {
            "candidate": {"status": "unavailable", "samples": 0, "scrape_errors": 3}
        }
        self.assertEqual(diagnose_result(quiet)["findings"], [])
        self.assertEqual(diagnose_result({"decision": {}})["findings"], [])


class TemplateTests(unittest.TestCase):
    def test_every_detector_template_is_documented_and_yields_an_honest_skeleton(self) -> None:
        documentation = (ROOT / "docs" / "detectors.md").read_text(encoding="utf-8")
        for name in ("tool-output-carryover.json", "serialized-independent-steps.json",
                     "step-overprovisioning.json", "structured-step-inflation.json"):
            for finding in _rules(_load(name)).values():
                if finding["experiment_template_id"] not in DETECTOR_TEMPLATES:
                    continue
                skeleton = initialize_experiment(finding)
                self.assertEqual(skeleton["generation_status"], "unresolved")
        for template_id in DETECTOR_TEMPLATES:
            self.assertIn(f"`{template_id}`", documentation)


if __name__ == "__main__":
    unittest.main()
