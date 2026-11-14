import json
import unittest
from pathlib import Path

from runeconomy.analysis import analyze_trace
from runeconomy.cache import eligible_prefix_tokens, longest_common_prefix
from runeconomy.critical_path import calculate_critical_path
from runeconomy.models import Trace, TraceValidationError


ROOT = Path(__file__).resolve().parents[1]


class CacheTests(unittest.TestCase):
    def test_longest_common_prefix(self):
        self.assertEqual(longest_common_prefix((1, 2, 3), (1, 2, 4)), 2)
        self.assertEqual(longest_common_prefix((1, 2), (1, 2, 3)), 2)
        self.assertEqual(longest_common_prefix((), (1,)), 0)

    def test_vllm_eligibility_uses_full_blocks_and_excludes_last_token(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        request = Trace.from_dict(payload).requests[0]
        self.assertEqual(eligible_prefix_tokens(request, 40), 36)
        self.assertEqual(eligible_prefix_tokens(request, 7), 4)


class AnalysisTests(unittest.TestCase):
    def load_demo(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        return Trace.from_dict(payload)

    def test_timestamp_divergence_is_attributed(self):
        report = analyze_trace(self.load_demo())
        second = report["requests"][1]
        self.assertEqual(second["baseline_request_id"], "req-1")
        self.assertEqual(second["shared_prefix_tokens"], 4)
        self.assertEqual(second["eligible_prefix_tokens"], 4)
        self.assertEqual(second["divergence_segment_kind"], "dynamic_metadata")
        self.assertTrue(second["on_critical_path"])
        self.assertEqual(report["critical_path"]["duration_ms"], 390)
        rules = [finding["rule_id"] for finding in report["findings"]]
        self.assertIn("prefix_divergence.dynamic_metadata", rules)
        finding = next(
            item
            for item in report["findings"]
            if item["rule_id"] == "prefix_divergence.dynamic_metadata"
        )
        self.assertTrue(finding["finding_id"].startswith("finding-"))
        self.assertEqual(
            finding["affected_request_ids"], ("req-1", "req-2")
        )
        self.assertEqual(
            finding["experiment_template_id"], "prompt.move_dynamic_metadata"
        )
        self.assertEqual(finding["predicted_direction"], "increase")
        self.assertIsNone(finding["estimated_reducible_ms"])

    def test_finding_ids_are_stable(self):
        first = analyze_trace(self.load_demo())
        second = analyze_trace(self.load_demo())
        self.assertEqual(
            [item["finding_id"] for item in first["findings"]],
            [item["finding_id"] for item in second["findings"]],
        )

    def test_missing_cache_telemetry_is_not_treated_as_zero(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        payload["requests"][1]["metrics"].pop("cached_input_tokens")
        report = analyze_trace(Trace.from_dict(payload))
        second = report["requests"][1]
        self.assertIsNone(second["actual_cached_tokens"])
        self.assertIsNone(second["unrealized_eligible_tokens"])
        self.assertIsNone(report["summary"]["total_actual_cached_tokens"])
        self.assertIsNone(report["summary"]["observed_cache_coverage"])
        self.assertFalse(report["summary"]["cache_telemetry_complete"])

    def test_unrealized_reuse_produces_inferred_finding(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        payload["requests"][1]["prompt_token_ids"] = payload["requests"][0][
            "prompt_token_ids"
        ].copy()
        payload["requests"][1]["metrics"]["cached_input_tokens"] = 0
        report = analyze_trace(Trace.from_dict(payload))
        rules = [finding["rule_id"] for finding in report["findings"]]
        self.assertIn("cache.unrealized_eligibility", rules)

    def test_overlapping_segments_are_rejected(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        payload["requests"][0]["segments"][1]["start_token"] = 3
        with self.assertRaisesRegex(TraceValidationError, "non-overlapping"):
            Trace.from_dict(payload)

    def test_cycle_in_span_graph_is_rejected_by_analysis(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        payload["spans"][0]["depends_on"] = ["infer-2"]
        with self.assertRaisesRegex(TraceValidationError, "cycle"):
            Trace.from_dict(payload)

    def test_dependency_timing_inconsistency_is_rejected(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        payload["spans"][1]["start_ms"] = 100
        with self.assertRaisesRegex(TraceValidationError, "starts before dependency"):
            Trace.from_dict(payload)

    def test_parallel_branch_is_not_on_critical_path(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        payload["spans"].append(
            {
                "span_id": "side-tool",
                "kind": "tool",
                "start_ms": 180,
                "end_ms": 190,
                "request_id": None,
                "depends_on": ["infer-1"],
            }
        )
        path = calculate_critical_path(Trace.from_dict(payload).spans)
        self.assertIsNotNone(path)
        self.assertNotIn("side-tool", path.span_ids)

    def test_unused_tool_exposure_is_found_on_critical_path(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        request = payload["requests"][1]
        request["available_tools"] = [
            "read_file",
            "search_repository",
            "run_tests",
            "write_file",
            "apply_patch",
            "create_branch",
        ]
        request["invoked_tools"] = ["read_file"]

        report = analyze_trace(Trace.from_dict(payload))
        finding = next(
            item
            for item in report["findings"]
            if item["rule_id"] == "tools.unused_exposure"
        )
        self.assertTrue(finding["on_critical_path"])
        self.assertEqual(finding["supporting_metrics"]["unused_tool_count"], 5)
        self.assertEqual(
            finding["supporting_metrics"]["optimization_target"], "input_tokens"
        )

    def test_invoked_tool_must_be_available(self):
        payload = json.loads(
            (ROOT / "examples/traces/unstable_timestamp.json").read_text()
        )
        payload["requests"][0]["available_tools"] = ["read_file"]
        payload["requests"][0]["invoked_tools"] = ["run_tests"]
        with self.assertRaisesRegex(TraceValidationError, "not present"):
            Trace.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
