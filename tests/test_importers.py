from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.analysis import analyze_trace
from runeconomy.experiment_init import initialize_experiment, select_finding
from runeconomy.importers import (
    import_openinference,
    import_prefixwise_jsonl,
    import_vllm_jsonl,
)
from runeconomy.models import Trace, TraceValidationError


class ImporterTests(unittest.TestCase):
    def test_vllm_jsonl_envelopes_produce_analyzable_trace(self) -> None:
        tool_catalog = [
            {"type": "function", "name": name, "parameters": {"type": "object"}}
            for name in ("read", "search", "test", "deploy")
        ]
        records = [
            {
                "request_id": "request-1",
                "request": {"model": "example/model", "tools": tool_catalog},
                "capture": {
                    "prompt_token_ids": list(range(32)),
                    "input_tokens": 32,
                    "output_tokens": 2,
                    "cached_tokens": 0,
                    "server_ttft_ms": 20.0,
                    "total_latency_ms": 40.0,
                },
                "invoked_tools": ["read"],
            },
            {
                "request_id": "request-2",
                "request": {"model": "example/model", "tools": tool_catalog},
                "capture": {
                    "prompt_token_ids": list(range(32)) + [99],
                    "input_tokens": 33,
                    "output_tokens": 2,
                    "cached_tokens": 32,
                    "server_ttft_ms": 10.0,
                    "total_latency_ms": 30.0,
                },
                "invoked_tools": ["search"],
            },
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "captures.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            payload = import_vllm_jsonl(
                path,
                trace_id="imported",
                workload="repository-agent",
                model="example/model",
                model_revision="model-sha",
                tokenizer="example/model",
                tokenizer_revision="tokenizer-sha",
                chat_template_digest="sha256:template",
                backend_version="0.33.0",
            )

        trace = Trace.from_dict(payload)
        report = analyze_trace(trace)
        self.assertEqual(len(trace.requests), 2)
        self.assertEqual(trace.requests[1].metrics.cached_input_tokens, 32)
        self.assertEqual(report["summary"]["request_count"], 2)
        self.assertEqual(trace.spans[1].depends_on, ("inference-1",))
        self.assertEqual(trace.schema_version, "0.2")
        self.assertEqual(
            trace.requests[0].segments[0].attribution_quality, "structural"
        )
        self.assertIsNone(trace.requests[0].segments[0].start_token)

    def test_verified_rendered_segments_produce_exact_boundaries(self) -> None:
        record = {
            "request": {"model": "example/model"},
            "capture": {"prompt_token_ids": [1, 2, 3, 4]},
            "rendered_segments": [
                {"kind": "system_instructions", "token_ids": [1, 2]},
                {"kind": "latest_user_input", "token_ids": [3, 4]},
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "captures.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            trace = Trace.from_dict(
                import_vllm_jsonl(
                    path, trace_id="exact", workload="test", model="example/model"
                )
            )
        self.assertEqual(trace.requests[0].segments[1].start_token, 2)
        self.assertEqual(
            trace.requests[0].segments[1].attribution_quality,
            "exact_token_boundary",
        )

    def test_unverified_rendered_segments_are_rejected(self) -> None:
        record = {
            "capture": {"prompt_token_ids": [1, 2, 3]},
            "rendered_segments": [{"kind": "system", "token_ids": [1, 9, 3]}],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "captures.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                TraceValidationError, "do not concatenate"
            ):
                import_vllm_jsonl(
                    path, trace_id="bad", workload="test", model="example/model"
                )

    def test_openinference_otlp_import_preserves_count_only_evidence(self) -> None:
        attributes = {
            "openinference.span.kind": "LLM",
            "llm.model_name": "example/model",
            "llm.provider": "vllm",
            "llm.token_count.prompt": 123,
            "llm.token_count.completion": 7,
            "llm.token_count.prompt_details.cache_read": 64,
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.1.message.role": "user",
            "llm.tools.0.tool.name": "search",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "search",
        }
        payload = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "trace-1",
                                    "spanId": "llm-1",
                                    "startTimeUnixNano": "1000000000",
                                    "endTimeUnixNano": "1250000000",
                                    "attributes": [
                                        {"key": key, "value": {"stringValue": str(value)}}
                                        for key, value in attributes.items()
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        # OTLP encodes integer attributes with intValue rather than stringValue.
        for item in payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0][
            "attributes"
        ]:
            if item["key"].startswith("llm.token_count"):
                item["value"] = {"intValue": attributes[item["key"]]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "otel.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = Trace.from_dict(
                import_openinference(path, workload="agent-run")
            )
        request = trace.requests[0]
        self.assertIsNone(request.prompt_token_ids)
        self.assertEqual(request.prompt_token_count, 123)
        self.assertEqual(request.token_evidence_quality, "count_only")
        self.assertEqual(request.metrics.cached_input_tokens, 64)
        self.assertEqual(request.available_tools, ("search",))
        self.assertEqual(request.invoked_tools, ("search",))
        self.assertEqual(request.segments[0].attribution_quality, "structural")
        self.assertEqual(trace.spans[0].duration_ms, 250.0)
        report = analyze_trace(trace)
        self.assertEqual(report["summary"]["count_only_request_count"], 1)
        self.assertFalse(report["requests"][0]["prefix_analysis_supported"])
        self.assertEqual(
            report["requests"][0]["attribution_qualities"], ("structural",)
        )

    def test_openinference_exact_extension_is_not_inferred_from_counts(self) -> None:
        payload = {
            "spans": [
                {
                    "trace_id": "trace-2",
                    "span_id": "llm-2",
                    "start_time_unix_nano": 0,
                    "end_time_unix_nano": 1000000,
                    "attributes": {
                        "openinference.span.kind": "LLM",
                        "runeconomy.prompt_token_ids": "[4, 5, 6]",
                    },
                }
            ]
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "otel.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            trace = Trace.from_dict(import_openinference(path))
        self.assertEqual(trace.requests[0].prompt_token_ids, (4, 5, 6))
        self.assertEqual(trace.requests[0].token_evidence_quality, "exact")

    def test_prefixwise_import_preserves_evidence_but_not_raw_prompt(self) -> None:
        row = {
            "session_id": "session-1",
            "request_id": "request-4",
            "provider": "vllm",
            "model": "example/model",
            "cause": "tool_reserialize",
            "prompt_tokens": 4096,
            "cached_tokens": 0,
            "avoidable_tokens": 3072,
            "divergence_role": "tools",
            "divergence_index": 0,
            "detail": "The same tools were serialized differently.",
            "body": {"messages": [{"content": "secret"}]},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prefixwise.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            artifact = import_prefixwise_jsonl(path)
        finding = artifact["findings"][0]
        self.assertEqual(finding["experiment_template_id"], "prompt.canonicalize_tools")
        self.assertEqual(finding["estimated_reducible_tokens"], 3072)
        self.assertNotIn("body", finding["supporting_metrics"])
        self.assertNotIn("secret", json.dumps(artifact))
        skeleton = initialize_experiment(select_finding(artifact))
        self.assertEqual(skeleton["generation_status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
