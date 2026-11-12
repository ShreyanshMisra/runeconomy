from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.analysis import analyze_trace
from runeconomy.importers import import_vllm_jsonl
from runeconomy.models import Trace


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


if __name__ == "__main__":
    unittest.main()
