from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy.comparison import run_comparison
from runeconomy.importers import import_openinference
from runeconomy.models import Trace


ROOT = Path(__file__).parents[1]


class _CompatibilityHandler(BaseHTTPRequestHandler):
    generation_count = 0
    answer_delay_seconds = 0.0
    plan_content = (
        '{"path":"src/runeconomy/privacy.py",'
        '"query":"calls sensitive_export_paths"}'
    )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions/render":
            self.send_error(404)
            return
        if self.path == "/tokenize":
            self._json({"count": 4, "tokens": [11, 12, 13, 14]})
            return
        if self.path == "/v1/chat/completions":
            type(self).generation_count += 1
            messages = payload.get("messages", [])
            is_answer = bool(
                messages
                and isinstance(messages[-1], dict)
                and str(messages[-1].get("content", "")).startswith(
                    "Tool read_repository_excerpt returned:"
                )
            )
            if is_answer and self.answer_delay_seconds:
                time.sleep(self.answer_delay_seconds)
            content = (
                "assert_safe_for_export"
                if is_answer
                else self.plan_content
            )
            events = [
                {
                    "id": f"chat-{self.generation_count}",
                    "choices": [
                        {"delta": {"content": content}, "finish_reason": None}
                    ],
                },
                {
                    "id": f"chat-{self.generation_count}",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    },
                    "metrics": {
                        "time_to_first_token_ms": 2.0,
                        "queue_time_ms": 0.5,
                    },
                },
            ]
            body = "".join(f"data: {json.dumps(item)}\n\n" for item in events)
            body += "data: [DONE]\n\n"
            encoded = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                if not self.answer_delay_seconds:
                    raise
            return
        self.send_error(404)

    def _json(self, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class CompatibilityAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        _CompatibilityHandler.generation_count = 0
        _CompatibilityHandler.answer_delay_seconds = 0.0
        _CompatibilityHandler.plan_content = (
            '{"path":"src/runeconomy/privacy.py",'
            '"query":"calls sensitive_export_paths"}'
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _CompatibilityHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_real_otel_agent_exports_explicit_stateful_dag(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            task = directory / "task.json"
            output = directory / "output.txt"
            trace_path = directory / "trace.json"
            status_path = directory / "status.json"
            task.write_text(
                json.dumps(
                    {
                        "task_id": "heldout-export-safety",
                        "question": (
                            "Which Python function calls sensitive_export_paths "
                            "and raises when findings are present?"
                        ),
                        "candidate_paths": [
                            "src/runeconomy/privacy.py",
                            "src/runeconomy/reports.py",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "VLLM_BASE_URL": f"http://127.0.0.1:{self.server.server_port}",
                "RUNECONOMY_TASK_PATH": str(task),
                "RUNECONOMY_TASK_ID": "heldout-export-safety",
                "RUNECONOMY_VARIANT": "baseline",
                "RUNECONOMY_REPETITION": "0",
                "RUNECONOMY_SEED": "17",
                "RUNECONOMY_ATTEMPT_DIR": str(directory),
                "RUNECONOMY_OUTPUT_PATH": str(output),
                "RUNECONOMY_TRACE_PATH": str(trace_path),
                "RUNECONOMY_STATUS_PATH": str(status_path),
                "RUNECONOMY_REPO_ROOT": str(ROOT),
            }
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "examples" / "compatibility" / "agent.py"),
                    "--metadata-position",
                    "prefix",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                json.loads(status_path.read_text(encoding="utf-8"))["status"],
                "completed",
            )
            self.assertEqual(output.read_text().strip(), "assert_safe_for_export")
            normalized = import_openinference(trace_path, workload="compatibility")
            trace = Trace.from_dict(normalized)
            self.assertEqual(normalized["dependency_evidence"], "complete_explicit")
            self.assertEqual(len(trace.requests), 2)
            self.assertEqual([span.kind for span in trace.spans], [
                "inference",
                "tool",
                "inference",
            ])
            self.assertTrue(all(request.prompt_token_ids for request in trace.requests))
            self.assertTrue(
                all(
                    request.metrics.client_ttft_ms is not None
                    for request in trace.requests
                )
            )
            self.assertEqual(trace.spans[1].depends_on, (trace.spans[0].span_id,))
            self.assertEqual(trace.spans[2].depends_on, (trace.spans[1].span_id,))

    def test_compare_contract_needs_only_documented_attempt_environment(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            dataset = directory / "tasks.json"
            dataset.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "heldout-export-safety",
                                "question": (
                                    "Which Python function calls sensitive_export_paths "
                                    "and raises when findings are present?"
                                ),
                                "candidate_paths": [
                                    "src/runeconomy/privacy.py",
                                    "src/runeconomy/reports.py",
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            agent = ROOT / "examples" / "compatibility" / "agent.py"
            evaluator = ROOT / "examples" / "compatibility" / "evaluator.py"
            with patch.dict(
                os.environ,
                {
                    "VLLM_BASE_URL": (
                        f"http://127.0.0.1:{self.server.server_port}"
                    ),
                    "RUNECONOMY_REPO_ROOT": str(ROOT),
                },
                clear=False,
            ):
                result = run_comparison(
                    baseline_command=(
                        f"{sys.executable} {agent} --metadata-position prefix"
                    ),
                    candidate_command=(
                        f"{sys.executable} {agent} --metadata-position suffix"
                    ),
                    dataset_path=dataset,
                    evaluator_path=evaluator,
                    output_path=directory / "result.json",
                    artifacts_dir=directory / "artifacts",
                    workdir=ROOT,
                    min_task_success_rate=0.0,
                    max_task_success_regression=1.0,
                    order="alternating",
                )
            self.assertTrue(
                all(attempt["status"] == "completed" for attempt in result["attempts"])
            )
            self.assertTrue(
                all(attempt["metrics"]["tool_span_ms"] is not None for attempt in result["attempts"])
            )
            self.assertTrue(
                all(attempt["metrics"]["client_ttft_ms"] is not None for attempt in result["attempts"])
            )
            self.assertNotIn("assert_safe_for_export", json.dumps(result))

    def test_timeout_retains_durable_partial_trace(self) -> None:
        _CompatibilityHandler.answer_delay_seconds = 2.0
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            dataset = directory / "tasks.json"
            dataset.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "heldout-export-safety",
                                "question": (
                                    "Which Python function calls sensitive_export_paths "
                                    "and raises when findings are present?"
                                ),
                                "candidate_paths": ["src/runeconomy/privacy.py"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            agent = ROOT / "examples" / "compatibility" / "agent.py"
            evaluator = ROOT / "examples" / "compatibility" / "evaluator.py"
            artifacts = directory / "artifacts"
            with patch.dict(
                os.environ,
                {
                    "VLLM_BASE_URL": (
                        f"http://127.0.0.1:{self.server.server_port}"
                    ),
                    "RUNECONOMY_REPO_ROOT": str(ROOT),
                },
                clear=False,
            ):
                result = run_comparison(
                    baseline_command=(
                        f"{sys.executable} {agent} --metadata-position prefix"
                    ),
                    candidate_command=(
                        f"{sys.executable} {agent} --metadata-position suffix"
                    ),
                    dataset_path=dataset,
                    evaluator_path=evaluator,
                    output_path=directory / "result.json",
                    artifacts_dir=artifacts,
                    workdir=ROOT,
                    timeout_seconds=0.7,
                    min_task_success_rate=0.0,
                    order="alternating",
                )
            self.assertTrue(
                all(attempt["status"] == "timeout" for attempt in result["attempts"])
            )
            for attempt in result["attempts"]:
                trace_name = attempt["artifacts"]["trace"]
                self.assertEqual(trace_name, "trace.json")
                attempt_dir = artifacts / attempt["artifacts"]["directory"]
                payload = json.loads((attempt_dir / trace_name).read_text())
                spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
                self.assertGreaterEqual(len(spans), 2)

    def test_model_output_failure_keeps_partial_trace_and_safe_provenance(self) -> None:
        _CompatibilityHandler.plan_content = "generated text that is not JSON"
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            dataset = directory / "tasks.json"
            dataset.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "heldout-export-safety",
                                "question": (
                                    "Which Python function calls sensitive_export_paths "
                                    "and raises when findings are present?"
                                ),
                                "candidate_paths": ["src/runeconomy/privacy.py"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "VLLM_BASE_URL": (
                        f"http://127.0.0.1:{self.server.server_port}"
                    ),
                    "RUNECONOMY_REPO_ROOT": str(ROOT),
                },
                clear=False,
            ):
                result = run_comparison(
                    baseline_command=(
                        f"{sys.executable} "
                        f"{ROOT / 'examples/compatibility/agent.py'} "
                        "--metadata-position prefix"
                    ),
                    candidate_command=(
                        f"{sys.executable} "
                        f"{ROOT / 'examples/compatibility/agent.py'} "
                        "--metadata-position suffix"
                    ),
                    dataset_path=dataset,
                    evaluator_path=ROOT / "examples/compatibility/evaluator.py",
                    output_path=directory / "result.json",
                    artifacts_dir=directory / "artifacts",
                    workdir=ROOT,
                    order="alternating",
                    min_task_success_rate=0.0,
                )

            self.assertTrue(
                all(
                    attempt["status"] == "model_output_failure"
                    for attempt in result["attempts"]
                )
            )
            for attempt in result["attempts"]:
                self.assertFalse(attempt["task_success"])
                self.assertEqual(
                    attempt["agent_status"]["provenance"],
                    {
                        "component": "model_output",
                        "stage": "agent_output_contract",
                        "code": "invalid_model_output",
                    },
                )
                self.assertEqual(attempt["metrics"]["request_count"], 1)
                self.assertIsNotNone(attempt["trace_digest"])
            self.assertNotIn("generated text that is not JSON", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
