from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy.comparison import run_comparison
from runeconomy.importers import import_openinference
from runeconomy.models import Trace


ROOT = Path(__file__).parents[1]
AGENT = ROOT / "examples" / "reliable_proof" / "agent.py"


class _ReliableProofHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    plan_query = "Which Python function directly contains the unique call 'which('?"
    answer = "_command_identity"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/chat/completions/render":
            self.send_error(404)
            return
        if self.path == "/tokenize":
            self._json({"count": 4, "tokens": [11, 12, 13, 14]})
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        type(self).requests.append(payload)
        schema_name = payload["response_format"]["json_schema"]["name"]
        if schema_name == "repository_search_plan":
            content = json.dumps({"query": self.plan_query})
        else:
            content = json.dumps({"answer": self.answer})
        events = [
            {
                "id": f"chat-{len(self.requests)}",
                "choices": [{"delta": {"content": content}, "finish_reason": None}],
            },
            {
                "id": f"chat-{len(self.requests)}",
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
        self.wfile.write(encoded)

    def _json(self, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class ReliableProofAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        _ReliableProofHandler.requests = []
        _ReliableProofHandler.plan_query = (
            "Which Python function directly contains the unique call 'which('?"
        )
        _ReliableProofHandler.answer = "_command_identity"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ReliableProofHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @staticmethod
    def _task() -> dict:
        return {
            "task_id": "proof-calibration-command-identity",
            "family": "direct_call",
            "question": "Which Python function directly contains the unique call 'which('?",
            "candidate_paths": [
                "src/runeconomy/backends.py",
                "src/runeconomy/comparison.py",
                "src/runeconomy/vllm.py",
            ],
        }

    def _environment(self, directory: Path, task_path: Path) -> dict[str, str]:
        return {
            **os.environ,
            "VLLM_BASE_URL": f"http://127.0.0.1:{self.server.server_port}",
            "RUNECONOMY_TASK_PATH": str(task_path),
            "RUNECONOMY_TASK_ID": "proof-calibration-command-identity",
            "RUNECONOMY_VARIANT": "baseline",
            "RUNECONOMY_REPETITION": "0",
            "RUNECONOMY_SEED": "17",
            "RUNECONOMY_ATTEMPT_DIR": str(directory),
            "RUNECONOMY_OUTPUT_PATH": str(directory / "output.txt"),
            "RUNECONOMY_TRACE_PATH": str(directory / "trace.json"),
            "RUNECONOMY_STATUS_PATH": str(directory / "status.json"),
            "RUNECONOMY_REPO_ROOT": str(ROOT),
        }

    def test_constrained_agent_searches_every_candidate_and_exports_dag(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            task_path = directory / "task.json"
            task_path.write_text(json.dumps(self._task()), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(AGENT), "--metadata-position", "prefix"],
                cwd=ROOT,
                env=self._environment(directory, task_path),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                (directory / "output.txt").read_text().strip(), "_command_identity"
            )
            self.assertEqual(
                json.loads((directory / "status.json").read_text())["status"],
                "completed",
            )
            self.assertEqual(len(_ReliableProofHandler.requests), 2)
            planning_prompt = _ReliableProofHandler.requests[0]["messages"][-1][
                "content"
            ]
            self.assertNotIn("Candidate paths", planning_prompt)
            self.assertNotIn("src/runeconomy/", planning_prompt)
            answer_prompt = _ReliableProofHandler.requests[1]["messages"][-1][
                "content"
            ]
            self.assertIn(
                "ENCLOSING_IDENTIFIER: _command_identity", answer_prompt
            )
            self.assertIn("Copy the ENCLOSING_IDENTIFIER value exactly", answer_prompt)
            for request in _ReliableProofHandler.requests:
                response_format = request["response_format"]
                self.assertEqual(response_format["type"], "json_schema")
                self.assertTrue(response_format["json_schema"]["strict"])
                self.assertFalse(
                    response_format["json_schema"]["schema"][
                        "additionalProperties"
                    ]
                )
            normalized = import_openinference(
                directory / "trace.json", workload="calibration"
            )
            trace = Trace.from_dict(normalized)
            self.assertEqual(normalized["dependency_evidence"], "complete_explicit")
            self.assertEqual(
                [span.kind for span in trace.spans],
                ["inference", "tool", "inference"],
            )
            self.assertEqual(trace.spans[1].depends_on, (trace.spans[0].span_id,))
            self.assertEqual(trace.spans[2].depends_on, (trace.spans[1].span_id,))

    def test_compare_contract_keeps_generated_json_private(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            dataset = directory / "tasks.json"
            dataset.write_text(
                json.dumps({"tasks": [self._task()]}), encoding="utf-8"
            )
            evaluator = directory / "evaluator.py"
            evaluator.write_text(
                """import json, sys
request = json.load(sys.stdin)
success = request['output_text'].strip() == '_command_identity'
json.dump({'success': success, 'score': float(success), 'details': {}}, sys.stdout)
""",
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
                        f"{sys.executable} {AGENT} --metadata-position prefix"
                    ),
                    candidate_command=(
                        f"{sys.executable} {AGENT} --metadata-position suffix"
                    ),
                    dataset_path=dataset,
                    evaluator_path=evaluator,
                    output_path=directory / "result.json",
                    artifacts_dir=directory / "artifacts",
                    workdir=ROOT,
                    order="alternating",
                    min_task_success_rate=0.0,
                    max_task_success_regression=1.0,
                )

            self.assertTrue(
                all(attempt["status"] == "completed" for attempt in result["attempts"])
            )
            serialized = json.dumps(result)
            self.assertNotIn('{"query": "which("}', serialized)
            self.assertNotIn('{"answer": "_command_identity"}', serialized)

    def test_invalid_constrained_query_is_model_output_failure_with_trace(self) -> None:
        _ReliableProofHandler.plan_query = "not copied from the question"
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            dataset = directory / "tasks.json"
            dataset.write_text(
                json.dumps({"tasks": [self._task()]}), encoding="utf-8"
            )
            evaluator = directory / "evaluator.py"
            evaluator.write_text(
                "import json, sys; json.dump({'success': False}, sys.stdout)\n",
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
                        f"{sys.executable} {AGENT} --metadata-position prefix"
                    ),
                    candidate_command=(
                        f"{sys.executable} {AGENT} --metadata-position suffix"
                    ),
                    dataset_path=dataset,
                    evaluator_path=evaluator,
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
                self.assertEqual(attempt["metrics"]["request_count"], 1)
                self.assertEqual(
                    attempt["agent_status"]["provenance"]["stage"],
                    "constrained_output",
                )
            self.assertNotIn(
                "not copied from the question", json.dumps(result)
            )


if __name__ == "__main__":
    unittest.main()
