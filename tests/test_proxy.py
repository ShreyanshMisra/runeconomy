"""Exercise the proxy wire contract and the public capture-to-workspace CLI."""

from __future__ import annotations

import http.client
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

from runeconomy.cli import main
from runeconomy.proxy import group_workflows, load_session


FIRST = b'data: {"choices":[{"delta":{"content":"O"}}]}\r\n\r\n'
SECOND = b'data: {"choices":[{"delta":{"content":"K"}}]}\n\n'
DONE = b"data: [DONE]\n\n"


def _tokens(request):
    tokens = [1]
    for message in request["messages"]:
        tokens.extend(message["content"].encode())
        tokens.append(2)
    if request.get("add_generation_prompt", True):
        tokens.append(3)
    return tokens


def _request(timestamp, **extra):
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": f"Request time: 2026-11-25T12:00:{timestamp:02d}Z"},
            {"role": "system", "content": "Stable instructions. Always reply with OK."},
            {"role": "user", "content": "Please confirm."},
        ],
        "temperature": 0,
        **extra,
    }


def _redact_fixture(part, value):
    if part == "request_body":
        value.pop("metadata", None)
    return value


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload, content_type="application/json", status=200):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        self.server.calls.append(self.path)
        if self.path == "/v1/models":
            self._send({"data": [{"id": "test-model", "owned_by": "vllm"}]})
        elif self.path == "/version":
            self._send({"version": "test-vllm"})
        elif self.path == "/metrics":
            self._send(
                b'vllm:cache_config_info{block_size="16",enable_prefix_caching="True"} 1\n',
                "text/plain",
            )
        else:
            self._send({"error": "missing"}, status=404)

    def do_POST(self):  # noqa: N802
        self.server.calls.append(self.path)
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(raw)
        if self.path.endswith("/render"):
            # Exercise the chat /tokenize fallback.
            self._send({"error": "missing"}, status=404)
            return
        if self.path == "/tokenize":
            self._send({"tokens": _tokens(request)})
            return
        self.server.forwarded.append((raw, self.headers.get("Authorization")))
        usage = {
            "prompt_tokens": len(_tokens(request)) + self.server.prompt_count_offset,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        if not request.get("stream"):
            self._send({
                "model": "test-model",
                "choices": [{"message": {"content": "OK"}}],
                "usage": usage,
            })
            return
        trailer = b'data: ' + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n"
        rest = SECOND + trailer + DONE
        self.server.sse = FIRST + rest
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(self.server.sse)))
        self.send_header("X-Request-Id", "upstream-stream")
        self.end_headers()
        # Stop mid-event until the test observes those bytes on the client.
        self.wfile.write(FIRST[:13])
        self.wfile.flush()
        if self.server.release_stream.wait(5) and not self.server.interrupt_stream:
            self.wfile.write(FIRST[13:] + rest)


@contextmanager
def _upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    server.calls = []
    server.forwarded = []
    server.release_stream = threading.Event()
    server.interrupt_stream = False
    server.prompt_count_offset = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.release_stream.set()
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def _capture(upstream, record, *extra, close_timeout=12):
    argv = [
        sys.executable, "-m", "runeconomy", "proxy",
        "--upstream", upstream, "--record", str(record), "--port", "0", *extra,
    ]
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            assert process.stdout is not None
            if not select.select([process.stdout], [], [], 8)[0]:
                raise AssertionError("proxy did not report readiness")
            line = process.stdout.readline().decode()
            if "OPENAI_BASE_URL=" not in line:
                stdout, stderr = process.communicate(timeout=5)
                raise AssertionError(f"proxy failed to start: {line}{stdout!r} {stderr!r}")
            yield line.split("OPENAI_BASE_URL=", 1)[1].strip()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=close_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
        if process.returncode:
            raise AssertionError(f"proxy exited {process.returncode}: {stdout!r} {stderr!r}")


@contextmanager
def _connection(base_url, timeout=3):
    address = urlsplit(base_url)
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=timeout)
    try:
        yield connection
    finally:
        connection.close()


def _prepare_replay(test, root, record, context):
    evidence = root / "replay-evidence.json"
    workspace = root / "experiment"
    report = StringIO()
    with redirect_stdout(report):
        test.assertEqual(main([
            "import", str(record), "--type", "proxy", "--replay-context",
            str(context), "--output", str(evidence),
        ]), 0)
    payload = json.loads(evidence.read_text())
    test.assertTrue(payload["capability_assessment"]["runnable_experiment"], report.getvalue())
    finding = next(
        item for item in payload["analysis"]["findings"]
        if item["rule_id"] == "prefix_divergence.dynamic_metadata"
    )
    test.assertIn(finding["finding_id"], report.getvalue())
    with redirect_stdout(StringIO()):
        test.assertEqual(main([
            "experiment", "init", "--from-evidence", str(evidence),
            "--finding-id", finding["finding_id"], "--output", str(workspace),
        ]), 0)
    output = StringIO()
    with redirect_stdout(output):
        test.assertEqual(main(["experiment", "validate", str(workspace), "--json"]), 0)
    test.assertEqual(json.loads(output.getvalue())["status"], "runnable")
    return json.loads((workspace / "manifest.json").read_text())


def _write_replay_context(root):
    context = root / "context.json"
    context.write_text(json.dumps({
        "experiment_template_id": "prompt.move_dynamic_metadata",
        "candidate_strategy": "move_volatile_messages_after_system_block",
        "backend": {
            "model_revision": "test-model-revision",
            "tokenizer_revision": "test-tokenizer-revision",
            "chat_template_digest": "sha256:" + "a" * 64,
            "hardware": "loopback test fixture",
        },
        "execution": {"repetitions": 2, "require_evaluation": True},
        "acceptance": {"primary_metric": "client_ttft_ms"},
        "evaluator": {"kind": "regex", "pattern": "OK", "full_match": True},
    }))
    return context


def _run_concurrent_streams(test, server, base_url, headers):
    ready = [threading.Event() for _ in headers]
    errors = []

    def request(index, request_headers):
        try:
            with _connection(base_url, timeout=8) as connection:
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    json.dumps(_request(index + 1, stream=True)),
                    request_headers,
                )
                response = connection.getresponse()
                test.assertEqual(response.status, 200)
                test.assertEqual(response.read(13), FIRST[:13])
                ready[index].set()
                response.read()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test thread
            errors.append(exc)
            ready[index].set()

    threads = [
        threading.Thread(target=request, args=(index, request_headers), daemon=True)
        for index, request_headers in enumerate(headers)
    ]
    for thread in threads:
        thread.start()
    for event in ready:
        test.assertTrue(event.wait(5), "concurrent request did not reach the upstream")
    server.release_stream.set()
    for thread in threads:
        thread.join(8)
        test.assertFalse(thread.is_alive(), "concurrent request did not finish")
    if errors:
        raise errors[0]


class ProxyTests(unittest.TestCase):
    def test_stream_forwards_partial_event_and_keeps_telemetry_off_path(self):
        with TemporaryDirectory() as directory, _upstream() as (server, upstream):
            record = Path(directory) / "session.jsonl"
            with _capture(
                upstream, record, "--redact", "tests.test_proxy:_redact_fixture"
            ) as base_url, _connection(base_url) as connection:
                request = json.dumps(_request(
                    1, stream=True, api_key="private-body-value",
                    metadata={"account": "private-hook-value"},
                )).encode()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer private-header-value",
                    "Cookie": "private-cookie-value",
                    "X-Internal-Secret": "private-custom-header-value",
                }
                connection.request("POST", "/v1/chat/completions", request, headers)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("X-Request-Id"), "upstream-stream")
                first = response.read(13)
                self.assertEqual(first, FIRST[:13])
                self.assertEqual(server.calls, ["/v1/chat/completions"])
                self.assertEqual(len(record.read_text().splitlines()), 1)
                server.release_stream.set()
                self.assertEqual(first + response.read(), server.sse)
            self.assertEqual(server.forwarded, [(request, headers["Authorization"])])
            session = load_session(record)
            self.assertTrue(session.close["clean"])
            self.assertEqual(len(session.exchanges), 1)
            exchange = session.exchanges[0]
            telemetry = exchange["telemetry"]
            self.assertIsNotNone(telemetry["client_ttft_ms"])
            self.assertIsNotNone(telemetry["itl_ms"])
            self.assertIsNotNone(telemetry["tpot_ms"])
            self.assertEqual(telemetry["usage"]["cached_tokens"], 4)
            resolution = session.resolutions[exchange["exchange_id"]]
            self.assertEqual(resolution["prompt_token_ids"], _tokens(json.loads(request)))
            self.assertEqual(resolution["source"], "vllm_tokenize_chat")
            self.assertEqual(exchange["request"]["body"]["api_key"], "[REDACTED]")
            self.assertNotIn("metadata", exchange["request"]["body"])
            self.assertNotIn("private-", record.read_text())
            self.assertEqual(record.stat().st_mode & 0o777, 0o600)
            schema = files("runeconomy").joinpath("data", "proxy-session-0.1.schema.json")
            validator = Draft202012Validator(json.loads(schema.read_text(encoding="utf-8")))
            for line in record.read_text().splitlines():
                validator.validate(json.loads(line))

    def test_capture_import_init_and_validate_with_operator_context(self):
        with TemporaryDirectory() as directory, _upstream() as (_server, upstream):
            root = Path(directory)
            record = root / "session.jsonl"
            with _capture(upstream, record) as base_url, _connection(base_url) as connection:
                for second in (1, 2):
                    connection.request("POST", "/v1/chat/completions", json.dumps(_request(second)))
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    response.read()
            evidence = root / "evidence.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["import", str(record), "--output", str(evidence)]), 0)
            payload = json.loads(evidence.read_text())
            self.assertEqual(payload["proxy_session"]["workflow_count"], 1)
            self.assertEqual(payload["proxy_session"]["ambiguous_workflow_count"], 0)
            self.assertFalse(payload["trace"]["workflow_grouping"]["ambiguous"])
            self.assertEqual(
                payload["trace"]["evidence_quality"]["fields"]["workflow_grouping"]["grade"],
                "weakly_inferred",
            )
            self.assertEqual(payload["trace"]["spans"], [])
            self.assertIn(
                "prefix_divergence.dynamic_metadata",
                {item["rule_id"] for item in payload["analysis"]["findings"]},
            )
            self.assertFalse(payload["capability_assessment"]["runnable_experiment"])
            context = _write_replay_context(root)
            manifest = _prepare_replay(self, root, record, context)
            variants = manifest["cases"][0]["variants"]
            self.assertIn("{{RUNECONOMY_TRIAL_ID}}", variants["baseline"]["request"]["messages"][0]["content"])
            self.assertEqual(
                variants["candidate"]["request"]["messages"][0],
                variants["baseline"]["request"]["messages"][1],
            )

    def test_concurrent_header_correlated_requests_remain_distinct_workflows(self):
        with TemporaryDirectory() as directory, _upstream() as (server, upstream):
            root = Path(directory)
            record = root / "session.jsonl"
            headers = [
                {"X-RunEconomy-Workflow-Id": "agent-one"},
                {"X-RunEconomy-Workflow-Id": "agent-two"},
            ]
            with _capture(upstream, record, "--token-resolution", "off") as base_url:
                _run_concurrent_streams(self, server, base_url, headers)

            workflows = group_workflows(load_session(record))
            self.assertEqual(
                {(item.workflow_id, item.grouping) for item in workflows},
                {("agent-one", "runeconomy_header"), ("agent-two", "runeconomy_header")},
            )
            self.assertTrue(all(not item.ambiguous for item in workflows))
            evidence = root / "evidence.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["import", str(record), "--output", str(evidence)]), 0)
            payload = json.loads(evidence.read_text())
            self.assertEqual(payload["proxy_session"]["workflow_count"], 2)
            self.assertEqual(payload["proxy_session"]["ambiguous_workflow_count"], 0)
            self.assertEqual(payload["proxy_session"]["warnings"], [])
            self.assertTrue(all(
                run["trace"]["evidence_quality"]["fields"]["workflow_grouping"]["grade"]
                == "directly_observed"
                for run in payload["runs"]
            ))

    def test_concurrent_uncorrelated_requests_warn_and_block_workspace(self):
        with TemporaryDirectory() as directory, _upstream() as (server, upstream):
            root = Path(directory)
            record = root / "session.jsonl"
            with _capture(upstream, record) as base_url:
                _run_concurrent_streams(self, server, base_url, [{}, {}])

            workflow, = group_workflows(load_session(record))
            self.assertTrue(workflow.ambiguous)
            self.assertEqual(workflow.overlap_count, 1)
            self.assertEqual(len(workflow.overlapping_exchange_ids), 2)

            context = _write_replay_context(root)
            evidence = root / "evidence.json"
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([
                    "import", str(record), "--replay-context", str(context),
                    "--output", str(evidence),
                ]), 0)
            self.assertIn("WARNING:", output.getvalue())
            self.assertIn("overlapping in-flight requests", output.getvalue())

            payload = json.loads(evidence.read_text())
            grouping = payload["trace"]["workflow_grouping"]
            self.assertTrue(grouping["ambiguous"])
            self.assertEqual(grouping["overlap_count"], 1)
            self.assertEqual(
                payload["trace"]["evidence_quality"]["fields"]["workflow_grouping"]["grade"],
                "unresolved",
            )
            assessment = payload["capability_assessment"]
            self.assertTrue(assessment["workflow_grouping"]["ambiguous"])
            self.assertFalse(assessment["runnable_experiment"])
            self.assertEqual(
                assessment["capabilities"]["static_request_replay"]["status"],
                "missing",
            )
            finding = next(
                item for item in payload["analysis"]["findings"]
                if item["rule_id"] == "prefix_divergence.dynamic_metadata"
            )
            self.assertTrue(finding["capture_workflow_grouping"]["ambiguous"])
            self.assertIn("capture workflow is ambiguous", finding["unsupported_reason"])

            workspace = root / "experiment"
            error = StringIO()
            with redirect_stderr(error):
                self.assertEqual(main([
                    "experiment", "init", "--from-evidence", str(evidence),
                    "--finding-id", finding["finding_id"], "--output", str(workspace),
                ]), 2)
            self.assertIn("Overlapping in-flight requests", error.getvalue())
            self.assertIn("X-RunEconomy-Workflow-Id", error.getvalue())
            self.assertFalse(workspace.exists())

    def test_short_upstream_stream_is_recorded_as_interrupted(self):
        with TemporaryDirectory() as directory, _upstream() as (server, upstream):
            record = Path(directory) / "session.jsonl"
            server.interrupt_stream = True
            server.release_stream.set()
            with _capture(upstream, record) as base_url, _connection(base_url) as connection:
                connection.request("POST", "/v1/chat/completions", json.dumps(_request(1, stream=True)))
                response = connection.getresponse()
                with self.assertRaises(http.client.IncompleteRead):
                    response.read()
            session = load_session(record)
            self.assertEqual(session.exchanges[0]["outcome"], "upstream_interrupted")
            self.assertFalse(session.resolutions)

    def test_traceparent_groups_requests_without_asserting_dependencies(self):
        with TemporaryDirectory() as directory, _upstream() as (_server, upstream):
            root = Path(directory)
            record = root / "session.jsonl"
            trace_id = "1234567890abcdef" * 2
            with _capture(upstream, record, "--token-resolution", "off") as base_url:
                with _connection(base_url) as connection:
                    for parent in ("a" * 16, "b" * 16):
                        connection.request("POST", "/v1/chat/completions", json.dumps(_request(1)), {
                            "traceparent": f"00-{trace_id}-{parent}-01",
                        })
                        connection.getresponse().read()
            workflow, = group_workflows(load_session(record), idle_gap_ms=0.001)
            self.assertEqual(workflow.workflow_id, trace_id)
            self.assertEqual(workflow.grouping, "w3c_traceparent")
            evidence = root / "evidence.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(main([
                    "import", str(record), "--idle-gap-ms", "0.001", "--output", str(evidence),
                ]), 0)
            trace = json.loads(evidence.read_text())["trace"]
            self.assertEqual(len(trace["requests"]), 2)
            self.assertEqual(trace["spans"], [])

    def test_token_mismatch_keeps_observed_count_and_explicit_dependencies(self):
        with TemporaryDirectory() as directory, _upstream() as (server, upstream):
            root = Path(directory)
            record = root / "session.jsonl"
            server.prompt_count_offset = 1
            with _capture(upstream, record, "--payload-capture", "digest_only") as base_url:
                with _connection(base_url) as connection:
                    for span, depends in (("first", "none"), ("second", '["first"]')):
                        connection.request("POST", "/v1/chat/completions", json.dumps(_request(1)), {
                            "X-RunEconomy-Workflow-Id": "explicit-workflow",
                            "X-RunEconomy-Span-Id": span,
                            "X-RunEconomy-Depends-On": depends,
                        })
                        connection.getresponse().read()
            session = load_session(record)
            self.assertEqual(group_workflows(session)[0].workflow_id, "explicit-workflow")
            self.assertEqual(session.close["counters"]["token_resolution_unknown"], 2)
            self.assertNotIn("Stable instructions", record.read_text())
            evidence = root / "evidence.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(main([
                    "import", str(record), "--trace-id", "explicit-workflow",
                    "--output", str(evidence),
                ]), 0)
            trace = json.loads(evidence.read_text())["trace"]
            self.assertEqual([span["depends_on"] for span in trace["spans"]], [[], ["first"]])
            for request in trace["requests"]:
                self.assertIsNone(request["prompt_token_ids"])
                self.assertEqual(request["prompt_token_count"], len(_tokens(_request(1))) + 1)

    def test_attempt_wrapper_preserves_child_status_and_exit_code(self):
        with TemporaryDirectory() as directory, _upstream() as (_server, upstream):
            root = Path(directory)
            status, trace = root / "status.json", root / "trace.json"
            task = root / "task.json"
            task.write_text("{}")
            environment = {
                **os.environ,
                "RUNECONOMY_TASK_ID": "test-task",
                "RUNECONOMY_TASK_PATH": str(task),
                "RUNECONOMY_OUTPUT_PATH": str(root / "output.json"),
                "RUNECONOMY_ATTEMPT_DIR": str(root),
                "RUNECONOMY_STATUS_PATH": str(status),
                "RUNECONOMY_TRACE_PATH": str(trace),
                "RUNECONOMY_VARIANT": "baseline",
                "RUNECONOMY_REPETITION": "0",
                "RUNECONOMY_SEED": "1",
            }
            agent = (
                "import os, sys, urllib.request\n"
                "from runeconomy.agent import Attempt\n"
                "url = os.environ['OPENAI_BASE_URL'] + '/chat/completions'\n"
                "with Attempt.from_environment() as attempt:\n"
                f"    urllib.request.urlopen(url, {json.dumps(_request(1)).encode()!r}).read()\n"
                "    attempt.application_failure(stage='test', code='fixture_failure')\n"
                "sys.exit(7)\n"
            )
            result = subprocess.run([
                sys.executable, "-m", "runeconomy", "proxy", "--upstream", upstream,
                "--attempt", "--", sys.executable, "-c", agent,
            ], env=environment, capture_output=True, timeout=15, check=False)
            self.assertEqual(result.returncode, 7, result.stderr.decode())
            terminal = json.loads(status.read_text())
            self.assertEqual(terminal["status"], "application_failure")
            self.assertEqual(terminal["provenance"]["code"], "fixture_failure")
            self.assertTrue(json.loads(trace.read_text())["resourceSpans"])
            self.assertEqual(load_session(root / "proxy-session.jsonl").scope["kind"], "attempt")

    def test_cli_preserves_an_existing_record(self):
        with TemporaryDirectory() as directory:
            record = Path(directory) / "session.jsonl"
            record.write_text("existing evidence")
            error = StringIO()
            with redirect_stderr(error):
                self.assertEqual(main([
                    "proxy", "--upstream", "http://127.0.0.1:8000", "--port", "0",
                    "--record", str(record),
                ]), 2)
            self.assertIn("already exists", error.getvalue())
            self.assertEqual(record.read_text(), "existing evidence")

    @unittest.skipUnless(
        all(os.environ.get(name) for name in (
            "RUNECONOMY_PROXY_LIVE_UPSTREAM", "VLLM_MODEL", "RUNECONOMY_PROXY_REPLAY_CONTEXT",
        )),
        "requires an explicitly configured live proxy replay target",
    )
    def test_live_vllm_prefix_finding_and_replay(self):
        upstream = os.environ["RUNECONOMY_PROXY_LIVE_UPSTREAM"]
        model = os.environ["VLLM_MODEL"]
        context = Path(os.environ["RUNECONOMY_PROXY_REPLAY_CONTEXT"])
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get("VLLM_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        started = time.monotonic()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "session.jsonl"
            headers["X-RunEconomy-Workflow-Id"] = "live-vllm-prefix-check"
            with _capture(upstream, record, close_timeout=150) as base_url:
                with _connection(base_url, timeout=120) as connection:
                    for second in (1, 2):
                        request = _request(second, model=model, max_tokens=8)
                        connection.request("POST", "/v1/chat/completions", json.dumps(request), headers)
                        response = connection.getresponse()
                        response.read()
                        self.assertEqual(response.status, 200)
            _prepare_replay(self, root, record, context)
        self.assertLess(time.monotonic() - started, 600)
