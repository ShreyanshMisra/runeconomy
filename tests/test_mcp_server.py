from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from runeconomy.mcp_server import TOOL_SPECS, RunEconomyMCPServer

from tests.test_comparison import AGENT, EVALUATOR
from tests.test_planning import _complete_planning


class _Client:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "runeconomy", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.next_id = 0

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        assert response["id"] == self.next_id, response
        return response

    def notify(self, method: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.process.stdin.flush()

    def tool(self, name: str, **arguments: Any) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})["result"]

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        self.process.wait(timeout=10)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


class MCPProtocolTests(unittest.TestCase):
    def test_initialize_list_and_unknown_method(self) -> None:
        server = RunEconomyMCPServer()
        init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
        self.assertEqual(init["result"]["protocolVersion"], "2024-11-05")
        self.assertIsNone(server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertEqual({tool["name"] for tool in tools}, set(TOOL_SPECS))
        read_only = {tool["name"] for tool in tools if tool["annotations"]["readOnlyHint"]}
        self.assertEqual(
            read_only,
            {"doctor", "analyze", "validate_experiment", "plan_experiment", "inspect_result"},
        )
        self.assertEqual(server.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"], -32601)
        bad = server.handle(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "analyze", "arguments": {"x": 1}}}
        )
        self.assertTrue(bad["result"]["isError"])

    def test_wrongly_typed_arguments_are_tool_errors(self) -> None:
        server = RunEconomyMCPServer()
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "patch_experiment",
                    "arguments": {"workspace": "experiment", "set": ["acceptance.min_task_success_rate=0.9"]},
                },
            }
        )
        self.assertNotIn("error", response)
        self.assertTrue(response["result"]["isError"])
        self.assertIn("set must be of type object", response["result"]["structuredContent"]["error"])


class MCPGoldenPathTests(unittest.TestCase):
    def test_agent_drives_frozen_experiment_through_mcp_with_approval(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
            (root / "evaluator.py").write_text(textwrap.dedent(EVALUATOR), encoding="utf-8")
            (root / "tasks.json").write_text(
                json.dumps({"tasks": [{"task_id": f"t{i}", "expected": f"v{i}"} for i in range(40)]}),
                encoding="utf-8",
            )
            workspace = str(root / "experiment")
            agent = f"{sys.executable} {root / 'agent.py'}"
            client = _Client()
            try:
                client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
                client.notify("notifications/initialized")
                init = client.tool(
                    "init_experiment",
                    output=workspace,
                    mode="agent-process",
                    baseline=agent,
                    candidate=agent,
                    dataset=str(root / "tasks.json"),
                    evaluator=str(root / "evaluator.py"),
                    order="alternating",
                    seed=17,
                    min_task_success_rate=0.8,
                    max_task_success_regression=0.1,
                )
                self.assertFalse(init["isError"], init)
                refused = client.tool("freeze_experiment", workspace=workspace)
                self.assertTrue(refused["isError"])
                self.assertEqual(refused["structuredContent"]["error"]["code"], "runeconomy.freeze.not_ready")

                patched = client.tool(
                    "patch_experiment",
                    workspace=workspace,
                    set={"design.planning": _complete_planning()},
                    reason="declare design",
                )
                self.assertEqual(patched["structuredContent"]["status"], "patched")
                frozen = client.tool("freeze_experiment", workspace=workspace, lock=[str(root / "agent.py")])
                self.assertEqual(frozen["structuredContent"]["status"], "frozen")
                plan = client.tool("plan_experiment", manifest=workspace)
                self.assertEqual(plan["structuredContent"]["state"], "ready_to_freeze")

                run_args = {"manifest": workspace, "output": str(root / "results"), "require_frozen": True}
                estimate = client.tool("run_experiment", **run_args)["structuredContent"]
                self.assertEqual(estimate["status"], "approval_required")
                self.assertFalse(estimate["executed"])
                self.assertEqual(estimate["total_attempts"], 80)
                self.assertEqual(estimate["local_commands"]["baseline"], agent)
                self.assertEqual(estimate["freeze"]["status"], "intact")
                self.assertFalse((root / "results").exists())

                wrong = client.tool("run_experiment", **run_args, approval_token="approve-forged")
                self.assertTrue(wrong["isError"])
                changed = client.tool(
                    "run_experiment", **{**run_args, "require_verified": True}, approval_token=estimate["approval_token"]
                )
                self.assertTrue(changed["isError"], "a token is bound to its exact arguments")

                previous_token = estimate["approval_token"]
                estimate = client.tool("run_experiment", **run_args)["structuredContent"]
                self.assertNotEqual(previous_token, estimate["approval_token"])
                ran = client.tool("run_experiment", **run_args, approval_token=estimate["approval_token"])
                self.assertFalse(ran["isError"], ran)
                self.assertEqual(ran["structuredContent"]["output_schema"], "runeconomy.cli.experiment-run.v1")
                self.assertEqual(ran["structuredContent"]["freeze"]["status"], "intact")
                reused = client.tool("run_experiment", **run_args, approval_token=estimate["approval_token"])
                self.assertTrue(reused["isError"], "tokens are single use")

                result = str(root / "results" / "result.json")
                report = client.tool("render_report", result=result, output=str(root / "report.md"))
                self.assertEqual(report["structuredContent"]["format"], "markdown")
                inspected = client.tool("inspect_result", result=result)["structuredContent"]
                self.assertIn(inspected["verdict"], {"verified", "regression", "inconclusive", "invalid"})
                self.assertEqual(inspected["preregistration"]["record_sha256"], frozen["structuredContent"]["record_sha256"])
            finally:
                client.close()
            manifest = json.loads((root / "experiment" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["design"]["planning"], _complete_planning())


if __name__ == "__main__":
    unittest.main()
