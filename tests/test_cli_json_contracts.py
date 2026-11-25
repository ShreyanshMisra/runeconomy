"""Every command's --json output validates against its versioned schema."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import socket
import threading
import unittest
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jsonschema import Draft202012Validator

from runeconomy import cli_contracts
from runeconomy.cli import _build_parser, main
from runeconomy.tutorial import create_tutorial_server


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _leaf_commands(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()) -> dict[str, argparse.ArgumentParser]:
    found: dict[str, argparse.ArgumentParser] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                found.update(_leaf_commands(child, (*prefix, name)))
            return found
    return {" ".join(prefix): parser}


class ContractRegistryTests(unittest.TestCase):
    def test_every_command_has_a_contract_or_a_documented_exemption(self) -> None:
        leaves = _leaf_commands(_build_parser())
        covered = set(cli_contracts.COMMAND_CONTRACTS) | set(cli_contracts.EXEMPT_COMMANDS)
        self.assertEqual(set(leaves), covered)
        self.assertFalse(set(cli_contracts.COMMAND_CONTRACTS) & set(cli_contracts.EXEMPT_COMMANDS))
        for command in cli_contracts.COMMAND_CONTRACTS:
            options = {option for action in leaves[command]._actions for option in action.option_strings}
            self.assertIn("--json", options, command)

    def test_schema_files_are_valid_versioned_and_complete(self) -> None:
        directory = files("runeconomy").joinpath("data", cli_contracts.SCHEMA_DIRECTORY)
        on_disk = {item.name for item in directory.iterdir() if item.name.endswith(".schema.json")}
        expected = {cli_contracts.schema_filename(name) for name in cli_contracts.CONTRACTS}
        self.assertTrue(expected <= on_disk, expected - on_disk)
        referenced = {name for names in cli_contracts.COMMAND_CONTRACTS.values() for name in names}
        self.assertEqual(referenced | {"error"}, set(cli_contracts.CONTRACTS))
        for name in cli_contracts.CONTRACTS:
            schema = cli_contracts.load_schema(name)
            Draft202012Validator.check_schema(schema)
            self.assertEqual(schema["$id"], cli_contracts.contract_id(name))
            self.assertEqual(schema["properties"]["output_schema"], {"const": schema["$id"]})
            self.assertIn("output_schema", schema["required"])
            self.assertNotEqual(schema.get("additionalProperties"), False, "contracts must allow additive keys")


class CommandContractTests(unittest.TestCase):
    """Run commands for real against the deterministic tutorial mock server."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        cls.port = _free_port()
        cls.server = create_tutorial_server(port=cls.port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.tutorial = cls.root / "tutorial"
        cls.evidence = cls.root / "evidence.json"
        cls.workspace = cls.root / "experiment"
        cls.results = cls.root / "results"
        cls.outputs: dict[str, dict[str, Any]] = {}
        steps = {
            "tutorial init": ["tutorial", "init", str(cls.tutorial), "--port", str(cls.port)],
            "import": ["import", str(cls.tutorial / "evidence.json"), "--output", str(cls.evidence)],
            "experiment init": [
                "experiment", "init", "--from-evidence", str(cls.evidence), "--output", str(cls.workspace)
            ],
            "experiment validate": ["experiment", "validate", str(cls.workspace)],
            "experiment plan": ["experiment", "plan", str(cls.workspace)],
            "experiment run": ["experiment", "run", str(cls.workspace), "--output", str(cls.results)],
            "experiment report": [
                "experiment", "report", str(cls.results / "result.json"), "--output", str(cls.root / "report.html")
            ],
        }
        for command, arguments in steps.items():
            code, stdout, stderr = cls._run(arguments + ["--json"])
            if code not in (0, 1):
                raise AssertionError(f"{command} failed ({code}): {stderr}")
            cls.outputs[command] = json.loads(stdout)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls._directory.cleanup()

    @staticmethod
    def _run(arguments: list[str]) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def assertContract(self, name: str, stdout: str) -> dict[str, Any]:
        payload = json.loads(stdout)  # exactly one JSON document, nothing else
        self.assertEqual(payload["output_schema"], cli_contracts.contract_id(name))
        errors = sorted(
            Draft202012Validator(cli_contracts.load_schema(name)).iter_errors(payload),
            key=lambda error: list(error.path),
        )
        self.assertEqual([f"{list(error.path)}: {error.message}" for error in errors], [])
        return payload

    def test_golden_path_commands(self) -> None:
        for command, contract in (
            ("tutorial init", "tutorial-init"),
            ("import", "import"),
            ("experiment init", "experiment-init"),
            ("experiment validate", "experiment-validate"),
            ("experiment plan", "experiment-plan"),
            ("experiment run", "experiment-run"),
            ("experiment report", "experiment-report"),
        ):
            with self.subTest(command=command):
                self.assertContract(contract, json.dumps(self.outputs[command]))
        self.assertEqual(self.outputs["experiment run"]["verdict"], "verified")
        self.assertEqual(self.outputs["experiment validate"]["freeze"]["status"], "not_frozen")

    def test_analyze_and_conversion_commands(self) -> None:
        trace = ROOT / "examples" / "traces" / "repository-unused-tools.json"
        code, stdout, _ = self._run(["analyze", str(trace), "--json"])
        self.assertEqual(code, 0)
        self.assertContract("analyze", stdout)

        otel = ROOT / "tests" / "fixtures" / "framework-traces" / "langchain-otel-synthetic.json"
        ledger = ROOT / "examples" / "connectors" / "prefixwise-ledger.jsonl"
        for arguments in (
            ["import-openinference", str(otel)],
            ["import-prefixwise-jsonl", str(ledger)],
            ["import-ledger-jsonl", str(ledger)],
        ):
            with self.subTest(command=arguments[0]):
                output = self.root / f"{arguments[0]}.json"
                code, stdout, stderr = self._run([*arguments, "--output", str(output), "--json"])
                self.assertEqual(code, 0, stderr)
                payload = self.assertContract("convert", stdout)
                self.assertEqual(payload["command"], arguments[0])

    def test_patch_freeze_and_plan_examples(self) -> None:
        code, stdout, _ = self._run(["experiment", "plan", "--examples", "--json"])
        self.assertEqual(code, 0)
        self.assertContract("experiment-plan-examples", stdout)

        code, stdout, stderr = self._run(
            ["experiment", "patch", str(self.workspace), "--set", "execution.warmup_trials=0", "--dry-run", "--json"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertContract("experiment-patch", stdout)

        frozen = self.root / "frozen-experiment"
        shutil.copytree(self.workspace, frozen)
        code, stdout, stderr = self._run(
            ["experiment", "freeze", str(frozen), "--allow-incomplete-design", "--json"]
        )
        self.assertEqual(code, 0, stderr)
        record = self.assertContract("experiment-freeze", stdout)
        self.assertEqual(record["mode"], "static_request")
        self.assertEqual(record["design_completeness"], "incomplete")
        code, stdout, _ = self._run(["experiment", "validate", str(frozen), "--json"])
        self.assertEqual(self.assertContract("experiment-validate", stdout)["freeze"]["status"], "intact")

    def test_demo_support_bundle_doctor_and_schema(self) -> None:
        code, stdout, _ = self._run(["demo", "--output", str(self.root / "demo.html"), "--json"])
        self.assertEqual(code, 0)
        self.assertContract("demo", stdout)

        code, stdout, stderr = self._run(
            ["support-bundle", str(self.workspace), "--output", str(self.root / "support.zip"), "--json"]
        )
        self.assertEqual(code, 0, stderr)
        self.assertContract("support-bundle", stdout)

        code, stdout, stderr = self._run(["doctor", "--base-url", f"http://127.0.0.1:{self.port}", "--json"])
        self.assertIn(code, (0, 1), stderr)
        self.assertContract("doctor", stdout)

        code, stdout, _ = self._run(["schema", "--json"])
        self.assertEqual(code, 0)
        listing = self.assertContract("schema-list", stdout)
        self.assertEqual({item["name"] for item in listing["contracts"]}, set(cli_contracts.CONTRACTS))
        code, stdout, _ = self._run(["schema", "experiment-freeze"])
        self.assertEqual(json.loads(stdout)["$id"], "runeconomy.cli.experiment-freeze.v1")

    def test_mocked_compare_and_capture(self) -> None:
        result = {"decision": {"status": "inconclusive"}}
        with patch("runeconomy.cli.run_comparison", return_value=result):
            code, stdout, _ = self._run(
                [
                    "compare", "--baseline", "true", "--candidate", "true",
                    "--dataset", str(self.evidence), "--evaluator", str(self.evidence),
                    "--output", str(self.root / "compare.json"), "--require-verified", "--json",
                ]
            )
        self.assertEqual(code, 3)
        payload = self.assertContract("compare", stdout)
        self.assertFalse(payload["verified"])

        request = self.root / "request.json"
        request.write_text('{"model":"m","input":"hi"}', encoding="utf-8")
        with patch("runeconomy.cli.VLLMClient") as client:
            client.return_value.capture.return_value.to_dict.return_value = {"output_text": "ok"}
            code, stdout, _ = self._run(
                ["capture-vllm", str(request), "--base-url", "http://127.0.0.1:1", "--json"]
            )
        self.assertEqual(code, 0)
        self.assertContract("capture-vllm", stdout)

    def test_failures_print_error_contract_on_stderr_only(self) -> None:
        for arguments in (
            ["analyze", str(self.root / "missing.json"), "--json"],
            ["experiment", "patch", str(self.workspace), "--set", "artifact_type=\"x\"", "--json"],
            ["experiment", "run", str(self.workspace), "--output", str(self.root / "r2"), "--require-frozen", "--json"],
            ["schema", "no-such-contract", "--json"],
        ):
            with self.subTest(command=arguments[:2]):
                code, stdout, stderr = self._run(arguments)
                self.assertEqual(code, 2, stderr)
                self.assertEqual(stdout, "")
                self.assertContract("error", stderr)

    def test_unreadable_manifest_is_an_invalid_validation_report_not_an_error(self) -> None:
        code, stdout, _ = self._run(
            ["experiment", "validate", str(self.root / "missing.json"), "--json"]
        )
        self.assertEqual(code, 2)
        self.assertEqual(self.assertContract("experiment-validate", stdout)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()


class ProxyContractTests(unittest.TestCase):
    def test_record_session_prints_one_json_object_after_flush(self) -> None:
        import select
        import signal
        import subprocess
        import sys

        with TemporaryDirectory() as raw:
            record = Path(raw) / "session.jsonl"
            argv = [
                sys.executable, "-m", "runeconomy", "proxy", "--upstream", "http://127.0.0.1:9/v1",
                "--record", str(record), "--port", "0", "--token-resolution", "off", "--json",
            ]
            with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                assert process.stderr is not None
                self.assertTrue(select.select([process.stderr], [], [], 8)[0], "proxy did not start")
                self.assertIn("OPENAI_BASE_URL=", process.stderr.readline().decode())
                process.send_signal(signal.SIGTERM)
                stdout, _ = process.communicate(timeout=15)
            payload = json.loads(stdout)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(payload["mode"], "record")
        self.assertEqual(payload["summary"]["exchanges_recorded"], 0)
        self.assertFalse(payload["gaps"])
        errors = list(Draft202012Validator(cli_contracts.load_schema("proxy")).iter_errors(payload))
        self.assertEqual(errors, [])
