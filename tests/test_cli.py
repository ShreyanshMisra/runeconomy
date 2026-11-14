from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy.backends import BackendProbeError
from runeconomy.cli import main
from runeconomy.reports import load_demo_result
from tests.test_experiments import _manifest


class CLITests(unittest.TestCase):
    def test_invalid_trace_returns_validation_exit_code(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{}\n", encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(main(["analyze", str(path)]), 2)
            rendered = stderr.getvalue()
            self.assertIn("What failed:", rendered)
            self.assertIn("Evidence:", rendered)
            self.assertIn("What to do next:", rendered)

    def test_html_report_path_uses_generic_renderer(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "result.json"
            output = Path(directory) / "report.html"
            source.write_text(
                json.dumps(load_demo_result("historical")), encoding="utf-8"
            )
            self.assertEqual(
                main(
                    ["negative-split", "report", str(source), "--output", str(output)]
                ),
                0,
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Generated from recorded RunEconomy aggregate", rendered)
            self.assertNotIn("fixture-backed repository-analysis replay", rendered)

    def test_executable_evaluators_require_explicit_opt_in(self) -> None:
        payload = _manifest()
        payload["cases"][0]["evaluator"] = {
            "kind": "command",
            "argv": ["true"],
        }
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(main(["negative-split", "validate", str(manifest)]), 2)
            self.assertEqual(
                main(
                    [
                        "negative-split",
                        "validate",
                        str(manifest),
                        "--allow-exec-evaluators",
                    ]
                ),
                0,
            )

    def test_require_verified_returns_three_for_inconclusive_result(self) -> None:
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            output = Path(directory) / "result.json"
            manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
            result = {"decision": {"status": "inconclusive"}}
            with (
                patch("runeconomy.cli.VLLMClient"),
                patch("runeconomy.cli.run_experiment", return_value=result),
            ):
                exit_code = main(
                    [
                        "negative-split",
                        "run",
                        str(manifest),
                        "--output",
                        str(output),
                        "--require-verified",
                    ]
                )
            self.assertEqual(exit_code, 3)
            self.assertEqual(
                json.loads(output.read_text())["decision"]["status"], "inconclusive"
            )

    def test_demo_open_uses_local_file_uri(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "demo.html"
            with patch("runeconomy.cli.webbrowser.open") as opened:
                self.assertEqual(
                    main(["demo", "--output", str(output), "--open"]), 0
                )
            opened.assert_called_once_with(output.resolve().as_uri())

    def test_demo_open_failure_preserves_report_and_gives_next_action(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "demo.html"
            stderr = StringIO()
            with (
                patch("runeconomy.cli.webbrowser.open", return_value=False),
                redirect_stderr(stderr),
            ):
                self.assertEqual(
                    main(["demo", "--output", str(output), "--open"]), 0
                )
            self.assertTrue(output.is_file())
            self.assertIn("report was written", stderr.getvalue())
            self.assertIn("What to do next", stderr.getvalue())

    def test_demo_can_render_historical_case_explicitly(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "historical.html"
            self.assertEqual(
                main(
                    [
                        "demo",
                        "--case",
                        "historical",
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("Runner decision:</strong> inconclusive", rendered)

    def test_doctor_json_does_not_echo_base_url_or_api_key(self) -> None:
        capabilities = {
            "backend": "vllm",
            "backend_version": "0.33.0",
            "model_ids": [],
            "experiment_modes": [],
            "readiness": {"status": "ready"},
        }
        with (
            patch.dict("os.environ", {"SECRET_TEST_KEY": "do-not-print"}),
            patch("runeconomy.cli.discover_vllm_capabilities") as discover,
            patch("builtins.print") as output,
        ):
            discover.return_value.to_dict.return_value = capabilities
            self.assertEqual(
                main(
                    [
                        "doctor",
                        "--base-url",
                        "https://private.example/path?token=secret",
                        "--api-key-env",
                        "SECRET_TEST_KEY",
                        "--format",
                        "json",
                    ]
                ),
                0,
            )
        discover.assert_called_once_with(
            "https://private.example/path?token=secret",
            api_key="do-not-print",
            timeout_seconds=10.0,
        )
        rendered = output.call_args.args[0]
        self.assertNotIn("do-not-print", rendered)
        self.assertNotIn("private.example", rendered)

    def test_doctor_blocked_is_structured_and_returns_one(self) -> None:
        capabilities = {
            "backend": "vllm",
            "backend_version": None,
            "model_ids": [],
            "experiment_modes": [],
            "readiness": {
                "status": "blocked",
                "summary": "No safe mode is available.",
                "supported_modes": [],
                "blockers": ["No generation API."],
                "next_actions": ["Expose `/v1/chat/completions` and rerun doctor."],
            },
        }
        with (
            patch("runeconomy.cli.discover_vllm_capabilities") as discover,
            patch("builtins.print") as output,
        ):
            discover.return_value.to_dict.return_value = capabilities
            exit_code = main(
                ["doctor", "--base-url", "http://localhost:8000", "--format", "json"]
            )
        self.assertEqual(exit_code, 1)
        rendered = json.loads(output.call_args.args[0])
        self.assertEqual(rendered["readiness"]["status"], "blocked")
        self.assertTrue(rendered["readiness"]["next_actions"])

    def test_doctor_failure_json_is_safe_and_actionable(self) -> None:
        error = BackendProbeError(
            "the server rejected the metadata request with HTTP 401",
            category="authentication",
            next_action="Set the API key and rerun doctor.",
        )
        with (
            patch.dict("os.environ", {"SECRET_TEST_KEY": "do-not-print"}),
            patch("runeconomy.cli.discover_vllm_capabilities", side_effect=error),
            patch("builtins.print") as output,
        ):
            exit_code = main(
                [
                    "doctor",
                    "--base-url",
                    "https://private.example/path?token=secret",
                    "--api-key-env",
                    "SECRET_TEST_KEY",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(exit_code, 2)
        rendered = json.loads(output.call_args.args[0])
        self.assertEqual(rendered["error"]["category"], "authentication")
        text = json.dumps(rendered)
        self.assertNotIn("do-not-print", text)
        self.assertNotIn("private.example", text)
        self.assertIn("next_action", rendered["error"])

    def test_doctor_human_output_is_accessible_without_color(self) -> None:
        capability = {
            "status": "supported",
            "evidence": "advertised by /openapi.json",
            "detail": None,
            "required": False,
            "next_action": "No action required.",
        }
        capabilities = {
            "backend": "vllm",
            "backend_version": "0.33.0",
            "model_ids": ["example/model"],
            **{
                key: dict(capability)
                for key in (
                    "responses",
                    "chat_completions",
                    "responses_render",
                    "chat_completions_render",
                    "tokenize",
                    "cache_telemetry",
                    "per_request_timing",
                    "cache_reset",
                    "cache_salt",
                )
            },
            "experiment_modes": [
                {
                    "mode": "warm-cache",
                    "status": "supported",
                    "supported": True,
                    "safe": True,
                    "note": "Measures normal cache reuse.",
                }
            ],
            "readiness": {
                "status": "ready",
                "summary": "Safe modes available: warm-cache.",
                "supported_modes": ["warm-cache"],
                "blockers": [],
                "next_actions": ["Pin backend identity, then validate the manifest."],
            },
        }
        stdout = StringIO()
        with (
            patch.dict("os.environ", {"NO_COLOR": "1"}),
            patch("runeconomy.cli.discover_vllm_capabilities") as discover,
            redirect_stdout(stdout),
        ):
            discover.return_value.to_dict.return_value = capabilities
            self.assertEqual(
                main(["doctor", "--base-url", "http://localhost:8000"]), 0
            )
        rendered = stdout.getvalue()
        self.assertIn("RunEconomy doctor · READY", rendered)
        self.assertIn("What to do next", rendered)
        self.assertIn("supported", rendered)
        self.assertNotIn("\x1b[", rendered)
        self.assertNotIn("localhost", rendered)

    def test_doctor_json_alias_preserves_machine_output(self) -> None:
        capabilities = {
            "schema_version": "0.2",
            "backend": "vllm",
            "backend_version": "0.33.0",
            "model_ids": [],
            "experiment_modes": [],
            "readiness": {"status": "ready"},
        }
        stdout = StringIO()
        with (
            patch("runeconomy.cli.discover_vllm_capabilities") as discover,
            redirect_stdout(stdout),
        ):
            discover.return_value.to_dict.return_value = capabilities
            self.assertEqual(
                main(["doctor", "--base-url", "http://localhost:8000", "--json"]),
                0,
            )
        self.assertEqual(json.loads(stdout.getvalue()), capabilities)

    def test_compare_writes_result_and_honors_require_verified(self) -> None:
        result = {"decision": {"status": "inconclusive"}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tasks.json"
            evaluator = root / "evaluator.py"
            output = root / "result.json"
            dataset.write_text('{"tasks": [{"task_id": "one"}]}', encoding="utf-8")
            evaluator.write_text("# fixture\n", encoding="utf-8")
            with patch("runeconomy.cli.run_comparison", return_value=result) as run:
                exit_code = main(
                    [
                        "compare",
                        "--baseline",
                        "python baseline.py",
                        "--candidate",
                        "python candidate.py",
                        "--dataset",
                        str(dataset),
                        "--evaluator",
                        str(evaluator),
                        "--output",
                        str(output),
                        "--require-verified",
                    ]
                )
            self.assertEqual(exit_code, 3)
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(run.call_args.kwargs["order"], "randomized")
            self.assertEqual(
                run.call_args.kwargs["artifacts_dir"], root / "result-artifacts"
            )


if __name__ == "__main__":
    unittest.main()
