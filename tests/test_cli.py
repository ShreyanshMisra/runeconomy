from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy.cli import main
from runeconomy.reports import load_demo_result
from tests.test_experiments import _manifest


class CLITests(unittest.TestCase):
    def test_invalid_trace_returns_validation_exit_code(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(main(["analyze", str(path)]), 2)

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
