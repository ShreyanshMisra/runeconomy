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
            source.write_text(json.dumps(load_demo_result()), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
