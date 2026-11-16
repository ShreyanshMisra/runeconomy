from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.cli import main
from runeconomy.tutorial import TutorialError, create_tutorial_server


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 5
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"tutorial server exited with {process.returncode}: {stdout} {stderr}"
            )
        try:
            with urllib.request.urlopen(url, timeout=0.2) as response:
                payload = json.load(response)
            if payload == {"status": "ok", "tutorial_schema_version": "0.1"}:
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError("tutorial server did not become ready")


class GoldenPathTests(unittest.TestCase):
    def test_materialization_is_deterministic_and_refuses_existing_output(self) -> None:
        port = _free_port()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "runeconomy-tutorial"
            second = root / "runeconomy-tutorial-copy"
            self.assertEqual(
                main(["tutorial", "init", str(first), "--port", str(port)]), 0
            )
            self.assertEqual(
                main(["tutorial", "init", str(second), "--port", str(port)]), 0
            )
            self.assertEqual(
                (first / "evidence.json").read_bytes(),
                (second / "evidence.json").read_bytes(),
            )
            metadata = json.loads((first / "tutorial.json").read_text())
            self.assertEqual(metadata["schema_version"], "0.1")
            self.assertEqual(metadata["host"], "127.0.0.1")
            self.assertEqual(metadata["port"], port)
            readme = (first / "README.md").read_text()
            for command in (
                "runeconomy import runeconomy-tutorial/evidence.json",
                "runeconomy experiment init",
                "runeconomy experiment validate experiment/",
                "runeconomy experiment run experiment/",
                "runeconomy experiment report results/result.json",
            ):
                self.assertIn(command, readme)
            self.assertIn("RunEconomy-enriched OpenInference", readme)
            self.assertIn("does not run a model", readme)
            self.assertEqual(
                main(["tutorial", "init", str(first), "--port", str(port)]), 2
            )

    def test_port_in_use_has_clear_failure(self) -> None:
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            port = int(occupied.getsockname()[1])
            with self.assertRaisesRegex(TutorialError, "address already in use"):
                create_tutorial_server(port=port)

    def test_import_init_validate_run_report(self) -> None:
        port = _free_port()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tutorial = root / "runeconomy-tutorial"
            evidence = root / "evidence.json"
            workspace = root / "experiment"
            results = root / "results"
            report = root / "report.html"
            self.assertEqual(
                main(["tutorial", "init", str(tutorial), "--port", str(port)]), 0
            )
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "runeconomy.cli",
                    "tutorial",
                    "serve",
                    "--port",
                    str(port),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                _wait_for_server(server, port)
                self.assertEqual(
                    main(
                        [
                            "import",
                            str(tutorial / "evidence.json"),
                            "--output",
                            str(evidence),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "experiment",
                            "init",
                            "--from-evidence",
                            str(evidence),
                            "--output",
                            str(workspace),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(["experiment", "validate", str(workspace)]), 0
                )
                self.assertEqual(
                    main(
                        [
                            "experiment",
                            "run",
                            str(workspace),
                            "--output",
                            str(results),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "experiment",
                            "report",
                            str(results / "result.json"),
                            "--output",
                            str(report),
                        ]
                    ),
                    0,
                )
            finally:
                server.terminate()
                try:
                    server.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.communicate(timeout=3)

            result = json.loads((results / "result.json").read_text())
            html = report.read_text()
            self.assertEqual(result["decision"]["status"], "verified")
            self.assertEqual(
                result["summary"]["effects"]["cached_tokens"][
                    "candidate_minus_baseline_mean"
                ],
                8.0,
            )
            self.assertIn("Verdict: Verified", html)
            self.assertNotIn("Stable instructions", html)
            for forbidden in (
                "/" + "Users/",
                "modal.direct",
                "Authorization: Bearer",
                "RUNECONOMY_STATUS_PATH",
            ):
                self.assertNotIn(forbidden, html)
                self.assertNotIn(forbidden, (tutorial / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
