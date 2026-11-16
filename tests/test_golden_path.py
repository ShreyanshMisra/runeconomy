from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.cli import main


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "examples" / "golden_path" / "openinference-evidence.json"
SERVER = ROOT / "examples" / "golden_path" / "mock_vllm.py"


def _wait_for_server(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"mock server exited with {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", 8765), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError("mock server did not start")


class GoldenPathTests(unittest.TestCase):
    def test_import_init_validate_run_report(self) -> None:
        server = subprocess.Popen(
            [sys.executable, str(SERVER)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _wait_for_server(server)
            with TemporaryDirectory() as directory:
                root = Path(directory)
                evidence = root / "evidence.json"
                workspace = root / "experiment"
                results = root / "results"
                report = root / "report.html"
                self.assertEqual(
                    main(["import", str(FIXTURE), "--output", str(evidence)]), 0
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
                result = json.loads((results / "result.json").read_text())
                html = report.read_text()

            self.assertEqual(result["decision"]["status"], "verified")
            self.assertEqual(
                result["summary"]["effects"]["cached_tokens"]["candidate_minus_baseline_mean"],
                8.0,
            )
            self.assertIn("Verdict: Verified", html)
            self.assertNotIn("Stable instructions", html)
        finally:
            server.terminate()
            try:
                server.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
