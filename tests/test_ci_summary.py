from __future__ import annotations

import contextlib
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from runeconomy.ci import COMMENT_MARKER, CIError, post_pull_request_comment, render_summary
from runeconomy.cli import main


ROOT = Path(__file__).resolve().parents[1]
RESULT = {
    "decision": {
        "status": "regression",
        "reason": "candidate task success is significantly worse",
        "checks": [
            {"name": "task_success_noninferior", "passed": False, "required": True, "observed": {"one_sided_lower_bound": -0.12}},
            {"name": "completion_rate", "passed": True, "required": True},
            {"name": "cached_token_gain", "passed": None, "required": False},
        ],
    },
    "preregistration": {"status": "intact", "record_sha256": "sha256:" + "a" * 64},
}


class _GitHub(BaseHTTPRequestHandler):
    comments: list[dict[str, Any]] = []
    seen: list[tuple[str, str, str | None]] = []

    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        type(self).seen.append(("GET", self.path, self.headers.get("Authorization")))
        self._send(type(self).comments)

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(("POST", self.path, None))
        comment = {"id": len(type(self).comments) + 1, "body": body["body"]}
        type(self).comments.append(comment)
        self._send(comment, 201)

    def do_PATCH(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(("PATCH", self.path, None))
        identifier = int(self.path.rsplit("/", 1)[1])
        for comment in type(self).comments:
            if comment["id"] == identifier:
                comment["body"] = body["body"]
        self._send({"id": identifier})

    def log_message(self, format: str, *args: object) -> None:
        pass


class SummaryTests(unittest.TestCase):
    def test_frozen_summary_lists_gates_and_never_upgrades_claims(self) -> None:
        summary = render_summary(RESULT)
        self.assertEqual(summary["verdict"], "regression")
        self.assertFalse(summary["verified"])
        self.assertEqual(summary["failed_required_checks"], ["task_success_noninferior"])
        markdown = summary["markdown"]
        self.assertTrue(markdown.startswith(COMMENT_MARKER))
        self.assertIn("| task_success_noninferior | yes | fail |", markdown)
        self.assertIn("| cached_token_gain | no | unknown |", markdown)
        self.assertIn("a" * 64, markdown)

    def test_unfrozen_and_compatibility_runs_are_labelled(self) -> None:
        verified = {"decision": {"status": "verified", "checks": []}}
        self.assertIn("Not run under an intact preregistration record", render_summary(verified)["markdown"])
        compatibility = render_summary(verified, mode="compatibility")
        self.assertFalse(compatibility["verified"])
        self.assertIn("makes no verdict claim", compatibility["markdown"])
        with self.assertRaises(CIError):
            render_summary({"no": "decision"})


class CommentTests(unittest.TestCase):
    def test_comment_is_created_once_then_updated_in_place(self) -> None:
        _GitHub.comments = [{"id": 99, "body": "unrelated review comment"}]
        _GitHub.seen = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _GitHub)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with TemporaryDirectory() as raw:
                event = Path(raw) / "event.json"
                event.write_text(json.dumps({"pull_request": {"number": 7}}), encoding="utf-8")
                environment = {
                    "GITHUB_TOKEN": "test-token",
                    "GITHUB_REPOSITORY": "owner/repo",
                    "GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}",
                    "GITHUB_EVENT_PATH": str(event),
                }
                first = post_pull_request_comment(COMMENT_MARKER + "\nfirst", environment)
                second = post_pull_request_comment(COMMENT_MARKER + "\nsecond", environment)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(first["action"], "created")
        self.assertEqual(second, {"action": "updated", "pull_request": 7, "comment_id": first["comment_id"]})
        self.assertEqual(len(_GitHub.comments), 2)
        self.assertTrue(_GitHub.comments[-1]["body"].endswith("second"))
        self.assertEqual(_GitHub.seen[0], ("GET", "/repos/owner/repo/issues/7/comments?per_page=100&page=1", "Bearer test-token"))

    def test_missing_pull_request_context_is_an_error(self) -> None:
        with self.assertRaises(CIError):
            post_pull_request_comment("x", {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"})

    def test_cli_json_contract(self) -> None:
        with TemporaryDirectory() as raw:
            result = Path(raw) / "result.json"
            result.write_text(json.dumps(RESULT), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["ci-summary", str(result), "--output", str(Path(raw) / "s.md"), "--json"]), 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["output_schema"], "runeconomy.cli.ci-summary.v1")
            self.assertEqual((Path(raw) / "s.md").read_text(encoding="utf-8"), payload["markdown"])


class ActionDefinitionTests(unittest.TestCase):
    def test_action_runs_frozen_experiment_and_posts_summary(self) -> None:
        text = (ROOT / "action.yml").read_text(encoding="utf-8")
        for required in (
            "using: composite",
            "--require-frozen",
            "runeconomy experiment run",
            "runeconomy ci-summary",
            "--github-comment",
            "GITHUB_STEP_SUMMARY",
            "require-verified",
        ):
            self.assertIn(required, text)
        # Inputs reach shell steps only through env, never by template expansion inside scripts.
        for block in text.split("run: |")[1:]:
            script = block.split("\n    - name:")[0]
            self.assertNotIn("${{", script)
        workflow = (ROOT / "examples" / "github-action" / "runeconomy-verify.yml").read_text(encoding="utf-8")
        self.assertIn("pull_request:", workflow)
        self.assertIn("pull-requests: write", workflow)


if __name__ == "__main__":
    unittest.main()
