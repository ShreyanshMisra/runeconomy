"""Agent-process pairing: capture one RunEconomy attempt without agent changes.

``runeconomy compare`` and agent-process workspaces launch each attempt with
``RUNECONOMY_*`` variables and read ``RUNECONOMY_TRACE_PATH`` afterwards.
:class:`AttemptCapture` wraps an unmodified agent command. It starts a capture
proxy on an ephemeral local port, points the agent at it through the standard
OpenAI base-URL variables and, once the agent exits, writes the attempt's
trace in the OpenInference form that the comparison runner imports.

The agent still owns its terminal status. It can keep using
:class:`runeconomy.agent.Attempt` for that, and the capture never writes
``RUNECONOMY_STATUS_PATH``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import TraceValidationError
from .contract import GROUPING_ATTEMPT_SCOPE
from .recorder import SessionRecorder
from .server import CaptureProxy
from .session import build_workflow_trace, group_workflows, load_session, trace_to_openinference
from .upstream import UpstreamTarget


SESSION_FILENAME = "proxy-session.jsonl"
FALLBACK_TRACE_FILENAME = "proxy-trace.json"
BASE_URL_VARIABLES = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "RUNECONOMY_PROXY_BASE_URL")


class AttemptCaptureError(RuntimeError):
    """The attempt environment cannot be captured safely."""


def _absolute(environment: Mapping[str, str], name: str) -> Path:
    value = environment.get(name)
    if not value:
        raise AttemptCaptureError(f"{name} is required")
    path = Path(value)
    if not path.is_absolute():
        raise AttemptCaptureError(f"{name} must be an absolute path")
    return path


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> bool:
    """Write JSON to ``path`` only if nothing exists there. Returns success."""

    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass
class AttemptCaptureResult:
    trace_path: Path | None
    wrote_runner_trace: bool
    request_count: int
    session_path: Path
    reason: str | None
    close_summary: dict[str, Any]


class AttemptCapture:
    """Capture every OpenAI-compatible call made during one attempt."""

    def __init__(
        self,
        upstream: str | UpstreamTarget,
        *,
        attempt_dir: Path,
        trace_path: Path,
        task_id: str,
        attempt_id: str | None = None,
        variant: str | None = None,
        repetition: int | None = None,
        session_path: Path | None = None,
        host: str = "127.0.0.1",
        recorder_options: Mapping[str, Any] | None = None,
        upstream_timeout: float | None = 600.0,
    ) -> None:
        self.target = upstream if isinstance(upstream, UpstreamTarget) else UpstreamTarget.parse(upstream)
        self.attempt_dir = attempt_dir
        self.trace_path = trace_path
        resolved_dir = attempt_dir.resolve()
        for label, path in (("trace path", trace_path), ("session path", session_path)):
            if path is None:
                continue
            try:
                path.resolve().relative_to(resolved_dir)
            except ValueError as exc:
                raise AttemptCaptureError(f"{label} must remain inside the attempt directory") from exc
        self.session_path = session_path or attempt_dir / SESSION_FILENAME
        self.task_id = task_id
        self.attempt_id = attempt_id or attempt_dir.name
        options = dict(recorder_options or {})
        options.setdefault("token_resolution", "background")
        self.recorder = SessionRecorder(
            self.session_path,
            upstream=self.target,
            scope={
                "kind": "attempt",
                "grouping": GROUPING_ATTEMPT_SCOPE,
                "task_id": task_id,
                "attempt_id": self.attempt_id,
                "variant": variant,
                "repetition": repetition,
            },
            **options,
        )
        self.proxy = CaptureProxy(
            self.target, self.recorder, host=host, port=0, upstream_timeout=upstream_timeout
        )
        self.result: AttemptCaptureResult | None = None

    @classmethod
    def from_environment(
        cls,
        upstream: str | UpstreamTarget,
        environment: Mapping[str, str] | None = None,
        **options: Any,
    ) -> "AttemptCapture":
        values = os.environ if environment is None else environment
        attempt_dir = _absolute(values, "RUNECONOMY_ATTEMPT_DIR")
        if attempt_dir.is_symlink() or not attempt_dir.is_dir():
            raise AttemptCaptureError(
                "RUNECONOMY_ATTEMPT_DIR must be an existing non-symlink directory"
            )
        trace_path = _absolute(values, "RUNECONOMY_TRACE_PATH")
        task_id = values.get("RUNECONOMY_TASK_ID")
        if not task_id:
            raise AttemptCaptureError("RUNECONOMY_TASK_ID is required")
        repetition_value = values.get("RUNECONOMY_REPETITION")
        try:
            repetition = int(repetition_value) if repetition_value is not None else None
        except ValueError as exc:
            raise AttemptCaptureError("RUNECONOMY_REPETITION must be an integer") from exc
        return cls(
            upstream,
            attempt_dir=attempt_dir,
            trace_path=trace_path,
            task_id=task_id,
            variant=values.get("RUNECONOMY_VARIANT"),
            repetition=repetition,
            **options,
        )

    @property
    def base_url(self) -> str:
        return self.proxy.base_url

    def child_environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return the agent's environment, pointed at the capture proxy.

        Only base-URL variables change. Credentials pass through untouched, and
        the agent's own client sends its own Authorization header.
        """

        environment = dict(os.environ if base is None else base)
        for name in BASE_URL_VARIABLES:
            environment[name] = self.base_url
        return environment

    def __enter__(self) -> "AttemptCapture":
        self.proxy.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.finish()

    def finish(self) -> AttemptCaptureResult:
        if self.result is not None:
            return self.result
        summary = self.proxy.stop()
        session = load_session(self.session_path)
        workflows = group_workflows(session)
        if not workflows:
            self.result = AttemptCaptureResult(
                None, False, 0, self.session_path, "the agent made no inference requests", summary
            )
            return self.result
        try:
            trace = build_workflow_trace(session, workflows[0], workload=self.task_id)
        except TraceValidationError as exc:
            self.result = AttemptCaptureResult(
                None, False, 0, self.session_path, str(exc), summary
            )
            return self.result
        exported = trace_to_openinference(trace)
        if _write_exclusive(self.trace_path, exported):
            self.result = AttemptCaptureResult(
                self.trace_path, True, len(trace["requests"]), self.session_path, None, summary
            )
            return self.result
        fallback = self.attempt_dir / FALLBACK_TRACE_FILENAME
        written = _write_exclusive(fallback, exported)
        self.result = AttemptCaptureResult(
            fallback if written else None,
            False,
            len(trace["requests"]),
            self.session_path,
            "the agent wrote RUNECONOMY_TRACE_PATH itself; the proxy trace was kept "
            "beside it and the agent's trace was not overwritten",
            summary,
        )
        return self.result


def run_agent_under_capture(
    argv: Sequence[str],
    upstream: str | UpstreamTarget,
    *,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    **options: Any,
) -> tuple[int, AttemptCaptureResult]:
    """Run an unmodified agent command for one attempt behind a capture proxy."""

    if not argv:
        raise AttemptCaptureError("an agent command is required")
    base = dict(os.environ if environment is None else environment)
    capture = AttemptCapture.from_environment(upstream, base, **options)
    with capture:
        completed = subprocess.run(
            list(argv), env=capture.child_environment(base), cwd=cwd, check=False
        )
    assert capture.result is not None
    return completed.returncode, capture.result
