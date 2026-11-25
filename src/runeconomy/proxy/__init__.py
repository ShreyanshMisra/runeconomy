"""Zero-code capture: an OpenAI-compatible reverse proxy that records evidence.

Point any agent's OpenAI base URL at :class:`CaptureProxy`. The proxy forwards
every byte immediately and records each inference exchange off the request
path. It resolves exact prompt tokens against vLLM, and the resulting session
imports straight into the normalized trace format, so ``import``, ``experiment
init`` and ``experiment validate`` work without agent code changes.

:mod:`runeconomy.proxy.contract` holds the normative wire contract.
"""

from .attempt import AttemptCapture, AttemptCaptureError, AttemptCaptureResult, run_agent_under_capture
from .contract import PROXY_SESSION_VERSION
from .recorder import RedactionPolicy, SessionRecorder, load_redaction_hook
from .server import CaptureProxy
from .session import (
    DEFAULT_IDLE_GAP_MS,
    build_workflow_trace,
    group_workflows,
    import_proxy_session,
    is_proxy_session,
    load_session,
    session_traces,
    trace_to_openinference,
)
from .upstream import UpstreamConfigurationError, UpstreamTarget


def start_capture(
    upstream: str,
    record: str,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    **recorder_options: object,
) -> CaptureProxy:
    """Start a capture proxy in the background and return it.

    ``record`` must not exist yet. Sessions are never overwritten. Call
    ``stop()`` on the returned proxy to drain telemetry and close the session.
    """

    from pathlib import Path

    target = UpstreamTarget.parse(upstream)
    recorder = SessionRecorder(Path(record), upstream=target, **recorder_options)  # type: ignore[arg-type]
    return CaptureProxy(target, recorder, host=host, port=port).start()


__all__ = [
    "AttemptCapture",
    "AttemptCaptureError",
    "AttemptCaptureResult",
    "CaptureProxy",
    "DEFAULT_IDLE_GAP_MS",
    "PROXY_SESSION_VERSION",
    "RedactionPolicy",
    "SessionRecorder",
    "UpstreamConfigurationError",
    "UpstreamTarget",
    "build_workflow_trace",
    "group_workflows",
    "import_proxy_session",
    "is_proxy_session",
    "load_redaction_hook",
    "load_session",
    "run_agent_under_capture",
    "session_traces",
    "start_capture",
    "trace_to_openinference",
]
