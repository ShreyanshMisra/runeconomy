"""Explicit workflow DAG tracing without OpenTelemetry boilerplate.

Agent-process attempts must write a trace with explicit ``depends_on`` edges
for critical-path analysis. These helpers produce that trace directly::

    import runeconomy

    @runeconomy.track_workflow
    def main() -> None:
        with runeconomy.step("plan", kind="LLM") as plan:
            reply = client.chat.completions.create(...)
            plan.record_llm(model="qwen", input_tokens=reply.usage.prompt_tokens,
                            output_tokens=reply.usage.completion_tokens)
        with runeconomy.step("search", kind="TOOL", depends_on=[plan]) as search:
            hits = search_repository(...)
        with runeconomy.step("answer", kind="LLM", depends_on=[plan, search]) as answer:
            ...

``track_workflow`` works as a decorator or a context manager. On exit it writes
OTLP JSON with OpenInference attributes (the format ``runeconomy import`` and
the agent-process runner already read) to ``RUNECONOMY_TRACE_PATH`` or an
explicit ``trace_path``. Dependencies are recorded only where you declare them
with ``depends_on``; nothing is inferred from timing. A step that declares no
dependencies is a root. Every step carries the attempt's task ID when the
runner provides one.

Unknown values stay unknown: a step records token counts only when you pass
them.
"""

from __future__ import annotations

import contextvars
import functools
import json
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar


SPAN_KINDS = ("LLM", "TOOL", "RETRIEVER", "CHAIN", "AGENT")
_current: contextvars.ContextVar["WorkflowTracker | None"] = contextvars.ContextVar(
    "runeconomy_workflow", default=None
)
F = TypeVar("F", bound=Callable[..., Any])
# Worker threads (for example a ThreadPoolExecutor running parallel steps) do
# not inherit context variables. When exactly one workflow is active in the
# process, step() falls back to it; with several active it refuses to guess.
_active: list["WorkflowTracker"] = []
_active_lock = threading.Lock()


class WorkflowTraceError(RuntimeError):
    """A step or dependency declaration that cannot be traced faithfully."""


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        typed = {"boolValue": value}
    elif isinstance(value, int):
        typed = {"intValue": value}
    elif isinstance(value, float):
        typed = {"doubleValue": value}
    else:
        typed = {"stringValue": str(value)}
    return {"key": key, "value": typed}


class Step:
    """One traced unit of work. Use :meth:`record_llm` on LLM steps."""

    def __init__(self, tracker: "WorkflowTracker", name: str, kind: str, depends: tuple[str, ...]) -> None:
        self.tracker = tracker
        self.name = name
        self.kind = kind
        self.id = secrets.token_hex(8)
        self.depends_on = depends
        self.attributes: dict[str, Any] = {}
        self.start_ns: int | None = None
        self.end_ns: int | None = None
        self.error: str | None = None

    def record_llm(
        self,
        *,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        provider: str | None = None,
        model_revision: str | None = None,
        backend_version: str | None = None,
        client_ttft_ms: float | None = None,
        server_ttft_ms: float | None = None,
        cost_usd: float | None = None,
        step_role: str | None = None,
        tokenizer: str | None = None,
        tokenizer_revision: str | None = None,
        chat_template_digest: str | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if self.kind != "LLM":
            raise WorkflowTraceError(f"record_llm applies only to LLM steps, not {self.kind}")
        values = {
            "llm.model_name": model,
            "llm.token_count.prompt": input_tokens,
            "llm.token_count.completion": output_tokens,
            "llm.token_count.prompt_details.cache_read": cached_tokens,
            "llm.provider": provider,
            "llm.model_revision": model_revision,
            "llm.backend_version": backend_version,
            "runeconomy.client_ttft_ms": client_ttft_ms,
            "runeconomy.server_ttft_ms": server_ttft_ms,
            "runeconomy.cost_usd": cost_usd,
            # Declare a role when steps deliberately use different models, so
            # identity stability is judged per role rather than as drift.
            "runeconomy.step_role": step_role,
            "runeconomy.tokenizer": tokenizer,
            "runeconomy.tokenizer_revision": tokenizer_revision,
            "runeconomy.chat_template_digest": chat_template_digest,
            "runeconomy.prompt_token_ids": (
                None if prompt_token_ids is None else json.dumps(list(prompt_token_ids))
            ),
        }
        self.attributes.update({key: value for key, value in values.items() if value is not None})

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        self.attributes[key] = value

    def __enter__(self) -> "Step":
        self.start_ns = time.time_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.end_ns = time.time_ns()
        if exc_type is not None:
            self.error = exc_type.__name__
        self.tracker._finish(self)


def depends_on(*steps: Step | str | Iterable[Step | str]) -> tuple[str, ...]:
    """Normalize steps or span IDs into a dependency tuple."""

    identifiers: list[str] = []
    for item in steps:
        values = [item] if isinstance(item, (Step, str)) else list(item)
        for value in values:
            if isinstance(value, Step):
                identifiers.append(value.id)
            elif isinstance(value, str) and value:
                identifiers.append(value)
            else:
                raise WorkflowTraceError("depends_on accepts steps or non-empty span IDs")
    return tuple(dict.fromkeys(identifiers))


_dependency_ids = depends_on


class WorkflowTracker:
    """Collect steps and write one OTLP JSON trace."""

    def __init__(
        self,
        name: str = "workflow",
        *,
        trace_path: str | Path | None = None,
        task_id: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        environment = os.environ if environment is None else environment
        self.name = name
        path = trace_path if trace_path is not None else environment.get("RUNECONOMY_TRACE_PATH")
        self.trace_path = None if path in (None, "") else Path(path)
        self.task_id = task_id if task_id is not None else environment.get("RUNECONOMY_TASK_ID")
        self.attempt_id = environment.get("RUNECONOMY_ATTEMPT_ID")
        self.trace_id = secrets.token_hex(16)
        self._steps: list[Step] = []
        self._ids: set[str] = set()
        self._lock = threading.Lock()
        self._token: contextvars.Token | None = None
        self.written: Path | None = None

    def step(
        self,
        name: str,
        *,
        kind: str = "LLM",
        depends_on: Iterable[Step | str] | None = None,
    ) -> Step:
        kind = kind.upper()
        if kind not in SPAN_KINDS:
            raise WorkflowTraceError(f"step kind must be one of {', '.join(SPAN_KINDS)}")
        depends = _dependency_ids(depends_on or ())
        with self._lock:
            unknown = [item for item in depends if item not in self._ids]
        if unknown:
            raise WorkflowTraceError(
                "depends_on must name steps already started in this workflow; "
                "declare the edge after its upstream step begins"
            )
        created = Step(self, name, kind, depends)
        with self._lock:
            self._ids.add(created.id)
        return created

    def _finish(self, step: Step) -> None:
        with self._lock:
            self._steps.append(step)

    def to_otlp(self) -> dict[str, Any]:
        spans = []
        with self._lock:
            steps = sorted(self._steps, key=lambda item: item.start_ns or 0)
        for item in steps:
            attributes = [
                _attribute("openinference.span.kind", item.kind),
                _attribute("runeconomy.depends_on", json.dumps(list(item.depends_on))),
                *(_attribute(key, value) for key, value in sorted(item.attributes.items())),
            ]
            if self.task_id:
                attributes.append(_attribute("runeconomy.task_id", self.task_id))
            if self.attempt_id:
                attributes.append(_attribute("runeconomy.attempt_id", self.attempt_id))
            span: dict[str, Any] = {
                "traceId": self.trace_id,
                "spanId": item.id,
                "name": item.name,
                "startTimeUnixNano": str(item.start_ns),
                "endTimeUnixNano": str(item.end_ns),
                "attributes": attributes,
            }
            if item.error:
                span["status"] = {"code": 2, "message": item.error}
            spans.append(span)
        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": [_attribute("service.name", self.name)]},
                    "scopeSpans": [{"scope": {"name": "runeconomy.workflow"}, "spans": spans}],
                }
            ]
        }

    def write(self) -> Path | None:
        if self.trace_path is None:
            return None
        payload = json.dumps(self.to_otlp(), sort_keys=True)
        temporary = self.trace_path.with_name(self.trace_path.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.trace_path)
        self.written = self.trace_path
        return self.trace_path

    def __enter__(self) -> "WorkflowTracker":
        self._token = _current.set(self)
        with _active_lock:
            _active.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            # The trace is written even when the workflow fails, so the
            # runner keeps the evidence of the steps that did run.
            self.write()
        finally:
            with _active_lock:
                if self in _active:
                    _active.remove(self)
            if self._token is not None:
                _current.reset(self._token)
                self._token = None


def current_workflow() -> WorkflowTracker:
    tracker = _current.get()
    if tracker is None:
        with _active_lock:
            tracker = _active[0] if len(_active) == 1 else None
    if tracker is None:
        raise WorkflowTraceError("no workflow is active; use @runeconomy.track_workflow or `with track_workflow()`")
    return tracker


def step(
    name: str,
    *,
    kind: str = "LLM",
    depends_on: Iterable[Step | str] | None = None,
) -> Step:
    """Start a step in the active workflow."""

    return current_workflow().step(name, kind=kind, depends_on=depends_on)


class _TrackWorkflow:
    def __init__(self, name: str, options: dict[str, Any]) -> None:
        self.name = name
        self.options = options
        self.tracker: WorkflowTracker | None = None

    def __enter__(self) -> WorkflowTracker:
        self.tracker = WorkflowTracker(self.name, **self.options)
        return self.tracker.__enter__()

    def __exit__(self, *exc_info: Any) -> None:
        assert self.tracker is not None
        self.tracker.__exit__(*exc_info)

    def __call__(self, function: F) -> F:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with WorkflowTracker(self.name if self.name != "workflow" else function.__name__, **self.options):
                return function(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


def track_workflow(
    function: F | str | None = None,
    *,
    trace_path: str | Path | None = None,
    task_id: str | None = None,
) -> Any:
    """Decorator or context manager that traces one workflow run.

    ``@track_workflow``, ``@track_workflow("name")``, and
    ``with track_workflow() as workflow:`` are all supported.
    """

    options = {"trace_path": trace_path, "task_id": task_id}
    if callable(function):
        return _TrackWorkflow("workflow", options)(function)
    return _TrackWorkflow(function or "workflow", options)
