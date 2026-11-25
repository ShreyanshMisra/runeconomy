"""Hosted OpenAI-compatible API versus self-hosted vLLM agent example.

One agent, two serving arms. Both arms send identical messages, generation
parameters, and tool evidence through the same plan -> tool -> answer workflow.
Only the serving endpoint differs:

* ``--arm hosted`` streams Chat Completions from any OpenAI-compatible API
  (OpenAI, Fireworks, Together, a gateway). It never calls vLLM-only endpoints.
  Exact prompt token IDs, server TTFT, and queue time are not observable there,
  so they are omitted from the trace and RunEconomy records them as unknown.
* ``--arm self-hosted`` uses :class:`runeconomy.vllm.VLLMClient`, which renders
  exact prompt tokens through vLLM and records whatever server timing vLLM
  reports.

The agent depends only on the Python standard library and RunEconomy. It
writes one OTLP JSON trace with OpenInference attributes and explicit
``runeconomy.depends_on`` edges.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from runeconomy.agent import Attempt
from runeconomy.vllm import VLLMClient, VLLMError


EXAMPLE_DIR = Path(__file__).resolve().parent
RECORDS_PATH = EXAMPLE_DIR / "orders.json"
SERVICE_NAME = "runeconomy-hosted-vs-self-hosted-agent"
TOOL_NAME = "lookup_order"
# Identity policy 0.1 grades (see runeconomy.identity).
OPERATOR_VERIFIED = "operator_verified"
PROVIDER_DECLARED = "provider_declared"
# Fields a hosted provider does not publish. They are recorded as undisclosed,
# never filled with placeholder strings.
HOSTED_UNDISCLOSED_DEFAULT = (
    "backend_version",
    "tokenizer",
    "tokenizer_revision",
    "chat_template_digest",
)
_ORDER_ID = re.compile(r"\b[A-Z]-\d{4}\b")
_ANSWER = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}")
SYSTEM_PROMPT = """You are an order-support agent. You may call one tool,
lookup_order, which returns a JSON order record. Never answer from memory.

Step 1: reply with exactly one JSON object {"order_id": "<id>"} naming the order
the customer asks about. Order IDs look like A-1042.
Step 2: after the tool result arrives, reply with only the requested field
value copied exactly from the record: no quotes, punctuation, or explanation.
"""


class ModelOutputFailure(ValueError):
    """The model response did not satisfy the agent's output contract."""


class HostedAPIError(RuntimeError):
    """The hosted API was unreachable or returned an unusable response."""


@dataclass(frozen=True)
class Capture:
    """One generation. ``None`` means the serving arm did not expose the value."""

    output_text: str | None
    response_model: str | None
    prompt_token_ids: tuple[int, ...] | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    client_ttft_ms: float | None
    server_ttft_ms: float | None
    server_queue_ms: float | None
    system_fingerprint: str | None


# --------------------------------------------------------------------------
# Serving arms
# --------------------------------------------------------------------------


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sse_events(response: Any) -> Iterator[dict[str, Any]]:
    data: list[str] = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                payload = "\n".join(data)
                data.clear()
                if payload != "[DONE]":
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise HostedAPIError("malformed SSE event") from exc
                    if isinstance(event, dict):
                        yield event
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data) != "[DONE]":
        try:
            event = json.loads("\n".join(data))
        except json.JSONDecodeError as exc:
            raise HostedAPIError("malformed final SSE event") from exc
        if isinstance(event, dict):
            yield event


class HostedChatClient:
    """Stream Chat Completions from a hosted OpenAI-compatible API.

    Only what the provider returns is recorded: client-observed TTFT, the
    resolved model name, the system fingerprint, and provider-reported usage.
    No tokenizer, render, or server-metrics endpoint is called.
    """

    def __init__(
        self, base_url: str, *, api_key: str | None, timeout_seconds: float
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def capture(self, body: Mapping[str, Any]) -> Capture:
        payload = {
            **body,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        if self.api_key:
            request.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        started = time.perf_counter()
        first_output: float | None = None
        pieces: list[str] = []
        usage: dict[str, Any] = {}
        model: str | None = None
        fingerprint: str | None = None
        finished = False
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                for event in _sse_events(response):
                    if isinstance(event.get("model"), str):
                        model = event["model"]
                    if isinstance(event.get("system_fingerprint"), str):
                        fingerprint = event["system_fingerprint"]
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        content = (
                            delta.get("content") if isinstance(delta, dict) else None
                        )
                        if isinstance(content, str) and content:
                            if first_output is None:
                                first_output = time.perf_counter()
                            pieces.append(content)
                        if isinstance(choice.get("finish_reason"), str):
                            finished = True
        except urllib.error.HTTPError as exc:
            raise HostedAPIError(f"hosted API returned HTTP {exc.code}") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise HostedAPIError("hosted API was unreachable or interrupted") from exc
        if not finished:
            raise HostedAPIError("hosted stream ended without a finish_reason")
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        return Capture(
            output_text="".join(pieces) or None,
            response_model=model,
            prompt_token_ids=None,
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            cached_tokens=_int_or_none(details.get("cached_tokens")),
            client_ttft_ms=(
                None if first_output is None else (first_output - started) * 1000
            ),
            server_ttft_ms=None,
            server_queue_ms=None,
            system_fingerprint=fingerprint,
        )


class SelfHostedVLLMClient:
    """Capture through RunEconomy's vLLM client, keeping exact prompt tokens."""

    def __init__(
        self, base_url: str, *, api_key: str | None, timeout_seconds: float
    ) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._client = VLLMClient(
            root,
            api_key=api_key,
            api_dialect="chat_completions",
            timeout_seconds=timeout_seconds,
        )

    def capture(self, body: Mapping[str, Any]) -> Capture:
        captured = self._client.capture(body)
        return Capture(
            output_text=captured.output_text,
            response_model=None,
            prompt_token_ids=captured.prompt_token_ids,
            input_tokens=captured.input_tokens,
            output_tokens=captured.output_tokens,
            cached_tokens=captured.cached_tokens,
            client_ttft_ms=captured.client_ttft_ms,
            server_ttft_ms=captured.server_ttft_ms,
            server_queue_ms=captured.server_queue_ms,
            system_fingerprint=None,
        )


@dataclass(frozen=True)
class Arm:
    name: str
    client: HostedChatClient | SelfHostedVLLMClient
    requested_model: str
    infrastructure_errors: tuple[type[BaseException], ...]

    def identity(self, capture: Capture) -> dict[str, str]:
        """Graded identity attributes for one inference span.

        Hosted spans are ``provider_declared``: the model snapshot is what the
        provider returns, and fields it does not publish are listed as
        undisclosed rather than filled in. Self-hosted spans are
        ``operator_verified``: every field is pinned by the operator, and a
        missing value is omitted rather than guessed, which RunEconomy treats
        as incomplete identity and an ``invalid`` verdict.
        """

        environment = os.environ
        if self.name == "hosted":
            model = capture.response_model or self.requested_model
            declared = {
                "llm.model_name": model,
                "llm.provider": environment.get(
                    "HOSTED_PROVIDER", "openai_compatible"
                ),
                # A dated snapshot name is the only revision a provider declares.
                "llm.model_revision": environment.get("HOSTED_MODEL_REVISION", model),
                "llm.backend_version": environment.get("HOSTED_BACKEND_VERSION"),
            }
            values = {key: value for key, value in declared.items() if value}
            undisclosed = [
                field
                for field in HOSTED_UNDISCLOSED_DEFAULT
                if not (
                    field == "backend_version"
                    and environment.get("HOSTED_BACKEND_VERSION")
                )
            ]
            return {
                **values,
                "runeconomy.identity_grade": PROVIDER_DECLARED,
                "runeconomy.identity_undisclosed": json.dumps(undisclosed),
            }
        declared = {
            "llm.model_name": self.requested_model,
            "llm.provider": "vllm",
            "llm.model_revision": environment.get("SELF_HOSTED_MODEL_REVISION"),
            "llm.backend_version": environment.get("SELF_HOSTED_BACKEND_VERSION"),
            "runeconomy.tokenizer": environment.get(
                "SELF_HOSTED_TOKENIZER", self.requested_model
            ),
            "runeconomy.tokenizer_revision": environment.get(
                "SELF_HOSTED_TOKENIZER_REVISION",
                environment.get("SELF_HOSTED_MODEL_REVISION"),
            ),
            "runeconomy.chat_template_digest": environment.get(
                "SELF_HOSTED_CHAT_TEMPLATE_DIGEST"
            ),
        }
        return {
            **{key: value for key, value in declared.items() if value},
            "runeconomy.identity_grade": OPERATOR_VERIFIED,
        }


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise KeyError(f"{name} must be set")
    return value


def build_arm(name: str) -> Arm:
    timeout = float(os.environ.get("RUNECONOMY_EXAMPLE_REQUEST_TIMEOUT", "120"))
    if name == "hosted":
        return Arm(
            name=name,
            client=HostedChatClient(
                _required_env("HOSTED_BASE_URL"),
                api_key=os.environ.get("HOSTED_API_KEY") or None,
                timeout_seconds=timeout,
            ),
            requested_model=_required_env("HOSTED_MODEL"),
            infrastructure_errors=(HostedAPIError,),
        )
    return Arm(
        name=name,
        client=SelfHostedVLLMClient(
            _required_env("SELF_HOSTED_BASE_URL"),
            api_key=os.environ.get("SELF_HOSTED_API_KEY") or None,
            timeout_seconds=timeout,
        ),
        requested_model=_required_env("SELF_HOSTED_MODEL"),
        infrastructure_errors=(VLLMError,),
    )


# --------------------------------------------------------------------------
# Trace writing (OTLP JSON, no OpenTelemetry SDK dependency)
# --------------------------------------------------------------------------


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


class TraceWriter:
    """Durably rewrite one OTLP JSON trace after every finished span."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.trace_id = secrets.token_hex(16)
        self._spans: list[dict[str, Any]] = []

    def span(
        self,
        name: str,
        *,
        parent: str | None,
        attributes: Mapping[str, Any],
    ) -> "_Span":
        return _Span(self, name, parent, dict(attributes))

    def _finish(self, record: dict[str, Any]) -> None:
        self._spans.append(record)
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": SERVICE_NAME},
                            }
                        ]
                    },
                    "scopeSpans": [{"spans": list(self._spans)}],
                }
            ]
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class _Span:
    def __init__(
        self,
        writer: TraceWriter,
        name: str,
        parent: str | None,
        attributes: dict[str, Any],
    ) -> None:
        self.writer = writer
        self.name = name
        self.parent = parent
        self.attributes = attributes
        self.span_id = secrets.token_hex(8)
        self.start_ns = 0

    def __enter__(self) -> "_Span":
        self.start_ns = time.time_ns()
        return self

    def set(self, key: str, value: Any) -> None:
        if value is not None:
            self.attributes[key] = value

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.writer._finish(
            {
                "traceId": self.writer.trace_id,
                "spanId": self.span_id,
                "parentSpanId": self.parent or "",
                "name": self.name,
                "startTimeUnixNano": str(self.start_ns),
                "endTimeUnixNano": str(time.time_ns()),
                "status": {
                    "code": "STATUS_CODE_ERROR" if exc_type else "STATUS_CODE_OK"
                },
                "attributes": [
                    {"key": key, "value": _otlp_value(value)}
                    for key, value in sorted(self.attributes.items())
                ],
            }
        )
        return False


def _record_capture(span: _Span, capture: Capture) -> None:
    if capture.prompt_token_ids is not None:
        span.set("runeconomy.prompt_token_ids", json.dumps(list(capture.prompt_token_ids)))
    span.set("llm.token_count.prompt", capture.input_tokens)
    span.set("llm.token_count.completion", capture.output_tokens)
    span.set("llm.token_count.prompt_details.cache_read", capture.cached_tokens)
    span.set("runeconomy.client_ttft_ms", capture.client_ttft_ms)
    span.set("llm.time_to_first_token_ms", capture.server_ttft_ms)
    span.set("llm.queue_time_ms", capture.server_queue_ms)
    # A drift indicator only. RunEconomy digests it and reports per-arm counts;
    # it is never part of identity and never affects validity.
    span.set("runeconomy.provider_system_fingerprint", capture.system_fingerprint)


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------


def _request(model: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _extract_order_id(text: str) -> str:
    for candidate in [text, *re.findall(r"\{[^{}]*\}", text)]:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("order_id"), str):
            order_id = value["order_id"].strip()
            if _ORDER_ID.fullmatch(order_id):
                return order_id
    raise ModelOutputFailure("plan did not name one order_id")


def _normalize_answer(text: str) -> str:
    stripped = text.strip().strip("`\"'.").strip()
    if not _ANSWER.fullmatch(stripped):
        raise ModelOutputFailure("answer was not one bare field value")
    return stripped


def lookup_order(order_id: str) -> dict[str, Any]:
    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))["orders"]
    record = records.get(order_id)
    if record is None:
        return {"error": "order not found", "order_id": order_id}
    return {"order_id": order_id, **record}


def run_workflow(attempt: Attempt, arm: Arm) -> str:
    task = json.loads(attempt.task_path.read_text(encoding="utf-8"))
    question = task.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("task question must be a non-empty string")
    writer = TraceWriter(attempt.trace_path)
    tool_schema = {"llm.tools.0.tool.name": TOOL_NAME}

    with writer.span(
        "order_support.workflow",
        parent=None,
        attributes={
            "openinference.span.kind": "CHAIN",
            "runeconomy.task_id": attempt.task_id,
        },
    ) as root:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        with writer.span(
            "order_support.plan",
            parent=root.span_id,
            attributes={
                "openinference.span.kind": "LLM",
                "runeconomy.depends_on": "[]",
                **tool_schema,
            },
        ) as plan_span:
            plan = arm.client.capture(_request(arm.requested_model, messages, 48))
            _record_capture(plan_span, plan)
            for key, value in arm.identity(plan).items():
                plan_span.set(key, value)
            plan_span.set(
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
                TOOL_NAME,
            )
        if plan.output_text is None:
            raise ModelOutputFailure("plan response contained no text")
        (attempt.attempt_dir / "model-output-plan.txt").write_text(
            plan.output_text, encoding="utf-8"
        )
        order_id = _extract_order_id(plan.output_text)

        with writer.span(
            f"order_support.{TOOL_NAME}",
            parent=root.span_id,
            attributes={
                "openinference.span.kind": "TOOL",
                "runeconomy.depends_on": json.dumps([plan_span.span_id]),
            },
        ) as tool_span:
            record = lookup_order(order_id)

        messages = [
            *messages,
            {"role": "assistant", "content": plan.output_text},
            {
                "role": "user",
                "content": (
                    f"Tool {TOOL_NAME} returned:\n"
                    f"{json.dumps(record, sort_keys=True)}\n\n"
                    "Reply with only the requested field value."
                ),
            },
        ]
        with writer.span(
            "order_support.answer",
            parent=root.span_id,
            attributes={
                "openinference.span.kind": "LLM",
                "runeconomy.depends_on": json.dumps([tool_span.span_id]),
                **tool_schema,
            },
        ) as answer_span:
            answer = arm.client.capture(_request(arm.requested_model, messages, 24))
            _record_capture(answer_span, answer)
            for key, value in arm.identity(answer).items():
                answer_span.set(key, value)
        if answer.output_text is None:
            raise ModelOutputFailure("answer response contained no text")
        (attempt.attempt_dir / "model-output-answer.txt").write_text(
            answer.output_text, encoding="utf-8"
        )
        return _normalize_answer(answer.output_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("hosted", "self-hosted"), required=True)
    args = parser.parse_args()

    with Attempt.from_environment() as attempt:
        arm = build_arm(args.arm)
        try:
            result = run_workflow(attempt, arm)
        except ModelOutputFailure:
            attempt.model_output_failure(
                stage="agent_output_contract", code="invalid_model_output"
            )
            raise
        except arm.infrastructure_errors:
            attempt.infrastructure_failure(
                stage="model_request",
                code=(
                    "hosted_request_failed"
                    if arm.name == "hosted"
                    else "vllm_request_failed"
                ),
            )
            raise
        attempt.output_path.write_text(result + "\n", encoding="utf-8")
        attempt.completed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
