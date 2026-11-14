"""Small, dependency-free adapter for vLLM's OpenAI-compatible APIs.

The adapter asks the serving backend to render or tokenize each request before
generation so the analyzer records the token sequence produced by the server's
tokenizer and chat template rather than reproducing it locally.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Mapping


class VLLMError(RuntimeError):
    """An actionable error returned by, or while contacting, vLLM."""


class VLLMHTTPError(VLLMError):
    """An HTTP response from vLLM with a non-success status."""

    def __init__(self, path: str, status: int, body: str) -> None:
        super().__init__(f"vLLM {path} returned HTTP {status}: {body}")
        self.path = path
        self.status = status
        self.body = body

    def __reduce__(self) -> tuple[Any, tuple[str, int, str]]:
        return (type(self), (self.path, self.status, self.body))


@dataclass(frozen=True)
class CapturedResponse:
    prompt_token_ids: tuple[int, ...]
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    client_ttft_ms: float | None
    total_latency_ms: float
    server_ttft_ms: float | None
    server_queue_ms: float | None
    generation_time_ms: float | None
    mean_itl_ms: float | None
    tokens_per_second: float | None
    response_id: str | None
    status: str | None
    output_digest: str | None
    output_text: str | None

    @property
    def cache_coverage(self) -> float | None:
        if self.input_tokens in (None, 0) or self.cached_tokens is None:
            return None
        return self.cached_tokens / self.input_tokens

    @property
    def client_server_ttft_gap_ms(self) -> float | None:
        if self.client_ttft_ms is None or self.server_ttft_ms is None:
            return None
        return self.client_ttft_ms - self.server_ttft_ms

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Generated content is available in memory for local evaluation but is
        # not persisted in raw trial exports by default.
        result.pop("output_text", None)
        result["prompt_token_ids"] = list(self.prompt_token_ids)
        result["cache_coverage"] = self.cache_coverage
        result["client_server_ttft_gap_ms"] = self.client_server_ttft_gap_ms
        return result


def _as_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _iter_sse_json(response: Any) -> Iterator[dict[str, Any]]:
    """Yield JSON payloads from a server-sent event byte stream."""

    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines.clear()
                if payload != "[DONE]":
                    try:
                        value = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise VLLMError("vLLM returned a malformed SSE JSON event") from exc
                    if isinstance(value, dict):
                        yield value
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise VLLMError("vLLM returned a malformed final SSE event") from exc
            if isinstance(value, dict):
                yield value


def _is_output_delta(event: Mapping[str, Any]) -> bool:
    event_type = event.get("type")
    return (
        isinstance(event_type, str)
        and event_type.endswith(".delta")
        and event.get("delta") not in (None, "", [], {})
    )


def _extract_output_text(output: Any) -> str | None:
    if not isinstance(output, list):
        return None
    pieces: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
    joined = "".join(pieces)
    return joined if joined else None


class VLLMClient:
    """Capture exact prompts and timing from a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        startup_retry_seconds: float = 0.0,
        api_dialect: str = "responses",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.startup_retry_seconds = startup_retry_seconds
        if api_dialect not in {"responses", "chat_completions"}:
            raise ValueError("api_dialect must be 'responses' or 'chat_completions'")
        self.api_dialect = api_dialect
        self._clock = clock

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _open(self, path: str, payload: Mapping[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        retry_deadline = time.monotonic() + self.startup_retry_seconds
        while True:
            try:
                return urllib.request.urlopen(request, timeout=self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                if exc.code in {502, 503} and time.monotonic() < retry_deadline:
                    exc.close()
                    time.sleep(2)
                    continue
                body = exc.read(2048).decode("utf-8", errors="replace")
                raise VLLMHTTPError(path, exc.code, body) from exc
            except urllib.error.URLError as exc:
                if time.monotonic() < retry_deadline:
                    time.sleep(2)
                    continue
                raise VLLMError(
                    f"cannot reach vLLM at {self.base_url}: {exc.reason}"
                ) from exc

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._open(path, payload) as response:
            try:
                result = json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise VLLMError(f"vLLM {path} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise VLLMError(f"vLLM {path} returned a non-object JSON value")
        return result

    def render(self, request_body: Mapping[str, Any]) -> tuple[int, ...]:
        payload = dict(request_body)
        payload.pop("stream", None)
        path = (
            "/v1/responses/render"
            if self.api_dialect == "responses"
            else "/v1/chat/completions/render"
        )
        result = self._post_json(path, payload)
        token_ids = result.get("token_ids", result.get("prompt_token_ids"))
        if (
            token_ids is None
            and isinstance(result.get("engine_input"), Mapping)
        ):
            token_ids = result["engine_input"].get("prompt_token_ids")
        if not isinstance(token_ids, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in token_ids
        ):
            raise VLLMError(
                "vLLM render response did not include integer token_ids; start vLLM "
                "with --enable-scale-out"
            )
        return tuple(token_ids)

    def tokenize_chat(self, request_body: Mapping[str, Any]) -> tuple[int, ...]:
        """Render a chat request through vLLM's standard tokenizer endpoint."""

        allowed = {
            "model",
            "messages",
            "tools",
            "add_generation_prompt",
            "continue_final_message",
            "add_special_tokens",
            "chat_template",
            "chat_template_kwargs",
            "mm_processor_kwargs",
        }
        payload = {key: value for key, value in request_body.items() if key in allowed}
        result = self._post_json("/tokenize", payload)
        token_ids = result.get("tokens")
        if not isinstance(token_ids, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in token_ids
        ):
            raise VLLMError("vLLM tokenize response did not include integer tokens")
        return tuple(token_ids)

    def prompt_token_ids(self, request_body: Mapping[str, Any]) -> tuple[int, ...]:
        """Prefer exact render APIs, falling back to chat-form /tokenize on 404."""

        try:
            return self.render(request_body)
        except VLLMHTTPError as exc:
            scale_out_disabled = (
                exc.status == 403 and "scale-out endpoints are disabled" in exc.body.lower()
            )
            if self.api_dialect != "chat_completions" or not (
                exc.status == 404 or scale_out_disabled
            ):
                raise
            return self.tokenize_chat(request_body)

    def capture(self, request_body: Mapping[str, Any]) -> CapturedResponse:
        if self.api_dialect == "chat_completions":
            return self._capture_chat_completions(request_body)
        token_ids = self.prompt_token_ids(request_body)
        payload = dict(request_body)
        payload["stream"] = True

        started = self._clock()
        first_output_at: float | None = None
        completed_at: float | None = None
        final_response: dict[str, Any] | None = None
        with self._open("/v1/responses", payload) as response:
            for event in _iter_sse_json(response):
                now = self._clock()
                if first_output_at is None and _is_output_delta(event):
                    first_output_at = now
                event_type = event.get("type")
                if event_type in {"response.completed", "response.incomplete"}:
                    candidate = event.get("response")
                    if isinstance(candidate, dict):
                        final_response = candidate
                        completed_at = now
                elif event_type in {"response.failed", "error"}:
                    raise VLLMError(f"vLLM generation failed: {event.get('error', event)}")

        ended = completed_at if completed_at is not None else self._clock()
        if final_response is None:
            raise VLLMError(
                "vLLM stream ended without a response.completed or "
                "response.incomplete event"
            )

        usage = final_response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("input_tokens_details")
        details = details if isinstance(details, dict) else {}
        metrics = final_response.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        output = final_response.get("output")
        output_digest = None
        if output is not None:
            canonical = json.dumps(output, sort_keys=True, separators=(",", ":"))
            output_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        return CapturedResponse(
            prompt_token_ids=token_ids,
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            cached_tokens=_as_int(details.get("cached_tokens")),
            client_ttft_ms=(
                None
                if first_output_at is None
                else (first_output_at - started) * 1000
            ),
            total_latency_ms=(ended - started) * 1000,
            server_ttft_ms=_as_number(metrics.get("time_to_first_token_ms")),
            server_queue_ms=_as_number(metrics.get("queue_time_ms")),
            generation_time_ms=_as_number(metrics.get("generation_time_ms")),
            mean_itl_ms=_as_number(metrics.get("mean_itl_ms")),
            tokens_per_second=_as_number(metrics.get("tokens_per_second")),
            response_id=(
                final_response.get("id")
                if isinstance(final_response.get("id"), str)
                else None
            ),
            status=(
                final_response.get("status")
                if isinstance(final_response.get("status"), str)
                else None
            ),
            output_digest=output_digest,
            output_text=_extract_output_text(output),
        )

    def _capture_chat_completions(
        self, request_body: Mapping[str, Any]
    ) -> CapturedResponse:
        token_ids = self.prompt_token_ids(request_body)
        payload = dict(request_body)
        payload["stream"] = True
        stream_options = payload.get("stream_options")
        stream_options = (
            dict(stream_options) if isinstance(stream_options, Mapping) else {}
        )
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options

        started = self._clock()
        first_output_at: float | None = None
        ended_at: float | None = None
        usage: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        response_id: str | None = None
        output_pieces: list[str] = []
        output_deltas: list[dict[str, Any]] = []
        terminal_reason: str | None = None
        with self._open("/v1/chat/completions", payload) as response:
            for event in _iter_sse_json(response):
                now = self._clock()
                if isinstance(event.get("id"), str):
                    response_id = event["id"]
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage
                event_metrics = event.get("metrics")
                if isinstance(event_metrics, dict):
                    metrics = event_metrics
                choices = event.get("choices")
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        if delta:
                            output_deltas.append(dict(delta))
                        content = delta.get("content")
                        has_tool_delta = delta.get("tool_calls") not in (None, [], {})
                        if isinstance(content, str) and content:
                            output_pieces.append(content)
                        if first_output_at is None and (
                            (isinstance(content, str) and content) or has_tool_delta
                        ):
                            first_output_at = now
                    finish_reason = choice.get("finish_reason")
                    if isinstance(finish_reason, str):
                        terminal_reason = finish_reason
                        ended_at = now

        ended = ended_at if ended_at is not None else self._clock()
        if terminal_reason is None:
            raise VLLMError(
                "vLLM Chat Completions stream ended without a finish_reason"
            )
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        text = "".join(output_pieces) or None
        canonical_output = json.dumps(
            output_deltas, sort_keys=True, separators=(",", ":")
        )
        output_digest = (
            hashlib.sha256(canonical_output.encode("utf-8")).hexdigest()
            if output_deltas
            else None
        )
        completed_reasons = {"stop", "tool_calls", "function_call"}
        return CapturedResponse(
            prompt_token_ids=token_ids,
            input_tokens=_as_int(usage.get("prompt_tokens")),
            output_tokens=_as_int(usage.get("completion_tokens")),
            cached_tokens=_as_int(details.get("cached_tokens")),
            client_ttft_ms=(
                None if first_output_at is None else (first_output_at - started) * 1000
            ),
            total_latency_ms=(ended - started) * 1000,
            server_ttft_ms=_as_number(metrics.get("time_to_first_token_ms")),
            server_queue_ms=_as_number(metrics.get("queue_time_ms")),
            generation_time_ms=_as_number(metrics.get("generation_time_ms")),
            mean_itl_ms=_as_number(metrics.get("mean_itl_ms")),
            tokens_per_second=_as_number(metrics.get("tokens_per_second")),
            response_id=response_id,
            status=(
                "completed" if terminal_reason in completed_reasons else "incomplete"
            ),
            output_digest=output_digest,
            output_text=text,
        )

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        suffix = "?reset_external=true" if reset_external else ""
        result = self._post_json(f"/reset_prefix_cache{suffix}", {})
        if result.get("success") is not True:
            raise VLLMError(
                "vLLM did not reset the prefix cache; ensure no request holds cache "
                "blocks and start the trusted test server with VLLM_SERVER_DEV_MODE=1"
            )
