"""Small, dependency-free adapter for vLLM's Responses API.

The adapter deliberately uses ``/v1/responses/render`` before generation so
the analyzer records the token sequence produced by the serving tokenizer and
chat template, rather than attempting to reproduce it locally.
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

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Generated content is available in memory for local evaluation but is
        # not persisted in raw trial exports by default.
        result.pop("output_text", None)
        result["prompt_token_ids"] = list(self.prompt_token_ids)
        result["cache_coverage"] = self.cache_coverage
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
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.startup_retry_seconds = startup_retry_seconds
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
                raise VLLMError(
                    f"vLLM {path} returned HTTP {exc.code}: {body}"
                ) from exc
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
        result = self._post_json("/v1/responses/render", payload)
        token_ids = result.get("token_ids")
        if not isinstance(token_ids, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in token_ids
        ):
            raise VLLMError(
                "vLLM render response did not include integer token_ids; start vLLM "
                "with --enable-scale-out"
            )
        return tuple(token_ids)

    def capture(self, request_body: Mapping[str, Any]) -> CapturedResponse:
        token_ids = self.render(request_body)
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

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        suffix = "?reset_external=true" if reset_external else ""
        result = self._post_json(f"/reset_prefix_cache{suffix}", {})
        if result.get("success") is not True:
            raise VLLMError(
                "vLLM did not reset the prefix cache; ensure no request holds cache "
                "blocks and start the trusted test server with VLLM_SERVER_DEV_MODE=1"
            )
