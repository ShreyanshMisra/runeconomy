"""Exact prompt-token resolution against the serving backend, off the request path.

The resolver asks vLLM to render or tokenize each captured request, so the
token IDs come from the server's own tokenizer and chat template. When no such
endpoint answers, or when the server's own usage count disagrees with the
resolved IDs, the request is marked ``unknown``. Tokens are never estimated or
produced locally.

Segment boundaries are found by prefix rendering. The boundary after message
``k`` is the length of the render of ``messages[:k]`` without a generation
prompt. It is accepted only if that render is a literal token prefix of the
complete render. Boundaries that fail the check become gaps. When removing the
``tools`` field deletes exactly one contiguous token block, that block is
recorded as the exact tools segment.
"""

from __future__ import annotations

import hashlib
import json
import threading
from array import array
from collections import OrderedDict
from typing import Any, Mapping, Sequence

from .upstream import JSONResult, OffPathClient


TOKEN_RESOLUTION_VERSION = "0.1"

_CHAT_TOKENIZE_KEYS = frozenset(
    {
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
)
_RUNNER_ONLY_KEYS = ("stream", "stream_options")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )


def _token_bytes(tokens: Sequence[int]) -> bytes:
    return array("Q", tokens).tobytes()


def _token_digest(tokens: Sequence[int]) -> str:
    return hashlib.sha256(_token_bytes(tokens)).hexdigest()


def _integer_tokens(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in value
    ):
        return None
    return value


def _tokens_from_render(payload: Any) -> list[int] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("token_ids", "prompt_token_ids"):
        tokens = _integer_tokens(payload.get(key))
        if tokens is not None:
            return tokens
    engine_input = payload.get("engine_input")
    if isinstance(engine_input, dict):
        return _integer_tokens(engine_input.get("prompt_token_ids"))
    return None


def _scale_out_disabled(result: JSONResult) -> bool:
    return result.status == 403 and "scale-out" in (result.text or "").lower()


def _endpoint_missing(result: JSONResult) -> bool:
    return result.status in {404, 405, 501} or _scale_out_disabled(result)


class _LRU:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)


def _message_kind(message: Mapping[str, Any], index: int, latest_user: int | None) -> str:
    role = str(message.get("role", "unknown"))
    if role in {"system", "developer"}:
        return "system_instructions"
    if role == "tool" or role == "function":
        return "tool_results"
    if role == "user" and index == latest_user:
        return "latest_user_input"
    return "history"


def _lcp(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


class TokenResolver:
    """Resolve exact prompt tokens and segments for captured requests.

    The resolver remembers which endpoints the server lacks, so a missing
    ``/render`` route is probed once and not on every request. Prefix renders
    are memoized by content digest. An agent loop that appends one message per
    step therefore costs a few renders per request rather than one per message.
    """

    def __init__(
        self,
        client: OffPathClient,
        *,
        segment: bool = True,
        max_segment_messages: int = 256,
        cache_entries: int = 65536,
    ) -> None:
        self.client = client
        self.segment = segment
        self.max_segment_messages = max_segment_messages
        self._prefix_cache = _LRU(cache_entries)
        self._tools_cache = _LRU(max(1024, cache_entries // 16))
        self._unavailable: set[str] = set()
        self._lock = threading.Lock()

    def _available(self, endpoint: str) -> bool:
        with self._lock:
            return endpoint not in self._unavailable

    def _mark_unavailable(self, endpoint: str) -> None:
        with self._lock:
            self._unavailable.add(endpoint)

    def unavailable_endpoints(self) -> list[str]:
        with self._lock:
            return sorted(self._unavailable)

    # -- primitive calls -------------------------------------------------

    def _chat_render(
        self, payload: Mapping[str, Any], authorization: str | None
    ) -> tuple[list[int] | None, str | None, bool]:
        """Return ``(tokens, error, endpoint_missing)``."""

        result = self.client.request(
            "POST", "/v1/chat/completions/render", payload, authorization=authorization
        )
        if _endpoint_missing(result):
            return None, f"render endpoint unavailable (HTTP {result.status})", True
        if not result.ok:
            return None, result.error or f"render returned HTTP {result.status}", False
        tokens = _tokens_from_render(result.payload)
        if tokens is None:
            return None, "render response carried no integer token IDs", False
        return tokens, None, False

    def _chat_tokenize(
        self, payload: Mapping[str, Any], authorization: str | None
    ) -> tuple[list[int] | None, str | None, bool]:
        body = {key: value for key, value in payload.items() if key in _CHAT_TOKENIZE_KEYS}
        result = self.client.request("POST", "/tokenize", body, authorization=authorization)
        if result.status in {404, 405, 501}:
            return None, f"tokenize endpoint unavailable (HTTP {result.status})", True
        if not result.ok:
            return None, result.error or f"tokenize returned HTTP {result.status}", False
        tokens = (
            _integer_tokens(result.payload.get("tokens"))
            if isinstance(result.payload, dict)
            else None
        )
        if tokens is None:
            return None, "tokenize response carried no integer tokens", False
        return tokens, None, False

    def _render_with(
        self, endpoint: str, payload: Mapping[str, Any], authorization: str | None
    ) -> tuple[list[int] | None, str | None]:
        if endpoint == "vllm_chat_render":
            tokens, error, _ = self._chat_render(payload, authorization)
        else:
            tokens, error, _ = self._chat_tokenize(payload, authorization)
        return tokens, error

    # -- public entry point ---------------------------------------------

    def resolve(
        self,
        route: str,
        body: Any,
        *,
        authorization: str | None,
        usage_prompt_tokens: int | None,
    ) -> dict[str, Any]:
        """Return the body of one ``token_resolution`` record."""

        if not isinstance(body, dict):
            return self._unknown("request body is not a JSON object")
        if route == "chat_completions":
            outcome = self._resolve_chat(body, authorization)
        elif route == "completions":
            outcome = self._resolve_completion(body, authorization)
        elif route == "responses":
            outcome = self._resolve_responses(body, authorization)
        else:
            return self._unknown(f"route {route!r} has no prompt-token resolution")
        if outcome.get("status") != "exact":
            return outcome
        tokens = outcome["prompt_token_ids"]
        if usage_prompt_tokens is not None and usage_prompt_tokens != len(tokens):
            return self._unknown(
                "resolved token count does not match the server-reported prompt "
                f"tokens ({len(tokens)} != {usage_prompt_tokens}); the render did "
                "not reproduce the executed prompt",
                source=outcome.get("source"),
                verification="usage_prompt_tokens_mismatch",
            )
        outcome["verification"] = (
            "usage_prompt_tokens_match"
            if usage_prompt_tokens is not None
            else "usage_prompt_tokens_unavailable"
        )
        return outcome

    @staticmethod
    def _unknown(
        reason: str,
        *,
        source: str | None = None,
        verification: str = "not_resolved",
    ) -> dict[str, Any]:
        return {
            "resolution_version": TOKEN_RESOLUTION_VERSION,
            "status": "unknown",
            "source": source,
            "verification": verification,
            "reason": reason,
            "prompt_token_ids": None,
            "prompt_token_count": None,
            "segments": None,
            "segmentation": {"status": "none", "reason": reason},
        }

    @staticmethod
    def _exact(
        tokens: list[int],
        source: str,
        *,
        segments: list[dict[str, Any]] | None = None,
        segmentation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "resolution_version": TOKEN_RESOLUTION_VERSION,
            "status": "exact",
            "source": source,
            "verification": "pending",
            "reason": None,
            "prompt_token_ids": tokens,
            "prompt_token_count": len(tokens),
            "segments": segments,
            "segmentation": dict(segmentation or {"status": "none", "reason": None}),
        }

    def _resolve_completion(
        self, body: Mapping[str, Any], authorization: str | None
    ) -> dict[str, Any]:
        prompt = body.get("prompt")
        direct = _integer_tokens(prompt)
        if direct is not None:
            return self._exact(
                list(direct),
                "request_prompt_token_ids",
                segmentation={"status": "none", "reason": "raw completion prompt"},
            )
        if not isinstance(prompt, str):
            return self._unknown(
                "completion prompt is batched or not a string; one exchange cannot "
                "carry several prompts"
            )
        if not self._available("vllm_tokenize"):
            return self._unknown("vLLM /tokenize is unavailable on this upstream")
        payload = {
            "model": body.get("model"),
            "prompt": prompt,
            "add_special_tokens": body.get("add_special_tokens", True),
        }
        result = self.client.request("POST", "/tokenize", payload, authorization=authorization)
        if result.status in {404, 405, 501}:
            self._mark_unavailable("vllm_tokenize")
            return self._unknown("vLLM /tokenize is unavailable on this upstream")
        tokens = (
            _integer_tokens(result.payload.get("tokens"))
            if result.ok and isinstance(result.payload, dict)
            else None
        )
        if tokens is None:
            return self._unknown(
                result.error or f"tokenize returned HTTP {result.status}",
                source="vllm_tokenize_prompt",
            )
        return self._exact(
            tokens,
            "vllm_tokenize_prompt",
            segmentation={"status": "none", "reason": "raw completion prompt"},
        )

    def _resolve_responses(
        self, body: Mapping[str, Any], authorization: str | None
    ) -> dict[str, Any]:
        if not self._available("vllm_responses_render"):
            return self._unknown("vLLM /v1/responses/render is unavailable on this upstream")
        payload = {key: value for key, value in body.items() if key not in _RUNNER_ONLY_KEYS}
        result = self.client.request(
            "POST", "/v1/responses/render", payload, authorization=authorization
        )
        if _endpoint_missing(result):
            self._mark_unavailable("vllm_responses_render")
            return self._unknown(
                "vLLM /v1/responses/render is unavailable; Responses API prompts have "
                "no tokenize fallback"
            )
        tokens = _tokens_from_render(result.payload) if result.ok else None
        if tokens is None:
            return self._unknown(
                result.error or f"responses render returned HTTP {result.status}",
                source="vllm_responses_render",
            )
        return self._exact(
            tokens,
            "vllm_responses_render",
            segmentation={
                "status": "none",
                "reason": "Responses API input items are not prefix-rendered",
            },
        )

    def _resolve_chat(
        self, body: Mapping[str, Any], authorization: str | None
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or any(
            not isinstance(item, dict) for item in messages
        ):
            return self._unknown("chat request has no message array")
        payload = {key: value for key, value in body.items() if key not in _RUNNER_ONLY_KEYS}
        tokens: list[int] | None = None
        endpoint: str | None = None
        errors: list[str] = []
        if self._available("vllm_chat_render"):
            tokens, error, missing = self._chat_render(payload, authorization)
            if missing:
                self._mark_unavailable("vllm_chat_render")
            if tokens is not None:
                endpoint = "vllm_chat_render"
            elif error:
                errors.append(error)
        if tokens is None and self._available("vllm_tokenize"):
            tokens, error, missing = self._chat_tokenize(payload, authorization)
            if missing:
                self._mark_unavailable("vllm_tokenize")
            if tokens is not None:
                endpoint = "vllm_tokenize_chat"
            elif error:
                errors.append(error)
        if tokens is None or endpoint is None:
            return self._unknown(
                "; ".join(errors) or "no vLLM render or tokenize endpoint is available"
            )
        if not self.segment:
            return self._exact(
                tokens,
                endpoint,
                segmentation={"status": "none", "reason": "segmentation disabled"},
            )
        segments, segmentation = self._segment_chat(
            payload, messages, tokens, endpoint, authorization
        )
        return self._exact(tokens, endpoint, segments=segments, segmentation=segmentation)

    # -- segmentation ----------------------------------------------------

    def _prefix_payload(
        self, payload: Mapping[str, Any], messages: Sequence[Any], *, tools: bool = True
    ) -> dict[str, Any]:
        prefix = dict(payload)
        prefix["messages"] = list(messages)
        prefix["add_generation_prompt"] = False
        prefix["continue_final_message"] = False
        if not tools:
            prefix.pop("tools", None)
            prefix.pop("tool_choice", None)
            prefix.pop("parallel_tool_calls", None)
        return prefix

    def _segment_chat(
        self,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        full: list[int],
        endpoint: str,
        authorization: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        count = len(messages)
        if count == 0:
            return [], {"status": "none", "reason": "no messages"}
        if count > self.max_segment_messages:
            return [], {
                "status": "none",
                "reason": (
                    f"{count} messages exceed the prefix-render limit of "
                    f"{self.max_segment_messages}"
                ),
            }
        full_bytes = _token_bytes(full)
        params = {
            key: value
            for key, value in self._prefix_payload(payload, []).items()
            if key != "messages"
        }
        chain = hashlib.sha256(endpoint.encode() + b"\0" + _canonical(params))
        lengths: list[int | None] = [0] + [None] * count
        failures: list[str] = []
        for index in range(1, count + 1):
            chain = hashlib.sha256(
                chain.digest() + hashlib.sha256(_canonical(messages[index - 1])).digest()
            )
            key = chain.hexdigest()
            cached = self._prefix_cache.get(key)
            if cached is None:
                tokens, error = self._render_with(
                    endpoint, self._prefix_payload(payload, messages[:index]), authorization
                )
                cached = (
                    ("error", error or "prefix render failed")
                    if tokens is None
                    else ("ok", len(tokens), _token_digest(tokens))
                )
                self._prefix_cache.put(key, cached)
            if cached[0] != "ok":
                failures.append(f"message {index - 1}: {cached[1]}")
                continue
            _, length, digest = cached
            if length > len(full):
                continue
            if hashlib.sha256(full_bytes[: length * 8]).hexdigest() == digest:
                lengths[index] = length

        latest_user = max(
            (i for i, message in enumerate(messages) if message.get("role") == "user"),
            default=None,
        )
        source = "vllm_prefix_render" if endpoint == "vllm_chat_render" else "vllm_prefix_tokenize"
        segments: list[dict[str, Any]] = []
        for index in range(1, count + 1):
            start, end = lengths[index - 1], lengths[index]
            if start is None or end is None or end <= start:
                continue
            message = messages[index - 1]
            role = str(message.get("role", "unknown"))
            segments.append(
                {
                    "kind": _message_kind(message, index - 1, latest_user),
                    "label": f"{role} message {index - 1}",
                    "start_token": start,
                    "end_token": end,
                    "attribution_quality": "exact_token_boundary",
                    "source": source,
                    "message_index": index - 1,
                }
            )

        tools_note: str | None = None
        tools = payload.get("tools")
        final_length = lengths[count]
        if isinstance(tools, list) and tools and final_length is not None:
            block = self._tools_block(
                payload, messages, full[:final_length], endpoint, authorization, chain.hexdigest()
            )
            if block is None:
                tools_note = "tool definitions are not one removable token block"
            else:
                segments, placed = _split_for_tools(segments, block, source)
                if not placed:
                    tools_note = "tool block straddles message boundaries"
        elif isinstance(tools, list) and tools:
            tools_note = "complete prefix render was not verified"

        covered = sum(item["end_token"] - item["start_token"] for item in segments)
        verified = sum(1 for value in lengths[1:] if value is not None)
        status = (
            "exact"
            if verified == count and not tools_note
            else "partial"
            if segments
            else "none"
        )
        reason_parts = []
        if verified != count:
            reason_parts.append(
                f"{count - verified} of {count} message boundaries were not a verified "
                "prefix of the complete render"
            )
        if tools_note:
            reason_parts.append(tools_note)
        if failures:
            reason_parts.append("; ".join(failures[:3]))
        return segments, {
            "status": status,
            "reason": "; ".join(reason_parts) or None,
            "verified_message_boundaries": verified,
            "message_count": count,
            "covered_tokens": covered,
            "generation_prompt_tokens": (
                len(full) - final_length if final_length is not None else None
            ),
        }

    def _tools_block(
        self,
        payload: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        with_tools: list[int],
        endpoint: str,
        authorization: str | None,
        chain_key: str,
    ) -> tuple[int, int] | None:
        key = hashlib.sha256(chain_key.encode() + b"\0tools-removed").hexdigest()
        cached = self._tools_cache.get(key)
        if cached is not None:
            return None if cached == "none" else cached
        without, _ = self._render_with(
            endpoint, self._prefix_payload(payload, messages, tools=False), authorization
        )
        block: tuple[int, int] | None = None
        if without is not None and len(without) < len(with_tools):
            prefix = _lcp(with_tools, without)
            suffix = _lcp(with_tools[::-1][: len(with_tools) - prefix], without[::-1][: len(without) - prefix])
            if prefix + suffix == len(without):
                end = len(with_tools) - suffix
                if end > prefix:
                    block = (prefix, end)
        self._tools_cache.put(key, "none" if block is None else block)
        return block


def _split_for_tools(
    segments: list[dict[str, Any]], block: tuple[int, int], source: str
) -> tuple[list[dict[str, Any]], bool]:
    start, end = block
    containing = [
        item for item in segments if item["start_token"] <= start and end <= item["end_token"]
    ]
    overlapping = [
        item for item in segments if item["start_token"] < end and start < item["end_token"]
    ]
    if overlapping and (len(containing) != 1 or overlapping != containing):
        return segments, False
    tools_segment = {
        "kind": "tools",
        "label": "tool definitions",
        "start_token": start,
        "end_token": end,
        "attribution_quality": "exact_token_boundary",
        "source": f"{source}_tools_removal",
    }
    result: list[dict[str, Any]] = []
    for item in segments:
        if containing and item is containing[0]:
            if item["start_token"] < start:
                result.append({**item, "end_token": start, "label": item["label"] + " (before tools)"})
            result.append(tools_segment)
            if end < item["end_token"]:
                result.append({**item, "start_token": end, "label": item["label"] + " (after tools)"})
        else:
            result.append(item)
    if not containing:
        result.append(tools_segment)
        result.sort(key=lambda item: item["start_token"])
    return result, True
