"""Upstream addressing and a small off-path JSON client.

The capture proxy talks to exactly one upstream: the one the operator named.
Token resolution and identity probes go to that same server and nowhere else.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import urllib.parse
from dataclasses import dataclass
from typing import Any, Mapping


class UpstreamConfigurationError(ValueError):
    """The upstream URL cannot be used safely."""


@dataclass(frozen=True)
class UpstreamTarget:
    """An OpenAI-compatible server root.

    ``base_path`` is the server root *without* a trailing ``/v1``. Clients
    address the proxy exactly as they would address the server's ``/v1`` base
    URL, and the incoming path, which already starts with ``/v1``, is appended
    to ``base_path``. The vLLM utility routes ``/tokenize``, ``/version`` and
    ``/metrics`` live directly under ``base_path``.
    """

    scheme: str
    host: str
    port: int
    base_path: str
    stripped_v1: bool = False

    @classmethod
    def parse(cls, url: str) -> "UpstreamTarget":
        parsed = urllib.parse.urlsplit(url.strip())
        if parsed.scheme not in {"http", "https"}:
            raise UpstreamConfigurationError("upstream URL must use http or https")
        if not parsed.hostname:
            raise UpstreamConfigurationError("upstream URL must include a host")
        if parsed.username is not None or parsed.password is not None:
            raise UpstreamConfigurationError(
                "upstream URL must not embed credentials; clients send their own "
                "Authorization header"
            )
        if parsed.query or parsed.fragment:
            raise UpstreamConfigurationError(
                "upstream URL must not carry a query string or fragment"
            )
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise UpstreamConfigurationError("upstream URL has an invalid port") from exc
        path = parsed.path.rstrip("/")
        stripped = False
        if path.endswith("/v1"):
            path = path[: -len("/v1")]
            stripped = True
        return cls(
            scheme=parsed.scheme,
            host=parsed.hostname,
            port=port,
            base_path=path,
            stripped_v1=stripped,
        )

    @property
    def origin(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        default = 443 if self.scheme == "https" else 80
        suffix = "" if self.port == default else f":{self.port}"
        return f"{self.scheme}://{host}{suffix}"

    @property
    def root_url(self) -> str:
        return self.origin + self.base_path

    @property
    def openai_base_url(self) -> str:
        return self.root_url + "/v1"

    @property
    def host_header(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        default = 443 if self.scheme == "https" else 80
        return host if self.port == default else f"{host}:{self.port}"

    def forward_path(self, incoming_path: str) -> str:
        if not incoming_path.startswith("/"):
            incoming_path = "/" + incoming_path
        return self.base_path + incoming_path

    def connect(self, timeout: float | None) -> http.client.HTTPConnection:
        connection: http.client.HTTPConnection
        if self.scheme == "https":
            connection = http.client.HTTPSConnection(self.host, self.port, timeout=timeout)
        else:
            connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        connection.connect()
        if connection.sock is not None:
            try:
                connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        return connection

    def describe(self) -> dict[str, Any]:
        """Return a record-safe description without credentials or queries."""

        return {
            "origin": self.origin,
            "base_path": self.base_path,
            "openai_base_url": self.openai_base_url,
            "stripped_trailing_v1": self.stripped_v1,
        }


@dataclass(frozen=True)
class JSONResult:
    status: int | None
    payload: Any
    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


class OffPathClient:
    """A keep-alive JSON client for recorder threads.

    Each thread keeps its own connection. The client is never used on the
    request path.
    """

    def __init__(self, target: UpstreamTarget, *, timeout: float = 30.0) -> None:
        self.target = target
        self.timeout = timeout
        self._local = threading.local()

    def _connection(self, fresh: bool = False) -> http.client.HTTPConnection:
        connection = getattr(self._local, "connection", None)
        if fresh and connection is not None:
            connection.close()
            connection = None
        if connection is None:
            connection = self.target.connect(self.timeout)
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        authorization: str | None = None,
        accept: str = "application/json",
        max_bytes: int = 64 * 1024 * 1024,
    ) -> JSONResult:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": accept, "Host": self.target.host_header}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if authorization:
            headers["Authorization"] = authorization
        last_error: str | None = None
        for attempt in range(2):
            try:
                connection = self._connection(fresh=attempt > 0)
                connection.request(
                    method, self.target.forward_path(path), body=body, headers=headers
                )
                response = connection.getresponse()
                raw = response.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    self.close()
                    return JSONResult(response.status, None, None, "response too large")
                if response.will_close:
                    self.close()
                text = raw.decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(text) if text else None
                except json.JSONDecodeError:
                    parsed = None
                return JSONResult(response.status, parsed, text)
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError) as exc:
                # A stale keep-alive connection. Retry once on a fresh socket.
                self.close()
                last_error = type(exc).__name__
                continue
            except (OSError, http.client.HTTPException) as exc:
                self.close()
                return JSONResult(None, None, None, f"{type(exc).__name__}: {exc}")
        return JSONResult(None, None, None, last_error or "connection failed")


_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _prometheus_info(text: str, metric: str) -> dict[str, str] | None:
    prefix = metric + "{"
    for line in text.splitlines():
        if line.startswith(prefix):
            labels = line[len(prefix) : line.rfind("}")]
            return {
                key: value.replace('\\"', '"').replace("\\\\", "\\")
                for key, value in _LABEL.findall(labels)
            }
    return None


def probe_upstream(
    client: OffPathClient, *, authorization: str | None
) -> dict[str, Any]:
    """Collect only identity facts that the upstream reports about itself.

    Model revisions, tokenizer revisions and chat-template digests are not
    exposed by OpenAI-compatible APIs. They stay null rather than being
    inferred from the model name.
    """

    errors: dict[str, str] = {}
    models: list[dict[str, Any]] = []
    owned_by: set[str] = set()
    result = client.request("GET", "/v1/models", authorization=authorization)
    if result.ok and isinstance(result.payload, dict) and isinstance(
        result.payload.get("data"), list
    ):
        for item in result.payload["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            entry: dict[str, Any] = {"id": item["id"]}
            if isinstance(item.get("root"), str):
                entry["root"] = item["root"]
            if isinstance(item.get("max_model_len"), int):
                entry["max_model_len"] = item["max_model_len"]
            if isinstance(item.get("owned_by"), str):
                entry["owned_by"] = item["owned_by"]
                owned_by.add(item["owned_by"])
            models.append(entry)
    else:
        errors["/v1/models"] = result.error or f"HTTP {result.status}"

    cache_config: dict[str, Any] | None = None
    metrics_mention_vllm = False
    metrics = client.request(
        "GET", "/metrics", authorization=authorization, accept="text/plain"
    )
    if metrics.ok and metrics.text:
        metrics_mention_vllm = "vllm:" in metrics.text
        labels = _prometheus_info(metrics.text, "vllm:cache_config_info")
        if labels is not None:
            cache_config = {}
            block_size = labels.get("block_size")
            if block_size is not None and block_size.isdigit():
                cache_config["block_size"] = int(block_size)
            prefix = labels.get("enable_prefix_caching")
            if prefix in {"True", "False", "true", "false"}:
                cache_config["enable_prefix_caching"] = prefix.lower() == "true"
            for key in ("cache_dtype", "prefix_caching_hash_algo", "gpu_memory_utilization"):
                if key in labels:
                    cache_config[key] = labels[key]
    else:
        errors["/metrics"] = metrics.error or f"HTTP {metrics.status}"

    is_vllm = "vllm" in owned_by or metrics_mention_vllm
    version: str | None = None
    version_result = client.request("GET", "/version", authorization=authorization)
    if (
        is_vllm
        and version_result.ok
        and isinstance(version_result.payload, dict)
        and isinstance(version_result.payload.get("version"), str)
    ):
        version = version_result.payload["version"]
    elif not version_result.ok:
        errors["/version"] = version_result.error or f"HTTP {version_result.status}"
    return {
        "backend": "vllm" if is_vllm else "openai_compatible" if models else "unknown",
        "backend_evidence": (
            "models owned_by vllm"
            if "vllm" in owned_by
            else "vllm Prometheus metrics"
            if metrics_mention_vllm
            else "OpenAI-compatible /v1/models"
            if models
            else "no identity endpoint answered"
        ),
        "backend_version": version,
        "models": models,
        "cache_config": cache_config,
        "probe_errors": errors,
    }
