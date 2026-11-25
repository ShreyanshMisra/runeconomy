"""Zero-buffer OpenAI-compatible reverse proxy with off-path evidence capture.

See :mod:`runeconomy.proxy.contract` for the wire contract this server
implements. The request path forwards bytes and timestamps. Everything else
belongs to :class:`~runeconomy.proxy.recorder.SessionRecorder`.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .contract import (
    HOP_BY_HOP_HEADERS,
    OUTCOME_CLIENT_DISCONNECTED,
    OUTCOME_COMPLETED,
    OUTCOME_UPSTREAM_HTTP_ERROR,
    OUTCOME_UPSTREAM_INTERRUPTED,
    OUTCOME_UPSTREAM_UNREACHABLE,
    route_for_path,
)
from .recorder import Observation, RecorderError, SessionRecorder
from .upstream import UpstreamTarget


DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_READ_SIZE = 64 * 1024
_STALE_CONNECTION_ERRORS = (
    http.client.RemoteDisconnected,
    BrokenPipeError,
    ConnectionResetError,
)


class _RequestTooLarge(Exception):
    pass


class _InvalidRequestFraming(Exception):
    pass


def _connection_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "_ProxyHTTPServer"

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self._upstream_connection: http.client.HTTPConnection | None = None
        self._upstream_used = False

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self._drop_upstream()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = (
        lambda self: self._proxy()
    )

    # -- upstream connection --------------------------------------------

    def _drop_upstream(self) -> None:
        if self._upstream_connection is not None:
            try:
                self._upstream_connection.close()
            except OSError:
                pass
        self._upstream_connection = None
        self._upstream_used = False

    def _upstream(self) -> tuple[http.client.HTTPConnection, bool]:
        if self._upstream_connection is None:
            self._upstream_connection = self.server.target.connect(
                self.server.upstream_timeout
            )
            self._upstream_used = False
        return self._upstream_connection, self._upstream_used

    # -- request body ----------------------------------------------------

    def _read_body(self) -> bytes:
        limit = self.server.max_request_bytes
        encoding = _connection_tokens(self.headers.get("Transfer-Encoding"))
        if "chunked" in encoding:
            pieces: list[bytes] = []
            total = 0
            while True:
                line = self.rfile.readline(1024)
                if not line:
                    raise _InvalidRequestFraming("chunked request body ended early")
                size_text = line.split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError as exc:
                    raise _InvalidRequestFraming("invalid chunk size") from exc
                if size == 0:
                    while True:
                        trailer = self.rfile.readline(8192)
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                total += size
                if total > limit:
                    raise _RequestTooLarge()
                pieces.append(self.rfile.read(size))
                self.rfile.readline(8)
            return b"".join(pieces)
        length_text = self.headers.get("Content-Length")
        if length_text is None:
            return b""
        try:
            length = int(length_text)
        except ValueError as exc:
            raise _InvalidRequestFraming("invalid Content-Length") from exc
        if length < 0:
            raise _InvalidRequestFraming("negative Content-Length")
        if length > limit:
            raise _RequestTooLarge()
        return self.rfile.read(length) if length else b""

    # -- local error responses -------------------------------------------

    def _local_error(self, status: int, kind: str, message: str) -> None:
        body = json.dumps(
            {"error": {"message": message, "type": kind, "code": status}}
        ).encode("utf-8")
        try:
            self.send_response_only(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError:
            pass
        self.close_connection = True

    # -- the request path ------------------------------------------------

    def _proxy(self) -> None:
        proxy = self.server
        headers_received_ns = time.perf_counter_ns()
        received_unix_ns = time.time_ns()
        route = route_for_path(self.path) if self.command == "POST" else None
        recorder = proxy.recorder
        if proxy.stopping.is_set():
            self._local_error(
                503, "runeconomy_proxy_stopping", "RunEconomy capture proxy is stopping"
            )
            return
        try:
            body = self._read_body()
        except _RequestTooLarge:
            self._local_error(
                413, "runeconomy_proxy_request_too_large", "request body exceeds the proxy limit"
            )
            return
        except (_InvalidRequestFraming, OSError) as exc:
            self._local_error(400, "runeconomy_proxy_bad_request", str(exc))
            return
        received_ns = time.perf_counter_ns()

        observation: Observation | None = None
        if route is not None:
            sequence, exchange_id = recorder.next_identity()
            observation = Observation(
                exchange_id=exchange_id,
                sequence=sequence,
                method=self.command,
                path=self.path.split("?", 1)[0],
                route=route,
                received_unix_ns=received_unix_ns,
                headers_received_ns=headers_received_ns,
                received_ns=received_ns,
                request_headers=list(self.headers.items()),
                request_body=body,
            )
        else:
            recorder.note_passthrough()

        proxy.begin_exchange()
        try:
            self._forward(body, route, observation)
        finally:
            try:
                if observation is not None:
                    if observation.completed_ns is None:
                        observation.completed_ns = time.perf_counter_ns()
                    recorder.submit(observation)
            finally:
                proxy.end_exchange()

    def _forward_headers(self, body: bytes, route: str | None) -> list[tuple[str, str]]:
        nominated = _connection_tokens(self.headers.get("Connection"))
        skip = HOP_BY_HOP_HEADERS | nominated | {"host", "content-length", "expect"}
        if route is not None:
            skip = skip | {"accept-encoding"}
        result = [
            (name, value)
            for name, value in self.headers.items()
            if name.lower() not in skip
        ]
        result.append(("Host", self.server.target.host_header))
        if body or self.command in {"POST", "PUT", "PATCH"}:
            result.append(("Content-Length", str(len(body))))
        if route is not None:
            result.append(("Accept-Encoding", "identity"))
        return result

    def _send_upstream(
        self, body: bytes, route: str | None, observation: Observation | None
    ) -> http.client.HTTPResponse:
        upstream_path = self.server.target.forward_path(self.path)
        headers = self._forward_headers(body, route)
        for attempt in range(2):
            connection, reused = self._upstream()
            try:
                connection.putrequest(
                    self.command, upstream_path, skip_host=True, skip_accept_encoding=True
                )
                for name, value in headers:
                    connection.putheader(name, value)
                connection.endheaders(message_body=body if body else None)
                if observation is not None:
                    observation.upstream_sent_ns = time.perf_counter_ns()
                response = connection.getresponse()
                self._upstream_used = True
                return response
            except _STALE_CONNECTION_ERRORS:
                self._drop_upstream()
                if reused and attempt == 0:
                    if observation is not None:
                        observation.upstream_retried = True
                    continue
                raise
            except BaseException:
                self._drop_upstream()
                raise
        raise http.client.RemoteDisconnected("upstream closed the connection")

    def _forward(
        self, body: bytes, route: str | None, observation: Observation | None
    ) -> None:
        try:
            response = self._send_upstream(body, route, observation)
        except TimeoutError as exc:
            if observation is not None:
                observation.outcome = OUTCOME_UPSTREAM_UNREACHABLE
                observation.error = f"upstream timed out before responding ({type(exc).__name__})"
            self._local_error(
                504, "runeconomy_proxy_upstream_timeout", "upstream did not respond in time"
            )
            return
        except (OSError, http.client.HTTPException) as exc:
            if observation is not None:
                observation.outcome = OUTCOME_UPSTREAM_UNREACHABLE
                observation.error = type(exc).__name__
            self._local_error(
                502,
                "runeconomy_proxy_upstream_unreachable",
                f"RunEconomy capture proxy could not reach the upstream ({type(exc).__name__})",
            )
            return

        headers_ns = time.perf_counter_ns()
        status = response.status
        response_headers = response.getheaders()
        if observation is not None:
            observation.response_headers_ns = headers_ns
            observation.status = status
            observation.response_headers = list(response_headers)
            if status >= 400:
                observation.outcome = OUTCOME_UPSTREAM_HTTP_ERROR

        nominated = _connection_tokens(response.getheader("Connection"))
        no_body = self.command == "HEAD" or status in {204, 304} or 100 <= status < 200
        length_header = None if response.chunked else response.getheader("Content-Length")
        try:
            length = int(length_header) if length_header is not None else None
        except ValueError:
            length = None
        chunked_to_client = (
            not no_body and length is None and self.request_version == "HTTP/1.1"
        )
        try:
            self.send_response_only(status, response.reason)
            for name, value in response_headers:
                key = name.lower()
                if key in HOP_BY_HOP_HEADERS or key in nominated or key == "content-length":
                    continue
                self.send_header(name, value)
            if no_body:
                if length is not None and self.command == "HEAD":
                    self.send_header("Content-Length", str(length))
            elif length is not None:
                self.send_header("Content-Length", str(length))
            elif chunked_to_client:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
        except OSError:
            if observation is not None:
                observation.outcome = OUTCOME_CLIENT_DISCONNECTED
                observation.error = "client closed before response headers"
            self._drop_upstream()
            self.close_connection = True
            return

        if no_body:
            response.close()
            if response.will_close:
                self._drop_upstream()
            return

        chunks = observation.chunks if observation is not None else None
        first_byte_ns: int | None = None
        clean_end = False
        write = self.wfile.write
        while True:
            try:
                data = response.read1(_READ_SIZE)
            except (OSError, http.client.HTTPException) as exc:
                if observation is not None:
                    observation.outcome = OUTCOME_UPSTREAM_INTERRUPTED
                    observation.error = f"upstream body interrupted ({type(exc).__name__})"
                break
            arrived_ns = time.perf_counter_ns()
            if not data:
                clean_end = response.length in (None, 0)
                if not clean_end and observation is not None:
                    observation.outcome = OUTCOME_UPSTREAM_INTERRUPTED
                    observation.error = "upstream body ended before Content-Length"
                break
            if first_byte_ns is None:
                first_byte_ns = arrived_ns
                if observation is not None:
                    observation.first_byte_ns = arrived_ns
            if chunks is not None:
                chunks.append((arrived_ns, data))
            try:
                if chunked_to_client:
                    write(b"%x\r\n" % len(data) + data + b"\r\n")
                else:
                    write(data)
            except OSError:
                if observation is not None:
                    observation.outcome = OUTCOME_CLIENT_DISCONNECTED
                    observation.error = "client closed the connection mid-response"
                break
        if observation is not None:
            observation.completed_ns = time.perf_counter_ns()

        if clean_end:
            if chunked_to_client:
                try:
                    write(b"0\r\n\r\n")
                except OSError:
                    if observation is not None:
                        observation.outcome = OUTCOME_CLIENT_DISCONNECTED
                        observation.error = "client closed before the final chunk"
                    self.close_connection = True
            if response.will_close:
                self._drop_upstream()
            if observation is not None and observation.outcome not in {
                OUTCOME_CLIENT_DISCONNECTED,
                OUTCOME_UPSTREAM_HTTP_ERROR,
            }:
                observation.outcome = OUTCOME_COMPLETED
        else:
            # Abort the upstream request (vLLM cancels generation on disconnect)
            # and never send a terminating chunk for a broken body.
            self._drop_upstream()
            self.close_connection = True


class _ProxyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(
        self,
        address: tuple[str, int],
        *,
        target: UpstreamTarget,
        recorder: SessionRecorder,
        upstream_timeout: float | None,
        max_request_bytes: int,
    ) -> None:
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        self.target = target
        self.recorder = recorder
        self.upstream_timeout = upstream_timeout
        self.max_request_bytes = max_request_bytes
        self.stopping = threading.Event()
        self._in_flight = 0
        self._in_flight_condition = threading.Condition()
        super().__init__(address, _ProxyHandler)

    def begin_exchange(self) -> None:
        with self._in_flight_condition:
            self._in_flight += 1

    def end_exchange(self) -> None:
        with self._in_flight_condition:
            self._in_flight -= 1
            self._in_flight_condition.notify_all()

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._in_flight_condition:
            while self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._in_flight_condition.wait(remaining)
        return True


class CaptureProxy:
    """A local capture proxy bound to one upstream and one session recorder.

    ``start()`` serves on a background thread. ``stop()`` refuses new
    exchanges, waits for in-flight streams to finish, then closes the recorder,
    which drains telemetry and finishes token resolution. A foreground command
    should call ``start()`` and block on its own signal or event, then call
    ``stop()``.
    """

    def __init__(
        self,
        upstream: UpstreamTarget | str,
        recorder: SessionRecorder,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        upstream_timeout: float | None = 600.0,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        self.target = (
            upstream if isinstance(upstream, UpstreamTarget) else UpstreamTarget.parse(upstream)
        )
        if recorder.upstream != self.target:
            raise ValueError("recorder and proxy must use the same upstream target")
        self.recorder = recorder
        self._server = _ProxyHTTPServer(
            (host, port),
            target=self.target,
            recorder=recorder,
            upstream_timeout=upstream_timeout,
            max_request_bytes=max_request_bytes,
        )
        self._thread: threading.Thread | None = None
        self._stopped = False
        self.close_summary: dict[str, Any] | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def base_url(self) -> str:
        """The URL a client should use as its OpenAI ``base_url``."""

        host, port = self.address
        rendered = f"[{host}]" if ":" in host else host
        return f"http://{rendered}:{port}/v1"

    def start(self) -> "CaptureProxy":
        if self._thread is not None:
            raise RuntimeError("capture proxy already started")
        if not self.recorder.started:
            try:
                self.recorder.start()
            except (OSError, RecorderError):
                self._server.server_close()
                raise
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="runeconomy-proxy",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(
        self, *, drain_timeout: float = 30.0, resolution_timeout: float = 120.0
    ) -> dict[str, Any]:
        if self._stopped:
            return self.close_summary or {}
        self._stopped = True
        self._server.stopping.set()
        drained = self._server.wait_idle(drain_timeout)
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        summary = self.recorder.close(resolution_timeout=resolution_timeout)
        summary["in_flight_drained"] = drained
        self.close_summary = summary
        return summary

    def __enter__(self) -> "CaptureProxy":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
