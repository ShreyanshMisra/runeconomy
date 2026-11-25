"""Read-only HTTP helpers shared by adapter discovery (stdlib only)."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..backends import BackendProbeError
from ..startup import StartupRetry


def validate_base_url(base_url: str, *, label: str) -> None:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BackendProbeError(
            "the base URL must be an absolute HTTP or HTTPS URL",
            category="configuration",
            next_action="Rerun with `--base-url http://HOST:PORT` or an HTTPS equivalent.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise BackendProbeError(
            "the base URL must not contain embedded credentials",
            category="configuration",
            next_action="Remove credentials from the URL and supply the key with `--api-key-env NAME`.",
        )
    if parsed.query or parsed.fragment:
        raise BackendProbeError(
            "the base URL must not contain a query string or fragment",
            category="configuration",
            next_action=f"Pass the {label} base URL without query parameters or a fragment.",
        )


def _headers(api_key: str | None, accept: str) -> dict[str, str]:
    headers = {"Accept": accept}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get(
    base_url: str,
    path: str,
    *,
    api_key: str | None,
    timeout_seconds: float,
    label: str,
    startup: StartupRetry | None = None,
    parse_json: bool = True,
) -> tuple[int, Any]:
    """GET one metadata path. Returns (status, body); body is None on errors.

    Authentication failures and unreachable servers raise, with the target
    redacted. Other HTTP statuses are returned for the caller to classify.
    """

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers=_headers(api_key, "application/json" if parse_json else "text/plain"),
        method="GET",
    )
    retry = startup or StartupRetry(0)
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
                status = response.status
            break
        except urllib.error.HTTPError as exc:
            exc.read(2048)
            if exc.code in {502, 503} and retry.available():
                retry.wait()
                continue
            if exc.code in {401, 403}:
                raise BackendProbeError(
                    f"the {label} server rejected the {path} metadata request with HTTP {exc.code}",
                    category="authentication",
                    next_action=(
                        "Set the API key in an environment variable and rerun with "
                        "`--api-key-env NAME`; verify the key may read metadata."
                    ),
                ) from exc
            return exc.code, None
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ) as exc:
            if retry.available():
                retry.wait()
                continue
            reason = getattr(exc, "reason", exc)
            timed_out = isinstance(reason, TimeoutError)
            raise BackendProbeError(
                f"the {label} endpoint did not respond before the timeout"
                if timed_out
                else f"the {label} endpoint could not be reached",
                category="timeout" if timed_out else "connection",
                next_action=(
                    "Increase `--timeout`, verify the server is running, and rerun `runeconomy doctor`."
                    if timed_out
                    else "Verify the base URL, network access, and server process, then rerun `runeconomy doctor`."
                ),
            ) from exc
    if not parse_json:
        return status, raw.decode("utf-8", errors="replace")
    try:
        return status, json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BackendProbeError(
            f"the {label} server returned malformed JSON for {path}",
            category="malformed_response",
            next_action=(
                f"Confirm the URL points to a {label} server rather than a login page or HTML error."
            ),
        ) from exc


def post(
    base_url: str,
    path: str,
    payload: Any,
    *,
    api_key: str | None,
    timeout_seconds: float,
) -> tuple[int, str]:
    """POST one request and return (status, body text). Never raises on HTTP errors."""

    headers = _headers(api_key, "application/json")
    headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(2048).decode("utf-8", errors="replace")
