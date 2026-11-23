"""Run one validation while a discovered Modal Server is pinned warm.

This operational helper never assumes the deployed Server name. It resolves
the App, requires exactly one deployed Server, changes that Server by ID, and
verifies the autoscaler after both the opening and closing transitions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _required_json(
    runner: Runner, command: Sequence[str], *, action: str
) -> dict[str, Any]:
    completed = runner(command)
    if completed.returncode != 0:
        raise RuntimeError(f"{action} failed; inspect the Modal CLI error locally")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{action} returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{action} returned a non-object JSON value")
    return payload


def resolve_server(
    app_name: str, runner: Runner = _run
) -> tuple[str, str | None]:
    """Confirm the live App and return its sole Server ID and optional URL."""

    payload = _required_json(
        runner,
        ["uv", "run", "modal", "app", "info", app_name, "--json"],
        action="Modal App discovery",
    )
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("state") != "deployed":
        raise RuntimeError("the selected Modal App is not currently deployed")
    servers = payload.get("servers")
    if not isinstance(servers, dict) or len(servers) != 1:
        raise RuntimeError(
            "the deployed App must expose exactly one Server; select the App explicitly"
        )
    entry = next(iter(servers.values()))
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
        raise RuntimeError("Modal App discovery did not return a Server ID")
    summary = entry.get("summary")
    url = summary.get("web_url") if isinstance(summary, dict) else None
    return entry["id"], url if isinstance(url, str) and url else None


def autoscaler_minimum(server_id: str, runner: Runner = _run) -> int:
    payload = _required_json(
        runner,
        [
            "uv",
            "run",
            "modal",
            "server",
            "autoscaler",
            "settings",
            server_id,
            "--json",
        ],
        action="Modal autoscaler inspection",
    )
    settings = payload.get("current_settings")
    minimum = settings.get("min_containers") if isinstance(settings, dict) else None
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise RuntimeError("Modal autoscaler inspection omitted min_containers")
    return minimum


def set_autoscaler_minimum(
    server_id: str, minimum: int, runner: Runner = _run
) -> None:
    completed = runner(
        [
            "uv",
            "run",
            "modal",
            "server",
            "autoscaler",
            "update",
            server_id,
            "--min-containers",
            str(minimum),
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError("Modal autoscaler update failed; inspect the CLI error locally")
    observed = autoscaler_minimum(server_id, runner)
    if observed != minimum:
        raise RuntimeError(
            f"Modal autoscaler verification expected {minimum} warm container(s), observed {observed}"
        )


def wait_until_healthy(
    url: str,
    *,
    timeout_seconds: float,
    interval_seconds: float,
    runner: Runner = _run,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = clock() + timeout_seconds
    health_url = url.rstrip("/") + "/health"
    while True:
        completed = runner(
            [
                "uv",
                "run",
                "modal",
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                health_url,
            ]
        )
        if completed.returncode == 0:
            return
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("the Modal Server did not become healthy before the deadline")
        sleep(min(interval_seconds, remaining))


def validation_window(
    *,
    app_name: str,
    validation_command: Sequence[str],
    health_url: str | None = None,
    health_timeout_seconds: float = 600.0,
    health_interval_seconds: float = 5.0,
    runner: Runner = _run,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    server_id, discovered_url = resolve_server(app_name, runner)
    url = health_url or discovered_url
    if not url:
        raise RuntimeError(
            "Modal did not expose a Server URL; pass --health-url from the private deployment record"
        )
    if not validation_command:
        raise ValueError("a validation command is required")
    opened = False
    validation_returncode = 1
    try:
        opened = True
        set_autoscaler_minimum(server_id, 1, runner)
        wait_until_healthy(
            url,
            timeout_seconds=health_timeout_seconds,
            interval_seconds=health_interval_seconds,
            runner=runner,
            clock=clock,
            sleep=sleep,
        )
        completed = runner(validation_command)
        validation_returncode = completed.returncode
    finally:
        if opened:
            set_autoscaler_minimum(server_id, 0, runner)
    return validation_returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default="runeconomy-vllm")
    parser.add_argument("--health-url")
    parser.add_argument("--health-timeout", type=float, default=600.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("validation_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.validation_command)
    if command[:1] == ["--"]:
        command = command[1:]
    try:
        return validation_window(
            app_name=args.app,
            validation_command=command,
            health_url=args.health_url,
            health_timeout_seconds=args.health_timeout,
            health_interval_seconds=args.health_interval,
        )
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"validation window failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
