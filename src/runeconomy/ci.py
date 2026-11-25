"""CI verdict summaries and pull-request comments.

``runeconomy ci-summary RESULT`` renders a result as a compact Markdown
verdict table for a CI job summary or a pull-request comment. With
``--github-comment`` it creates or updates one marked comment on the pull
request that triggered a GitHub Actions run, using the REST API and the job's
``GITHUB_TOKEN`` (standard library only).

The summary never upgrades a claim. A compatibility trial is labelled as such
and makes no verdict claim. A result without an intact preregistration record
is labelled as not frozen.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from .experiments import ExperimentValidationError


COMMENT_MARKER = "<!-- runeconomy-verdict -->"
SUMMARY_MODES = ("frozen", "compatibility")
_ICON = {"verified": "✅", "regression": "❌", "inconclusive": "⚠️", "invalid": "⛔"}


class CIError(ExperimentValidationError):
    """The CI summary could not be rendered or posted."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_code = "runeconomy.ci.failed"
        self.public_details = {"reason": message}


def _format(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(item) for item in value[:4]) + "]"
    if isinstance(value, Mapping):
        parts = []
        for key in ("one_sided_lower_bound", "lower_bound", "upper_bound", "difference", "candidate", "baseline", "status"):
            if key in value and not isinstance(value[key], (Mapping, list)):
                parts.append(f"{key.replace('_', ' ')} {_format(value[key])}")
        return "; ".join(parts[:3]) or "recorded"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_summary(result: Mapping[str, Any], *, mode: str = "frozen", title: str | None = None) -> dict[str, Any]:
    if mode not in SUMMARY_MODES:
        raise CIError(f"mode must be one of {', '.join(SUMMARY_MODES)}")
    decision = result.get("decision")
    if not isinstance(decision, Mapping):
        raise CIError("the file is not a RunEconomy result (no decision)")
    verdict = str(decision.get("status", "unknown"))
    checks = [check for check in decision.get("checks", []) if isinstance(check, Mapping)]
    required = [check for check in checks if check.get("required", True)]
    failed = [str(check.get("name")) for check in required if check.get("passed") is False]
    preregistration = result.get("preregistration")
    frozen = isinstance(preregistration, Mapping) and preregistration.get("status") == "intact"
    heading = title or "RunEconomy verdict"
    lines = [COMMENT_MARKER, f"### {heading}", ""]
    if mode == "compatibility":
        lines += [
            "**Compatibility trial.** This checks that the agent satisfies the process, status, trace and evaluator "
            "contract. It is not a benchmark and makes no verdict claim.",
            "",
            f"Recorded status: `{verdict}`",
        ]
    else:
        lines.append(f"**{_ICON.get(verdict, '❔')} `{verdict}`**: {_format(decision.get('reason'))}")
        if not frozen:
            lines += ["", "> Not run under an intact preregistration record. Treat this as an exploratory result."]
    lines += [
        "",
        "| Gate | Required | Result | Observed |",
        "| --- | --- | --- | --- |",
    ]
    for check in checks:
        passed = check.get("passed")
        lines.append(
            "| {name} | {required} | {result} | {observed} |".format(
                name=_format(check.get("name")),
                required="yes" if check.get("required", True) else "no",
                result="pass" if passed is True else "fail" if passed is False else "unknown",
                observed=_format(check.get("observed")),
            )
        )
    lines.append("")
    if frozen:
        lines.append(f"Preregistration record `{preregistration.get('record_sha256')}` (locks intact at run start).")
    lines.append(
        "Only `verified` supports an improvement claim, and only for the recorded workload, backend, hardware and gates."
    )
    return {
        "mode": mode,
        "verdict": verdict,
        "verified": verdict == "verified" and mode == "frozen" and frozen,
        "frozen": frozen,
        "failed_required_checks": failed,
        "markdown": "\n".join(lines) + "\n",
    }


def _pull_request_number(environment: Mapping[str, str]) -> int:
    explicit = environment.get("RUNECONOMY_PR_NUMBER")
    if explicit:
        return int(explicit)
    event_path = environment.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise CIError("GITHUB_EVENT_PATH is not set; run inside a pull_request workflow or set RUNECONOMY_PR_NUMBER")
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CIError("cannot read the GitHub event payload") from exc
    number = (event.get("pull_request") or {}).get("number") or event.get("number")
    if not isinstance(number, int):
        raise CIError("the triggering event is not a pull request")
    return number


def _request(method: str, url: str, token: str, body: Mapping[str, Any] | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "runeconomy-ci",
            **({"Content-Type": "application/json"} if data is not None else {}),
        },
    )
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload = response.read()
    except urllib.error.HTTPError as exc:
        raise CIError(f"GitHub API {method} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise CIError("GitHub API is unreachable") from exc
    return json.loads(payload) if payload else None


def post_pull_request_comment(markdown: str, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Create or update the single marked RunEconomy comment on the pull request."""

    environment = os.environ if environment is None else environment
    token = environment.get("GITHUB_TOKEN")
    repository = environment.get("GITHUB_REPOSITORY")
    if not token or not repository:
        raise CIError("GITHUB_TOKEN and GITHUB_REPOSITORY are required to comment")
    api = environment.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    number = _pull_request_number(environment)
    base = f"{api}/repos/{repository}/issues"
    existing = None
    page = 1
    while existing is None:
        comments = _request("GET", f"{base}/{number}/comments?per_page=100&page={page}", token) or []
        existing = next(
            (
                item for item in comments
                if (item.get("user") or {}).get("login") == "github-actions[bot]"
                and str(item.get("body", "")).startswith(COMMENT_MARKER)
            ),
            None,
        )
        if len(comments) < 100:
            break
        page += 1
    if existing is not None:
        _request("PATCH", f"{base}/comments/{existing['id']}", token, {"body": markdown})
        return {"action": "updated", "pull_request": number, "comment_id": existing["id"]}
    created = _request("POST", f"{base}/{number}/comments", token, {"body": markdown})
    return {"action": "created", "pull_request": number, "comment_id": (created or {}).get("id")}
