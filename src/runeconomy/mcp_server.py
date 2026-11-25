"""``runeconomy mcp``: a Model Context Protocol server over stdio.

The server exposes RunEconomy as typed MCP tools so a coding agent can drive
import -> init -> patch -> freeze -> plan -> run -> report without editing
JSON. It is implemented with the standard library (JSON-RPC 2.0, one message
per line on stdin/stdout) to keep RunEconomy's dependency footprint small.

Every tool runs the corresponding CLI command in a subprocess with ``--json``
and returns that command's versioned output contract as ``structuredContent``
(see :mod:`runeconomy.cli_contracts`). Running in a subprocess keeps agent and
evaluator child processes from ever writing into the protocol stream, and it
guarantees MCP and CLI behave identically.

Approval boundary
-----------------
``run_experiment`` executes agents and contacts serving backends, which can
spend money. Its first call never executes anything: it returns the planned
attempts, the commands that will run, the declared cost estimate and cost
model, the freeze status, and a single-use ``approval_token`` that is bound to
the exact arguments and to the current digests of the workspace files. Only a
second call that presents that token, after the human approved, executes. A
changed manifest, dataset, evaluator or argument invalidates the token.
Tokens expire after :data:`APPROVAL_TTL_SECONDS` and live only in this server
process.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Mapping, TextIO

from . import __version__


PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_NAME = "runeconomy"
APPROVAL_TTL_SECONDS = 15 * 60
TOOL_TIMEOUT_SECONDS = 6 * 60 * 60


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _bool(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _schema(properties: Mapping[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _annotations(*, read_only: bool, destructive: bool = False, open_world: bool = False, idempotent: bool = False) -> dict[str, bool]:
    return {
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": idempotent,
        "openWorldHint": open_world,
    }


class ToolError(Exception):
    """A tool call that failed before or outside the CLI contract."""


def _argv_doctor(args: Mapping[str, Any]) -> list[str]:
    argv = ["doctor", "--base-url", args["base_url"]]
    if args.get("backend"):
        argv += ["--backend", args["backend"]]
    if args.get("api_key_env"):
        argv += ["--api-key-env", args["api_key_env"]]
    return argv


def _argv_import(args: Mapping[str, Any]) -> list[str]:
    argv = ["import", args["source"], "--output", args["output"]]
    for key, flag in (
        ("type", "--type"),
        ("trace_id", "--trace-id"),
        ("replay_context", "--replay-context"),
        ("proxy_identity", "--proxy-identity"),
        ("workload", "--workload"),
        ("model", "--model"),
    ):
        if args.get(key):
            argv += [flag, str(args[key])]
    return argv


def _argv_init(args: Mapping[str, Any]) -> list[str]:
    argv = ["experiment", "init", "--output", args["output"]]
    if args.get("mode") == "agent-process":
        argv += ["--mode", "agent-process"]
    for key, flag in (
        ("from_evidence", "--from-evidence"),
        ("finding_id", "--finding-id"),
        ("baseline", "--baseline"),
        ("candidate", "--candidate"),
        ("dataset", "--dataset"),
        ("evaluator", "--evaluator"),
        ("order", "--order"),
        ("optimization_target", "--optimization-target"),
        ("latency_semantics", "--latency-semantics"),
    ):
        if args.get(key):
            argv += [flag, str(args[key])]
    for key, flag in (
        ("repetitions", "--repetitions"),
        ("seed", "--seed"),
        ("min_task_success_rate", "--min-task-success-rate"),
        ("max_task_success_regression", "--max-task-success-regression"),
        ("min_completed_rate", "--min-completed-rate"),
        ("max_latency_regression_ms", "--max-latency-regression-ms"),
        ("cost_ceiling_usd", "--cost-ceiling-usd"),
    ):
        if args.get(key) is not None:
            argv += [flag, str(args[key])]
    return argv


def _argv_patch(args: Mapping[str, Any]) -> list[str]:
    argv = ["experiment", "patch", args["workspace"]]
    for path, value in (args.get("set") or {}).items():
        argv += ["--set", f"{path}={json.dumps(value)}"]
    for path in args.get("unset") or []:
        argv += ["--unset", path]
    if args.get("reason"):
        argv += ["--reason", args["reason"]]
    if args.get("repin"):
        argv.append("--repin")
    if args.get("dry_run"):
        argv.append("--dry-run")
    return argv


def _argv_freeze(args: Mapping[str, Any]) -> list[str]:
    argv = ["experiment", "freeze", args["workspace"]]
    for path in args.get("lock") or []:
        argv += ["--lock", path]
    if args.get("allow_incomplete_design"):
        argv.append("--allow-incomplete-design")
    if args.get("reason"):
        argv += ["--reason", args["reason"]]
    return argv


def _argv_plan(args: Mapping[str, Any]) -> list[str]:
    if args.get("examples"):
        return ["experiment", "plan", "--examples"]
    if not args.get("manifest"):
        raise ToolError("plan requires manifest unless examples is true")
    return ["experiment", "plan", args["manifest"]]


def _argv_validate(args: Mapping[str, Any]) -> list[str]:
    argv = ["experiment", "validate", args["manifest"]]
    if args.get("doctor_report"):
        argv += ["--doctor-report", args["doctor_report"]]
    if args.get("allow_exec_evaluators"):
        argv.append("--allow-exec-evaluators")
    return argv


def _argv_run(args: Mapping[str, Any]) -> list[str]:
    argv = ["experiment", "run", args["manifest"], "--output", args["output"]]
    for key, flag in (
        ("require_frozen", "--require-frozen"),
        ("require_verified", "--require-verified"),
        ("allow_exec_evaluators", "--allow-exec-evaluators"),
        ("server_metrics", "--server-metrics"),
    ):
        if args.get(key):
            argv.append(flag)
    for item in args.get("server_metrics_url") or []:
        argv += ["--server-metrics-url", item]
    return argv


def _argv_report(args: Mapping[str, Any]) -> list[str]:
    argv = ["experiment", "report", args["result"], "--output", args["output"]]
    if args.get("format"):
        argv += ["--format", args["format"]]
    return argv


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "doctor": {
        "description": "Read-only capability probe of a serving backend. Never generates text or mutates the server.",
        "inputSchema": _schema(
            {
                "base_url": _string("Backend base URL."),
                "backend": {"type": "string", "enum": ["vllm", "sglang", "openai_compat"]},
                "api_key_env": _string("Name of the environment variable that holds the API key (never the key)."),
            },
            ("base_url",),
        ),
        "annotations": _annotations(read_only=True, open_world=True, idempotent=True),
        "argv": _argv_doctor,
    },
    "import_evidence": {
        "description": "Import traces or a proxy session into an evidence bundle and assess what it can support.",
        "inputSchema": _schema(
            {
                "source": _string("Trace, export, directory or proxy session path."),
                "output": _string("Evidence bundle path to write."),
                "type": {"type": "string", "enum": ["auto", "openinference", "vllm", "prefixwise", "phoenix", "proxy"]},
                "trace_id": _string("Trace or proxy workflow to select."),
                "replay_context": _string("Static replay design JSON for a proxy session."),
                "proxy_identity": _string("Trace identity JSON for a proxy session."),
                "workload": _string("Workload label."),
                "model": _string("Model name for count-only sources."),
            },
            ("source", "output"),
        ),
        "annotations": _annotations(read_only=False, idempotent=True),
        "argv": _argv_import,
    },
    "analyze": {
        "description": "Analyze a normalized trace for inference waste findings.",
        "inputSchema": _schema({"trace": _string("Normalized trace JSON path.")}, ("trace",)),
        "annotations": _annotations(read_only=True, idempotent=True),
        "argv": lambda args: ["analyze", args["trace"]],
    },
    "init_experiment": {
        "description": "Create an experiment workspace from evidence (static) or from explicit agent commands (agent-process).",
        "inputSchema": _schema(
            {
                "output": _string("New workspace directory."),
                "mode": {"type": "string", "enum": ["static-request", "agent-process"]},
                "from_evidence": _string("Evidence bundle (static mode)."),
                "finding_id": _string("Finding to select when the evidence has several."),
                "baseline": _string("Baseline agent command (agent-process)."),
                "candidate": _string("Candidate agent command (agent-process)."),
                "dataset": _string("Task dataset JSON (agent-process)."),
                "evaluator": _string("Evaluator file (agent-process)."),
                "order": {"type": "string", "enum": ["randomized", "alternating"]},
                "optimization_target": {"type": "string", "enum": ["successful-workflow-latency", "engine-change-latency"]},
                "latency_semantics": {"type": "string", "enum": ["superiority", "noninferiority"]},
                "repetitions": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer"},
                "min_task_success_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "max_task_success_regression": {"type": "number", "minimum": 0, "maximum": 1},
                "min_completed_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "max_latency_regression_ms": {"type": "number", "minimum": 0},
                "cost_ceiling_usd": {"type": "number", "minimum": 0},
            },
            ("output",),
        ),
        "annotations": _annotations(read_only=False),
        "argv": _argv_init,
    },
    "validate_experiment": {
        "description": "Validate a workspace or manifest and return corrective issues. Never executes.",
        "inputSchema": _schema(
            {
                "manifest": _string("Workspace directory or manifest JSON."),
                "doctor_report": _string("Saved doctor JSON to apply without probing."),
                "allow_exec_evaluators": _bool("Allow trusted python and command evaluators."),
            },
            ("manifest",),
        ),
        "annotations": _annotations(read_only=True, idempotent=True),
        "argv": _argv_validate,
    },
    "patch_experiment": {
        "description": "Change manifest fields with schema validation and a provenance entry. Refused once frozen.",
        "inputSchema": _schema(
            {
                "workspace": _string("Workspace directory."),
                "set": {"type": "object", "description": "Map of field path (a.b[0].c or /a/b/0/c) to its new JSON value."},
                "unset": {"type": "array", "items": {"type": "string"}},
                "reason": _string("Why the change is made (recorded in provenance)."),
                "repin": _bool("Re-pin agent-process dataset and evaluator digests."),
                "dry_run": _bool("Validate and report without writing."),
            },
            ("workspace",),
        ),
        "annotations": _annotations(read_only=False),
        "argv": _argv_patch,
    },
    "freeze_experiment": {
        "description": "Lock digests and gates and write the preregistration record. Irreversible for this workspace.",
        "inputSchema": _schema(
            {
                "workspace": _string("Workspace directory."),
                "lock": {"type": "array", "items": {"type": "string"}, "description": "Extra files to digest-lock (agent code)."},
                "allow_incomplete_design": _bool("Freeze a needs_design_input or ready_for_calibration design as incomplete."),
                "reason": _string("Why the design is frozen."),
            },
            ("workspace",),
        ),
        "annotations": _annotations(read_only=False, destructive=True),
        "argv": _argv_freeze,
    },
    "plan_experiment": {
        "description": "Assess design readiness, power and preregistration without executing anything.",
        "inputSchema": _schema(
            {"manifest": _string("Workspace directory or manifest JSON."), "examples": _bool("Return the worked planning examples instead.")}
        ),
        "annotations": _annotations(read_only=True, idempotent=True),
        "argv": _argv_plan,
    },
    "run_experiment": {
        "description": (
            "Execute an experiment. The first call returns a compute and cost estimate plus an approval_token and "
            "executes nothing. Show the estimate to the human; only after they approve, call again with the token."
        ),
        "inputSchema": _schema(
            {
                "manifest": _string("Workspace directory or manifest JSON."),
                "output": _string("Result directory or .json path."),
                "require_frozen": _bool("Refuse unless the workspace has an intact preregistration record."),
                "require_verified": _bool("Exit 3 unless the verdict is verified."),
                "allow_exec_evaluators": _bool("Allow trusted python and command evaluators."),
                "server_metrics": _bool("Sample the static backend's Prometheus metrics during load blocks."),
                "server_metrics_url": {"type": "array", "items": {"type": "string"}},
                "approval_token": _string("Token from the estimate call, presented only after human approval."),
            },
            ("manifest", "output"),
        ),
        "annotations": _annotations(read_only=False, open_world=True),
        "argv": _argv_run,
    },
    "render_report": {
        "description": "Render a Markdown or HTML report from a result JSON.",
        "inputSchema": _schema(
            {
                "result": _string("Result JSON path."),
                "output": _string("Report path (.md or .html)."),
                "format": {"type": "string", "enum": ["markdown", "html"]},
            },
            ("result", "output"),
        ),
        "annotations": _annotations(read_only=False, idempotent=True),
        "argv": _argv_report,
    },
    "inspect_result": {
        "description": "Read a result JSON and return its verdict, decisive gate, required checks and preregistration.",
        "inputSchema": _schema({"result": _string("Result JSON path.")}, ("result",)),
        "annotations": _annotations(read_only=True, idempotent=True),
        "argv": None,
    },
}


def _digest_file(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _manifest_state(manifest: Path) -> dict[str, Any]:
    """Digests that bind an approval to the exact files that will run."""

    state: dict[str, Any] = {}
    if manifest.is_dir():
        for name in ("manifest.json", "workspace.json", "preregistration.json"):
            state[name] = _digest_file(manifest / name)
        try:
            payload = json.loads((manifest / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        for label in ("dataset", "evaluator"):
            reference = payload.get(label) if isinstance(payload, Mapping) else None
            if isinstance(reference, Mapping) and isinstance(reference.get("path"), str):
                path = Path(reference["path"])
                state[label] = _digest_file(path if path.is_absolute() else manifest / path)
    else:
        state["manifest"] = _digest_file(manifest)
    return state


class RunEconomyMCPServer:
    def __init__(
        self,
        *,
        command: list[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.command = command or [sys.executable, "-m", "runeconomy"]
        self.clock = clock
        self.environment = dict(os.environ if environment is None else environment)
        self._approvals: dict[str, tuple[float, bytes]] = {}
        self.initialized = False

    # -- tool execution -------------------------------------------------

    def _cli(self, argv: list[str]) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
        completed = subprocess.run(
            [*self.command, *argv, "--json"],
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT_SECONDS,
            env=self.environment,
            stdin=subprocess.DEVNULL,
        )

        def parse(text: str) -> dict[str, Any] | None:
            text = text.strip()
            if not text:
                return None
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                # stderr may carry progress lines before the error object.
                start = text.rfind("\n{")
                try:
                    value = json.loads(text[start + 1 :] if start >= 0 else text)
                except json.JSONDecodeError:
                    return None
            return value if isinstance(value, dict) else None

        return completed.returncode, parse(completed.stdout), parse(completed.stderr)

    def _approval_digest(self, arguments: Mapping[str, Any]) -> bytes:
        bound = {key: value for key, value in arguments.items() if key != "approval_token"}
        message = json.dumps(
            {"arguments": bound, "state": _manifest_state(Path(arguments["manifest"]))},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(message).digest()

    def _estimate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        manifest = Path(arguments["manifest"])
        code, plan, error = self._cli(["experiment", "plan", str(manifest)])
        if plan is None:
            raise ToolError("the experiment could not be planned; validate it first")
        sample = plan.get("sample") or {}
        cost = plan.get("cost_estimate") or {}
        commands: dict[str, Any] = {}
        cost_model = None
        freeze = {"status": "not_frozen"}
        if manifest.is_dir():
            try:
                payload = json.loads((manifest / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            for variant in ("baseline", "candidate"):
                item = payload.get(variant)
                if isinstance(item, Mapping) and isinstance(item.get("command"), str):
                    commands[variant] = item["command"]
            cost_model = payload.get("cost_model")
            if (manifest / "preregistration.json").exists():
                from .freeze import verify_freeze

                state = verify_freeze(manifest)
                freeze = {"status": state["status"], "record_sha256": state["record_sha256"]}
        now = self.clock()
        self._approvals = {key: entry for key, entry in self._approvals.items() if entry[0] > now}
        token = "approve-" + secrets.token_urlsafe(32)
        self._approvals[token] = (now + APPROVAL_TTL_SECONDS, self._approval_digest(arguments))
        return {
            "status": "approval_required",
            "executed": False,
            "planner_state": plan.get("state"),
            "planned_pairs": sample.get("planned_pairs"),
            "total_attempts": sample.get("total_attempts"),
            "cost_estimate": cost,
            "cost_model": cost_model,
            "local_commands": commands,
            "executes_local_code": bool(commands) or bool(arguments.get("allow_exec_evaluators")),
            "contacts_serving_backend": True,
            "freeze": freeze,
            "approval_token": token,
            "approval_expires_in_seconds": APPROVAL_TTL_SECONDS,
            "instructions": (
                "Show this estimate to the human. Call run_experiment again with the same arguments and "
                "approval_token only after they explicitly approve. Any change to the arguments or the "
                "workspace files invalidates the token."
            ),
        }

    def _inspect(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            result = json.loads(Path(arguments["result"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"cannot read result: {exc}") from exc
        decision = result.get("decision") if isinstance(result, Mapping) else None
        if not isinstance(decision, Mapping):
            raise ToolError("the file is not a RunEconomy result (no decision)")
        checks = [
            {
                "name": check.get("name"),
                "passed": check.get("passed"),
                "required": check.get("required", True),
            }
            for check in decision.get("checks", [])
            if isinstance(check, Mapping)
        ]
        return {
            "verdict": decision.get("status"),
            "reason": decision.get("reason"),
            "decisive_gate": decision.get("decisive_gate"),
            "failed_required_checks": [c["name"] for c in checks if c["required"] and c["passed"] is False],
            "checks": checks,
            "preregistration": result.get("preregistration"),
            "experiment_id": result.get("experiment_id"),
        }

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            raise ToolError(f"unknown tool {name!r}")
        properties = spec["inputSchema"]["properties"]
        unknown = sorted(set(arguments) - set(properties))
        missing = [key for key in spec["inputSchema"]["required"] if key not in arguments]
        expected = {"object": Mapping, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}
        wrong = sorted(
            key
            for key, value in arguments.items()
            if key in properties
            and properties[key].get("type") in expected
            and (
                not isinstance(value, expected[properties[key]["type"]])
                or (isinstance(value, bool) and properties[key]["type"] in {"integer", "number"})
            )
        )
        if wrong:
            raise ToolError(
                "invalid arguments: "
                + "; ".join(f"{key} must be of type {properties[key]['type']}" for key in wrong)
            )
        if unknown or missing:
            raise ToolError(
                "invalid arguments: "
                + "; ".join(
                    part
                    for part in (
                        f"unknown {', '.join(unknown)}" if unknown else "",
                        f"missing {', '.join(missing)}" if missing else "",
                    )
                    if part
                )
            )
        if name == "inspect_result":
            return self._result(self._inspect(arguments), is_error=False)
        if name == "run_experiment":
            token = arguments.get("approval_token")
            if not token:
                return self._result(self._estimate(arguments), is_error=False)
            approval = self._approvals.pop(token, None)
            if approval is None or self.clock() >= approval[0] or self._approval_digest(arguments) != approval[1]:
                raise ToolError(
                    "approval_token is unknown, expired, already used, or no longer matches the arguments "
                    "and workspace files; request a new estimate"
                )
        code, payload, error = self._cli(spec["argv"](arguments))
        if payload is not None:
            return self._result({**payload, "exit_code": code}, is_error=False)
        if error is not None:
            return self._result({**error, "exit_code": code}, is_error=True)
        raise ToolError(f"{name} produced no JSON result (exit code {code})")

    @staticmethod
    def _result(payload: Mapping[str, Any], *, is_error: bool) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}],
            "structuredContent": dict(payload),
            "isError": is_error,
        }

    # -- JSON-RPC --------------------------------------------------------

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        identifier = message.get("id")
        params = message.get("params") or {}
        if identifier is None:  # notification
            if method == "notifications/initialized":
                self.initialized = True
            return None
        try:
            if method == "initialize":
                requested = params.get("protocolVersion")
                version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
                result: Any = {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": __version__},
                    "instructions": (
                        "RunEconomy verifies whether a serving change keeps an agent's task quality while making it "
                        "cheaper or faster. Use validate/plan/patch/freeze freely. run_experiment needs explicit "
                        "human approval of its estimate. Never invent evidence values; unknown stays unknown."
                    ),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": name,
                            "description": spec["description"],
                            "inputSchema": spec["inputSchema"],
                            "annotations": spec["annotations"],
                        }
                        for name, spec in TOOL_SPECS.items()
                    ]
                }
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if not isinstance(name, str) or not isinstance(arguments, Mapping):
                    return _error(identifier, -32602, "tools/call requires a name and an arguments object")
                if name not in TOOL_SPECS:
                    return _error(identifier, -32602, f"unknown tool {name!r}")
                try:
                    result = self.call_tool(name, arguments)
                except (ToolError, subprocess.TimeoutExpired, OSError) as exc:
                    result = self._result({"error": str(exc)}, is_error=True)
            else:
                return _error(identifier, -32601, f"method not found: {method}")
        except Exception as exc:  # pragma: no cover - last-resort protocol safety
            return _error(identifier, -32603, f"internal error: {type(exc).__name__}")
        return {"jsonrpc": "2.0", "id": identifier, "result": result}

    def serve(self, stdin: TextIO | BinaryIO, stdout: TextIO) -> None:
        for raw in stdin:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response: dict[str, Any] | None = _error(None, -32700, "parse error")
            else:
                if isinstance(message, list):
                    responses = [r for r in (self.handle(item) for item in message if isinstance(item, Mapping)) if r]
                    if responses:
                        stdout.write(json.dumps(responses) + "\n")
                        stdout.flush()
                    continue
                response = self.handle(message) if isinstance(message, Mapping) else _error(None, -32600, "invalid request")
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()


def _error(identifier: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}


def serve_stdio() -> int:
    RunEconomyMCPServer().serve(sys.stdin, sys.stdout)
    return 0
