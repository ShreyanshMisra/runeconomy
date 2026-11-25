"""Command-line interface for RunEconomy."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import __version__
from . import cli_contracts
from .analysis import analyze_trace
from .attempt_status import AttemptStatusError
from .adapters import create_adapter, format_report as format_adapter_report
from .backends import BackendProbeError, discover_vllm_capabilities
from .comparison import run_comparison
from .evidence import import_evidence, load_evidence_bundle
from .errors import ERROR_CODE_VERSION, classify_exception
from .evaluators import EvaluatorInfrastructureError
from .experiment_init import initialize_experiment, select_finding
from .cost_model import parse_cost_model
from .freeze import freeze_workspace, require_intact_freeze, verify_freeze
from .manifest_patch import (
    PatchOperation,
    load_patch_file,
    parse_assignment,
    patch_workspace,
)
from .experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    run_experiment,
)
from .importers import (
    import_openinference,
    import_ledger_jsonl,
    import_vllm_jsonl,
)
from .models import Trace, TraceValidationError
from .planning import plan_experiment, worked_examples
from .reports import (
    load_demo_result,
    render_demo_html,
    render_experiment_html,
    render_experiment_markdown,
)
from .tutorial import (
    DEFAULT_TUTORIAL_PORT,
    TutorialError,
    materialize_tutorial,
    serve_tutorial,
)
from .support_bundle import SupportBundleError, create_support_bundle
from .vllm import VLLMClient, VLLMError
from .workspaces import (
    agent_run_arguments,
    create_agent_workspace,
    create_workspace_from_evidence,
    create_workspace_from_finding,
    load_workspace,
    resolve_agent_manifest,
    resolve_static_manifest,
    validate_experiment_artifact,
)


def _load_trace(path: Path) -> Trace:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise TraceValidationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    return Trace.from_dict(payload)


def _format_human(report: dict[str, Any]) -> str:
    summary = report["summary"]
    coverage = summary["observed_cache_coverage"]
    coverage_text = "unknown" if coverage is None else f"{coverage:.1%}"
    lines = [
        f"RunEconomy analyze · {summary['finding_count']} FINDINGS",
        f"Trace: {report['trace_id']} ({report['workload']})",
        f"Requests: {summary['request_count']}",
        f"Input tokens: {summary['total_input_tokens']}",
        f"Eligible prefix tokens: {summary['total_eligible_prefix_tokens']}",
        f"Actual cached tokens: {summary['total_actual_cached_tokens']}",
        f"Observed cache coverage: {coverage_text}",
        "Critical path: "
        + (
            "not provided"
            if summary["critical_path_duration_ms"] is None
            else f"{summary['critical_path_duration_ms']:.1f} ms"
        ),
        f"Findings: {summary['finding_count']}",
    ]
    for finding in report["findings"]:
        lines.extend(
            [
                "",
                f"[{finding['evidence_level']}/{finding['confidence']}] "
                f"{finding['title']} ({finding.get('finding_id', finding['request_id'])})",
                "Critical path: "
                + (
                    "unknown"
                    if finding["on_critical_path"] is None
                    else "yes"
                    if finding["on_critical_path"]
                    else "no"
                ),
                finding["summary"],
                f"Target: {finding.get('optimization_target', 'unknown')}",
                f"Experiment: {finding['suggested_experiment']}",
            ]
        )
    return "\n".join(lines)


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print one versioned JSON result on stdout (see `runeconomy schema`)",
    )


def _json_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json_output", False)) or getattr(args, "format", None) == "json"


def _emit(contract: str, payload: Mapping[str, Any]) -> None:
    print(cli_contracts.render(contract, payload))


def _convert_payload(args: argparse.Namespace, payload: Mapping[str, Any]) -> dict[str, Any]:
    trace = "requests" in payload
    return {
        "command": args.command,
        "artifact": "normalized_trace" if trace else "findings",
        "output": str(args.output),
        "request_count": len(payload["requests"]) if trace else None,
        "finding_count": None if trace else len(payload.get("findings", [])),
        "next_command": (
            f"runeconomy analyze {args.output}"
            if trace
            else f"runeconomy experiment init --from-finding {args.output} --output experiment/"
        ),
    }


def _init_json(
    args: argparse.Namespace, descriptor: Mapping[str, Any], *, output_kind: str = "workspace"
) -> bool:
    if not _json_mode(args):
        return False
    _emit(
        "experiment-init",
        {
            "mode": descriptor.get("mode", "static_request"),
            "output": str(args.output),
            "output_kind": output_kind,
            "generation_status": descriptor.get("generation_status", "runnable"),
            "unresolved_fields": list(descriptor.get("unresolved_fields") or []),
            "next_command": descriptor["next_command"],
        },
    )
    return True


def _freeze_state(path: Path) -> dict[str, Any]:
    try:
        state = verify_freeze(path)
    except ExperimentValidationError:
        return {"status": "drifted", "record_sha256": None}
    return {"status": state["status"], "record_sha256": state["record_sha256"]}


def _metrics_observers(args: argparse.Namespace, client: Any = None) -> tuple[Any, ...]:
    from .telemetry import SERVER_TARGET, ServerMetricsSampler, prometheus_url_scraper

    targets: dict[str, Any] = {}
    for item in args.server_metrics_url:
        name, separator, url = item.partition("=")
        if not separator or name not in {"baseline", "candidate", SERVER_TARGET} or not url:
            raise ExperimentValidationError(
                "--server-metrics-url must be TARGET=URL with TARGET baseline, candidate, or server"
            )
        targets[name] = prometheus_url_scraper(url)
    if args.server_metrics:
        scrape = getattr(client, "scrape_metrics", None)
        if not callable(scrape):
            raise ExperimentValidationError(
                "--server-metrics requires a static backend adapter with Prometheus metrics; "
                "use --server-metrics-url for agent-process runs"
            )
        targets.setdefault(SERVER_TARGET, scrape)
    if not targets:
        return ()
    try:
        return (ServerMetricsSampler(targets, interval_seconds=args.server_metrics_interval),)
    except ValueError as exc:
        raise ExperimentValidationError(str(exc)) from exc


def _run_json(
    args: argparse.Namespace,
    mode: str,
    result: Mapping[str, Any],
    result_path: Path,
    artifacts_path: Path | None,
    checkpoint: Path | None,
    freeze_state: Mapping[str, Any],
) -> int:
    verified = result["decision"]["status"] == "verified"
    _emit(
        "experiment-run",
        {
            "mode": mode,
            "verdict": result["decision"]["status"],
            "result": str(result_path),
            "artifacts": str(artifacts_path) if artifacts_path else None,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "require_verified": bool(args.require_verified),
            "verified": verified,
            "freeze": {"status": freeze_state["status"], "record_sha256": freeze_state["record_sha256"]},
        },
    )
    return 3 if args.require_verified and not verified else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runeconomy")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a normalized trace")
    analyze.add_argument("trace", type=Path)
    analyze.add_argument("--format", choices=("human", "json"), default="human")
    analyze.add_argument("--json", dest="format", action="store_const", const="json")
    analyze.add_argument("--output", type=Path)

    unified_import = subparsers.add_parser(
        "import",
        help="import evidence and explain which RunEconomy workflows it can support",
    )
    unified_import.add_argument("source", type=Path)
    unified_import.add_argument("--output", type=Path, required=True)
    unified_import.add_argument(
        "--type",
        choices=("auto", "openinference", "vllm", "prefixwise", "phoenix", "proxy"),
        default="auto",
    )
    unified_import.add_argument("--trace-id", help="select a trace or proxy workflow")
    unified_import.add_argument("--workload", default="unknown")
    unified_import.add_argument("--backend", default="unknown")
    unified_import.add_argument("--model")
    unified_import.add_argument("--block-size", type=int, default=16)
    unified_import.add_argument(
        "--replay-context", type=Path, help="read operator-supplied static replay design JSON for a proxy session"
    )
    unified_import.add_argument(
        "--proxy-identity", type=Path, help="read operator-supplied trace identity JSON for a proxy session"
    )
    unified_import.add_argument(
        "--idle-gap-ms", type=float, help="override the proxy fallback grouping gap (default: 10000 ms)"
    )
    unified_import.add_argument("--format", choices=("human", "json"), default="human")
    unified_import.add_argument("--json", dest="format", action="store_const", const="json")

    proxy = subparsers.add_parser("proxy", help="capture OpenAI-compatible requests through a local proxy")
    proxy.add_argument("--upstream", required=True, help="upstream HTTP(S) URL, with or without /v1")
    proxy_mode = proxy.add_mutually_exclusive_group(required=True)
    proxy_mode.add_argument("--record", type=Path, help="path to a new JSONL session")
    proxy_mode.add_argument("--attempt", action="store_true", help="wrap an agent-process command after --")
    proxy.add_argument("--port", type=int, help="local listen port (default: 8767, use 0 for an available port)")
    proxy.add_argument("--token-resolution", choices=("background", "on_close", "off"), default="background")
    proxy.add_argument("--payload-capture", choices=("full", "digest_only"), default="full")
    proxy.add_argument("--redact", action="append", default=[], metavar="MODULE:FUNCTION", help="apply a trusted local body redaction hook")
    proxy.add_argument("agent_command", nargs=argparse.REMAINDER, metavar="COMMAND")

    import_vllm = subparsers.add_parser(
        "import-vllm-jsonl",
        help="convert exact-token vLLM capture records into a normalized trace",
    )
    import_vllm.add_argument("captures", type=Path)
    import_vllm.add_argument("--output", type=Path, required=True)
    import_vllm.add_argument("--trace-id", required=True)
    import_vllm.add_argument("--workload", default="unknown")
    import_vllm.add_argument("--model", required=True)
    import_vllm.add_argument("--model-revision")
    import_vllm.add_argument("--tokenizer")
    import_vllm.add_argument("--tokenizer-revision")
    import_vllm.add_argument("--chat-template-digest")
    import_vllm.add_argument("--backend-version")
    import_vllm.add_argument("--block-size", type=int, default=16)
    success = import_vllm.add_mutually_exclusive_group()
    success.add_argument("--success", dest="success", action="store_true")
    success.add_argument("--failure", dest="success", action="store_false")
    import_vllm.set_defaults(success=None)

    import_oi = subparsers.add_parser(
        "import-openinference",
        help="convert one OpenInference/OpenTelemetry OTLP JSON trace",
    )
    import_oi.add_argument("trace", type=Path)
    import_oi.add_argument("--output", type=Path, required=True)
    import_oi.add_argument("--trace-id")
    import_oi.add_argument("--workload", default="unknown")
    import_oi.add_argument("--backend", default="unknown")
    import_oi.add_argument("--block-size", type=int, default=16)

    import_ledger = subparsers.add_parser(
        "import-ledger-jsonl",
        help="convert cache-ledger JSONL exports into structured findings",
    )
    import_ledger.add_argument("export", type=Path)
    import_ledger.add_argument("--output", type=Path, required=True)

    import_prefixwise = subparsers.add_parser(
        "import-prefixwise-jsonl",
        help="import documented Prefixwise ledger evidence without reconstructing prompts",
    )
    import_prefixwise.add_argument("export", type=Path)
    import_prefixwise.add_argument("--output", type=Path, required=True)

    capture = subparsers.add_parser(
        "capture-vllm", help="render and execute one vLLM request"
    )
    capture.add_argument("request", type=Path)
    capture.add_argument("--base-url", required=True)
    capture.add_argument("--api-key-env")
    capture.add_argument(
        "--api-dialect",
        choices=("responses", "chat_completions"),
        default="responses",
    )
    capture.add_argument("--timeout", type=float, default=300.0)
    capture.add_argument(
        "--startup-timeout",
        type=float,
        default=0.0,
        help="seconds to keep retrying while a cold server starts (default: 0)",
    )
    capture.add_argument("--output", type=Path)
    capture.add_argument(
        "--append-trace-jsonl",
        type=Path,
        help="append an importable envelope containing the request and capture",
    )
    capture.add_argument("--request-id")
    capture.add_argument("--invoked-tool", action="append", default=[])

    demo = subparsers.add_parser(
        "demo", help="render the bundled CPU-only experiment report"
    )
    demo.add_argument(
        "--case",
        choices=("rejection", "historical"),
        default="rejection",
        help="bundled case to render (default: frozen invalid proof)",
    )
    demo.add_argument("--output", type=Path, default=Path("runeconomy-demo.html"))
    demo.add_argument("--open", action="store_true", help="open the report in a browser")

    tutorial = subparsers.add_parser(
        "tutorial", help="materialize or serve the deterministic local tutorial"
    )
    tutorial_commands = tutorial.add_subparsers(
        dest="tutorial_command", required=True
    )
    tutorial_init = tutorial_commands.add_parser(
        "init", help="create a sanitized tutorial directory"
    )
    tutorial_init.add_argument("output", type=Path)
    tutorial_init.add_argument(
        "--port", type=int, default=DEFAULT_TUTORIAL_PORT
    )
    tutorial_serve = tutorial_commands.add_parser(
        "serve", help="start the localhost-only deterministic tutorial server"
    )
    tutorial_serve.add_argument(
        "--port", type=int, default=DEFAULT_TUTORIAL_PORT
    )

    support = subparsers.add_parser(
        "support-bundle",
        help="create a local privacy-minimized diagnostic archive",
    )
    support.add_argument("workspace", type=Path)
    support.add_argument("--output", type=Path, required=True)
    support.add_argument(
        "--doctor-report",
        type=Path,
        help="include only allowlisted states from a saved doctor JSON report",
    )
    support.add_argument(
        "--validation-report",
        type=Path,
        help="include only allowlisted states from a saved validation JSON report",
    )
    support.add_argument("--format", choices=("human", "json"), default="human")
    support.add_argument("--json", dest="format", action="store_const", const="json")

    doctor = subparsers.add_parser(
        "doctor", help="inspect a serving backend without running inference"
    )
    doctor.add_argument("--base-url", required=True)
    doctor.add_argument(
        "--backend",
        choices=("vllm", "sglang", "openai_compat"),
        default="vllm",
        help=(
            "backend kind (default vllm). openai_compat takes the provider's API "
            "base URL, usually ending in /v1"
        ),
    )
    doctor.add_argument("--api-key-env")
    doctor.add_argument("--timeout", type=float, default=10.0)
    doctor.add_argument(
        "--startup-timeout",
        type=float,
        default=0.0,
        help="seconds to retry 502/503 and connection failures during cold start",
    )
    doctor.add_argument("--format", choices=("human", "json"), default="human")
    doctor.add_argument("--json", dest="format", action="store_const", const="json")

    compare = subparsers.add_parser(
        "compare", help="execute and compare real baseline and candidate agents"
    )
    compare.add_argument("--baseline", required=True, help="baseline agent command")
    compare.add_argument("--candidate", required=True, help="candidate agent command")
    compare.add_argument("--dataset", type=Path, required=True)
    compare.add_argument("--evaluator", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--artifacts-dir", type=Path)
    compare.add_argument("--workdir", type=Path, default=Path.cwd())
    compare.add_argument("--repetitions", type=int, default=1)
    compare.add_argument(
        "--order", choices=("randomized", "alternating"), default="randomized"
    )
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--timeout", type=float, default=300.0)
    compare.add_argument("--evaluator-timeout", type=float, default=30.0)
    compare.add_argument("--max-task-success-regression", type=float, default=0.05)
    compare.add_argument("--min-task-success-rate", type=float, default=0.8)
    compare.add_argument("--min-completed-rate", type=float, default=1.0)
    compare.add_argument(
        "--optimization-target",
        choices=("successful-workflow-latency", "engine-change-latency"),
        default="successful-workflow-latency",
    )
    compare.add_argument(
        "--latency-semantics",
        choices=("superiority", "noninferiority"),
        default="superiority",
    )
    compare.add_argument("--max-latency-regression-ms", type=float, default=0.0)
    compare.add_argument("--resume", action="store_true")
    compare.add_argument(
        "--require-verified",
        action="store_true",
        help="return exit code 3 unless the closed-loop verdict is verified",
    )

    experiment = subparsers.add_parser(
        "experiment",
        help="run a controlled baseline-versus-candidate comparison",
    )
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    initialize = experiment_commands.add_parser(
        "init", help="create an experiment workspace from evidence or explicit agent commands"
    )
    initialize.add_argument("--from-finding", type=Path)
    initialize.add_argument("--from-evidence", type=Path)
    initialize.add_argument(
        "--mode", choices=("static-request", "agent-process"), default="static-request"
    )
    initialize.add_argument("--finding-id")
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--baseline", help="baseline agent command")
    initialize.add_argument("--candidate", help="candidate agent command")
    initialize.add_argument("--dataset", type=Path)
    initialize.add_argument("--evaluator", type=Path)
    initialize.add_argument("--repetitions", type=int, default=1)
    initialize.add_argument("--order", choices=("randomized", "alternating"), default="randomized")
    initialize.add_argument("--seed", type=int, default=0)
    initialize.add_argument("--timeout", type=float, default=300.0)
    initialize.add_argument("--evaluator-timeout", type=float, default=30.0)
    initialize.add_argument("--max-task-success-regression", type=float, default=0.05)
    initialize.add_argument("--min-task-success-rate", type=float, default=0.8)
    initialize.add_argument("--min-completed-rate", type=float, default=1.0)
    initialize.add_argument(
        "--optimization-target",
        choices=("successful-workflow-latency", "engine-change-latency"),
        default="successful-workflow-latency",
    )
    initialize.add_argument(
        "--latency-semantics",
        choices=("superiority", "noninferiority"),
        default="superiority",
    )
    initialize.add_argument("--max-latency-regression-ms", type=float, default=0.0)
    initialize.add_argument("--cost-ceiling-usd", type=float)
    initialize.add_argument(
        "--min-identity-grade",
        choices=("operator-verified", "provider-declared"),
        help=(
            "preregister identity policy 0.1 (agent-process): the weakest "
            "serving-identity grade a valid verdict may rest on"
        ),
    )
    validate = experiment_commands.add_parser("validate", help="validate a manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--format", choices=("human", "json"), default="human")
    validate.add_argument("--json", dest="format", action="store_const", const="json")
    validate.add_argument(
        "--doctor-report",
        type=Path,
        help="read a saved `runeconomy doctor --json` result without probing the server",
    )
    validate.add_argument(
        "--allow-exec-evaluators",
        action="store_true",
        help="allow trusted python and command evaluators",
    )
    plan = experiment_commands.add_parser(
        "plan", help="assess claim-design readiness without executing the experiment"
    )
    plan_input = plan.add_mutually_exclusive_group(required=True)
    plan_input.add_argument("manifest", type=Path, nargs="?")
    plan_input.add_argument(
        "--examples", action="store_true", help="show worked sample-size and completion scenarios"
    )
    plan.add_argument("--format", choices=("human", "json"), default="human")
    plan.add_argument("--json", dest="format", action="store_const", const="json")
    run = experiment_commands.add_parser("run", help="run a manifest against vLLM")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument(
        "--startup-timeout",
        type=float,
        default=0.0,
        help="seconds to keep retrying while a cold server starts (default: 0)",
    )
    run.add_argument(
        "--checkpoint",
        type=Path,
        help="JSONL recovery journal (defaults beside the output file)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="resume the deterministic schedule from an existing checkpoint",
    )
    run.add_argument(
        "--allow-exec-evaluators",
        action="store_true",
        help="allow trusted python and command evaluators",
    )
    run.add_argument(
        "--require-verified",
        action="store_true",
        help="return exit code 3 unless every configured acceptance gate passes",
    )
    report = experiment_commands.add_parser(
        "report", help="render a static experiment report"
    )
    report.add_argument("result", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--format", choices=("markdown", "html"))
    report.add_argument("--open", action="store_true", help="open an HTML report in a browser")

    patch_command = experiment_commands.add_parser(
        "patch",
        help="apply schema-validated manifest field updates and record their provenance",
    )
    patch_command.add_argument("workspace", type=Path)
    patch_command.add_argument(
        "--set",
        dest="patch_set",
        action="append",
        default=[],
        metavar="PATH=JSON",
        help="set a field to a JSON value, e.g. acceptance.min_task_success_rate=0.9",
    )
    patch_command.add_argument(
        "--set-string",
        dest="patch_set_string",
        action="append",
        default=[],
        metavar="PATH=TEXT",
        help="set a field to a literal string without JSON quoting",
    )
    patch_command.add_argument(
        "--unset", dest="patch_unset", action="append", default=[], metavar="PATH", help="remove a field"
    )
    patch_command.add_argument(
        "--patch-file", type=Path, help="read operations from a patch JSON file (applied before flags)"
    )
    patch_command.add_argument("--reason", help="short reason recorded in provenance.jsonl")
    patch_command.add_argument(
        "--repin",
        action="store_true",
        help="re-pin agent-process dataset and evaluator digests to the current file contents",
    )
    patch_command.add_argument(
        "--dry-run", action="store_true", help="validate and report the patch without writing it"
    )

    freeze_command = experiment_commands.add_parser(
        "freeze",
        help="lock digests and gates and write the preregistration record",
    )
    freeze_command.add_argument("workspace", type=Path)
    freeze_command.add_argument(
        "--lock",
        dest="lock_files",
        action="append",
        type=Path,
        default=[],
        metavar="FILE",
        help="also digest-lock a file the run depends on (agent code, runner, allocator)",
    )
    freeze_command.add_argument(
        "--allow-incomplete-design",
        action="store_true",
        help="freeze a needs_design_input or ready_for_calibration design and record it as incomplete",
    )
    freeze_command.add_argument("--reason", help="short reason recorded in the preregistration record")

    run.add_argument(
        "--server-metrics",
        action="store_true",
        help="sample the static backend's Prometheus metrics during concurrent load blocks",
    )
    run.add_argument(
        "--server-metrics-url",
        action="append",
        default=[],
        metavar="TARGET=URL",
        help="sample a /metrics URL during load blocks; TARGET is baseline, candidate, or server",
    )
    run.add_argument(
        "--server-metrics-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="seconds between metric samples (default: 1)",
    )
    run.add_argument(
        "--require-frozen",
        action="store_true",
        help="refuse to run unless the workspace has an intact preregistration record",
    )

    subparsers.add_parser(
        "mcp",
        help="serve RunEconomy as Model Context Protocol tools over stdio",
    )

    ci_summary = subparsers.add_parser(
        "ci-summary", help="render a result as a Markdown verdict table, optionally as a PR comment"
    )
    ci_summary.add_argument("result", type=Path)
    ci_summary.add_argument("--mode", choices=("frozen", "compatibility"), default="frozen")
    ci_summary.add_argument("--title")
    ci_summary.add_argument("--output", type=Path, help="also write the Markdown to this file")
    ci_summary.add_argument(
        "--github-comment",
        action="store_true",
        help="create or update the marked RunEconomy comment on the triggering pull request",
    )

    schema_command = subparsers.add_parser(
        "schema", help="list the versioned --json output contracts or print one JSON Schema"
    )
    schema_command.add_argument("name", nargs="?", help="contract name, e.g. experiment-validate")

    for command_parser in (
        import_vllm,
        import_oi,
        import_ledger,
        import_prefixwise,
        capture,
        demo,
        tutorial_init,
        compare,
        initialize,
        run,
        report,
        proxy,
        ci_summary,
        patch_command,
        freeze_command,
        schema_command,
    ):
        _add_json_flag(command_parser)
    return parser


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ExperimentValidationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExperimentValidationError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExperimentValidationError(f"{path} must contain a JSON object")
    return payload


def _api_key(environment_name: str | None) -> str | None:
    if environment_name is None:
        return None
    value = os.environ.get(environment_name)
    if not value:
        raise ExperimentValidationError(
            f"API key environment variable {environment_name!r} is not set"
        )
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExperimentValidationError(f"cannot write {path}: {exc}") from exc


def _write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ExperimentValidationError(f"cannot write {path}: {exc}") from exc


def _progress(message: str) -> None:
    if sys.stderr.isatty():
        print(f"RunEconomy · {message}", file=sys.stderr, flush=True)


def _startup_retry_notice(remaining_seconds: float) -> None:
    print(
        "RunEconomy · the server is not accepting requests yet; retrying for up "
        f"to {remaining_seconds:.0f} more seconds (--startup-timeout)",
        file=sys.stderr,
        flush=True,
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ExperimentValidationError(f"cannot append to {path}: {exc}") from exc


def _require_trusted_evaluators(
    manifest: ExperimentManifest, allow_exec_evaluators: bool
) -> None:
    if manifest.uses_executable_evaluators() and not allow_exec_evaluators:
        raise ExperimentValidationError(
            "manifest contains python or command evaluators that execute local code; "
            "review the manifest and pass --allow-exec-evaluators to continue"
        )


def _format_capabilities(base_url: str, capabilities: Mapping[str, Any]) -> str:
    del base_url  # Endpoint details are intentionally omitted from presentation output.

    def line(label: str, key: str) -> str:
        value = capabilities[key]
        return f"{label:<27} {value['status']:<11} {value['evidence']}"

    models = capabilities.get("model_ids") or []
    readiness = capabilities["readiness"]
    status = readiness["status"].upper()
    capability_labels = (
        ("Responses API", "responses"),
        ("Chat Completions API", "chat_completions"),
        ("Responses render", "responses_render"),
        ("Chat Completions render", "chat_completions_render"),
        ("Tokenize", "tokenize"),
        ("Cache telemetry", "cache_telemetry"),
        ("Per-request timing", "per_request_timing"),
        ("Cache reset", "cache_reset"),
        ("Cache salt", "cache_salt"),
    )
    lines = [
        f"RunEconomy doctor · {status}",
        readiness["summary"],
        "Target details: redacted",
        f"Backend: vLLM {capabilities.get('backend_version') or 'unknown version'}",
        f"Models: {', '.join(models) if models else 'unknown'}",
        "",
        "Capability                  Status      Evidence",
        "--------------------------  ----------  ----------------------------------------",
        *(line(label, key) for label, key in capability_labels),
        "",
        "Identity",
        "Model revision, tokenizer/template identity, and engine configuration "
        "must be pinned separately in the experiment manifest unless reported above.",
        "",
        "Experiment modes:",
    ]
    for mode in capabilities["experiment_modes"]:
        if mode["status"] != "supported":
            lines.append(f"  {mode['mode']}: {mode['status']}")
        else:
            safety = "safe" if mode["safe"] else "manual/private only"
            lines.append(f"  {mode['mode']}: {mode['status']} ({safety})")
        lines.append(f"    {mode['note']}")
    lines.extend(["", "What to do next"])
    # Only actions that change what the operator can do are listed here. The
    # complete per-capability and identity guidance stays in the JSON output.
    actions = list(readiness.get("next_actions") or [])
    if readiness["status"] != "ready":
        for label, key in capability_labels[:5]:
            capability = capabilities[key]
            if capability["status"] == "unknown" and capability.get("next_action"):
                actions.append(f"{label}: {capability['next_action']}")
    if not actions:
        actions.append(
            "Record the detected identities in the manifest, then validate the experiment before execution."
        )
    actions = list(dict.fromkeys(actions))
    lines.extend(f"  {index}. {action}" for index, action in enumerate(actions, 1))
    lines.append("  Run with --json for guidance on every capability and identity field.")
    lines.extend(
        [
            "",
            "Safety: doctor performed metadata and non-inference render probes only; "
            "it did not generate text, reset caches, or mutate the server.",
        ]
    )
    return "\n".join(lines)


def _doctor_error_payload(exc: Exception) -> dict[str, Any]:
    descriptor = classify_exception(exc, stage="doctor")
    if isinstance(exc, BackendProbeError):
        category = exc.category
        next_action = exc.next_action
    else:
        category = "configuration"
        next_action = (
            "Set the named environment variable, then rerun `runeconomy doctor` "
            "with the same arguments."
        )
    return {
        "schema_version": "0.1",
        "error_code_version": ERROR_CODE_VERSION,
        "status": "error",
        "error": {
            "code": descriptor.code,
            "category": category,
            "stage": "doctor",
            "message": descriptor.summary,
            "evidence_complete": False,
            "next_action_code": descriptor.next_action_code,
            "next_action": next_action,
        },
    }


def _format_doctor_error(payload: Mapping[str, Any]) -> str:
    error = payload["error"]
    return "\n".join(
        [
            "RunEconomy doctor · ERROR",
            f"Code: {error['code']}",
            f"What failed: {error['category'].replace('_', ' ')}.",
            f"Why: {error['message']}",
            "Evidence: capability discovery is incomplete; no compatibility verdict was produced.",
            f"What to do next: {error['next_action']}",
            "Safety: no inference, cache reset, or server mutation was attempted.",
        ]
    )


def _format_evidence_assessment(payload: Mapping[str, Any]) -> str:
    assessment = payload["capability_assessment"]
    trace = payload.get("trace", {})
    analysis = payload.get("analysis", {})
    runs = payload.get("runs", [])
    if isinstance(runs, list) and runs:
        requests = [
            request
            for run in runs
            if isinstance(run, Mapping)
            for request in run.get("trace", {}).get("requests", [])
        ]
        spans = [
            span
            for run in runs
            if isinstance(run, Mapping)
            for span in run.get("trace", {}).get("spans", [])
        ]
    else:
        requests = trace.get("requests", []) if isinstance(trace, Mapping) else []
        spans = trace.get("spans", []) if isinstance(trace, Mapping) else []
    findings = (
        payload.get("findings", []) if runs else analysis.get("findings", [])
    )
    lines = [
        "RunEconomy import",
        f"Imported {len(requests)} inference request(s), {len(spans)} work span(s), and {len(findings)} finding(s)",
    ]
    proxy_session = payload.get("proxy_session")
    if isinstance(proxy_session, Mapping):
        lines.append(f"Proxy workflows: {proxy_session['workflow_count']}.")
        warnings = proxy_session.get("warnings", [])
        for warning in warnings:
            lines.append(f"WARNING: {warning}")
        for limitation in proxy_session["limitations"]:
            if limitation in warnings:
                continue
            lines.append(f"Capture limitation: {limitation}")
        for finding in findings:
            lines.append(f"Finding {finding['finding_id']} ({finding['rule_id']})")
            if reason := finding.get("unsupported_reason"):
                lines.append(f"  Replay input: {reason}")
        if len(findings) > 1:
            lines.append("Select a finding with experiment init --finding-id.")
    quality = assessment.get("evidence_quality")
    if isinstance(quality, Mapping):
        lines.extend(
            [
                f"Overall evidence grade: {quality.get('overall_grade', 'unresolved')}",
                "",
                "Field evidence                     Grade               Source",
                "---------------------------------  ------------------  ----------------------------------------",
            ]
        )
        fields = quality.get("fields", {})
        if isinstance(fields, Mapping):
            for name, field in fields.items():
                if isinstance(field, Mapping):
                    lines.append(
                        f"{str(name).replace('_', ' ').title():<34} "
                        f"{str(field.get('grade', 'unresolved')):<19} "
                        f"{field.get('source', 'unknown')}"
                    )
    lines.extend(
        [
            "",
            "Evidence capability                 Status      Evidence",
            "----------------------------------  ----------  ----------------------------------------",
        ]
    )
    labels = {
        "basic_diagnosis": "Basic diagnosis",
        "structural_prompt_attribution": "Structural prompt attribution",
        "exact_token_cache_analysis": "Exact token cache analysis",
        "critical_path_analysis": "Critical-path analysis",
        "static_request_replay": "Static request replay",
        "complete_agent_comparison": "Complete agent comparison",
        "public_verification_claim": "Public verification claim",
    }
    for key, capability in assessment["capabilities"].items():
        lines.append(
            f"{labels.get(key, key):<34} {str(capability['status']).title():<11} {capability['evidence']}"
        )
    lines.extend(
        [
            "",
            "Runnable experiment: " + ("Yes" if assessment["runnable_experiment"] else "No"),
            f"Next action: {assessment['next_action']}",
        ]
    )
    return "\n".join(lines)


def _format_validation_report(report: Mapping[str, Any], path: Path) -> str:
    status = str(report["status"])
    headline = {
        "runnable": "Valid and runnable",
        "unresolved": "Structurally valid but unresolved",
        "invalid": "Invalid",
        "unsupported_backend": "Valid but unsupported by the current backend",
        "claim_evidence_incomplete": "Valid but missing claim evidence",
    }.get(status, status.replace("_", " ").title())
    lines = [
        f"RunEconomy experiment validate · {headline}",
        f"Artifact: {path}",
        f"Mode: {report.get('mode', 'unknown')}",
    ]
    for issue in report.get("issues", []):
        lines.extend(
            [
                "",
                f"Code: {issue['code']}",
                f"Field: {issue['field_path']}",
                f"Why it matters: {issue['reason']}",
                "Blocks execution: " + ("Yes" if issue["execution_blocking"] else "No"),
                "Weakens claim: " + ("Yes" if issue["claim_weakening"] else "No"),
                f"Safest next action: {issue['next_action']}",
            ]
        )
        if issue.get("command"):
            lines.append(f"Command: {issue['command']}")
    if report.get("next_command"):
        lines.extend(["", f"What to do next: {report['next_command']}"])
    return "\n".join(lines)


def _format_planning_examples(examples: Mapping[str, Any]) -> str:
    small = examples["n40"]
    bounds = small["best_case_all_pairs_agree"]["margin_0.05"]
    historical = examples["n180"]
    completion = historical["completion"]
    failure = completion["scenario_failure_model"]
    rate = failure["infrastructure_failure_rate"] + failure["application_failure_rate"]
    historical_power = historical["scenario_power"]["margin_0.05"]
    historical_assumptions = historical["scenario_power"]["assumptions"]
    proof = examples["proof_3_scenario"]
    methods = proof["minimum_pairs"]
    power = proof["power_at_minimum"]["paired_score_exact"]["quality_gates"]
    assumptions = proof["assumptions"]
    contract = proof["contract"]
    return "\n".join(
        [
            "RunEconomy planning examples",
            "",
            "40 pairs, all successful: at a 5 pp margin the one-sided lower bounds are "
            f"{bounds['paired_exact']['one_sided_lower_bound'] * 100:.2f} pp (paired_exact) "
            f"and {bounds['paired_score_exact']['one_sided_lower_bound'] * 100:.2f} pp "
            "(paired_score_exact). Both miss the gate.",
            "180 pairs: Proof 2 tolerated "
            f"{completion['declared_tolerance_per_variant']} non-completions per variant. "
            f"An illustrative {rate:.2%} failure rate gives a model-derived tolerance of "
            f"{completion['model_derived_tolerance_per_variant']} at a 95% per-variant target.",
            f"At {historical_assumptions['expected_success_rate']:.0%} expected success and "
            f"{historical_assumptions['expected_discordance_rate']:.0%} discordance, "
            "joint task-quality power is "
            f"{historical_power['paired_exact']['quality_gates']:.1%} with paired_exact "
            f"and {historical_power['paired_score_exact']['quality_gates']:.1%} with paired_score_exact.",
            "Proof 2 remains invalid under its frozen contract.",
            "",
            "Proof 3 FP8 vs BF16 scenario: "
            f"{assumptions['expected_success_rate']:.0%} expected success, "
            f"{assumptions['expected_discordance_rate']:.0%} discordance, "
            f"{assumptions['expected_difference'] * 100:g} pp expected success difference, "
            f"absolute floor {contract['min_task_success_rate']:.0%}, "
            f"non-inferiority margin {contract['max_task_success_regression'] * 100:g} pp.",
            f"At {proof['target_power']:.0%} target task-quality power: "
            f"{methods['paired_score_exact']} pairs with paired_score_exact "
            f"({power:.1%} power), versus {methods['paired_exact']} with paired_exact.",
            f"Model-derived tolerance at the score-method sample: "
            f"{proof['completion']['model_derived_tolerance_per_variant']} per variant.",
            f"Illustrative cost: ${proof['cost_estimate']['estimated_usd']:.2f} "
            "at Proof 2's measured cost per attempt. Calibrate on the new workload before freezing.",
        ]
    )


def _format_experiment_plan(plan: Mapping[str, Any]) -> str:
    sample = plan.get("sample")
    success = plan.get("task_success")
    performance = plan.get("performance")
    unit = plan.get("experimental_unit")
    claim = plan.get("proposed_claim", {})
    declarations = plan.get("design_declarations")
    lines = [
        f"RunEconomy experiment plan · {str(plan['state']).replace('_', ' ').title()}",
        "This is a design-readiness assessment, not an experiment verdict.",
        "",
        "Proposed claim: "
        f"{claim.get('level', 'unknown')} claim using {claim.get('primary_metric', 'unknown')}.",
    ]
    if isinstance(unit, Mapping) and isinstance(sample, Mapping):
        lines.extend(
            [
                f"Experimental unit: {unit['name']}.",
                "Planned sample: "
                f"{sample['unique_task_count']} unique task(s), "
                f"{sample['repetitions']} repetition(s), "
                f"{sample['planned_pairs']} pair(s), "
                f"{sample['total_attempts']} attempt(s).",
            ]
        )
    if isinstance(success, Mapping):
        required = success.get("minimum_successful_units")
        required_text = (
            "not achievable with this sample"
            if required is None
            else f"at least {required} of {success['planned_units']} planned units"
        )
        lines.extend(
            [
                "Task-success requirement: "
                f"{required_text} must clear a {success['minimum_lower_bound']:.2%} "
                "exact lower-confidence-bound gate.",
                "All observed successes still do not imply a true 100% success rate.",
                f"Paired task-success method: {success['method_id']}.",
            ]
        )
    sizing = plan.get("sample_size")
    if sizing and sizing["status"] == "scenario_estimate":
        gate = sizing["gate"]
        minimum = sizing["minimum_pairs_for_target_power"]
        minimum_text = (
            f"{minimum} pairs" if minimum is not None else "no recommendation within the search limit"
        )
        lines.append(
            f"Task-success power ({gate.replace('_', ' ')}): "
            f"{sizing['planned_power'][gate]:.1%} at {sizing['planned_pairs']} pairs "
            f"using {sizing['assumptions']['source']} assumptions. "
            f"At {sizing['target_power']:.0%} target power: {minimum_text}."
        )
        all_gates = sizing["planned_power"]["all_gates_lower_bound"]
        if all_gates is not None and len(sizing["planned_power"]["components"]) == 3:
            lines.append(f"Modeled all-gates power lower bound: {all_gates:.1%}.")
    elif sizing:
        lines.append(f"Task-success power unknown: {sizing['reason']}.")
    completion = plan.get("completion")
    completion_contract = plan.get("completion_model")
    model = completion_contract.get("expected_failure_model") if completion_contract else None
    if completion:
        tolerance = completion["maximum_failed_or_incomplete_attempts_per_variant"]
        lines.append(f"Declared completion tolerance: {tolerance} per variant.")
    if model and model["status"] == "derived":
        lines.append(
            f"Model-derived tolerance: {model['derived_max_noncompleted_attempts_per_variant']} "
            f"per variant at {model['pass_probability_target_per_variant']:.0%} per-variant "
            f"pass probability (min_completed_rate {model['derived_min_completed_rate']!r}, "
            f"source: {model['source']})."
        )
    if completion_contract:
        classification = completion_contract["common_mode_paired_failures"]
        lines.append(f"Common-mode completion policy: {classification}.")
    cost = plan.get("cost_estimate")
    if cost and cost["status"] == "estimate":
        lines.append(
            f"Estimated cost at planned sample: ${cost['estimated_usd']:.2f} "
            f"({cost['source']})."
        )
        if cost["estimated_usd_at_minimum_pairs"] is not None:
            lines.append(
                f"Estimated cost at recommended sample: ${cost['estimated_usd_at_minimum_pairs']:.2f}."
            )
    if plan.get("preregistration"):
        lines.append("Preregistration draft: available under preregistration in the JSON output.")
    if isinstance(performance, Mapping):
        lines.extend(
            [
                "Performance evidence: "
                f"{performance['primary_metric']} across at most "
                f"{performance['maximum_analyzable_pairs']} pairs where both variants succeed.",
                "Task failures reduce the analyzable latency sample.",
            ]
        )
        if performance["power"]["status"] == "unknown":
            lines.append(
                "Unknown: latency power requires a calibration effect and paired variability estimate."
            )
    if isinstance(declarations, Mapping):
        lines.append(
            "Held-out status: "
            f"{str(declarations['heldout_status']).replace('_', ' ')}. "
            "RunEconomy cannot independently verify whether data was inspected."
        )
    issues = plan.get("issues", [])
    unknown_count = sum(
        issue.get("severity") == "warning" for issue in issues
    )
    lines.append(f"Unknown or unresolved design items: {unknown_count}.")
    if issues:
        lines.extend(["", "Design issues:"])
        for issue in issues:
            lines.append(
                f"- [{str(issue['severity']).title()}] {issue['code']} · "
                f"{issue['why']} Next: {issue['next_action']}"
            )
    lines.extend(
        [
            "",
            f"Planning state: {plan['state']}",
            f"What to do next: {plan['next_action_code']}",
        ]
    )
    return "\n".join(lines)


def _result_output_path(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        return output, output.with_name(output.stem + "-artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output / "result.json", output / "artifacts"


def _next_action_for_error(args: argparse.Namespace) -> str:
    command = getattr(args, "command", None)
    if command == "analyze":
        return "Check that the input is normalized trace JSON, then rerun `runeconomy analyze TRACE`."
    if command == "import":
        return "Pass the correct --type when auto-detection is ambiguous, then rerun `runeconomy import SOURCE --output evidence.json`."
    if command and command.startswith("import-"):
        return "Review the source format and required flags, then rerun the same import command."
    if command == "compare":
        return "Inspect the preserved artifacts and status files, correct the agent or evaluator failure, then rerun `runeconomy compare`."
    if command == "experiment":
        subcommand = getattr(args, "experiment_command", None)
        if subcommand == "init":
            return "Import evidence first, then run `runeconomy experiment init --from-evidence evidence.json --output experiment/`."
        if subcommand == "validate":
            return "Correct the manifest field named above, then rerun `runeconomy experiment validate MANIFEST`."
        if subcommand == "patch":
            return "Correct the patch path or value named above; nothing was written. Check with `runeconomy experiment patch WORKSPACE --dry-run` before applying."
        if subcommand == "freeze":
            return "Resolve the validation or planning issues named above with `runeconomy experiment patch`, then rerun `runeconomy experiment freeze WORKSPACE`."
        if subcommand == "plan":
            return "Correct the workspace named above, then rerun `runeconomy experiment plan WORKSPACE`."
        if subcommand == "run":
            return "Preserve the checkpoint and partial result, run `runeconomy doctor` against the backend, then retry or resume only after correcting the cause."
        return "Verify the result JSON, then rerun `runeconomy experiment report`."
    if command == "capture-vllm":
        return "Run `runeconomy doctor` against the same backend, correct the reported capability or authentication issue, then retry the capture."
    if command == "tutorial":
        if getattr(args, "tutorial_command", None) == "init":
            return "Choose a new empty directory, then rerun `runeconomy tutorial init DIRECTORY`."
        return "Free the port or create a matching tutorial and server on another port."
    if command == "support-bundle":
        return "Review the support input, choose a new .zip output path, and rerun without uploading any private artifacts."
    return "Correct the reported input or configuration and rerun the command."


def _error_stage(args: argparse.Namespace) -> str:
    command = str(getattr(args, "command", "command"))
    if command == "experiment":
        return f"experiment.{getattr(args, 'experiment_command', 'unknown')}"
    if command == "tutorial":
        return f"tutorial.{getattr(args, 'tutorial_command', 'unknown')}"
    return command


_FREEZE_NEXT_ACTIONS = {
    "runeconomy.freeze.drift": "Restore the locked files listed in error.details, or create a new workspace and freeze it; a frozen contract never changes.",
    "runeconomy.freeze.required": "Run `runeconomy experiment freeze WORKSPACE`, then rerun with --require-frozen.",
    "runeconomy.workspace.frozen": "Create a new workspace for the changed design; the frozen workspace and its preregistration record stay as they are.",
}


def _cli_error_payload(exc: Exception, args: argparse.Namespace) -> dict[str, Any]:
    stage = _error_stage(args)
    descriptor = classify_exception(exc, stage=stage)
    next_action = (
        "Recapture with a unique X-RunEconomy-Workflow-Id per concurrent workflow "
        "or propagate a distinct W3C traceparent trace ID."
        if descriptor.code == "runeconomy.evidence.workflow_ambiguous"
        else _FREEZE_NEXT_ACTIONS.get(descriptor.code) or _next_action_for_error(args)
    )
    error: dict[str, Any] = {
        "code": descriptor.code,
        "category": descriptor.category,
        "stage": stage,
        "message": descriptor.summary,
        "evidence_complete": False,
        "next_action_code": descriptor.next_action_code,
        "next_action": next_action,
    }
    # Only exceptions that opt in (patch and freeze) carry content-free details.
    details = getattr(exc, "public_details", None)
    if isinstance(details, Mapping):
        error["details"] = dict(details)
    return {
        "schema_version": "0.1",
        "error_code_version": ERROR_CODE_VERSION,
        "status": "error",
        "error": error,
    }


def _format_cli_error(exc: Exception, args: argparse.Namespace) -> str:
    error = _cli_error_payload(exc, args)["error"]
    detail = error.get("details", {}).get("reason")
    return "\n".join(
        [
            "RunEconomy · ERROR",
            f"Code: {error['code']}",
            f"What failed: {getattr(args, 'command', 'command')} did not complete.",
            f"Why: {error['message']}",
            *([f"Detail: {detail}"] if detail else []),
            "Evidence: output may be absent or incomplete; do not treat a partial artifact as a verdict.",
            f"What to do next: {error['next_action']}",
        ]
    )


def _run_capture_proxy(args: argparse.Namespace) -> int:
    from .proxy import (
        AttemptCaptureError,
        CaptureProxy,
        RedactionPolicy,
        SessionRecorder,
        UpstreamTarget,
        load_redaction_hook,
        run_agent_under_capture,
    )
    from .proxy.recorder import RecorderError

    command = args.agent_command
    if command and command[0] == "--":
        command = command[1:]
    try:
        if args.attempt and (not command or args.port is not None):
            raise ValueError("--attempt requires an agent command after -- and uses an available local port")
        if not args.attempt and command:
            raise ValueError("an agent command requires --attempt")
        if args.port is not None and not 0 <= args.port <= 65535:
            raise ValueError("--port must be between 0 and 65535")
        target = UpstreamTarget.parse(args.upstream)
        options = {
            "token_resolution": args.token_resolution,
            "payload_capture": args.payload_capture,
            "redaction": RedactionPolicy(hooks=tuple(load_redaction_hook(spec) for spec in args.redact)),
        }
        if args.attempt:
            code, capture = run_agent_under_capture(command, target, recorder_options=options)
            if capture.reason:
                print(f"RunEconomy proxy · {capture.reason}", file=sys.stderr)
            exit_code = 128 - code if code < 0 else code
            if _json_mode(args):
                _emit(
                    "proxy",
                    {"mode": "attempt", "agent_exit_code": exit_code, "reason": capture.reason, "summary": None, "session": None, "gaps": None},
                )
            return exit_code

        recorder = SessionRecorder(args.record, upstream=target, **options)
        stop_requested = threading.Event()

        def request_stop(_signum: int, _frame: object) -> None:
            stop_requested.set()

        signals = (signal.SIGINT, signal.SIGTERM)
        previous = {sig: signal.signal(sig, request_stop) for sig in signals}
        try:
            proxy = CaptureProxy(target, recorder, port=8767 if args.port is None else args.port).start()
            try:
                # In --json mode stdout is reserved for the single result object.
                notice = sys.stderr if _json_mode(args) else sys.stdout
                print(f"RunEconomy proxy · OPENAI_BASE_URL={proxy.base_url}", file=notice, flush=True)
                print(
                    f"Recording to {args.record}. Press Ctrl-C to finish and flush the session.",
                    file=notice,
                    flush=True,
                )
                stop_requested.wait()
            finally:
                summary = proxy.stop()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except (AttemptCaptureError, RecorderError, OSError, ValueError, ImportError) as exc:
        if _json_mode(args):
            payload = _cli_error_payload(exc, args)
            payload["error"]["details"] = {"reason": str(exc)}
            print(cli_contracts.render("error", payload), file=sys.stderr)
        else:
            print(f"error: proxy: {exc}", file=sys.stderr)
        return 2

    gaps = not summary["in_flight_drained"] or any(
        summary[key] for key in (
            "observations_dropped", "observations_after_close",
            "token_resolution_incomplete_at_close", "record_errors",
        )
    )
    if _json_mode(args):
        _emit(
            "proxy",
            {
                "mode": "record",
                "session": str(args.record),
                "summary": dict(summary),
                "gaps": bool(gaps),
                "agent_exit_code": None,
                "reason": None,
            },
        )
        return 1 if gaps else 0
    print(
        f"Recorded {summary['exchanges_recorded']} exchange(s), "
        f"{summary['token_resolution_exact']} exact prompt(s), "
        f"{summary['token_resolution_unknown']} unknown token resolution(s)."
    )
    if not summary["in_flight_drained"] or any(
        summary[key] for key in (
            "observations_dropped", "observations_after_close",
            "token_resolution_incomplete_at_close", "record_errors",
        )
    ):
        print("Capture has gaps. Inspect the session close counters before using the evidence.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "proxy":
            return _run_capture_proxy(args)

        if args.command == "mcp":
            from .mcp_server import serve_stdio

            return serve_stdio()

        if args.command == "ci-summary":
            from .ci import post_pull_request_comment, render_summary

            summary = render_summary(_load_json_object(args.result), mode=args.mode, title=args.title)
            if args.output:
                _write_text(args.output, summary["markdown"])
            comment = post_pull_request_comment(summary["markdown"]) if args.github_comment else None
            if _json_mode(args):
                _emit("ci-summary", {**summary, "comment": comment, "output": str(args.output) if args.output else None})
            else:
                print(summary["markdown"], end="")
            return 0

        if args.command == "schema":
            if args.name:
                try:
                    schema = cli_contracts.load_schema(args.name)
                except KeyError as exc:
                    raise ExperimentValidationError(
                        f"unknown CLI contract {args.name!r}; run `runeconomy schema` to list contracts"
                    ) from exc
                print(json.dumps(schema, indent=2, sort_keys=True))
                return 0
            listing = cli_contracts.list_contracts()
            if _json_mode(args):
                _emit("schema-list", listing)
                return 0
            print("RunEconomy --json output contracts")
            for item in listing["contracts"]:
                print(f"  {item['id']:<46} {', '.join(item['commands']) or '(all commands, on failure)'}")
            for item in listing["exempt_commands"]:
                print(f"  no --json: {item['command']} ({item['reason']})")
            print("Print one schema with `runeconomy schema NAME`.")
            return 0

        if args.command == "analyze":
            raw = _load_json_object(args.trace) if args.trace.suffix == ".json" else None
            if isinstance(raw, Mapping) and isinstance(raw.get("decision"), Mapping):
                from .detectors import diagnose_result

                # A RunEconomy result: run the result-level (server telemetry) detectors.
                report = diagnose_result(raw)
            else:
                trace = Trace.from_dict(raw) if raw is not None else _load_trace(args.trace)
                report = analyze_trace(
                    trace,
                    dependency_evidence=(raw or {}).get("dependency_evidence"),
                )
            rendered = (
                cli_contracts.render("analyze", report)
                if args.format == "json"
                else _format_human(report)
            )
            if args.output:
                _write_text(args.output, rendered + "\n")
            else:
                print(rendered)
            return 0

        if args.command == "import":
            if args.idle_gap_ms is not None and (
                not math.isfinite(args.idle_gap_ms) or args.idle_gap_ms <= 0
            ):
                raise TraceValidationError("--idle-gap-ms must be finite and positive")
            identity = _load_json_object(args.proxy_identity) if args.proxy_identity else None
            if identity is not None and any(
                not isinstance(value, str) or not value for value in identity.values()
            ):
                raise TraceValidationError("--proxy-identity values must be nonempty strings")
            payload = import_evidence(
                args.source,
                source_type=args.type,
                trace_id=args.trace_id,
                workload=args.workload,
                backend=args.backend,
                model=args.model,
                block_size=args.block_size,
                context=_load_json_object(args.replay_context) if args.replay_context else None,
                identity=identity,
                idle_gap_ms=args.idle_gap_ms,
            )
            _write_json(args.output, payload)
            next_command = None
            if payload["capability_assessment"]["runnable_experiment"]:
                findings = payload.get("analysis", payload).get("findings", [])
                selection = (
                    " --finding-id FINDING_ID"
                    if "proxy_session" in payload and len(findings) > 1
                    else ""
                )
                next_command = (
                    f"runeconomy experiment init --from-evidence {args.output}{selection} --output experiment/"
                )
            if args.format == "json":
                _emit(
                    "import",
                    {
                        **payload["capability_assessment"],
                        "evidence_bundle": str(args.output),
                        "next_command": next_command,
                    },
                )
                return 0
            print(_format_evidence_assessment(payload))
            print(f"Evidence bundle: {args.output}")
            if next_command:
                print(f"What to do next: run `{next_command}`.")
            return 0

        if args.command == "import-vllm-jsonl":
            payload = import_vllm_jsonl(
                args.captures,
                trace_id=args.trace_id,
                workload=args.workload,
                model=args.model,
                model_revision=args.model_revision,
                tokenizer=args.tokenizer,
                tokenizer_revision=args.tokenizer_revision,
                chat_template_digest=args.chat_template_digest,
                backend_version=args.backend_version,
                block_size=args.block_size,
                success=args.success,
            )
            _write_json(args.output, payload)
            if _json_mode(args):
                _emit("convert", _convert_payload(args, payload))
                return 0
            print(
                f"RunEconomy import · wrote normalized trace with {len(payload['requests'])} requests "
                f"to {args.output}"
            )
            print(f"What to do next: run `runeconomy analyze {args.output}`.")
            return 0

        if args.command == "import-openinference":
            payload = import_openinference(
                args.trace,
                trace_id=args.trace_id,
                workload=args.workload,
                backend=args.backend,
                block_size=args.block_size,
            )
            _write_json(args.output, payload)
            if _json_mode(args):
                _emit("convert", _convert_payload(args, payload))
                return 0
            print(
                f"RunEconomy import · wrote normalized trace with {len(payload['requests'])} requests "
                f"to {args.output}"
            )
            print(f"What to do next: run `runeconomy analyze {args.output}`.")
            return 0

        if args.command == "import-ledger-jsonl":
            payload = import_ledger_jsonl(args.export)
            _write_json(args.output, payload)
            if _json_mode(args):
                _emit("convert", _convert_payload(args, payload))
                return 0
            print(
                f"RunEconomy import · wrote {len(payload['findings'])} imported findings to {args.output}"
            )
            print(
                "What to do next: inspect the evidence and run "
                f"`runeconomy experiment init --from-finding {args.output} ...`."
            )
            return 0

        if args.command == "import-prefixwise-jsonl":
            payload = import_ledger_jsonl(args.export, source_name="prefixwise")
            _write_json(args.output, payload)
            if _json_mode(args):
                _emit("convert", _convert_payload(args, payload))
                return 0
            print(
                f"RunEconomy import · wrote {len(payload['findings'])} Prefixwise finding(s) to {args.output}"
            )
            print(
                "What to do next: initialize an unresolved workspace, then supply replay inputs and an evaluator only when available."
            )
            return 0

        if args.command == "capture-vllm":
            request = _load_json_object(args.request)
            client = VLLMClient(
                args.base_url,
                api_key=_api_key(args.api_key_env),
                timeout_seconds=args.timeout,
                startup_retry_seconds=args.startup_timeout,
                api_dialect=args.api_dialect,
                on_startup_retry=_startup_retry_notice,
            )
            capture_result = client.capture(request).to_dict()
            rendered = json.dumps(capture_result, indent=2)
            if args.output:
                _write_text(args.output, rendered + "\n")
            elif not _json_mode(args):
                print(rendered)
            if args.append_trace_jsonl:
                record: dict[str, Any] = {
                    "request": request,
                    "capture": capture_result,
                    "invoked_tools": args.invoked_tool,
                }
                if args.request_id:
                    record["request_id"] = args.request_id
                _append_jsonl(args.append_trace_jsonl, record)
            if _json_mode(args):
                _emit(
                    "capture-vllm",
                    {
                        "capture": capture_result,
                        "output": str(args.output) if args.output else None,
                        "appended_to": str(args.append_trace_jsonl) if args.append_trace_jsonl else None,
                    },
                )
            return 0

        if args.command == "demo":
            demo_result = load_demo_result(args.case)
            rendered = render_demo_html(demo_result)
            _write_text(args.output, rendered)
            status = str(demo_result["decision"]["status"]).title()
            if _json_mode(args):
                opened = bool(args.open) and webbrowser.open(args.output.resolve().as_uri())
                _emit(
                    "demo",
                    {
                        "case": args.case,
                        "verdict": demo_result["decision"]["status"],
                        "report": str(args.output),
                        "opened": opened,
                    },
                )
                return 0
            print(
                f"RunEconomy demo · Verdict: {status} · report written to {args.output}"
            )
            print(
                "What to do next: open the report and read the failed quality gates "
                "before interpreting the conditional latency measurements."
            )
            if args.open:
                if not webbrowser.open(args.output.resolve().as_uri()):
                    print(
                        "RunEconomy · WARNING · the report was written, but no browser "
                        "accepted the open request.",
                        file=sys.stderr,
                    )
                    print(
                        f"What to do next: open {args.output.resolve()} in a browser.",
                        file=sys.stderr,
                    )
            return 0

        if args.command == "tutorial":
            if args.tutorial_command == "init":
                created = materialize_tutorial(args.output, port=args.port)
                serve_command = "runeconomy tutorial serve"
                if args.port != DEFAULT_TUTORIAL_PORT:
                    serve_command += f" --port {args.port}"
                if _json_mode(args):
                    _emit(
                        "tutorial-init",
                        {
                            "directory": str(args.output),
                            "tutorial_schema_version": "0.1",
                            "files": sorted(path.name for path in created),
                            "port": args.port,
                            "next_command": serve_command,
                        },
                    )
                    return 0
                print(
                    "RunEconomy tutorial · Created deterministic local tutorial · "
                    f"{args.output}"
                )
                print(f"Tutorial schema: 0.1 · Files: {len(created)}")
                print(f"What to do next: run `{serve_command}` in another terminal.")
                return 0
            serve_tutorial(port=args.port)
            return 0

        if args.command == "support-bundle":
            summary = create_support_bundle(
                args.workspace,
                args.output,
                doctor_report=args.doctor_report,
                validation_report=args.validation_report,
            )
            if args.format == "json":
                _emit("support-bundle", {**summary, "output": str(args.output)})
            else:
                print(f"RunEconomy support bundle · Created · {args.output}")
                print("Included:")
                for name in summary["included_sections"]:
                    print(f"  - {name}")
                print(
                    "Private experiment content was excluded by a strict export allowlist."
                )
                print(
                    "RunEconomy did not upload this archive. Inspect every file before sharing it."
                )
            return 0

        if args.command == "doctor":
            try:
                doctor_options: dict[str, Any] = {
                    "api_key": _api_key(args.api_key_env),
                    "timeout_seconds": args.timeout,
                }
                if args.startup_timeout:
                    doctor_options.update(
                        {
                            "startup_retry_seconds": args.startup_timeout,
                            "on_startup_retry": _startup_retry_notice,
                        }
                    )
                if args.backend == "vllm":
                    capabilities = discover_vllm_capabilities(
                        args.base_url, **doctor_options
                    ).to_dict()
                else:
                    capabilities = create_adapter(
                        args.backend, args.base_url, **doctor_options
                    ).discover()
            except (BackendProbeError, ExperimentValidationError) as exc:
                payload = _doctor_error_payload(exc)
                rendered = (
                    cli_contracts.render("error", payload)
                    if args.format == "json"
                    else _format_doctor_error(payload)
                )
                print(rendered, file=sys.stderr)
                return 2
            print(
                cli_contracts.render("doctor", capabilities)
                if args.format == "json"
                else _format_capabilities(args.base_url, capabilities)
                if args.backend == "vllm"
                else format_adapter_report(capabilities)
            )
            return 0 if capabilities["readiness"]["status"] == "ready" else 1

        if args.command == "compare":
            print(
                "RunEconomy · WARNING · `runeconomy compare` is a compatibility wrapper. "
                "New workflows should use `runeconomy experiment init --mode agent-process`.",
                file=sys.stderr,
            )
            artifacts_dir = args.artifacts_dir or args.output.with_name(
                args.output.stem + "-artifacts"
            )
            _progress(
                "starting paired agent comparison; partial traces and status files "
                f"will be preserved under {artifacts_dir}"
            )
            result = run_comparison(
                baseline_command=args.baseline,
                candidate_command=args.candidate,
                dataset_path=args.dataset,
                evaluator_path=args.evaluator,
                output_path=args.output,
                artifacts_dir=artifacts_dir,
                workdir=args.workdir,
                repetitions=args.repetitions,
                order=args.order,
                seed=args.seed,
                timeout_seconds=args.timeout,
                evaluator_timeout_seconds=args.evaluator_timeout,
                max_task_success_regression=args.max_task_success_regression,
                min_task_success_rate=args.min_task_success_rate,
                min_completed_rate=args.min_completed_rate,
                optimization_target=args.optimization_target.replace("-", "_"),
                latency_semantics=args.latency_semantics,
                max_latency_regression_ms=args.max_latency_regression_ms,
                resume=args.resume,
            )
            _write_json(args.output, result)
            verified = result["decision"]["status"] == "verified"
            if _json_mode(args):
                _emit(
                    "compare",
                    {
                        "verdict": result["decision"]["status"],
                        "result": str(args.output),
                        "artifacts": str(artifacts_dir),
                        "require_verified": bool(args.require_verified),
                        "verified": verified,
                    },
                )
                return 3 if args.require_verified and not verified else 0
            print(
                f"RunEconomy compare · Verdict: {str(result['decision']['status']).title()} · wrote "
                f"to {args.output}"
            )
            print(
                "What to do next: inspect task-success gates before interpreting "
                "successful-pair latency."
            )
            if args.require_verified and result["decision"]["status"] != "verified":
                print("error: comparison did not reach a verified decision", file=sys.stderr)
                return 3
            return 0

        if args.experiment_command == "report":
            result = _load_json_object(args.result)
            try:
                report_format = args.format or (
                    "html" if args.output.suffix.lower() == ".html" else "markdown"
                )
                rendered = (
                    render_experiment_html(result)
                    if report_format == "html"
                    else render_experiment_markdown(result)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ExperimentValidationError(
                    f"cannot render experiment report: {exc}"
                ) from exc
            _write_text(args.output, rendered)
            if _json_mode(args):
                if args.open and report_format != "html":
                    raise ExperimentValidationError("--open requires an HTML report")
                opened = bool(args.open) and webbrowser.open(args.output.resolve().as_uri())
                _emit(
                    "experiment-report",
                    {"report": str(args.output), "format": report_format, "opened": opened},
                )
                return 0
            print(f"RunEconomy report · wrote experiment report to {args.output}")
            print(
                "What to do next: read the verdict and quality gates before "
                "interpreting conditional performance measurements."
            )
            if args.open:
                if report_format != "html":
                    raise ExperimentValidationError("--open requires an HTML report")
                if not webbrowser.open(args.output.resolve().as_uri()):
                    print(
                        "RunEconomy · WARNING · the report was written, but no browser accepted the open request.",
                        file=sys.stderr,
                    )
            return 0

        if args.experiment_command == "init":
            if args.mode == "agent-process":
                if args.from_finding or args.from_evidence:
                    raise ExperimentValidationError(
                        "agent-process initialization accepts explicit commands and files, not --from-finding or --from-evidence"
                    )
                missing = [
                    name
                    for name, value in (
                        ("--baseline", args.baseline),
                        ("--candidate", args.candidate),
                        ("--dataset", args.dataset),
                        ("--evaluator", args.evaluator),
                    )
                    if value is None
                ]
                if missing:
                    raise ExperimentValidationError(
                        "agent-process initialization requires " + ", ".join(missing)
                    )
                descriptor = create_agent_workspace(
                    output=args.output,
                    baseline_command=args.baseline,
                    candidate_command=args.candidate,
                    dataset_path=args.dataset,
                    evaluator_path=args.evaluator,
                    repetitions=args.repetitions,
                    order=args.order,
                    seed=args.seed,
                    timeout_seconds=args.timeout,
                    evaluator_timeout_seconds=args.evaluator_timeout,
                    max_task_success_regression=args.max_task_success_regression,
                    min_task_success_rate=args.min_task_success_rate,
                    min_completed_rate=args.min_completed_rate,
                    optimization_target=args.optimization_target.replace("-", "_"),
                    latency_semantics=args.latency_semantics,
                    max_latency_regression_ms=args.max_latency_regression_ms,
                    cost_ceiling_usd=args.cost_ceiling_usd,
                    identity_policy=(
                        None
                        if args.min_identity_grade is None
                        else {
                            "schema_version": "0.1",
                            "minimum_grade": args.min_identity_grade.replace("-", "_"),
                        }
                    ),
                )
                if _init_json(args, descriptor):
                    return 0
                print(f"RunEconomy experiment init · Runnable agent-process workspace · {args.output}")
                print(f"What to do next: {descriptor['next_command']}")
                return 0

            if bool(args.from_finding) == bool(args.from_evidence):
                raise ExperimentValidationError(
                    "static initialization requires exactly one of --from-evidence or --from-finding"
                )
            if args.from_evidence:
                evidence = load_evidence_bundle(args.from_evidence)
                descriptor = create_workspace_from_evidence(
                    evidence,
                    evidence_path=args.from_evidence,
                    output=args.output,
                    finding_id=args.finding_id,
                )
                if _init_json(args, descriptor):
                    return 0
                print(
                    "RunEconomy experiment init · "
                    f"{str(descriptor['generation_status']).title()} workspace · {args.output}"
                )
                print(f"What to do next: {descriptor['next_command']}")
                return 0

            source = _load_json_object(args.from_finding)
            if args.output.suffix.lower() == ".json":
                finding = select_finding(source, args.finding_id)
                artifact = initialize_experiment(finding)
                _write_json(args.output, artifact)
                status = artifact.get("generation_status", "runnable")
                if _init_json(
                    args,
                    {
                        "mode": "static_request",
                        "generation_status": status,
                        "unresolved_fields": artifact.get("unresolved_fields", []),
                        "next_command": f"runeconomy experiment validate {args.output}",
                    },
                    output_kind="artifact",
                ):
                    return 0
                print(f"RunEconomy experiment init · {str(status).title()} artifact · {args.output}")
                print(f"What to do next: run `runeconomy experiment validate {args.output}`.")
                return 0
            descriptor = create_workspace_from_finding(
                source,
                source_path=args.from_finding,
                output=args.output,
                finding_id=args.finding_id,
            )
            if _init_json(args, descriptor):
                return 0
            print(
                "RunEconomy experiment init · "
                f"{str(descriptor['generation_status']).title()} workspace · {args.output}"
            )
            print(f"What to do next: {descriptor['next_command']}")
            return 0

        if args.experiment_command == "validate":
            doctor_report = (
                _load_json_object(args.doctor_report) if args.doctor_report else None
            )
            report = validate_experiment_artifact(
                args.manifest, doctor_report=doctor_report
            )
            if report["runnable"] and report["mode"] == "static_request":
                manifest = resolve_static_manifest(args.manifest)
                _require_trusted_evaluators(manifest, args.allow_exec_evaluators)
            print(
                cli_contracts.render(
                    "experiment-validate", {**report, "freeze": _freeze_state(args.manifest)}
                )
                if args.format == "json"
                else _format_validation_report(report, args.manifest)
            )
            return 0 if report["status"] == "runnable" else 2 if report["status"] == "invalid" else 1

        if args.experiment_command == "plan":
            if args.examples:
                examples = worked_examples()
                print(
                    cli_contracts.render("experiment-plan-examples", examples)
                    if args.format == "json"
                    else _format_planning_examples(examples)
                )
                return 0
            planning = plan_experiment(args.manifest)
            print(
                cli_contracts.render("experiment-plan", planning)
                if args.format == "json"
                else _format_experiment_plan(planning)
            )
            return 2 if planning["state"] == "blocked" else 1 if planning["state"] == "needs_design_input" else 0

        if args.experiment_command == "patch":
            operations = load_patch_file(args.patch_file) if args.patch_file else []
            operations.extend(parse_assignment(item) for item in args.patch_set)
            operations.extend(parse_assignment(item, raw_string=True) for item in args.patch_set_string)
            operations.extend(PatchOperation("unset", path) for path in args.patch_unset)
            outcome = patch_workspace(
                args.workspace,
                operations,
                reason=args.reason,
                repin=args.repin,
                dry_run=args.dry_run,
            )
            if _json_mode(args):
                _emit("experiment-patch", outcome)
                return 0
            label = {"patched": "Patched", "unchanged": "Unchanged", "dry_run": "Dry run (nothing written)"}
            print(f"RunEconomy experiment patch · {label[outcome['status']]} · {args.workspace}")
            for change in outcome["operations"]:
                origin = " (derived digest)" if change["source"] == "derived_digest" else ""
                print(f"  {change['op']} {change['path']}{origin}")
            if outcome["untracked_change_before"]:
                print(
                    "WARNING: manifest.json changed outside `experiment patch` since the last "
                    "provenance entry; that edit is recorded as untracked.",
                    file=sys.stderr,
                )
            if outcome["written"]:
                print(f"Provenance: {outcome['provenance']['file']} entry {outcome['provenance']['entries']}")
            print(f"What to do next: run `{outcome['next_command']}`.")
            return 0

        if args.experiment_command == "freeze":
            outcome = freeze_workspace(
                args.workspace,
                lock_files=args.lock_files,
                allow_incomplete_design=args.allow_incomplete_design,
                reason=args.reason,
            )
            if _json_mode(args):
                _emit("experiment-freeze", outcome)
                return 0
            headline = "Frozen" if outcome["status"] == "frozen" else "Already frozen (locks intact)"
            print(f"RunEconomy experiment freeze · {headline} · {args.workspace}")
            print(f"Preregistration record: {outcome['record']} ({outcome['record_sha256']})")
            print(f"Design completeness: {outcome['design_completeness']} (planner: {outcome['planner_state']})")
            print(f"Locked: manifest, gates, execution design, {len(outcome['locks']['inputs'])} input file(s), "
                  f"{len(outcome['locks']['additional_files'])} additional file(s).")
            for item in outcome["unlocked_inputs"]:
                print(f"Not digest-locked: {item['field']}: {item['reason']}")
            if outcome["unknown_fields"]:
                print("Unknown preregistration fields: " + ", ".join(outcome["unknown_fields"]))
            print("Patches are now refused; any change needs a new workspace.")
            print(f"What to do next: run `{outcome['next_command']}`.")
            return 0

        freeze_state = require_intact_freeze(args.manifest, require_frozen=args.require_frozen)
        result_path, artifacts_path = _result_output_path(args.output)
        preregistration = (
            {
                "status": freeze_state["status"],
                "record_sha256": freeze_state["record_sha256"],
                "frozen_at": freeze_state.get("frozen_at"),
                "design_completeness": freeze_state.get("design_completeness"),
            }
            if freeze_state["status"] == "intact"
            else None
        )
        if args.manifest.is_dir():
            descriptor, _ = load_workspace(args.manifest)
            if descriptor.get("mode") == "agent_process":
                manifest_data, workspace = resolve_agent_manifest(args.manifest)
                run_args = agent_run_arguments(manifest_data, workspace)
                _progress(
                    "starting paired agent-process experiment; partial traces and status files "
                    f"will be preserved under {artifacts_path}"
                )
                result = run_comparison(
                    **run_args,
                    observers=_metrics_observers(args),
                    output_path=result_path,
                    artifacts_dir=artifacts_path,
                    workdir=Path.cwd(),
                    resume=args.resume,
                )
                if preregistration is not None:
                    result["preregistration"] = preregistration
                _write_json(result_path, result)
                if _json_mode(args):
                    return _run_json(args, "agent_process", result, result_path, artifacts_path, None, freeze_state)
                print(
                    "RunEconomy experiment · Agent process · Verdict: "
                    f"{str(result['decision']['status']).title()} · wrote {result_path}"
                )
                print(
                    "What to do next: render the report and inspect task-quality gates before latency."
                )
                if args.require_verified and result["decision"]["status"] != "verified":
                    print("error: experiment did not reach a verified decision", file=sys.stderr)
                    return 3
                return 0

        manifest = resolve_static_manifest(args.manifest)
        _require_trusted_evaluators(manifest, args.allow_exec_evaluators)

        client = create_adapter(
            manifest.backend.kind,
            manifest.backend.base_url,
            api_key=_api_key(manifest.backend.api_key_env),
            timeout_seconds=args.timeout,
            startup_retry_seconds=args.startup_timeout,
            api_dialect=manifest.backend.api_dialect,
            on_startup_retry=_startup_retry_notice,
            **(
                {
                    "provider": manifest.backend.provider,
                    "undisclosed_fields": manifest.backend.undisclosed_identity,
                }
                if manifest.backend.kind == "openai_compat"
                else {}
            ),
        )
        observers = _metrics_observers(args, client)
        checkpoint = args.checkpoint or result_path.with_name(
            result_path.name + ".trials.jsonl"
        )
        _progress(
            "starting static experiment; completed trial records will be preserved "
            f"in {checkpoint}"
        )
        result = run_experiment(
            manifest,
            client,
            checkpoint_path=checkpoint,
            resume=args.resume,
            attempt_economics=(
                None if manifest.cost_model is None else parse_cost_model(manifest.cost_model)
            ),
            observers=observers,
        )
        if preregistration is not None:
            result["preregistration"] = preregistration
        _write_json(result_path, result)
        if _json_mode(args):
            return _run_json(args, "static_request", result, result_path, None, checkpoint, freeze_state)
        print(
            f"RunEconomy experiment · Verdict: {str(result['decision']['status']).title()} · "
            f"wrote result to {result_path}"
        )
        print(
            "What to do next: render a report and interpret performance only after "
            "checking task-quality gates."
        )
        if args.require_verified and result["decision"]["status"] != "verified":
            print(
                "error: experiment did not reach a verified decision",
                file=sys.stderr,
            )
            return 3
        return 0
    except (
        TraceValidationError,
        ExperimentValidationError,
        VLLMError,
        BackendProbeError,
        TutorialError,
        SupportBundleError,
        AttemptStatusError,
        EvaluatorInfrastructureError,
    ) as exc:
        rendered = (
            cli_contracts.render("error", _cli_error_payload(exc, args))
            if _json_mode(args)
            else _format_cli_error(exc, args)
        )
        print(rendered, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
