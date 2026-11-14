"""Command-line interface for local trace analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .analysis import analyze_trace
from .backends import BackendProbeError, discover_vllm_capabilities
from .comparison import run_comparison
from .experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    run_experiment,
)
from .experiment_init import initialize_experiment, select_finding
from .importers import (
    import_openinference,
    import_prefixwise_jsonl,
    import_vllm_jsonl,
)
from .models import Trace, TraceValidationError
from .reports import (
    load_demo_result,
    render_demo_html,
    render_experiment_html,
    render_experiment_markdown,
)
from .vllm import VLLMClient, VLLMError


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runeconomy")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a normalized trace")
    analyze.add_argument("trace", type=Path)
    analyze.add_argument("--format", choices=("human", "json"), default="human")
    analyze.add_argument("--json", dest="format", action="store_const", const="json")
    analyze.add_argument("--output", type=Path)

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

    import_prefixwise = subparsers.add_parser(
        "import-prefixwise-jsonl",
        help="convert Prefixwise ledger exports into structured findings",
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
    capture.add_argument("--startup-timeout", type=float, default=600.0)
    capture.add_argument("--output", type=Path)
    capture.add_argument(
        "--append-trace-jsonl",
        type=Path,
        help="append an importable envelope containing the request and capture",
    )
    capture.add_argument("--request-id")
    capture.add_argument("--invoked-tool", action="append", default=[])

    demo = subparsers.add_parser(
        "demo", help="render the bundled CPU-only NegativeSplit demonstration"
    )
    demo.add_argument(
        "--case",
        choices=("rejection", "historical"),
        default="rejection",
        help="bundled case to render (default: quality-gated rejection)",
    )
    demo.add_argument("--output", type=Path, default=Path("runeconomy-demo.html"))
    demo.add_argument("--open", action="store_true", help="open the report in a browser")

    doctor = subparsers.add_parser(
        "doctor", help="inspect a vLLM server without running inference"
    )
    doctor.add_argument("--base-url", required=True)
    doctor.add_argument("--api-key-env")
    doctor.add_argument("--timeout", type=float, default=10.0)
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
        "--require-verified",
        action="store_true",
        help="return exit code 3 unless the closed-loop verdict is verified",
    )

    experiment = subparsers.add_parser(
        "experiment",
        aliases=["negative-split"],
        help="run a controlled baseline-versus-candidate comparison",
    )
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    initialize = experiment_commands.add_parser(
        "init", help="create a manifest or skeleton from a structured finding"
    )
    initialize.add_argument("--from-finding", type=Path, required=True)
    initialize.add_argument("--finding-id")
    initialize.add_argument("--output", type=Path, required=True)
    validate = experiment_commands.add_parser("validate", help="validate a manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument(
        "--allow-exec-evaluators",
        action="store_true",
        help="allow trusted python and command evaluators",
    )
    run = experiment_commands.add_parser("run", help="run a manifest against vLLM")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--startup-timeout", type=float, default=600.0)
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
        "report", help="render a static NegativeSplit report"
    )
    report.add_argument("result", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--format", choices=("markdown", "html"))
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
        safety = "safe" if mode["safe"] else "manual/private only"
        lines.append(f"  {mode['mode']}: {mode['status']} ({safety})")
        lines.append(f"    {mode['note']}")
    lines.extend(["", "What to do next"])
    actions = list(readiness.get("next_actions") or [])
    for label, key in capability_labels:
        capability = capabilities[key]
        if capability["status"] != "supported" and capability.get("next_action"):
            actions.append(f"{label}: {capability['next_action']}")
    for label, identity in capabilities.get("identity", {}).items():
        if identity["status"] != "supported":
            actions.append(f"{label.replace('_', ' ').title()}: {identity['next_action']}")
    if not actions:
        actions.append(
            "Record the detected identities in the manifest, then validate the experiment before execution."
        )
    actions = list(dict.fromkeys(actions))
    lines.extend(f"  {index}. {action}" for index, action in enumerate(actions, 1))
    lines.extend(
        [
            "",
            "Safety: doctor performed metadata and non-inference render probes only; "
            "it did not generate text, reset caches, or mutate the server.",
        ]
    )
    return "\n".join(lines)


def _doctor_error_payload(exc: Exception) -> dict[str, Any]:
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
        "status": "error",
        "error": {
            "category": category,
            "message": str(exc),
            "evidence_complete": False,
            "next_action": next_action,
        },
    }


def _format_doctor_error(payload: Mapping[str, Any]) -> str:
    error = payload["error"]
    return "\n".join(
        [
            "RunEconomy doctor · ERROR",
            f"What failed: {error['category'].replace('_', ' ')}.",
            f"Why: {error['message']}",
            "Evidence: capability discovery is incomplete; no compatibility verdict was produced.",
            f"What to do next: {error['next_action']}",
            "Safety: no inference, cache reset, or server mutation was attempted.",
        ]
    )


def _next_action_for_error(args: argparse.Namespace) -> str:
    command = getattr(args, "command", None)
    if command == "analyze":
        return "Check that the input is normalized trace JSON, then rerun `runeconomy analyze TRACE`."
    if command and command.startswith("import-"):
        return "Review the source format and required flags, then rerun the same import command."
    if command == "compare":
        return "Inspect the preserved artifacts and status files, correct the agent or evaluator failure, then rerun `runeconomy compare`."
    if command in {"experiment", "negative-split"}:
        subcommand = getattr(args, "experiment_command", None)
        if subcommand == "init":
            return "Run `runeconomy analyze` or a supported importer first, then pass its report to `runeconomy experiment init --from-finding ...`."
        if subcommand == "validate":
            return "Correct the manifest field named above, then rerun `runeconomy experiment validate MANIFEST`."
        if subcommand == "run":
            return "Preserve the checkpoint and partial result, run `runeconomy doctor` against the backend, then retry or resume only after correcting the cause."
        return "Verify the result JSON, then rerun `runeconomy experiment report`."
    if command == "capture-vllm":
        return "Run `runeconomy doctor` against the same backend, correct the reported capability or authentication issue, then retry the capture."
    return "Correct the reported input or configuration and rerun the command."


def _format_cli_error(exc: Exception, args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "RunEconomy · ERROR",
            f"What failed: {getattr(args, 'command', 'command')} did not complete.",
            f"Why: {exc}",
            "Evidence: output may be absent or incomplete; do not treat a partial artifact as a verdict.",
            f"What to do next: {_next_action_for_error(args)}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            trace = _load_trace(args.trace)
            report = analyze_trace(trace)
            rendered = (
                json.dumps(report, indent=2, sort_keys=True)
                if args.format == "json"
                else _format_human(report)
            )
            if args.output:
                _write_text(args.output, rendered + "\n")
            else:
                print(rendered)
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
            print(
                f"RunEconomy import · wrote normalized trace with {len(payload['requests'])} requests "
                f"to {args.output}"
            )
            print(f"What to do next: run `runeconomy analyze {args.output}`.")
            return 0

        if args.command == "import-prefixwise-jsonl":
            payload = import_prefixwise_jsonl(args.export)
            _write_json(args.output, payload)
            print(
                f"RunEconomy import · wrote {len(payload['findings'])} imported findings to {args.output}"
            )
            print(
                "What to do next: inspect the evidence and run "
                f"`runeconomy experiment init --from-finding {args.output} ...`."
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
            )
            capture_result = client.capture(request).to_dict()
            rendered = json.dumps(capture_result, indent=2)
            if args.output:
                _write_text(args.output, rendered + "\n")
            else:
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
            return 0

        if args.command == "demo":
            rendered = render_demo_html(load_demo_result(args.case))
            _write_text(args.output, rendered)
            print(f"RunEconomy demo · INVALID proof written to {args.output}")
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

        if args.command == "doctor":
            try:
                capabilities = discover_vllm_capabilities(
                    args.base_url,
                    api_key=_api_key(args.api_key_env),
                    timeout_seconds=args.timeout,
                ).to_dict()
            except (BackendProbeError, ExperimentValidationError) as exc:
                payload = _doctor_error_payload(exc)
                rendered = (
                    json.dumps(payload, indent=2, sort_keys=True)
                    if args.format == "json"
                    else _format_doctor_error(payload)
                )
                print(rendered, file=sys.stderr)
                return 2
            print(
                json.dumps(capabilities, indent=2, sort_keys=True)
                if args.format == "json"
                else _format_capabilities(args.base_url, capabilities)
            )
            return 0 if capabilities["readiness"]["status"] == "ready" else 1

        if args.command == "compare":
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
            )
            _write_json(args.output, result)
            print(
                f"RunEconomy compare · {result['decision']['status'].upper()} · wrote "
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
            print(f"RunEconomy report · wrote NegativeSplit report to {args.output}")
            print(
                "What to do next: read the verdict and quality gates before "
                "interpreting conditional performance measurements."
            )
            return 0

        if args.experiment_command == "init":
            source = _load_json_object(args.from_finding)
            finding = select_finding(source, args.finding_id)
            artifact = initialize_experiment(finding)
            _write_json(args.output, artifact)
            status = artifact.get("generation_status", "runnable")
            if status == "unresolved":
                print(
                    "wrote unresolved experiment skeleton to "
                    f"{args.output}; resolve every listed field before validation"
                )
                print(
                    "What to do next: fill every unresolved field, then run "
                    f"`runeconomy experiment validate {args.output}`."
                )
            else:
                print(f"wrote runnable experiment manifest to {args.output}")
                print(
                    "What to do next: review the frozen gates and run "
                    f"`runeconomy experiment validate {args.output}`."
                )
            return 0

        manifest = ExperimentManifest.from_path(args.manifest)
        _require_trusted_evaluators(manifest, args.allow_exec_evaluators)
        if args.experiment_command == "validate":
            print(f"RunEconomy experiment · VALID · {manifest.experiment_id}")
            print(
                "What to do next: run the manifest only after reviewing its evaluator, "
                "quality gates, cache-isolation mode, and backend identity."
            )
            return 0

        client = VLLMClient(
            manifest.backend.base_url,
            api_key=_api_key(manifest.backend.api_key_env),
            timeout_seconds=args.timeout,
            startup_retry_seconds=args.startup_timeout,
            api_dialect=manifest.backend.api_dialect,
        )
        checkpoint = args.checkpoint or args.output.with_name(
            args.output.name + ".trials.jsonl"
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
        )
        _write_json(args.output, result)
        print(
            f"RunEconomy experiment · {result['decision']['status'].upper()} · "
            f"wrote result to {args.output}"
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
    ) as exc:
        print(_format_cli_error(exc, args), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
