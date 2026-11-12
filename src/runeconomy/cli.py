"""Command-line interface for local trace analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .analysis import analyze_trace
from .experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    run_experiment,
)
from .importers import import_vllm_jsonl
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
                f"{finding['title']} ({finding['request_id']})",
                "Critical path: "
                + (
                    "unknown"
                    if finding["on_critical_path"] is None
                    else "yes"
                    if finding["on_critical_path"]
                    else "no"
                ),
                finding["summary"],
                f"Experiment: {finding['suggested_experiment']}",
            ]
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runeconomy")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a normalized trace")
    analyze.add_argument("trace", type=Path)
    analyze.add_argument("--format", choices=("human", "json"), default="human")
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

    capture = subparsers.add_parser(
        "capture-vllm", help="render and execute one vLLM Responses request"
    )
    capture.add_argument("request", type=Path)
    capture.add_argument("--base-url", required=True)
    capture.add_argument("--api-key-env")
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
    demo.add_argument("--output", type=Path, default=Path("runeconomy-demo.html"))

    experiment = subparsers.add_parser(
        "negative-split",
        aliases=["experiment"],
        help="run a controlled baseline-versus-candidate comparison",
    )
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
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
                args.output.write_text(rendered + "\n", encoding="utf-8")
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
                f"wrote normalized trace with {len(payload['requests'])} requests "
                f"to {args.output}"
            )
            return 0

        if args.command == "capture-vllm":
            request = _load_json_object(args.request)
            client = VLLMClient(
                args.base_url,
                api_key=_api_key(args.api_key_env),
                timeout_seconds=args.timeout,
                startup_retry_seconds=args.startup_timeout,
            )
            capture_result = client.capture(request).to_dict()
            rendered = json.dumps(capture_result, indent=2)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
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
            rendered = render_demo_html(load_demo_result())
            args.output.write_text(rendered, encoding="utf-8")
            print(f"wrote RunEconomy demo to {args.output}")
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
            args.output.write_text(rendered, encoding="utf-8")
            print(f"wrote NegativeSplit report to {args.output}")
            return 0

        manifest = ExperimentManifest.from_path(args.manifest)
        _require_trusted_evaluators(manifest, args.allow_exec_evaluators)
        if args.experiment_command == "validate":
            print(f"valid experiment manifest: {manifest.experiment_id}")
            return 0

        client = VLLMClient(
            manifest.backend.base_url,
            api_key=_api_key(manifest.backend.api_key_env),
            timeout_seconds=args.timeout,
            startup_retry_seconds=args.startup_timeout,
        )
        checkpoint = args.checkpoint or args.output.with_name(
            args.output.name + ".trials.jsonl"
        )
        result = run_experiment(
            manifest,
            client,
            checkpoint_path=checkpoint,
            resume=args.resume,
        )
        _write_json(args.output, result)
        print(f"wrote experiment result to {args.output}")
        if args.require_verified and result["decision"]["status"] != "verified":
            print(
                "error: experiment did not reach a verified decision",
                file=sys.stderr,
            )
            return 3
        return 0
    except (TraceValidationError, ExperimentValidationError, VLLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
