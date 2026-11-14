"""Build the predeclared held-out repository-analysis experiment manifest."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTPUT = (
    ROOT
    / "examples"
    / "experiments"
    / "repository-agent-tool-pruning-heldout.json"
)
CALIBRATION_OUTPUT = (
    ROOT
    / "examples"
    / "experiments"
    / "repository-agent-tool-pruning-prompt-calibration.json"
)
CALIBRATION = (
    ROOT / "examples" / "calibration" / "repository-readonly-tool-selection.json"
)
BROAD_TOOLS = ROOT / "examples" / "fixtures" / "coding-tools-broad.json"
SCOPED_TOOLS = ROOT / "examples" / "fixtures" / "coding-tools-scoped.json"
FROZEN_EXCERPTS = {
    ("src/runeconomy/cli.py", "_build_parser"): (
        ROOT / "examples" / "fixtures" / "recorded-cli-build-parser.txt"
    ),
    ("src/runeconomy/analysis.py", "Finding"): (
        ROOT / "examples" / "fixtures" / "recorded-analysis-finding.txt"
    ),
}
FROZEN_HELDOUT_CONTEXTS = {
    ("src/runeconomy/analysis.py", "_segment_at"): (
        "SOURCE_CONTEXT_01",
        "57a93507f670690f83a716cc9bc36d2a231d86c2229fcd65b8616ab05fdc2629",
    ),
    ("src/runeconomy/analysis.py", "_best_baseline"): (
        "SOURCE_CONTEXT_02",
        "7502c8f0276a1ddc38df72088e6e8bdf52f26e8b2323a778e0d418cc8bf6ddec",
    ),
    ("src/runeconomy/analysis.py", "_finding_for_divergence"): (
        "SOURCE_CONTEXT_03",
        "e039b99b23c86225d8cb20ef9f7df0e0ad65190d87bc21895aa76fc2fa4f9a5a",
    ),
    ("src/runeconomy/analysis.py", "_finding_for_runtime_gap"): (
        "SOURCE_CONTEXT_04",
        "11c6f8fe0b4c00f5a5c1c12d3e4f70def1fffefa1d2db361c6aaaa2c4d4cb898",
    ),
    ("src/runeconomy/analysis.py", "_finding_for_unused_tools"): (
        "SOURCE_CONTEXT_05",
        "2aa0beb2dd624045dd0c567fe65754b7847b8c8247d1951d8f000cfabbf7f27d",
    ),
    ("src/runeconomy/analysis.py", "analyze_trace"): (
        "SOURCE_CONTEXT_06",
        "6de33988c1d8d9bcf8b1679fbb92d322f35c05b75964dff8903cd70146722107",
    ),
    ("src/runeconomy/cli.py", "_format_human"): (
        "SOURCE_CONTEXT_10",
        "033f25f0f7f2ae702d2ef41a1f732c2956ef9e4c9b496fc7b2633306552d9486",
    ),
}

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
CHAT_TEMPLATE_DIGEST = (
    "sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)

MODULE_SUMMARIES = {
    "src/runeconomy/analysis.py": "trace-level diagnosis and finding generation",
    "src/runeconomy/cache.py": "static prompt-prefix cache calculations",
    "src/runeconomy/cli.py": "command-line parsing, validation, and dispatch",
    "src/runeconomy/critical_path.py": "duration-weighted execution DAG analysis",
    "src/runeconomy/evaluators.py": "deterministic task-output evaluators",
    "src/runeconomy/experiments.py": "paired experiment manifests, statistics, and execution",
}

# These task intents were frozen before the held-out GPU evaluation. They do
# not overlap the historical two-question proof bundled with RunEconomy.
TASKS = [
    ("src/runeconomy/analysis.py", "RequestAnalysis", "stores per-request cache-analysis evidence"),
    ("src/runeconomy/analysis.py", "Finding", "stores one structured diagnostic finding"),
    ("src/runeconomy/analysis.py", "_segment_at", "finds the semantic prompt segment containing a token offset"),
    ("src/runeconomy/analysis.py", "_best_baseline", "selects the best prior request for prefix comparison"),
    ("src/runeconomy/analysis.py", "_finding_for_divergence", "creates a finding for semantic prefix divergence"),
    ("src/runeconomy/analysis.py", "_finding_for_runtime_gap", "creates a finding when observed cache reuse trails eligibility"),
    ("src/runeconomy/analysis.py", "_finding_for_unused_tools", "creates a finding for exposed tools that were not invoked"),
    ("src/runeconomy/analysis.py", "analyze_trace", "turns a normalized trace into a serializable analysis report"),
    ("src/runeconomy/cache.py", "longest_common_prefix", "counts identical leading token identifiers"),
    ("src/runeconomy/cache.py", "eligible_prefix_tokens", "computes the conservative full-block reusable prefix bound"),
    ("src/runeconomy/cli.py", "_load_trace", "loads and validates a normalized trace file"),
    ("src/runeconomy/cli.py", "_format_human", "formats an analysis report for terminal output"),
    ("src/runeconomy/cli.py", "_build_parser", "constructs the command-line argument parser"),
    ("src/runeconomy/cli.py", "_load_json_object", "loads a JSON file while requiring an object root"),
    ("src/runeconomy/cli.py", "_api_key", "reads a required API key from an environment variable"),
    ("src/runeconomy/cli.py", "_write_json", "atomically writes a JSON result file"),
    ("src/runeconomy/cli.py", "_append_jsonl", "durably appends one record to a JSONL file"),
    ("src/runeconomy/cli.py", "_require_trusted_evaluators", "blocks executable evaluators until explicitly authorized"),
    ("src/runeconomy/critical_path.py", "CriticalPath", "stores a calculated critical path and its duration"),
    ("src/runeconomy/critical_path.py", "calculate_critical_path", "finds the longest duration-weighted path through an execution DAG"),
    ("src/runeconomy/evaluators.py", "EvaluatorInfrastructureError", "distinguishes evaluator execution failure from task failure"),
    ("src/runeconomy/evaluators.py", "EvaluationResult", "stores evaluator success, score, kind, and details"),
    ("src/runeconomy/evaluators.py", "_require_object", "requires an evaluator value to be an object"),
    ("src/runeconomy/evaluators.py", "validate_evaluator_spec", "validates supported evaluator configurations"),
    ("src/runeconomy/evaluators.py", "_normalize_text", "applies configured whitespace and case normalization"),
    ("src/runeconomy/evaluators.py", "_coerce_python_result", "normalizes a Python evaluator return value"),
    ("src/runeconomy/evaluators.py", "evaluate_output", "runs a configured evaluator against generated output"),
    ("src/runeconomy/experiments.py", "ExperimentValidationError", "represents an invalid experiment manifest or run"),
    ("src/runeconomy/experiments.py", "_object", "requires a manifest value to be an object"),
    ("src/runeconomy/experiments.py", "_string", "requires a manifest value to be a non-empty string"),
    ("src/runeconomy/experiments.py", "_positive_int", "validates a bounded non-negative or positive integer"),
    ("src/runeconomy/experiments.py", "_expected_environment_identity", "derives the pinned environment identity from a manifest"),
    ("src/runeconomy/experiments.py", "_read_environment_identity", "reads optional identity evidence from a capture client"),
    ("src/runeconomy/experiments.py", "_environment_identity_check", "compares expected, before, and after environment identity"),
    ("src/runeconomy/experiments.py", "_materialize_trial", "replaces deterministic placeholders in a request"),
    ("src/runeconomy/experiments.py", "_tool_name", "extracts a tool name from either supported tool schema"),
    ("src/runeconomy/experiments.py", "_apply_request_transform", "applies a deterministic tool-list transformation"),
    ("src/runeconomy/experiments.py", "_predicted_eligible_tokens", "predicts reusable full-block tokens from prior prompts"),
    ("src/runeconomy/experiments.py", "_clopper_pearson_interval", "calculates an exact two-sided binomial interval"),
    ("src/runeconomy/experiments.py", "_paired_success_statistics", "bounds the paired task-success difference without bootstrap collapse"),
    ("src/runeconomy/experiments.py", "_percentile", "selects a nearest-rank percentile from numeric values"),
    ("src/runeconomy/experiments.py", "_metric_summary", "summarizes a metric with count, mean, median, and p95"),
]

PROMPT_CALIBRATION_TASKS = TASKS[:2]
HELDOUT_TASKS = TASKS[2:]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _tool_names(path: Path) -> list[str]:
    tools = json.loads(path.read_text(encoding="utf-8"))
    return [tool.get("name") or tool.get("function", {}).get("name") for tool in tools]


def _source_excerpt(relative_path: str, symbol: str) -> str:
    frozen = FROZEN_EXCERPTS.get((relative_path, symbol))
    if frozen is not None:
        return frozen.read_text(encoding="utf-8").rstrip("\n")
    heldout = FROZEN_HELDOUT_CONTEXTS.get((relative_path, symbol))
    if heldout is not None:
        context_name, expected_digest = heldout
        recorded = json.loads(OUTPUT.read_text(encoding="utf-8"))
        excerpt = recorded["constants"][context_name]
        observed_digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"frozen held-out excerpt {context_name} digest changed"
            )
        return excerpt
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == symbol
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one top-level {symbol!r} in {relative_path}")
    excerpt = ast.get_source_segment(source, matches[0])
    if excerpt is None:
        raise RuntimeError(f"could not extract {symbol!r} from {relative_path}")
    return excerpt


def build_manifest(
    tasks: list[tuple[str, str, str]] | None = None,
    *,
    experiment_id: str = "repository-agent-tool-pruning-heldout-v1",
    evaluation_split: str = "held_out_from_tool_and_prompt_calibration",
    eligible_for_final_claim: bool = True,
) -> dict:
    tasks = HELDOUT_TASKS if tasks is None else tasks
    calibration_bytes = CALIBRATION.read_bytes()
    calibration = json.loads(calibration_bytes)
    selected = calibration["selected_tools"]
    if _tool_names(SCOPED_TOOLS) != selected:
        raise RuntimeError("scoped tool fixture differs from frozen calibration selection")
    if not set(selected).issubset(_tool_names(BROAD_TOOLS)):
        raise RuntimeError("calibration selected a tool absent from the broad catalog")

    symbols_by_path: dict[str, list[str]] = {}
    for path, symbol, _purpose in tasks:
        symbols_by_path.setdefault(path, []).append(symbol)
    repository_map = "\n".join(
        f"- {path}: {MODULE_SUMMARIES[path]}; top-level symbols: "
        + ", ".join(symbols)
        for path, symbols in symbols_by_path.items()
    )
    constants: dict[str, str] = {"REPOSITORY_MAP": repository_map}
    cases: list[dict] = []
    for index, (path, symbol, purpose) in enumerate(tasks, 1):
        workflow_id = f"heldout-{index:02d}"
        context_name = f"SOURCE_CONTEXT_{index:02d}"
        constants[context_name] = _source_excerpt(path, symbol)
        shared_tools = {
            "baseline": {"$fixture": "BROAD_CODING_TOOLS"},
            "candidate": {"$fixture": "SCOPED_CODING_TOOLS"},
        }
        locate_variants = {}
        inspect_variants = {}
        for variant in ("baseline", "candidate"):
            locate_variants[variant] = {
                "request": {
                    "instructions": (
                        "You are a read-only coding agent. Use only the supplied "
                        "repository map; do not call a tool. Return only the bare "
                        "Python filename such as cache.py, with no directory, "
                        "markdown, or explanation.\n\n"
                        "{{REPOSITORY_MAP}}"
                    ),
                    "input": (
                        f"Which Python filename defines the top-level symbol {symbol}, "
                        f"which {purpose}?"
                    ),
                    "tools": shared_tools[variant],
                    "tool_choice": "none",
                }
            }
            inspect_variants[variant] = {
                "request": {
                    "instructions": (
                        "You are a read-only coding agent. Inspect only the supplied "
                        "Python source excerpt; do not call a tool. Return only the "
                        "bare top-level Python identifier, with no markdown or "
                        f"explanation.\n\n--- {path} ---\n{{{{{context_name}}}}}"
                    ),
                    "input": f"Which top-level symbol {purpose}?",
                    "tools": shared_tools[variant],
                    "tool_choice": "none",
                }
            }
        cases.extend(
            [
                {
                    "case_id": f"{workflow_id}-locate",
                    "workflow_id": workflow_id,
                    "evaluator": {"kind": "exact", "expected": Path(path).name},
                    "variants": locate_variants,
                },
                {
                    "case_id": f"{workflow_id}-inspect",
                    "workflow_id": workflow_id,
                    "evaluator": {"kind": "exact", "expected": symbol},
                    "variants": inspect_variants,
                },
            ]
        )

    return {
        "schema_version": "0.2",
        "experiment_id": experiment_id,
        "backend": {
            "kind": "vllm",
            "base_url": "https://vllm.example.invalid",
            "api_key_env": "VLLM_API_KEY",
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "chat_template_digest": CHAT_TEMPLATE_DIGEST,
            "backend_version": "0.33.0",
            "hardware": "1x NVIDIA A10 24GB on Modal",
            "block_size": 16,
            "engine_config": {
                "enable_prefix_caching": True,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 8,
            },
        },
        "sampling": {"temperature": 0.0, "max_output_tokens": 48, "seed": 404},
        "execution": {
            "repetitions": 1,
            "warmup_repetitions": 0,
            "order": "randomized",
            "seed": 404,
            "cache_treatment": "cold",
            "arrival_pattern": "sequential",
            "reset_external": False,
            "require_evaluation": True,
            "cases_form_workflow": True,
        },
        "acceptance": {
            "primary_metric": "total_latency_ms",
            "optimization_target": "input_tokens",
            "max_task_success_regression": 0.10,
            "task_success_method": "paired_exact",
            "min_completed_rate": 1.0,
            "min_input_token_reduction": 500,
            "require_latency_improvement": True,
            "require_environment_identity": True,
        },
        "design": {
            "evaluation_split": evaluation_split,
            "eligible_for_final_claim": eligible_for_final_claim,
            "workflow_unit_count": len(tasks),
            "steps_per_workflow": 2,
            "prompt_format_calibration": {
                "excluded_symbols": [
                    symbol for _path, symbol, _purpose in PROMPT_CALIBRATION_TASKS
                ],
                "change": (
                    "Repository maps now expose top-level symbol inventories and "
                    "location questions name the target symbol explicitly. Because "
                    "filenames are unique in this benchmark, both the requested "
                    "filename and inspected identifier are exact-match outputs."
                ),
                "excluded_from_heldout_evaluation": True,
            },
            "candidate_selection": {
                "calibration_id": calibration["calibration_id"],
                "calibration_sha256": "sha256:"
                + hashlib.sha256(calibration_bytes).hexdigest(),
                "selection_rule": calibration["selection_rule"],
                "selected_tools": selected,
                "evaluation_data_used": calibration["evaluation_data_used"],
            },
            "predeclared_primary_outcome": "summed total_latency_ms per workflow",
            "mechanism_metrics": [
                "input_tokens",
                "server_ttft_ms",
                "client_ttft_ms",
                "client_server_ttft_gap_ms",
                "server_queue_ms",
                "cached_tokens",
            ],
            "prediction": (
                "The scoped tool catalog will reduce input tokens and cold-cache "
                "workflow latency without reducing paired workflow success by more "
                "than 10 percentage points."
            ),
            "claim_boundary": (
                f"{len(tasks)} distinct fixture-backed, two-step repository-analysis units; "
                "no live tool execution or candidate-local state propagation."
            ),
        },
        "constants": constants,
        "json_fixtures": [
            {
                "name": "BROAD_CODING_TOOLS",
                "path": "../fixtures/coding-tools-broad.json",
                "sha256": _sha256(BROAD_TOOLS),
            },
            {
                "name": "SCOPED_CODING_TOOLS",
                "path": "../fixtures/coding-tools-scoped.json",
                "sha256": _sha256(SCOPED_TOOLS),
            },
        ],
        "cases": cases,
    }


def main() -> None:
    payload = build_manifest()
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    calibration_payload = build_manifest(
        PROMPT_CALIBRATION_TASKS,
        experiment_id="repository-agent-tool-pruning-prompt-calibration-v1",
        evaluation_split="prompt_format_calibration_only",
        eligible_for_final_claim=False,
    )
    CALIBRATION_OUTPUT.write_text(
        json.dumps(calibration_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(HELDOUT_TASKS)} held-out workflow units to {OUTPUT}")
    print(
        f"wrote {len(PROMPT_CALIBRATION_TASKS)} prompt-calibration units to "
        f"{CALIBRATION_OUTPUT}"
    )


if __name__ == "__main__":
    main()
