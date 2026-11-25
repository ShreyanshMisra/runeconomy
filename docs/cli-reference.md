# CLI reference

Run `runeconomy COMMAND --help` for every flag. Human output is plain text and
does not rely on color or animation. Machine-readable commands support JSON
where documented.

## Canonical workflow

| Command | Purpose |
| --- | --- |
| `runeconomy proxy --upstream URL --record session.jsonl` | Capture OpenAI-compatible traffic on loopback and write a new session. |
| `runeconomy import SOURCE --output evidence.json` | Detect and normalize evidence, then assess its capabilities. |
| `runeconomy experiment init --from-evidence evidence.json --output experiment/` | Create a static experiment workspace or an honest unresolved workspace. |
| `runeconomy experiment init --mode agent-process ... --output experiment/` | Create a complete-agent experiment workspace. |
| `runeconomy experiment validate experiment/` | Report runnable, unresolved, invalid, unsupported-backend, or claim-incomplete status. |
| `runeconomy experiment plan experiment/` | Assess claim-design readiness without execution or network access. |
| `runeconomy experiment plan --examples` | Show the 40-pair and 180-pair examples and the prospective FP8 vs BF16 design. |
| `runeconomy experiment patch experiment/ --set PATH=JSON` | Change manifest fields with validation and a provenance entry, with no hand-editing. |
| `runeconomy experiment freeze experiment/` | Lock digests and gates and write `preregistration.json`. |
| `runeconomy experiment run experiment/ --output results/` | Execute the workspace and preserve durable results. |
| `runeconomy experiment report results/result.json --output report.html --open` | Render the evidence-derived report. |
| `runeconomy support-bundle experiment/ --output runeconomy-support.zip` | Create a local privacy-minimized diagnostic archive. |
| `runeconomy mcp` | Serve the workflow as MCP tools over stdio, with an approval step before execution. [MCP server](mcp.md). |
| `runeconomy ci-summary results/result.json` | Render the Markdown verdict table used by the GitHub Action. |

`experiment validate --json` emits structured issues for coding agents and CI.
`--doctor-report doctor.json` applies a saved read-only capability diagnosis
without probing the backend again.

Every command that finishes with a result accepts `--json` and prints one
object with a versioned `output_schema` on stdout. Failures print the
`runeconomy.cli.error.v1` object on stderr. `runeconomy schema` lists the
contracts. `experiment patch` and `experiment freeze` change and lock a
workspace without hand-editing JSON. [JSON contracts, patch and freeze](json-contracts-patch-freeze.md).

Proxy capture accepts `--token-resolution background\|on_close\|off`,
`--payload-capture full\|digest_only` and repeatable `--redact module:function`.
Standalone capture exits `0` after a graceful drain, `1` when the summary
reports recording loss or unfinished drain or resolution, and `2` for
configuration or startup errors. `proxy --attempt -- COMMAND` runs inside a
runner-managed attempt and propagates the child status after exporting trace
evidence. For import, use `--replay-context` to supply replay design fields or
`--trace-id` to select a captured workflow. [Capture guide](proxy.md).

`experiment plan --json` reports `blocked`, `needs_design_input`,
`ready_for_calibration`, or `ready_to_freeze`. Planning reuses validation and
the exact task-success methods. It never produces an experiment verdict.
Planning exits `0` for calibration or freeze readiness, `1` when design input
is missing, and `2` when the design is blocked.

With planning schema `0.2`, the human report shows the declared task-success
method, estimated power at the planned sample, recommended pair count,
completion tolerance and estimated cost when available. JSON includes both
methods in `sample_size.method_comparison` and the generated draft in
`preregistration`. Use `--examples --json` for the example assumptions and
calculations. [Configure a powered plan](experiment-format.md).

For agent-process engine changes, initialize with
`--optimization-target engine-change-latency`. The default latency semantics
are superiority. `--latency-semantics noninferiority` additionally requires a
predeclared non-negative `--max-latency-regression-ms`. Both modes retain exact
task-quality and completion gates. `experiment run --resume` validates and
continues either static or agent-process checkpoints.

`support-bundle` never contacts a backend or uploads anything. Optional
`--doctor-report` and `--validation-report` inputs must already exist. Run with
`--json` for a machine-readable creation summary. The command refuses to
overwrite an existing archive and never follows workspace symlinks.

## Evidence and compatibility commands

| Command | Purpose |
| --- | --- |
| `analyze` | Advanced analysis of a normalized trace. |
| `doctor` | Inspect vLLM, SGLang (`--backend sglang`) or an OpenAI-compatible API (`--backend openai_compat`) without inference or cache reset. See [backends](backends.md). |
| `import-vllm-jsonl` | Compatibility importer for exact-token vLLM captures. |
| `import-openinference` | Compatibility importer for one OpenInference or OTLP trace. |
| `import-prefixwise-jsonl` | Named Prefixwise ledger importer. |
| `import-ledger-jsonl` | Backward-compatible generic ledger importer. |
| `capture-vllm` | Render and execute one supported vLLM request. |
| `compare` | Compatibility wrapper for direct agent-process execution. |

New integrations should prefer the unified `import` and `experiment` workflow.
The compatibility commands continue to read existing artifacts without
silently upgrading their evidence quality.

`doctor --startup-timeout SECONDS`, `capture-vllm --startup-timeout SECONDS`,
and static `experiment run --startup-timeout SECONDS` share one bounded cold
start retry policy for connection failures and HTTP 502/503 responses.

## First use

| Command | Purpose |
| --- | --- |
| `runeconomy demo --open` | Render the bundled frozen proof with an `invalid` verdict. |
| `runeconomy tutorial init DIRECTORY` | Materialize the sanitized installed tutorial without overwriting files. |
| `runeconomy tutorial serve` | Start the deterministic localhost-only tutorial server. |
| `runeconomy doctor ...` | Diagnose vLLM compatibility safely. |
| `runeconomy --version` | Print the installed package version. |

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The command completed successfully. For `doctor`, a safe mode is ready. |
| `1` | Diagnosis or validation completed, but the artifact is not runnable. |
| `2` | Input, configuration, authentication, connection, or validation failure. |
| `3` | `--require-verified` was set and the verdict was not `verified`. |

An exit code does not replace a verdict. Read `decision.status`, the decisive
gate, attempt classifications, and evidence integrity.

## Error contract

Human errors state a stable `runeconomy.*` code, what failed, why RunEconomy
believes it failed, whether evidence may be incomplete, and the next action.
JSON errors expose the same code separately from mutable prose. JSON output
never includes an API-key value. Partial checkpoints and private artifacts are
evidence, not final verdicts.
