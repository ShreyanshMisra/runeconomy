# CLI reference

Run `runeconomy COMMAND --help` for every flag. Human output is plain text and
does not rely on color or animation. Machine-readable commands support JSON
where documented.

## Canonical workflow

| Command | Purpose |
| --- | --- |
| `runeconomy import SOURCE --output evidence.json` | Detect and normalize evidence, then assess its capabilities. |
| `runeconomy experiment init --from-evidence evidence.json --output experiment/` | Create a static experiment workspace or an honest unresolved workspace. |
| `runeconomy experiment init --mode agent-process ... --output experiment/` | Create a complete-agent experiment workspace. |
| `runeconomy experiment validate experiment/` | Report runnable, unresolved, invalid, unsupported-backend, or claim-incomplete status. |
| `runeconomy experiment run experiment/ --output results/` | Execute the workspace and preserve durable results. |
| `runeconomy experiment report results/result.json --output report.html --open` | Render the evidence-derived report. |

`experiment validate --json` emits structured issues for coding agents and CI.
`--doctor-report doctor.json` applies a saved read-only capability diagnosis
without probing the backend again.

## Evidence and compatibility commands

| Command | Purpose |
| --- | --- |
| `analyze` | Advanced analysis of a normalized trace. |
| `doctor` | Inspect vLLM without inference or cache reset. |
| `import-vllm-jsonl` | Compatibility importer for exact-token vLLM captures. |
| `import-openinference` | Compatibility importer for one OpenInference or OTLP trace. |
| `import-prefixwise-jsonl` | Named Prefixwise ledger importer. |
| `import-ledger-jsonl` | Backward-compatible generic ledger importer. |
| `capture-vllm` | Render and execute one supported vLLM request. |
| `compare` | Compatibility wrapper for direct agent-process execution. |

New integrations should prefer the unified `import` and `experiment` workflow.
The compatibility commands continue to read existing artifacts without
silently upgrading their evidence quality.

## First use

| Command | Purpose |
| --- | --- |
| `runeconomy demo --open` | Render the bundled frozen rejection proof. |
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

Human errors state what failed, why RunEconomy believes it failed, whether
evidence may be incomplete, and the next action. JSON output never includes an
API-key value. Partial checkpoints and private artifacts are evidence, not
final verdicts.
