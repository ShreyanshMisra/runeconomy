# CLI reference

Run `runeconomy COMMAND --help` for complete flags. Human output is plain text,
does not rely on color, respects non-interactive streams, and uses no animation.
Set `NO_COLOR=1` when standardizing a wider toolchain; RunEconomy itself emits
no ANSI color in this alpha.

## First use

| Command | Purpose |
| --- | --- |
| `runeconomy demo --open` | Render the bundled frozen rejection proof. |
| `runeconomy doctor ...` | Diagnose vLLM compatibility without inference or cache reset. |
| `runeconomy --version` | Print the installed package version. |

## Evidence

| Command | Purpose |
| --- | --- |
| `analyze` | Analyze a normalized trace and emit structured findings. |
| `import-vllm-jsonl` | Normalize exact-token vLLM capture envelopes. |
| `import-openinference` | Normalize structural OpenInference/OTLP JSON evidence. |
| `import-prefixwise-jsonl` | Import evidence present in Prefixwise ledgers. |
| `capture-vllm` | Render and execute one supported vLLM request. |

`analyze` and `doctor` accept `--json` as an alias for `--format json`.

## Experiments

| Command | Purpose |
| --- | --- |
| `experiment init` | Generate a manifest or honest unresolved skeleton from a finding. |
| `experiment validate` | Validate schema, identities, execution controls, and evaluators. |
| `experiment run` | Execute a static paired request experiment. |
| `experiment report` | Render a NegativeSplit HTML or Markdown report. |
| `compare` | Execute complete baseline and candidate agent processes. |

`negative-split` remains a backward-compatible alias for `experiment`.
NegativeSplit is the methodology and report name. `compare` is reserved for
closed-loop agent execution.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | The command completed successfully. For `doctor`, a safe mode is ready. |
| `1` | `doctor` completed its diagnosis but could not establish a safe mode. |
| `2` | Input, configuration, authentication, connection, or validation failure. |
| `3` | `--require-verified` was set and the result was not `verified`. |

An exit code does not replace the result verdict. Always read the result's
`decision.status`, decisive gate, attempt classifications, and evidence
integrity fields.

## Error contract

Human errors state what failed, why RunEconomy believes it failed, whether
output may be incomplete, and the exact next action. `doctor --json` emits the
same information as structured JSON on stderr when the probe fails and never
includes the API-key value or configured endpoint. Successful or partially
supported diagnoses remain on stdout. Partial checkpoints and private
artifacts are evidence, not final verdicts.
