# RunEconomy 0.2.0 release notes

RunEconomy `0.2.0` is the first complete evidence-to-verdict release. It is the
default installable release and remains alpha-stage software for controlled
experiments and private testing. This release is not a production-readiness or
stable-API commitment.

## What changed

- `runeconomy import` detects supported evidence and emits a human and JSON
  capability assessment.
- `experiment init --from-evidence` creates a versioned workspace with source
  provenance, a selected finding, intervention, manifest, unresolved fields,
  and the exact next command.
- Static request and complete agent-process comparisons now share
  `experiment init`, `validate`, `run`, and `report`.
- Validation distinguishes runnable, unresolved, invalid, unsupported backend,
  and incomplete public-claim evidence, with corrective actions for every issue.
- A realistic OpenInference fixture supports the complete zero-edit workflow
  through a deterministic mock vLLM integration test.
- `runeconomy tutorial init` and `runeconomy tutorial serve` make the complete
  workflow available from an installed wheel without cloning the repository.
- Package CI installs the wheel outside the checkout and runs import,
  initialization, validation, controlled trials, and report generation.
- Prefixwise ledger evidence imports without reconstructing prompts or tokens.
- A narrow Phoenix JSON adapter preserves documented structural evidence while
  keeping raw input and output out of the bundle.
- The official agent skill is version-neutral, installs through the portable
  Agent Skills CLI, and loads focused evidence, experiment, verdict, privacy,
  and troubleshooting references as needed.
- The documentation site publishes canonical latest and versioned skill files,
  a SHA-256 checksum, `llms.txt`, and `llms-full.txt`.

## Installation

```bash
python -m pip install runeconomy==0.2.0
```

## Canonical workflow

```bash
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve

runeconomy import otlp.json --output evidence.json
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

The tutorial uses a RunEconomy-enriched OpenInference fixture with explicit
prompt tokens, prompt segments, dependencies, and experiment context. Ordinary
OpenTelemetry evidence may support diagnosis while still producing an
unresolved workspace. The deterministic server validates product plumbing. It
does not run a model or measure real inference performance.

## Compatibility

Existing trace, finding, static manifest, result, and importer-specific commands
remain supported. `compare` remains a compatibility wrapper and now directs new
integrations to `experiment init --mode agent-process`.

Evidence bundle and workspace schema `0.1` wrap existing artifacts. They do not
reinterpret structural parentage as workflow dependencies, hashes as prompt
tokens, or missing metrics as zero.

## Connector boundaries

OpenInference and OpenTelemetry are the golden path. vLLM capture and Prefixwise
ledger imports are supported within their recorded evidence. Phoenix support is
limited to the documented CLI JSON trace shape. AgenticLens and native LangSmith
adapters remain pending. LangSmith OpenTelemetry export can use the standard
OpenInference path.

## Known limitations

The bundled golden path uses a deterministic mock server and supports an
engineering plumbing claim only. No new live-vLLM performance claim is included.
Run `doctor` before using a real backend.

Keep prompts, traces, generated output, evaluator data, credentials, private
endpoints, and local result artifacts private unless explicitly sanitized.

## Documentation

- [Start with RunEconomy](getting-started.md)
- [Evidence formats](evidence-formats.md)
- [Experiment contract](experiment-format.md)
- [Connector support](connectors.md)
- [Install the coding-agent skill](agent-skill.md)
