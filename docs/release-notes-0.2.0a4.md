# RunEconomy 0.2.0a4 release notes

RunEconomy `0.2.0a4` is the first prerelease with a coherent
evidence-to-verdict workflow.

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
python -m pip install --pre runeconomy==0.2.0a4
```

## Canonical workflow

```bash
runeconomy import otlp.json --output evidence.json
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

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
