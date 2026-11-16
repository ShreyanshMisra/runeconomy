# RunEconomy 0.2.0a4 alpha guide

`0.2.0a4` is an invited-tester prerelease focused on one coherent
evidence-to-verdict workflow. RunEconomy remains a local CLI and Python package,
not a production service or general observability platform.

## Install

```bash
python -m pip install --pre runeconomy==0.2.0a4
runeconomy --version
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

The tutorial is deterministic and local. It uses RunEconomy-enriched
OpenInference evidence and validates product plumbing only. Ordinary telemetry
may produce an unresolved workspace. Use `runeconomy demo --open` to inspect the
immutable aggregate-only rejection proof.

## Evidence-to-report workflow

```bash
runeconomy import SOURCE --output evidence.json
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

The unified importer assesses basic diagnosis, structural attribution, exact
cache analysis, critical-path analysis, static replay, complete agent
comparison, and public-claim readiness. Missing evidence remains unknown.

Experiment initialization creates a small workspace with source provenance,
the selected finding, intervention, mode, gates, metrics, identity, unresolved
fields, and the exact next command. An incomplete workspace is not runnable.

Both static requests and complete agent processes use the same experiment
command family. `compare` remains a compatibility wrapper and prints migration
guidance.

## Connectors

- OpenInference and OpenTelemetry are the golden path. Exact token boundaries
  and workflow dependencies require explicit evidence.
- vLLM capture JSONL supports exact-token analysis when the capture contains it.
- Prefixwise ledger JSONL preserves findings, hashes, counts, and provenance.
  It does not reconstruct prompts or tokens and normally creates an unresolved
  workspace.
- Phoenix CLI JSON has a narrow adapter. Parentage is structural only.
- LangSmith users should use its documented OpenTelemetry export or fan-out.
- AgenticLens has no claimed native adapter until a stable export contract is
  available.

## Compatibility and migration from 0.2.0a3

Existing normalized traces, structured findings, static manifests, results, and
the legacy importer commands remain supported. `analyze` remains available for
advanced use.

New workflows should replace importer-specific first steps with
`runeconomy import`, use a directory workspace, and use agent-process mode
instead of calling `compare` directly. A `.json` output path still creates the
legacy monolithic finding artifact where supported.

Evidence bundle schema `0.1` and workspace schema `0.1` are new. They wrap
existing artifacts without silently strengthening their evidence. Validation
statuses now distinguish runnable, unresolved, invalid, unsupported backend,
and incomplete claim evidence.

## Privacy and limitations

Raw prompts, replay requests, traces, generated output, evaluator data, task
answers, credentials, endpoints, and local experiment artifacts remain the
operator's responsibility. Source evidence is referenced by name and digest in
the workspace rather than copied into it.

The zero-edit public fixture uses a deterministic mock vLLM server. It validates
workflow plumbing, not real serving performance. Live vLLM behavior still
depends on deployment capabilities reported by `doctor`.

RunEconomy does not provide a hosted dashboard, managed collector,
framework-specific SDK, automatic production changes, automatic resume, SGLang
runner, or broad diagnosis catalog.
