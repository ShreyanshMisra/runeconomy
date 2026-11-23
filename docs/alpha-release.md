# RunEconomy 0.2.1 alpha guide

RunEconomy `0.2.1` adds privacy-safe tester support to the complete
evidence-to-verdict workflow. RunEconomy remains alpha-stage software for
controlled experiments and private testing. It is a local CLI and Python
package, not a production service or general observability platform.

## Install

```bash
python -m pip install runeconomy==0.2.1
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

Real-agent integrations may use the optional dependency-free
`runeconomy.agent.Attempt` helper at the existing entry point. It implements
only the environment and status contract. `runeconomy experiment plan` performs
a local read-only design assessment before calibration or held-out execution.

When a workflow fails, public commands display a stable `runeconomy.*` error
code. A local support bundle can preserve safe capability, workspace,
validation, and error states without copying raw experiment content:

```bash
runeconomy support-bundle experiment/ --output runeconomy-support.zip
```

The archive is never uploaded automatically. Inspect it before sharing.

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

## Compatibility with 0.2.0

Existing normalized traces, structured findings, static manifests, results, and
the legacy importer commands remain supported. `analyze` remains available for
advanced use.

New workflows should replace importer-specific first steps with
`runeconomy import`, use a directory workspace, and use agent-process mode
instead of calling `compare` directly. A `.json` output path still creates the
legacy monolithic finding artifact where supported.

Evidence bundle schema `0.1`, workspace schema `0.1`, experiment behavior, and
validation statuses are unchanged from `0.2.0`. Support-bundle schema `0.1` and
stable error-code metadata are additive.

## Privacy and limitations

Raw prompts, replay requests, traces, generated output, evaluator data, task
answers, credentials, endpoints, and local experiment artifacts remain the
operator's responsibility. Source evidence is referenced by name and digest in
the workspace rather than copied into it.

The zero-edit public fixture uses a deterministic mock vLLM server. It validates
workflow plumbing, not real serving performance. Live vLLM behavior still
depends on deployment capabilities reported by `doctor`.

Support bundles are built from a strict allowlist and exclude raw traces,
results, prompts, responses, evaluator data, endpoints, credentials, paths, and
arbitrary workspace files. Privacy minimization is not a guarantee of
anonymity.

RunEconomy does not provide a hosted dashboard, managed collector,
framework-specific SDK, automatic production changes, automatic resume, SGLang
runner, or broad diagnosis catalog.
