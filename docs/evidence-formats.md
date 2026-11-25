# Evidence formats

`runeconomy import` is the public entry point for evidence:

```bash
runeconomy import SOURCE --output evidence.json
```

It detects the supported format, normalizes only established fields, records
source provenance, and emits a human or JSON capability assessment.

## Capability levels

Every bundle explains whether it can support:

1. Basic diagnosis
2. Structural prompt attribution
3. Exact token-level cache analysis
4. Critical-path analysis
5. Static request replay
6. Complete agent comparison
7. A public verification claim

Each capability is supported, unsupported, or unknown. A missing metric remains
unknown. Unsupported replay does not prevent diagnosis, but it prevents a
runnable replay workspace.

## Field evidence grades

Every normalized trace now carries evidence-quality schema `0.1`. Human import
output and machine-readable bundles show the same field grades:

| Grade | Meaning |
| --- | --- |
| `directly_observed` | The source recorded the field with unambiguous semantics. |
| `reliably_derived` | RunEconomy deterministically derived the field from directly recorded values. |
| `weakly_inferred` | The source field is usable only with an explicit semantic caveat. |
| `unresolved` | The evidence does not justify a value or claim. |

The overall grade is the weakest recorded field. A stronger field cannot mask
an unresolved one. Generic `gen_ai.ttft_ms` is weakly inferred because the
clock boundary may be framework- or provider-specific. Only the explicit
`runeconomy.server_ttft_ms` extension is treated as directly observed server
TTFT. Conflicting aliases are rejected instead of selected silently.

## Tutorial evidence

The installed tutorial uses a RunEconomy-enriched OpenInference fixture. It
contains explicit prompt token arrays, verified prompt segments, complete
workflow dependencies, replay requests, a synthetic evaluator, acceptance
gates, and deterministic mock-backend context.

This is intentionally stronger than an ordinary OpenTelemetry export. Generic
user traces may support diagnosis while still producing an unresolved
workspace. Importing a historical trace does not create a verification report.
The report is generated only after the runner executes newly declared baseline
and candidate trials.

## OpenInference and OpenTelemetry

Current GenAI and OpenInference exports can provide model and provider identity,
token counts, timing, task metadata, tool and retrieval spans, and selected
serving measurements when the source recorded them. Ordinary parent-child span
relationships are structural evidence, not proof of an execution dependency.
Critical-path analysis requires explicit `runeconomy.depends_on` declarations
for every retained work span.

Rendered token IDs and exact semantic segments are RunEconomy extensions. Exact
segment boundaries are accepted only when their token arrays reproduce the
complete prompt token sequence. Otherwise attribution stays structural or
unknown.

Deterministic compatibility fixtures cover RunEconomy-enriched OpenInference,
a synthetic LangChain/LangSmith-style OpenTelemetry GenAI export, and the
documented Phoenix CLI shape. They test import semantics and do not represent
live framework or workload evidence.

A directory may contain several JSON traces. The imported bundle lists each run
and requires deliberate selection before runnable initialization.

## Proxy sessions

RunEconomy proxy JSONL sessions normalize into one evidence run per captured
workflow. Grouping uses explicit trace IDs or a configurable idle gap.
Dependency edges require explicit assertions. Exact prompt IDs come from
the serving backend or an explicit token-ID request. Missing evidence stays
unknown. [Capture and prepare a replay](proxy.md).

## vLLM captures

Exact-token vLLM JSONL captures can support exact cache analysis. Static replay
also requires the request body, workload, evaluator, gates, backend controls,
and identity. Prompt token IDs and request bodies are sensitive.

## Prefixwise ledgers

Prefixwise ledgers can preserve classified causes, hashes, counts, divergence
roles, and source provenance. RunEconomy imports only the evidence present. It
does not reconstruct prompts or token arrays from hashes. A ledger can produce
a hypothesis and unresolved workspace, but not a runnable experiment unless
replay inputs and evaluation evidence are supplied separately.

## Phoenix JSON traces

RunEconomy supports the documented Phoenix CLI JSON trace shape. It imports
span counts, timing, identity, and allowed attributes without copying raw input
or output. Phoenix parentage remains structural. Explicit RunEconomy dependency
attributes are required for a complete workflow DAG.

See the [normalized trace reference](trace-format.md) for the lower-level schema
and [connector support](connectors.md) for current boundaries.
