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

A directory may contain several JSON traces. The imported bundle lists each run
and requires deliberate selection before runnable initialization.

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
