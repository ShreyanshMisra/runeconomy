# Connector support

Connectors preserve the strength and limitations of their source evidence. An
imported observation does not automatically become causal, paired, replayable,
or eligible for a public verification claim.

| Source | Status | Preserved | Important boundary |
| --- | --- | --- | --- |
| OpenTelemetry and OpenInference | Golden path with deterministic OpenInference and LangChain/LangSmith-style OTEL fixtures | Current OTLP structures, GenAI and OpenInference attributes, explicit dependency extensions, optional exact token extensions | Fixtures are synthetic compatibility evidence, not live framework validation. Span nesting is not a workflow DAG. Exact prompt boundaries require explicit verified tokens. |
| vLLM capture JSONL | Supported | Rendered prompt tokens, request envelope, timing and cache fields when present | Replay and public claims still need evaluator, workload, gates, and identity. |
| Prefixwise ledger JSONL | Supported hypothesis import | Findings, hashes, counts, divergence evidence, source rule IDs | Prompts and token arrays are not reconstructed. Most imports create unresolved workspaces. |
| Phoenix JSON trace | Narrow supported adapter with recorded deterministic fixture | Documented CLI JSON trace shape, structural spans, allowed metrics and identity | The fixture is not live evidence. Raw input and output are excluded. Parentage is structural only. |
| LangSmith | Use OpenTelemetry export | Standard OpenTelemetry fan-out when configured by the operator | Native bulk export is not a stable, universally available RunEconomy contract. |
| AgenticLens | Pending | OpenTelemetry evidence can use the OpenInference path when compatible | No stable native export contract is claimed. |

## Prefixwise positioning

Prefixwise diagnoses why cache reuse failed. RunEconomy verifies whether a
proposed correction improves successful workflows. The tools are complementary.

## Deferred adapters

RunEconomy does not scrape private APIs or couple to unstable internal schemas.
AgenticLens and native LangSmith adapters remain pending until a stable public
artifact establishes a useful contract beyond OpenTelemetry.
