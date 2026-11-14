# Normalized trace format 0.2

The trace format is local-first JSON. Raw prompt text is optional and is not
part of the initial schema. Token IDs can often be decoded and must still be
treated as sensitive data.

```json
{
  "schema_version": "0.2",
  "trace_id": "trace-example",
  "workload": "coding-agent",
  "success": true,
  "spans": [
    {
      "span_id": "step-1",
      "kind": "inference",
      "start_ms": 0,
      "end_ms": 30,
      "request_id": "req-1",
      "depends_on": []
    }
  ],
  "requests": [
    {
      "request_id": "req-1",
      "sequence": 1,
      "model": "example/model",
      "model_revision": "revision-or-digest",
      "backend": "vllm",
      "backend_version": "0.x",
      "tokenizer": "example/model",
      "tokenizer_revision": "revision-or-digest",
      "chat_template_digest": "sha256:...",
      "prompt_token_ids": [1, 2, 3],
      "prompt_token_count": 3,
      "token_evidence_quality": "exact",
      "segments": [
        {
          "kind": "system",
          "label": "agent instructions",
          "start_token": 0,
          "end_token": 3,
          "attribution_quality": "exact_token_boundary",
          "source": "server_render"
        }
      ],
      "available_tools": ["read_file", "search_repository", "run_tests"],
      "invoked_tools": ["read_file"],
      "cache": {
        "enabled": true,
        "block_size": 16,
        "cache_salt_digest": null
      },
      "metrics": {
        "input_tokens": 3,
        "output_tokens": 1,
        "cached_input_tokens": 0,
        "ttft_ms": 12.3,
        "queue_ms": null,
        "prefill_ms": null,
        "decode_ms": null,
        "cost_usd": null,
        "gpu_time_ms": null,
        "total_latency_ms": 30.0
      }
    }
  ]
}
```

Version `0.2` also accepts count-only requests with `prompt_token_ids: null`, a
known `prompt_token_count`, and `token_evidence_quality: "count_only"`. When
neither tokens nor a reliable count is available, both are null and the quality
is `unknown`. Prefix matching and static cache-eligibility analysis run only on
exact token evidence. Version `0.1` remains readable for compatibility.
Analysis reports expose per-request `token_evidence_quality`,
`attribution_qualities`, and `prefix_analysis_supported`, plus summary counts by
token-evidence quality, so unsupported prefix analysis is visible rather than
silently reported as a zero-length match.

Attribution quality is explicit:

- `exact_token_boundary`: token offsets were verified against the complete
  rendered prompt.
- `block_level`: offsets identify a known rendered or cache block, but not a
  more precise semantic boundary within it.
- `structural`: the request or trace identifies a semantic block, but the chat
  template does not provide defensible token offsets. `start_token` and
  `end_token` must be null.
- `unknown`: neither a reliable block nor token boundary is available. Offsets
  must be null.

Known segment ranges are half-open: `start_token` is included and `end_token`
is excluded. They may be adjacent and must not overlap. Gaps are allowed for
template tokens that cannot be assigned safely.

Requests are comparable for cache analysis only when cache-relevant identity
matches. The analyzer checks the backend, model and revision, tokenizer and
revision, chat-template digest, block size, and cache-salt digest.

`available_tools` records the unique tool names exposed to the model for that
inference request. `invoked_tools` records the subset selected by the resulting
model output. When at least half of a catalog of four or more tools is unused,
the analyzer emits a medium-confidence `tools.unused_exposure` finding. This is
observed evidence for a scoped-tool experiment, not proof that the unused tools
are unnecessary on every representative task.

## Structured findings

Analysis report version `0.2` retains the human-readable summary and adds a
machine-readable experiment intent to every finding. Important fields include:

- `finding_id`: a stable identifier derived from the trace, rule, and affected
  requests.
- `affected_request_ids`, `evidence_level`, `confidence`, and
  `on_critical_path`.
- `estimated_reducible_tokens` and `estimated_reducible_ms`. Unknown estimates
  remain `null`; RunEconomy does not derive milliseconds from tokens without a
  measured model.
- `optimization_target`, `experiment_template_id`, and
  `experiment_parameters`.
- `required_capabilities` and `unsupported_reason`.
- `causal_mechanism`, `predicted_metric`, and `predicted_direction`.

`suggested_experiment` remains for backward-compatible human output. The
structured fields—not that prose—drive `runeconomy experiment init`.

`spans` are non-nested executable workflow steps. `depends_on` contains direct
predecessor span IDs and defines a directed acyclic graph. The analyzer computes
the longest duration-weighted path through that graph. Nested tracing spans may
be retained by an importer, but must be normalized into non-overlapping work
steps before entering this field; otherwise parent and child time would be
double-counted.

## Importing vLLM capture records

`runeconomy import-vllm-jsonl` converts one JSON object per inference request
into this schema. Each line can be a raw `capture-vllm` result or an envelope
with `capture`, the original `request`, and optional `request_id`,
`invoked_tools`, `segments`, timing, and dependency fields. Exact
`prompt_token_ids` are required. When dependency fields are absent, the
importer models the records as sequential inference spans.

If an envelope supplies `segments`, those claims are validated normally. A
source may instead supply `rendered_segments`, each with `token_ids`; the
importer accepts exact boundaries only when those arrays concatenate exactly
to the complete captured prompt. Otherwise, the importer derives structural
blocks from message roles, tool definitions, retrieved-context fields, and
metadata. It never tokenizes request text locally to guess template boundaries.
Prompt token IDs and request bodies remain sensitive even when raw prompt text
is omitted.

The vLLM adapter can append compatible envelopes while capturing requests:

```bash
runeconomy capture-vllm request.json \
  --base-url https://your-benchmark-server.example \
  --api-key-env VLLM_API_KEY \
  --request-id request-1 \
  --append-trace-jsonl captures.jsonl

runeconomy import-vllm-jsonl captures.jsonl \
  --output trace.json \
  --trace-id agent-run-1 \
  --workload coding-agent \
  --model example/model
```

The JSONL envelope contains the original request body. Keep it local or
sanitize it before sharing.

## Importing OpenInference/OpenTelemetry

`runeconomy import-openinference` reads one trace from an OTLP JSON export. It
recognizes OpenInference LLM, tool, and retriever spans, model/provider and token
attributes, flattened input-message roles, tool definitions, output tool calls,
and cache-read counts. Message content is not copied into the normalized trace.

OpenInference normally does not contain the server's rendered token sequence,
so imported LLM requests are explicitly `count_only` or `unknown` and semantic
segments are `structural`. An instrumentation source may provide the extension
attribute `runeconomy.prompt_token_ids` as an integer array or JSON array string;
only then is the request marked exact. This extension is not an OpenInference
standard.

```bash
runeconomy import-openinference otlp-trace.json \
  --output trace.json \
  --workload coding-agent
```

The importer retains selected LLM/tool/retriever work spans but does not infer
dependencies from nested telemetry parents. `depends_on` therefore stays empty
unless spans provide the RunEconomy extension attribute
`runeconomy.depends_on`, containing an array or JSON array string of direct
predecessor span IDs. Critical-path results are available only when every
retained work span declares that attribute; otherwise the dependency evidence
is `partial_explicit` or `absent` and must not be presented as a reconstructed
workflow DAG.

## Importing Prefixwise evidence

Prefixwise documents `prefixwise export --format jsonl` as raw ledger rows.
Those rows contain causes, counts, costs, divergence roles/indexes, and other
evidence, while Prefixwise intentionally stores hashes and counts rather than
prompt text. RunEconomy imports them as structured finding evidence:

```bash
prefixwise export --format jsonl > prefixwise.jsonl
runeconomy import-prefixwise-jsonl prefixwise.jsonl \
  --output findings.json
runeconomy experiment init \
  --from-finding findings.json \
  --finding-id FINDING_ID \
  --output experiment-skeleton.json
```

The import allowlists documented ledger fields, does not copy arbitrary request
bodies, and never reconstructs raw prompts or token arrays. Because ledger rows
are not replayable workloads and do not establish workflow critical-path
membership, `experiment init` produces an unresolved skeleton until the user
provides requests, a candidate change, backend controls, acceptance gates, and
an evaluator.
