# Normalized trace format 0.1

The trace format is local-first JSON. Raw prompt text is optional and is not
part of the initial schema. Token IDs can often be decoded and must still be
treated as sensitive data.

```json
{
  "schema_version": "0.1",
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
      "segments": [
        {
          "kind": "system",
          "label": "agent instructions",
          "start_token": 0,
          "end_token": 3
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
        "total_latency_ms": 30.0
      }
    }
  ]
}
```

Segment ranges are half-open: `start_token` is included and `end_token` is
excluded. Segments may be adjacent and must not overlap. Gaps are allowed for
template tokens that cannot be assigned safely.

Requests are comparable for cache analysis only when cache-relevant identity
matches. Version 0.1 checks the backend, model and revision, tokenizer and
revision, chat-template digest, block size, and cache-salt digest.

`available_tools` records the unique tool names exposed to the model for that
inference request. `invoked_tools` records the subset selected by the resulting
model output. When at least half of a catalog of four or more tools is unused,
the analyzer emits a medium-confidence `tools.unused_exposure` finding. This is
observed evidence for a scoped-tool experiment, not proof that the unused tools
are unnecessary on every representative task.

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

The importer does not invent semantic segment boundaries or invoked tools.
Those analyses are available only when the capture source supplies that
evidence. Prompt token IDs and request bodies remain sensitive even when raw
prompt text is omitted.

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
