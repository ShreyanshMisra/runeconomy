# Experiment manifest and vLLM runner

The `0.1` experiment format defines a paired baseline/candidate replay against
one vLLM backend. It is intentionally narrow: requests use vLLM's Responses
API, run sequentially, and contain their complete history inline.

## Why the runner renders first

Before every generation, the adapter sends the same request to
`/v1/responses/render`. The returned `token_ids` are the exact prompt tokens
produced by the server's tokenizer and chat template. This avoids a misleading
comparison caused by recreating prompt rendering in the benchmark driver.

The subsequent `/v1/responses` call is streamed. The driver measures client
time to the first non-empty delta and total latency, then reads cached tokens
and optional server timing metrics from the completed response. Generated
content is not saved; the raw result contains a SHA-256 digest of the response
output for later evaluator wiring.

## Required vLLM configuration

Use a dedicated, trusted benchmark server. The runner requires the render and
development cache-reset endpoints, cached-token details, and per-request
metrics:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve MODEL \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enable-per-request-metrics \
  --enable-scale-out \
  --api-key "$VLLM_API_KEY"
```

Do not expose this server publicly. vLLM's development mode enables endpoints
that are unsafe for an untrusted production network.

The included Modal deployment keeps exactly one authenticated replica during
an active validation window because Modal Servers do not cold-start from a
request at zero replicas. Stop the deployed app as soon as the run finishes:

```bash
uv run modal app stop runeconomy-vllm
```

The checked-in manifests use `https://vllm.example.invalid` deliberately so
they cannot contact a private deployment. Copy a manifest and replace
`backend.base_url` with your authenticated vLLM endpoint before a live run.
Credentials are read only from the environment variable named by
`backend.api_key_env`; the examples use `VLLM_API_KEY`.

## Manifest fields

- `backend` pins the model, tokenizer, chat-template digest, vLLM version,
  hardware, cache block size, engine configuration, endpoint, and API-key
  environment variable. Prefix caching must be explicitly enabled in the
  recorded engine configuration.
- `sampling` is merged into every request, preventing either variant from
  silently changing decoding parameters.
- `execution.repetitions` controls measured pairs; warmups are retained in raw
  results but excluded from the comparison.
- `execution.order` is `alternating` or seeded `randomized`.
- `execution.cases_form_workflow` treats every case as a sequential
  critical-path inference step. Workflow token and latency metrics are summed
  per repetition, while workflow success requires every case to pass.
- `cache_treatment: warm` resets once, then permits the workload to populate
  the cache. `cold` resets before every request.
- `arrival_pattern` is pinned to `sequential` in version `0.1`.
- Every case contains exactly a `baseline` and `candidate` request.
- Each case may define a deterministic evaluator. Set
  `execution.require_evaluation` to reject unevaluated cases.
- `acceptance` defines the primary latency metric, task-success non-inferiority
  margin, minimum completion rate, minimum cached-token gain, and whether a
  statistically clear latency improvement is required.
- `acceptance.optimization_target` is `cache_reuse` by default. Set it to
  `input_tokens` with `min_input_token_reduction` when verifying interventions
  such as task-specific tool exposure or context pruning. The paired
  input-token interval must clear that threshold; cache gain is then reported
  but is not a required gate.
- `text_fixtures` loads large UTF-8 files relative to the manifest and verifies
  their SHA-256 digest before use. Their names become request placeholders such
  as `{{INCIDENT_RUNBOOK}}`.
- `json_fixtures` loads and verifies structured JSON. A complete fixture can be
  inserted without string conversion using `{"$fixture": "INCIDENT_TOOLS"}`.
- A variant may define a seeded `request_transform`. `shuffle_tools` models an
  application that emits an unstable tool-list order; `canonicalize_tools`
  sorts the same catalog by function name and canonical JSON. Both operate on
  a copy and preserve tool definitions.

Any string in a request may contain the reserved `{{AID_TRIAL_ID}}`
placeholder. The runner replaces it with the same deterministic value for both
members of a pair and a different value for every case/repetition. This models
timestamps, correlation IDs, and other per-run data without making the two
variants semantically different. A timestamp-position experiment must use this
placeholder; a literal timestamp would become reusable after its first replay
and invalidate the intended test.

Stateful Responses fields such as `previous_response_id` and `background` are
rejected because they make replay dependent on server-side state. `stream` is
also rejected because the runner owns streaming.

See
[`examples/experiments/timestamp-placement.json`](../examples/experiments/timestamp-placement.json)
for a complete manifest.

## Commands

```bash
PYTHONPATH=src python3 -m runeconomy negative-split validate \
  examples/experiments/timestamp-placement.json

PYTHONPATH=src python3 -m runeconomy negative-split run \
  examples/experiments/timestamp-placement.json \
  --output reports/timestamp-placement.json \
  --require-verified
```

For adapter smoke-testing, one request can be captured directly:

```bash
PYTHONPATH=src python3 -m runeconomy capture-vllm request.json \
  --base-url http://127.0.0.1:8000 \
  --api-key-env VLLM_API_KEY
```

The experiment result includes every rendered token sequence and raw metric,
per-variant mean/median/p95 summaries, paired candidate-minus-baseline effects,
deterministic bootstrap 95% confidence intervals, and response-status outcome
rates. A negative latency effect favors the candidate; a positive cached-token
effect favors it. Exact output-digest agreement is reported separately and is
explicitly not treated as semantic correctness.

Supported evaluator kinds are `exact`, `regex`, `json_schema`, `python`,
`command`, and `all`. Python evaluators use `module:function`; command
evaluators receive generated text on standard input and never invoke a shell.
Both are local-code execution features and should only be used with trusted
manifests. The CLI rejects them by default; after reviewing a trusted manifest,
pass `--allow-exec-evaluators` to `validate` and `run`.

An evaluator infrastructure failure (for example, a missing executable, import
failure, timeout, or exception inside evaluator code) is not treated as a task
failure. The affected trial records an `evaluation_error`, its task success is
unknown, and the experiment cannot receive a verified decision.

The CLI writes each completed trial to a durable JSONL checkpoint before
starting the next request. By default this is
`RESULT.json.trials.jsonl`; `--checkpoint` selects another path. Resume an
interrupted deterministic schedule with `--resume`. Cold-cache trials already
in the checkpoint are skipped. For a warm-cache experiment, completed requests
are replayed after a cache reset to reconstruct cache state; their saved
measurements are retained and the replayed token IDs must match before the run
continues.

Checkpoint files contain prompt token IDs and per-trial measurements. They are
ignored by the repository's default `*.trials.jsonl` rule and should be treated
as sensitive local experiment data.

Every result includes a decision of `verified`, `inconclusive`, `regression`,
or `invalid`. A verified decision requires all configured quality and
performance guardrails. Static full-block cache predictions are compared with
observed cached-token deltas in the same report.

When `cases_form_workflow` is enabled, the decision uses aggregate workflow
metrics. The current aggregation assumes sequential inference steps and
excludes tool runtime. It is a fixture-backed replay claim, not a measurement
of live tool execution.

Render a shareable Markdown report from a raw result:

```bash
PYTHONPATH=src python3 -m runeconomy negative-split report result.json \
  --output experiment-report.md
```

## Current claim boundary

The runner verifies inference behavior and applies a workload-specific task
evaluator. Output digests remain a byte-equivalence diagnostic, not a semantic
quality metric. A `verified` decision therefore means the configured evaluator
and performance gates passed for this workload; it is not evidence that the
candidate preserves quality outside the represented cases.
