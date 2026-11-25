# Serving backends and identity grades

RunEconomy's static runner and `doctor` support three backend kinds:

| Kind | What it is | Identity grade |
| --- | --- | --- |
| `vllm` | A vLLM OpenAI-compatible server you run (default) | `operator_verified` |
| `sglang` | An SGLang server you run | `operator_verified` |
| `openai_compat` | Any hosted OpenAI-compatible Chat Completions API or gateway: OpenAI, Fireworks, Together, Baseten, Modal, llm-d and AIBrix gateways | `provider_declared` |

Agent-process experiments run any agent command, so they cover every backend
already. The agent's trace declares its identity grade (see
[Identity grades in traces](#identity-grades-in-agent-process-traces)). The
[hosted-versus-self-hosted example](../examples/hosted_vs_self_hosted/README.md)
compares a hosted API against self-hosted vLLM that way.

## The adapter protocol

Each kind is a `BackendAdapter` (`runeconomy.adapters`) with five parts:

- **Capability discovery**: a read-only probe. It never generates text and
  never resets a cache.
- **Identity**: what the backend reports about itself, together with the grade
  that says who stands behind it.
- **Request execution**: one streamed Chat Completions (or, for vLLM,
  Responses) request.
- **Cache reset** (optional): vLLM `/reset_prefix_cache` or SGLang
  `POST /flush_cache`. A backend without one declares
  `supports_cache_reset = False`.
- **Metrics** (optional): one Prometheus scrape, mapped from a fixed
  allowlist of vLLM or SGLang metrics with label values dropped.

The vLLM adapter delegates to the original client and discovery code. vLLM
manifests, fingerprints, trial records, results and `doctor` output are
unchanged.

## Capability tables

Every adapter reports the same nine capabilities. Each is `supported`,
`unsupported` or `unknown`, and every entry that isn't supported states what
it costs the experiment. Unknown is never treated as supported.

| Capability | vLLM | SGLang | openai_compat |
| --- | --- | --- | --- |
| Chat Completions API | discovered | discovered | unknown: doctor never generates |
| Responses API | discovered | unsupported by the adapter | unsupported by the adapter |
| Exact prompt tokens | render or `/tokenize` | `/tokenize` | **unsupported**: prompt tokens are count-only |
| Cache telemetry | smoke capture | `--enable-cache-report` | provider-reported usage, when present |
| Server timing | smoke capture | **unsupported**: the adapter reads none | **unsupported** |
| Cache reset | `/reset_prefix_cache` | `POST /flush_cache` | **unsupported** |
| Cache salt | discovered | **unsupported** | **unsupported** |
| Model metadata | `/v1/models` | `/get_model_info` | `{base}/models` |
| Prometheus metrics | optional | `--enable-metrics` | **unsupported** |

## Diagnose a backend

```bash
runeconomy doctor --base-url http://127.0.0.1:8000                      # vLLM (default)
runeconomy doctor --backend sglang --base-url http://127.0.0.1:30000
runeconomy doctor --backend openai_compat \
  --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
```

For `openai_compat`, pass the API base exactly as the provider documents it
for the OpenAI SDK. It usually ends in `/v1`. Doctor reads only
`{base}/models`. SGLang and vLLM take the server root. Use the final URL
for authenticated endpoints. Bearer credentials are sent on the initial
request only.

vLLM keeps doctor report schema `0.2`. SGLang and `openai_compat` reports use
schema `0.3`. It keeps every `0.2` capability key, so `experiment validate
--doctor-report` and `support-bundle --doctor-report` read it the same way,
and adds `backend_kind`, `capability_table` and graded `identity`. Human
output lists every downgraded capability and what it costs.

## Static manifests for SGLang and openai_compat

Both kinds need manifest `schema_version` `0.3` or later and the
`chat_completions` dialect. The manifest is checked against the kind's limits
before anything runs.

**SGLang** pins its identity like vLLM. `engine_config` must pin
`"disable_radix_cache": false`, and `block_size` should match `--page-size`
(the default is 1). Supported cache isolation is `shared` (warm) or
`physical_reset` (cold, via `/flush_cache`). `salt_isolated` and
`reset_external` are unavailable.

**openai_compat** must declare an identity policy and the fields the provider
doesn't disclose:

```json
{
  "schema_version": "0.3",
  "backend": {
    "kind": "openai_compat",
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "model": "gpt-4.1-mini-2025-04-14",
    "model_revision": "gpt-4.1-mini-2025-04-14",
    "api_dialect": "chat_completions",
    "block_size": 1,
    "undisclosed_identity": [
      "backend_version",
      "chat_template_digest",
      "engine_config",
      "hardware",
      "tokenizer_revision"
    ]
  },
  "execution": {"cache_treatment": "warm", "repetitions": 10},
  "acceptance": {
    "identity_policy": {"schema_version": "0.1", "minimum_grade": "provider_declared"}
  }
}
```

Undisclosed fields must be `null` or absent. `model` and `provider` are always
required. Only warm, shared execution is allowed. The run doesn't reset the
cache, so it starts from an unknown provider cache state, and the result says
so. `--resume` is refused, because replayed prompts can't be checked without
exact tokens. Pin a dated snapshot: if the model the provider says it served
ever differs from `backend.model`, the `serving_identity_grade` check fails
and the verdict is `invalid`.

Results from SGLang and `openai_compat` add a `serving_capabilities` record
(the backend kind, its identity grade and every downgrade). Reports show it
in a separate "Serving capabilities" section. Prompt token IDs and predicted
prefix reuse are `null` on `openai_compat` trials, not zero.

## Identity grades

Legacy contracts require every identity field to hold a value. They reject
only empty strings and `"unknown"`. They keep exactly that behavior.

Identity policy `0.1` is an opt-in, preregistered acceptance field that
grades identity instead:

```json
"identity_policy": {"schema_version": "0.1", "minimum_grade": "provider_declared"}
```

- **`operator_verified`**: the operator controls the serving artifact and
  pins every field. A verdict can be attributed to that pinned configuration.
- **`provider_declared`**: a provider the operator doesn't control reports
  some or all of the fields. Fields it doesn't publish are `null` and
  explicitly listed as undisclosed. A verdict describes the provider service
  as observed during the run.

Under a policy:

- A placeholder string (`"undisclosed"`, `"provider-managed"`, `"unknown"`,
  `"n/a"` and similar) is never an identity value. It counts as missing.
- A missing field that isn't declared undisclosed is incomplete identity, and
  the verdict is `invalid`.
- Only `provider_declared` identity may declare fields undisclosed. `backend`
  and `model` can never be undisclosed.
- A variant's grade is the weakest grade among its requests. It must meet
  `minimum_grade`, or identity is invalid.

The policy needs agent-process manifest schema `0.3` (`experiment init
--min-identity-grade provider-declared`) or static manifest schema `0.3`. It
enters the frozen design and fingerprint only when it's declared.

### What the decision records

A contract with a policy gets an `identity_assurance` block in its decision:

| Field | Meaning |
| --- | --- |
| `grades` | Grade per variant |
| `claim_scope` | `pinned_configuration` when every variant is operator-verified, else `observed_service_window` |
| `supported_claim` | The claim the verdict supports, in words |
| `claim_limits` | `identity.provider_declared`, `identity.undisclosed_fields` and `identity.provider_fingerprint_changed` entries |
| `variants` | Per-variant validity, missing, placeholder and undisclosed fields, and the fingerprint summary |

A `verified` verdict with scope `observed_service_window` holds only for the
declared workload and the provider service as it behaved during the run. It
isn't a claim about a pinned model or serving configuration, and the report's
recommendation says to re-verify before relying on it later. Reports show
identity assurance in its own section, separate from the acceptance gates.

### Provider fingerprints

A provider's `system_fingerprint` is a drift indicator only. RunEconomy
digests each value and reports per variant how many requests reported one and
how many distinct values appeared. It never stores the values. A change
during the run adds the `identity.provider_fingerprint_changed` claim limit.
It is never part of identity and never invalidates a run.

## Identity grades in agent-process traces

An agent declares grades on its OpenInference LLM spans:

| Attribute | Value |
| --- | --- |
| `runeconomy.identity_grade` | `operator_verified` or `provider_declared` |
| `runeconomy.identity_undisclosed` | JSON array of undisclosed fields, for example `["tokenizer", "chat_template_digest"]` |
| `runeconomy.provider_system_fingerprint` | The provider's `system_fingerprint`, digested on import |

A span without a grade is `operator_verified`, which is what trace identity
always meant. Traces without these attributes import exactly as before.

## Boundaries

- Sequential static replay on `sglang` and `openai_compat` is tested end to
  end against local fake servers. Concurrent arrival patterns use the same
  shared load executor as vLLM, but no end-to-end test covers them on these
  kinds yet. An `openai_compat` load run measures the provider under your
  load, not the provider's own concurrency.
- The SGLang adapter reads no per-request server timing, and doctor never
  invokes `/flush_cache`.
- `openai_compat` covers Chat Completions only. Responses-only providers and
  reasoning models that reject `max_tokens` need request fields set
  accordingly in the manifest.
- A manifest can declare one cost model for both variants and gate on cost
  per 1,000 successful workflows. The hosted-versus-self-hosted example needs
  different pricing models for its two arms and leaves cost unknown. See
  [cost models and server telemetry](load-pairing-and-tail-gates.md#cost-models-and-server-telemetry).
