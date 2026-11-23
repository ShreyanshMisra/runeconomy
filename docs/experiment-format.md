# Experiment manifest and vLLM runner

RunEconomy uses one experiment workflow for static requests and complete agent
processes. A workspace created by `experiment init` records the mode and points
to the mode-specific manifest. This document first describes the workspace and
then the static request manifest. See
[complete agent processes](closed-loop-comparison.md) for the agent contract.

## Experiment workspace

A workspace contains `workspace.json`, `manifest.json`, `intervention.md`, and a
short `README.md`. It creates only files meaningful to the selected mode. Source
evidence stays outside the directory and is referenced by name and SHA-256
digest.

`workspace.json` records the selected finding, intervention, mode, gates,
metrics, identity, cache treatment, order, cost ceiling, unresolved fields, and
next command. Missing values remain unresolved. They are not filled with
synthetic data.

```bash
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/
runeconomy experiment plan experiment/
```

Validation reports `runnable`, `unresolved`, `invalid`,
`unsupported_backend`, or `claim_evidence_incomplete`. Every issue identifies
the artifact and field, why it matters, whether it blocks execution or weakens
the claim, and the next safe action. A saved `doctor --json` result can be
applied with `--doctor-report` without changing server state.

The `0.1`, `0.2`, and `0.3` static manifest formats remain readable. A legacy
JSON manifest can still be passed directly to `validate` and `run`.

## Optional planning declarations

Static and agent-process manifests may add a versioned `design.planning`
object. It is optional and separate from acceptance gates:

```json
{
  "design": {
    "planning": {
      "schema_version": "0.1",
      "hypothesis": "A stable prefix lowers successful-task workflow latency.",
      "intervention": "Move changing metadata after stable instructions.",
      "predicted_direction": "decrease",
      "claim": {
        "level": "workflow",
        "primary_metric": "workflow_latency_ms"
      },
      "calibration": {"artifact_sha256": "sha256:..."},
      "heldout": {
        "artifact_sha256": "sha256:...",
        "operator_declared_unseen": true
      },
      "evaluator_frozen": true,
      "gates_frozen": true,
      "stopping_rule": {
        "kind": "fixed_sample",
        "favorable_result_dependent": false
      },
      "claim_boundary": "One declared local workflow and environment.",
      "intervention_variables": ["prompt_order"],
      "latency_assumptions": {
        "expected_difference_ms": -50,
        "paired_standard_deviation_ms": 100
      }
    }
  }
}
```

Free-form hypothesis and claim-boundary text is never copied into planning
output. The planner reports only a digest, presence, safe categories, counts,
and stable issue codes. Static-manifest fingerprints already include `design`.
For agent-process mode, the declaration remains in the versioned workspace and
its digest is recorded in the planning result. It is not copied into the
execution result.

Latency assumptions are declared calibration assumptions, not observations or
acceptance gates. When both values are present, planning shows a two-sided 95%
normal approximation at 80% target power. The executed comparison still uses
its paired Student-t interval. Without those assumptions, latency power remains
unknown.

RunEconomy can record `operator_declared_unseen`, but cannot independently know
whether a workload was inspected or consumed. Matching calibration and
held-out digests are treated as overlap.

The `0.1`, `0.2`, and `0.3` experiment formats define a paired
baseline/candidate replay against one vLLM backend. Versions `0.1` and `0.2`
remain readable for legacy manifests. New experiments should use `0.3`, which
defaults the absolute task-success floor to 80%. The runner is
intentionally narrow today: requests use either vLLM's Responses or Chat
Completions API, run sequentially, and contain their complete history inline.

## Why the runner renders first

Before every generation, the adapter sends the same request to the matching
Responses or Chat Completions render endpoint. The returned token IDs are the exact prompt tokens
produced by the server's tokenizer and chat template. This avoids a misleading
comparison caused by recreating prompt rendering in the benchmark driver.

For Chat Completions, a server without the optional render endpoint can use the
standard chat-form `/tokenize` endpoint. RunEconomy sends only
tokenization-relevant fields and does not reconstruct the server's chat
template locally. Responses requests still require their render endpoint.

The subsequent generation request uses the matching Responses or Chat
Completions endpoint and is streamed. The driver measures client time to the
first non-empty delta and total latency, then reads cached tokens and optional
server timing metrics from the completed response. Generated content is not
saved. The raw result contains a SHA-256 digest of the response output for
later evaluator wiring.

## Required vLLM configuration

Run `runeconomy doctor --base-url URL` before preparing a manifest. A normal
server can run warm-cache or salt-isolated experiments. A dedicated trusted
benchmark server can additionally provide physical resets, cached-token
details, and per-request metrics:

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

The included Modal deployment normally keeps zero warm containers. An
authenticated request can trigger a cold start. During a bounded validation
window, the operator temporarily pins one container and restores zero in a
`finally` block. Follow the [Modal validation-window runbook](modal-validation.md),
do not permanently stop the App as the normal closing operation.

The checked-in manifests use `https://vllm.example.invalid` deliberately so
they cannot contact a private deployment. Copy a manifest and replace
`backend.base_url` with your authenticated vLLM endpoint before a live run.
Credentials are read only from the environment variable named by
`backend.api_key_env`. The examples use `VLLM_API_KEY`.

## Manifest fields

- `backend` pins the model, tokenizer, chat-template digest, vLLM version,
  hardware, cache block size, engine configuration, endpoint, and API-key
  environment variable. Prefix caching must be explicitly enabled in the
  recorded engine configuration.
- `backend.api_dialect` is `responses` by default or `chat_completions` for a
  native Chat Completions request. Request and sampling field names must match
  the selected dialect. RunEconomy does not silently translate between APIs.
- `sampling` is merged into every request, preventing either variant from
  silently changing decoding parameters.
- `execution.repetitions` controls measured pairs. Warmups are retained in raw
  results but excluded from the comparison.
- `execution.order` is `alternating` or seeded `randomized`.
- `execution.cases_form_workflow` treats every case as a sequential
  critical-path inference step. Workflow token and latency metrics are summed
  per repetition, while workflow success requires every case to pass.
- Cases may declare `workflow_id` when `cases_form_workflow` is enabled. When
  present, every case must declare it. Cases sharing an ID form one sequential
  workflow unit, and uncertainty is calculated across the resulting workflow
  units rather than across individual steps. Manifests without `workflow_id`
  retain the legacy behavior of treating all cases as one workflow.
- `cache_treatment: warm` uses `cache_isolation: shared`, resets once, then
  permits the workload to populate the cache.
- `cache_treatment: cold` requires either `cache_isolation: physical_reset`
  (the legacy default, using the private development endpoint before every
  request) or `cache_isolation: salt_isolated`. Salt isolation generates a
  unique, unrecorded random `cache_salt` per trial. It prevents cross-trial
  prefix reuse but does not physically clear cached blocks, recover capacity,
  or remove eviction effects.
- `arrival_pattern` currently supports only `sequential`.
- Every case contains exactly a `baseline` and `candidate` request.
- Each case may define a deterministic evaluator. Set
  `execution.require_evaluation` to reject unevaluated cases.
- `acceptance` defines the primary latency metric, task-success non-inferiority
  margin, minimum completion rate, minimum cached-token gain, and whether a
  statistically clear latency improvement is required.
- `acceptance.task_success_method` defaults to `paired_exact`. It uses exact
  binomial bounds on the two discordant paired-outcome probabilities with
  Bonferroni coverage. `point_estimate` remains available only for reproducing
  legacy decisions and should not support a public verification claim.
- `acceptance.min_task_success_rate` defaults to `0` in legacy `0.1`/`0.2`
  manifests and `0.8` in `0.3`. New verification experiments should set a
  workload-appropriate absolute success floor explicitly. Both variants'
  exact 95% Clopper-Pearson lower bounds must
  clear it. A weak baseline makes the experiment invalid, while a candidate
  below the floor is a regression. Non-inferiority alone is insufficient when
  both variants perform poorly.
- `acceptance.require_environment_identity` invalidates a run unless all pinned
  identity fields are observed before and after the measurements, match the
  manifest, and remain unchanged. A normal vLLM server does not expose every
  field needed to prove model revision, tokenizer/template identity, and engine
  configuration, so strict identity mode
  requires a client or deployment that supplies those observations.
- `acceptance.optimization_target` is `cache_reuse` by default. Set it to
  `input_tokens` with `min_input_token_reduction` when verifying interventions
  such as task-specific tool exposure or context pruning. The paired
  input-token interval must clear that threshold. Cache gain is then reported
  but is not a required gate.
- `text_fixtures` loads large UTF-8 files relative to the manifest and verifies
  their SHA-256 digest before use. Their names become request placeholders such
  as `{{INCIDENT_RUNBOOK}}`.
- `json_fixtures` loads and verifies structured JSON. A complete fixture can be
  inserted without string conversion using `{"$fixture": "INCIDENT_TOOLS"}`.
- A variant may define a seeded `request_transform`. `shuffle_tools` models an
  application that emits an unstable tool-list order. `canonicalize_tools`
  sorts the same catalog by function name and canonical JSON. Both operate on
  a copy and preserve tool definitions.

Any string in a request may contain the reserved `{{RUNECONOMY_TRIAL_ID}}`
placeholder. The runner replaces it with the same deterministic value for both
members of a pair and a different value for every case/repetition. This models
timestamps, correlation IDs, and other per-run data without making the two
variants semantically different. A timestamp-position experiment must use this
placeholder. A literal timestamp would become reusable after its first replay
and invalidate the intended test.

Stateful Responses fields such as `previous_response_id` and `background` are
rejected because they make replay dependent on server-side state. `stream` is
also rejected because the runner owns streaming.

See
[`examples/experiments/timestamp-placement.json`](../examples/experiments/timestamp-placement.json)
for a complete manifest.

## Static request commands

```bash
runeconomy experiment validate examples/experiments/timestamp-placement.json

runeconomy experiment run examples/experiments/timestamp-placement.json \
  --output reports/timestamp-placement.json --require-verified

runeconomy experiment report reports/timestamp-placement.json \
  --output reports/timestamp-placement.html --open
```

For adapter smoke-testing, one request can be captured directly:

```bash
runeconomy capture-vllm request.json \
  --base-url http://127.0.0.1:8000 \
  --api-key-env VLLM_API_KEY
```

Add `--api-dialect chat_completions` for a native Chat Completions request.

The experiment result includes every rendered token sequence and raw metric,
per-variant mean/median/p95 summaries, paired candidate-minus-baseline effects,
deterministic bootstrap 95% confidence intervals for continuous measurements,
and response-status outcome rates. Task-success rates use exact
Clopper-Pearson intervals. Paired task-success non-inferiority uses conservative
exact marginal binomial bounds rather than a bootstrap, which would collapse to
`[0, 0]` when every observed pair agrees. Fifteen of fifteen successes therefore
does not imply a true 100% success rate or prove a zero-regression margin.

Client TTFT, server TTFT, and their difference remain separate. End-to-end
workflow latency is the product outcome when selected as `primary_metric`.
Server TTFT is a serving-mechanism metric, and the client/server gap exposes
network, proxy, and streaming-delivery overhead.

Supported evaluator kinds are `exact`, `regex`, `json_schema`, `python`,
`command`, and `all`. Python evaluators use `module:function`. Command
evaluators receive generated text on standard input and never invoke a shell.
Both are local-code execution features and should only be used with trusted
manifests. The CLI rejects them by default. After reviewing a trusted manifest,
pass `--allow-exec-evaluators` to `validate` and `run`.

An evaluator infrastructure failure (for example, a missing executable, import
failure, timeout, or exception inside evaluator code) is not treated as a task
failure. The affected trial records an `evaluation_error`, its task success is
unknown, and the experiment cannot receive a verified decision.

The CLI writes each completed trial to a durable JSONL checkpoint before
starting the next request. By default this is
`RESULT.json.trials.jsonl`. `--checkpoint` selects another path. Resume an
interrupted deterministic schedule with `--resume`. Cold-cache trials already
in the checkpoint are skipped. For a warm-cache experiment, completed requests
are replayed after a cache reset to reconstruct cache state. Their saved
measurements are retained and the replayed token IDs must match before the run
continues.

Checkpoint files contain prompt token IDs and per-trial measurements. They are
ignored by the repository's default `*.trials.jsonl` rule and should be treated
as sensitive local experiment data.

Every result includes a decision of `verified`, `inconclusive`, `regression`,
or `invalid`. A verified decision requires all configured quality and
performance guardrails. Static full-block cache predictions are compared with
observed cached-token deltas in the same report.

Results contain both a full manifest fingerprint and a reproducibility
fingerprint. The latter excludes only `backend.base_url` and
`backend.api_key_env`, allowing a private measured manifest to be compared with
a sanitized public copy without publishing endpoint or credential locations.
It does not exclude model, prompt, fixture content, sampling, evaluator,
hardware, engine configuration, or acceptance settings.

The optional `design` object is included in both fingerprints and copied into
result provenance. Use it for predeclared hypotheses, calibration-split
evidence, predicted direction, and claim boundaries. It is not interpreted as
an acceptance gate.

When `cases_form_workflow` is enabled, the decision uses aggregate workflow
metrics. The current aggregation assumes sequential inference steps and
excludes tool runtime. It is a fixture-backed replay claim, not a measurement
of live tool execution. Calling these cases a workflow does not make this a
closed-loop agent replay: candidate step 2 cannot consume candidate step 1's
newly generated output unless that state was already materialized in the
manifest.

Render a shareable Markdown report from a raw result:

```bash
runeconomy experiment report result.json \
  --output experiment-report.md
```

The same report command renders static and agent-process results.

## Advanced initialization from an analysis report

```bash
runeconomy experiment init \
  --from-finding analysis.json \
  --finding-id finding-0123456789abcdef \
  --output experiment/
```

`--from-finding` remains available for advanced and existing workflows. The
preferred public path is `--from-evidence`, which performs analysis and carries
the capability assessment into the workspace. The source can be one structured
finding or a complete analysis report. When a
report contains multiple findings, `--finding-id` is required. Safe templates
produce a runnable manifest only when the finding includes complete baseline
request evidence, candidate transformation inputs, evaluator, backend,
sampling, execution design, acceptance thresholds, and confirmed
capabilities. The generated manifest passes through the ordinary manifest
validator and must use the paired exact task-success method before it is
labeled runnable.

Otherwise the workspace records `generation_status: unresolved`, an explicit
`unresolved_fields` list, and a non-runnable manifest draft. A `.json` output
path retains the legacy monolithic skeleton. This is intentional: token IDs and
tool names are not enough evidence to reconstruct requests, tool definitions, a
representative dataset, or an evaluator.

Generated manifests store the source finding, expected causal mechanism, and
predicted metric direction under `design.hypothesis`. The final report compares
that prediction with the paired 95% interval. A prediction is marked supported
only when the interval lies wholly in the predeclared direction. Otherwise it
is reported as not established or contradicted.

## Current claim boundary

The runner verifies inference behavior and applies a workload-specific task
evaluator. Output digests remain a byte-equivalence diagnostic, not a semantic
quality metric. A `verified` decision therefore means the configured evaluator
and performance gates passed for this workload. It is not evidence that the
candidate preserves quality outside the represented cases.
