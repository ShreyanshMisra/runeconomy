# Hosted API versus self-hosted vLLM

This example compares the same agent on two serving arms:

- **baseline**: a hosted OpenAI-compatible API (OpenAI, Fireworks, Together, a
  gateway, or anything else that serves `POST /v1/chat/completions`)
- **candidate**: an open-weight model you serve yourself with vLLM

It answers the migration question: *if we move this agent off the hosted API,
does it still complete its tasks, and is it faster or no slower?* It uses the
existing agent-process mode. Nothing here is a separate code path.

The agent is a small order-support workflow: one inference call plans a lookup,
a deterministic `lookup_order` tool reads `orders.json`, and a second inference
call answers from the tool result. Both arms receive identical messages,
generation parameters (`temperature: 0`, fixed `max_tokens`) and tool evidence.
Only the endpoint differs. The agent uses the Python standard library and
RunEconomy, with no OpenTelemetry SDK or provider SDK.

The six bundled tasks are a **smoke workload**. They show that the contract
works end to end. They are far too few for a decision, and
`runeconomy experiment plan` blocks them for that reason (see
[Toward a real decision](#toward-a-real-decision)).

## What each arm can observe

The hosted arm records only what a hosted API exposes. Values it can't observe
are left out of the trace, so RunEconomy reports them as unknown (`null`), not
zero and not an estimate.

| Field | Hosted API | Self-hosted vLLM |
| --- | --- | --- |
| End-to-end workflow wall time (primary metric) | measured | measured |
| Client-observed TTFT | measured | measured |
| Input, output and cached-token counts | provider-reported, when the provider returns them | server-reported |
| Exact prompt token IDs | **unknown**: no tokenizer endpoint | exact, via vLLM render or `/tokenize` |
| Server TTFT and queue time | **unknown** | server-reported, when vLLM returns them |
| Prefill and decode time | **unknown** | unknown in this example |
| Cost, GPU time | unknown | unknown |

The hosted arm never calls `/tokenize`, render or metrics endpoints, and never
sends vLLM-only parameters such as `cache_salt`. The test suite checks this.

Cost is deliberately absent. Hosted per-token prices and self-hosted GPU-hour
costs are not comparable until a versioned cost model exists, and RunEconomy
doesn't estimate them.

## Identity on each arm

Each arm records a graded identity (identity policy `0.1`, see
[serving backends and identity grades](../../docs/backends.md#identity-grades)).
The workspace preregisters `minimum_grade: provider_declared`, so a hosted arm
can support a valid verdict without pretending to know what the provider
doesn't publish.

**Self-hosted vLLM: `operator_verified`.** You declare the pinned identity of
what you serve: model, model revision, backend version, tokenizer, tokenizer
revision and chat-template digest. If any value is missing, the agent omits
it, identity is incomplete, and the verdict is `invalid`. Nothing is guessed.

**Hosted API: `provider_declared`.** A hosted provider reports a model, not its
tokenizer, chat template or serving stack. The agent records what the
provider declares and lists the rest as undisclosed, with no placeholder
strings:

| Identity field | Hosted value |
| --- | --- |
| `model` | the `model` the provider returns in each response (a dated snapshot), else the requested model |
| `backend` | `HOSTED_PROVIDER` (default `openai_compatible`) |
| `model_revision` | `HOSTED_MODEL_REVISION`, default the returned model snapshot |
| `backend_version` | `HOSTED_BACKEND_VERSION` if you know it, else `null` and undisclosed |
| `tokenizer`, `tokenizer_revision`, `chat_template_digest` | `null` and undisclosed |

The spans carry `runeconomy.identity_grade` and `runeconomy.identity_undisclosed`.
Under the policy, a placeholder value such as `"provider-managed"` would count
as missing, not as an identity. Some consequences:

- If the provider returns a different model snapshot partway through a run,
  the hosted variant's identity is unstable and the verdict is `invalid`.
  Pin a dated snapshot (for example `gpt-4.1-mini-2025-04-14`, not
  `gpt-4.1-mini`).
- A provider can change its serving stack, tokenizer or template under the
  same snapshot name without notice. RunEconomy can't detect that. As a drift
  indicator, the provider's `system_fingerprint` is digested and summarized
  per arm as a count of distinct values. It isn't part of identity and never
  invalidates the run. If it changes, the decision records an
  `identity.provider_fingerprint_changed` claim limit.
- Because one arm is provider-declared, the decision's `identity_assurance`
  has claim scope `observed_service_window`. The verdict describes the hosted
  service as it behaved during the run. It is not a statement about the
  provider's model in general, and it can't be reproduced against a pinned
  configuration.
- Without the identity policy (a legacy contract), undisclosed fields are
  simply missing and the verdict is `invalid`. With `--min-identity-grade
  operator-verified` the hosted arm can never qualify.

## Run it

Set the endpoints and identities. Both base URLs use the OpenAI SDK
convention and end in `/v1`.

```bash
# Baseline: hosted OpenAI-compatible API
export HOSTED_BASE_URL="https://api.openai.com/v1"
export HOSTED_API_KEY="..."                      # read from the environment only
export HOSTED_MODEL="gpt-4.1-mini-2025-04-14"    # pin a dated snapshot
export HOSTED_PROVIDER="openai"

# Candidate: your vLLM server
export SELF_HOSTED_BASE_URL="http://127.0.0.1:8000/v1"
export SELF_HOSTED_API_KEY=""                    # optional
export SELF_HOSTED_MODEL="Qwen/Qwen2.5-7B-Instruct"
export SELF_HOSTED_MODEL_REVISION="<commit sha of the served checkpoint>"
export SELF_HOSTED_BACKEND_VERSION="<vllm version>"
export SELF_HOSTED_CHAT_TEMPLATE_DIGEST="sha256:<digest of the chat template>"
# Optional. They default to the model name and model revision.
# export SELF_HOSTED_TOKENIZER=...
# export SELF_HOSTED_TOKENIZER_REVISION=...
```

Check the vLLM side first. `runeconomy doctor --base-url http://127.0.0.1:8000`
reports the backend version and whether render or `/tokenize` is available.
`doctor` is vLLM-only today, so don't point it at the hosted API.

Then run the usual workspace workflow from the repository root:

```bash
runeconomy experiment init --mode agent-process \
  --baseline "python examples/hosted_vs_self_hosted/agent.py --arm hosted" \
  --candidate "python examples/hosted_vs_self_hosted/agent.py --arm self-hosted" \
  --dataset examples/hosted_vs_self_hosted/tasks.json \
  --evaluator examples/hosted_vs_self_hosted/evaluator.py \
  --optimization-target engine-change-latency \
  --latency-semantics noninferiority \
  --max-latency-regression-ms 250 \
  --repetitions 3 \
  --order randomized \
  --seed 2026 \
  --min-identity-grade provider-declared \
  --output experiment/

runeconomy experiment validate experiment/
runeconomy experiment plan experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

Choose the latency question before you run:

- `--latency-semantics superiority`: the self-hosted arm must be faster on
  successful workflows.
- `--latency-semantics noninferiority --max-latency-regression-ms N`: the
  self-hosted arm may be up to `N` ms slower. Use this when the migration is
  motivated by cost, privacy or control rather than speed, and justify `N`
  before running.

`engine-change-latency` requires each arm to have one complete, stable identity
and the two identities to differ. That's always true here, because the
backends differ.

The run makes real paid requests to the hosted API: 6 tasks × 3 repetitions ×
2 calls = 36 hosted requests with short prompts and `max_tokens` of 48 or 24.
Check the provider's pricing before you run it.

Reasoning models that reject `temperature` or `max_tokens` (some providers
require `max_completion_tokens`) need a small change to `_request` in
`agent.py`. Make the same change on both arms.

## Reading the result

The verdict means what it means for every agent-process experiment. Task
success and completion are gated before latency. Latency compares only pairs
in which both arms succeeded. A hosted outage or rate limit is recorded as an
`infrastructure_failure` with code `hosted_request_failed`, which invalidates
the completion gate rather than shrinking the hosted latency sample.

In the result JSON, look at:

- `decision.identity_assurance` shows each arm's grade, the claim scope
  (`observed_service_window` here), the supported claim, the claim limits, the
  undisclosed fields, and the fingerprint summary. The HTML report shows it in
  its own "Identity assurance" section.
- `attempts[].metrics.environment_identity` shows the exact identity values for
  each arm. Undisclosed hosted fields are `null`, and
  `attempts[].metrics.identity_declarations` lists the grade and undisclosed
  fields per request.
- `attempts[].metrics.evidence_quality` shows `server_ttft` as `unresolved` on
  the hosted arm, and `prompt_tokens` as count-only (`reliably_derived`) rather
  than exact.
- `decision.checks` includes `intentional_backend_identity_difference` and
  `trace_and_environment_identity`.

## Claim boundary

A verdict from this example covers only the declared workload, evaluator,
hosted service during the run, self-hosted model, hardware, sequential
execution and gates. In particular:

- **It is a bundled change.** A migration usually swaps the model and the
  serving stack together. The result applies to that pair of arms. It can't
  say whether a difference came from the model or from the infrastructure.
- **It is sequential.** One request at a time, from one client location.
  Hosted latency includes network distance to the provider, and the
  self-hosted arm runs on an otherwise idle server. Neither reflects production
  concurrency.
- **It doesn't cover cost.** There's no cost or goodput gate yet.
- **Hosted identity is provider-declared.** The verdict is limited to the
  observed service window, as recorded in `identity_assurance`.
- **The hosted internals are opaque.** Exact prompt tokens, server timing and
  serving configuration are unknown on the hosted side, so cache and prefill
  mechanisms can only be observed on the self-hosted arm.

## Toward a real decision

The bundled tasks show the integration. They can't support a decision. For a
real migration decision:

1. Replace `tasks.json`, `orders.json`, `evaluator.py` and the workflow in
   `agent.py` with your own agent and a task-level evaluator you trust. Keep
   the `--arm` split, the identity labels and the `Attempt` status handling.
2. Run a calibration workload first. Use it to estimate success, discordance
   and paired latency variability.
3. Declare the intervention in the workspace manifest's `design.planning`
   block. For this target the planner requires `identity_change: true` and
   `intervention_variables: ["backend_identity"]`. RunEconomy's backend
   identity covers the model, tokenizer and template fields as well as the
   serving stack.

   ```json
   {
     "design": {
       "planning": {
         "schema_version": "0.1",
         "hypothesis": "Self-hosted serving keeps task success and is no more than 250 ms slower.",
         "intervention": "Move the agent from the hosted API to the self-hosted vLLM model.",
         "predicted_direction": "no_change",
         "claim": {"level": "workflow", "primary_metric": "workflow_latency_ms"},
         "identity_change": true,
         "intervention_variables": ["backend_identity"],
         "claim_boundary": "One declared workflow, hosted snapshot, self-hosted model and hardware, run sequentially."
       }
     }
   }
   ```

4. Size the held-out sample with `runeconomy experiment plan` (planning schema
   `0.2`, see [experiment format](../../docs/experiment-format.md)) and freeze
   it before the one held-out run.

## Tests

`tests/test_hosted_vs_self_hosted_example.py` runs the full
`init → validate → run → report` workflow against a local fake hosted API (Chat
Completions only, no tokenizer or server metrics) and a local fake vLLM server.
It checks that:

- the hosted arm's server metrics are `null` while the vLLM arm's are present
- the hosted arm calls only `/v1/chat/completions` and gets only its own
  credential
- the two identities differ and the decision is valid, with the hosted arm
  graded `provider_declared`, the vLLM arm `operator_verified`, and claim scope
  `observed_service_window`
- fingerprint drift is counted and flagged as a claim limit but never
  invalidates the run, and raw fingerprints never appear in the result
- without the identity policy (a legacy contract), or with an
  `operator_verified` minimum, the hosted arm makes the verdict `invalid`
- a missing self-hosted identity value makes the verdict `invalid`
- hosted model-snapshot drift within a run makes the verdict `invalid`
- a hosted outage is recorded as `infrastructure_failure`

The fake servers don't run a model, and the test isn't a benchmark.
