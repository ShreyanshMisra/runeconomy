# RunEconomy 0.3.0 release notes (alpha)

RunEconomy `0.3.0` makes capture zero-code, runs experiments under concurrent
load, gates on cost and tail latency, and adds SGLang and hosted
OpenAI-compatible backends. Coding agents and CI can now drive the whole
workflow. RunEconomy stores capture and experiment artifacts locally. Live runs
send requests to the configured serving backend. An LLM judge also sends its
evaluation input to the configured judge endpoint.

## Zero-code capture

`runeconomy proxy` sits between an OpenAI-compatible client and its server.
It streams responses through unchanged and records timing, usage, cached
tokens and call order. It groups calls into workflows by correlation header
and can resolve exact token IDs against vLLM. Stop the proxy, check the session
gap counters, then run `runeconomy import session.jsonl --output evidence.json`.
Static replay also needs your serving identity, evaluator and replay context.
Full capture can contain prompts and responses. Use `--redact MODULE:FUNCTION`
for a local body-redaction hook.
Overlapping requests without correlation headers are flagged and never merged
silently. See [the capture guide](proxy.md).

## Concurrent load, tail gates and cost

Experiments can run closed-loop virtual agents, constant-rate arrivals or
Poisson arrivals at a declared concurrency, with variants balanced across time
and devices. Acceptance gates can now cover p95/p99 TTFT, TPOT and workflow
latency, goodput, and cost per 1,000 successful workflows through a versioned
`cost_model`. Server metrics (KV-cache usage, running and waiting requests,
preemptions, prefix-cache hits) are sampled from the server's `/metrics`
endpoint during each run. See [load, pairing and tail gates](load-pairing-and-tail-gates.md).

## Powered planning and new paired methods

`paired_score_exact` is an exact paired non-inferiority method based on
Tango's score statistic. It needs far fewer pairs than `paired_exact` for the
same design. `paired_exact` remains the default, and contracts that name no
method behave as before. `experiment plan` reports power, the recommended
number of pairs, the exact completion tolerance, a cost estimate and a filled
preregistration draft. An opt-in `completion_policy` can count a failure that
both arms share on the same task as a task outcome rather than non-completion,
when the contract declares it in advance.

## Backends and identity grades

The static runner and `doctor` support `vllm`, `sglang` and `openai_compat`.
Hosted providers get a `provider_declared` identity grade that is recorded in
decision artifacts and limits the claims a verdict can support. See
[serving backends](backends.md) and the
[hosted-versus-self-hosted example](../examples/hosted_vs_self_hosted/README.md).

## Agents and CI

- `runeconomy mcp` serves typed MCP tools. Execution runs only after an
  approved estimate and a single-use `approval_token`. See [MCP server](mcp.md).
- `experiment patch` records validated changes in `provenance.jsonl`.
  `experiment freeze` pins the reviewed inputs. Use `experiment run --require-frozen`
  for held-out execution. [Patch and freeze](json-contracts-patch-freeze.md).
- Result-producing commands support versioned `--json` output. List the
  available contracts with `runeconomy schema`.
- A GitHub Action posts a verdict table on pull requests. See
  [GitHub Action](github-action.md).
- `runeconomy.track_workflow` and `runeconomy.step` record explicit
  `depends_on` edges in an OpenInference trace.

## Diagnosis and evaluators

Five new detectors cover tool-output carry-over, critical-path serialization,
step-level over-provisioning, output-token inflation and KV pressure. Each
maps to an experiment template. See [detectors](detectors.md). New evaluator
kinds run repository tests, tool trajectories, calibrated LLM judges, and
Promptfoo assertions, Braintrust scorers and Inspect AI scorers unchanged.
See [evaluators](evaluators.md).

## Public results

- **Proof 3 · `verified`.** Online FP8 reduced mean workflow latency by 499 ms
  (27%) against BF16 at 16 concurrent agents per variant. Both arms passed
  200/200 tasks.
- **Proof 4 · `regression`.** Removing search halved median workflow latency,
  but task success fell from 100% to 54% against an 80% floor.

Both designs were frozen and executed once. [Proofs 3 and 4](proofs-and-studies.md) ·
[Proof 1 quality gates](public-proof.md) · [Proof 2 completion gates](proof-2-case-study.md).

## Fixes

- vLLM running, waiting and KV-usage gauges are now recorded. Earlier
  releases left them unknown.
- `runeconomy import` auto-detects a one-row Prefixwise ledger or vLLM capture.
- MCP tools return a tool error that names any wrongly typed argument.

## Compatibility

Every existing manifest, workspace, result and decision artifact keeps its
meaning. New behavior is versioned and applies only to contracts that declare
it.

## Installation

Python 3.11 or newer is required. Start the deterministic CPU-only tutorial:

```bash
python -m pip install runeconomy==0.3.0
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

Follow the generated README from a second terminal in the parent directory.
The alpha is intended for local, operator-run experiments. Hosted dashboards,
managed collectors and unattended production execution are outside its scope.
[Getting started](getting-started.md) · [Alpha boundaries](alpha-release.md).
