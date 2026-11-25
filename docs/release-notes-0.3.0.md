# RunEconomy 0.3.0 release notes

RunEconomy `0.3.0` makes capture zero-code, runs experiments under concurrent
load, gates on cost and tail latency, and adds SGLang and hosted
OpenAI-compatible backends. Coding agents and CI can now drive the whole
workflow. The privacy boundary is unchanged: raw tasks, traces, model output,
evaluator answers, credentials and endpoints stay local.

## Zero-code capture

`runeconomy proxy` sits between an OpenAI-compatible client and its server.
It streams responses through unchanged and records timing, usage, cached
tokens and call order. It groups calls into workflows by correlation header,
resolves exact token IDs against vLLM, and produces a runnable workspace.
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
  approved estimate. See [MCP server](mcp.md).
- `experiment patch` and `experiment freeze` change and lock a workspace
  without hand-edited JSON.
- Every command's `--json` output follows a versioned contract.
- A GitHub Action posts a verdict table on pull requests. See
  [GitHub Action](github-action.md).
- `runeconomy.workflow` helpers record explicit `depends_on` edges.

## Diagnosis and evaluators

Five new detectors cover tool-output carry-over, critical-path serialization,
step-level over-provisioning, output-token inflation and KV pressure. Each
maps to an experiment template. See [detectors](detectors.md). New evaluator
kinds run repository tests, tool trajectories, calibrated LLM judges, and
Promptfoo assertions, Braintrust scorers and Inspect AI scorers unchanged.
See [evaluators](evaluators.md).

## Public results

Proof 3 (BF16 vs online FP8 at 16 agents per variant) returned `verified`, and
Proof 4 (removing a search tool) returned `regression`. Both were frozen
before running and ran once. See [proofs and studies](proofs-and-studies.md).

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

```bash
python -m pip install runeconomy==0.3.0
```
