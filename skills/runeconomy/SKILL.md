---
name: runeconomy
description: Import agent inference evidence, assess what it can support, prepare a controlled baseline and candidate experiment, and interpret a quality-gated RunEconomy verdict. Use when onboarding a repository to RunEconomy, diagnosing vLLM compatibility, converting OpenInference, Phoenix, vLLM, or Prefixwise evidence, or running an experiment safely.
---

# RunEconomy

RunEconomy tests whether an inference change improves successful-task
performance. Treat cache reuse, tokens, TTFT, and latency as observations until
the declared task-quality and performance gates support the claim.

## Start safely

1. Inspect the repository and deployment without changing them.
2. Run `runeconomy --version`. Read documentation from the matching release or
   tag. If RunEconomy is missing, install the version requested by the user.
3. Never upgrade RunEconomy without asking first.
4. Never print or commit credentials, prompts, model output, evaluator answers,
   raw traces, private endpoints, or unsanitized result artifacts.
5. Ask before paid compute, cache reset, development-mode endpoints,
   infrastructure changes, publication, or deployment.

`runeconomy doctor` is read-only. It does not generate text or reset caches:

```bash
runeconomy doctor --base-url "$VLLM_BASE_URL" --api-key-env VLLM_API_KEY
```

Follow its exact next action. An optional capability is not a blocker unless the
selected experiment mode requires it.

When the user wants to learn the workflow before supplying real evidence, use
the installed deterministic tutorial:

```bash
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

Treat it as a product-plumbing check, never as real inference validation or a
performance benchmark.

## Evidence-first workflow

Import the evidence without guessing its format:

```bash
runeconomy import SOURCE --output evidence.json
```

Read the capability assessment. Unknown fields must remain unknown. Then create
an experiment workspace:

```bash
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/
```

Validation is corrective. Resolve only the fields it identifies and only from
real evidence. Do not make an unresolved workspace runnable by inventing
requests, dependencies, token boundaries, identity, an evaluator, or replay
context.

For a new real-agent experiment, initialize the agent-process mode directly:

```bash
runeconomy experiment init --mode agent-process \
  --baseline "python agent_v1.py" \
  --candidate "python agent_v2.py" \
  --dataset tasks.json \
  --evaluator evaluator.py \
  --output experiment/
```

Run and report through the same command family:

```bash
runeconomy experiment run experiment/ --output result/
runeconomy experiment report result/ --output report.html
```

Review executable agent commands and evaluators before running them. Keep every
failure and timeout in the result. Preserve variant-local state and explicit
`runeconomy.depends_on` relationships. Never infer a workflow dependency merely
from span nesting.

## Interpret the result

- `verified`: declared quality and performance requirements passed.
- `regression`: a declared quality or performance guardrail failed.
- `inconclusive`: evidence was valid but insufficient for the claim.
- `invalid`: evidence, identity, execution, or baseline quality could not
  support the comparison.

Only call a change an improvement when the verdict is `verified`. State which
pairs entered each latency or mechanism estimate. Missing metrics stay unknown,
not zero.

## Load detail only when needed

- [Evidence imports and capability levels](references/evidence-imports.md)
- [Experiment workspace and execution contract](references/experiment-contract.md)
- [Verdicts and claim language](references/verdicts.md)
- [Privacy and infrastructure safety](references/privacy.md)
- [Troubleshooting](references/troubleshooting.md)

## Completion report

Report the evidence source, privacy boundary, capability assessment, frozen
intervention and gates, attempt classifications, verdict and decisive gate,
conditional observations, remaining uncertainty, and next safe action.
