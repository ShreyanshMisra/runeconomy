---
name: runeconomy
description: Capture OpenAI-compatible agent traffic with runeconomy proxy, import workflow traces, and test a serving or agent change against task-quality and performance gates. Use for RunEconomy onboarding, vLLM or SGLang compatibility, hosted API comparisons, and controlled experiments.
---

# RunEconomy

RunEconomy tests whether an inference change improves successful-task
performance. Treat cache reuse, tokens, TTFT, and latency as observations until
the declared task-quality and performance gates support the claim.

## Start safely

1. Inspect the repository and deployment without changing them.
2. Run `runeconomy --version`. Read documentation from the matching release or
   tag. If the package is missing, install the requested or project-pinned
   version. If neither specifies a version, ask which release to install.
3. Never upgrade RunEconomy without asking first. Confirm downgrades as well.
4. Never print or commit credentials, prompts, model output, evaluator answers,
   raw traces, private endpoints, or unsanitized result artifacts.
5. Ask before paid compute, cache reset, development-mode endpoints,
   infrastructure changes, publication, or deployment.

`runeconomy doctor` is read-only. It does not generate text or reset caches:

```bash
runeconomy doctor --backend vllm --base-url "$VLLM_BASE_URL" --api-key-env VLLM_API_KEY --json > doctor.json
```

Use `--backend sglang` for an SGLang root or `--backend openai_compat` for a
provider API base URL. Follow the reported next action for capabilities
required by the selected experiment mode.

When the user wants to learn the workflow before supplying real evidence, use
the installed deterministic tutorial:

```bash
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

Treat it as a product-plumbing check, never as real inference validation or a
performance benchmark.

When a RunEconomy command fails, record its `runeconomy.*` error code. If a
maintainer needs diagnostic context, create a local support bundle:

```bash
runeconomy support-bundle experiment/ --output runeconomy-support.zip
```

Inspect every JSON member before sharing it. Bundle creation never authorizes
uploading the archive, opening an issue, or sharing it publicly. Do not attach
raw traces, private results, prompts, generated output, evaluator answers, or
credentials alongside it. Read the privacy and troubleshooting references
before handling a support artifact.

## Evidence-first workflow

To capture new client traffic, follow the [proxy recording and import steps](references/evidence-imports.md)
and [capture privacy rules](references/privacy.md). For existing evidence:

```bash
runeconomy import SOURCE --output evidence.json
```

Read the capability assessment. Unknown fields must remain unknown. Then create
an experiment workspace:

```bash
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/ --doctor-report doctor.json --json
```

If import produced an unresolved workspace, complete its source evidence or
replay context, re-import, and initialize a fresh directory. Keep missing
requests, dependencies, token boundaries, identity and evaluators explicit.

For a new real-agent experiment, initialize the agent-process mode directly:

```bash
runeconomy experiment init --mode agent-process \
  --baseline "python agent_v1.py" \
  --candidate "python agent_v2.py" \
  --dataset tasks.json \
  --evaluator evaluator.py \
  --output experiment/
```

Locate the existing entry point and tracing and explain the smallest edit.
Use a runner-only wrapper or guard the shared entry point. Adapt the function
names below to the repository. The normal branch returns the existing CLI’s
exit value. The runner branch assumes the agent returns text and its existing
instrumentation flushes the trace to `RUNECONOMY_TRACE_PATH` before returning.

```python
import os

def main():
    if not os.environ.get("RUNECONOMY_STATUS_PATH"):
        return run_agent_normal_cli()

    from runeconomy.agent import Attempt

    with Attempt.from_environment() as attempt:
        result = run_agent(attempt.task_path)
        attempt.output_path.write_text(result, encoding="utf-8")
        attempt.completed()
```

A missing or invalid runner variable raises `AttemptEnvironmentError`. Let
environment and lifecycle errors propagate. Call exactly one terminal method
per attempt. Call `completed()` after writing its output and trace. A second
terminal call raises
`AttemptLifecycleError`. A clean exit with no call records
`terminal_status_not_declared` and raises. The context manager also handles
`SystemExit` and `KeyboardInterrupt`: when no status exists it attempts to
record `application_failure` and propagates the original exception. Keep
`stage` and `code` fixed and content-free (for example, `stage="tool_call"`
and `code="timeout"`). Map only failures the application can identify.

Change a workspace only through `experiment patch`, never by editing
`manifest.json`. Each patch is validated and recorded in `provenance.jsonl`:

```bash
runeconomy experiment patch experiment/ --set acceptance.min_task_success_rate=0.9 \
  --reason "match the declared quality floor" --json
```

For explicit DAG edges without OpenTelemetry, use `runeconomy.track_workflow`
and `runeconomy.step(..., depends_on=[...])` around the agent's real steps.

Before benchmarking, review the read-only doctor report and run local readiness
checks. For an agent-process integration, use a stubbed backend to test both
the ordinary CLI invocation and the runner branch with its output, trace and
terminal status. Validate the workspace and inspect design blockers and power:

```bash
runeconomy experiment validate experiment/ --json
runeconomy experiment plan experiment/ --json
```

Before held-out execution, freeze the reviewed design and lock the code that
will run. Add a `--lock` for each additional agent or runner file. Changes
after freezing require a new workspace. Obtain explicit approval for paid
compute, non-local GPU runs, held-out execution, backend or CI changes, and
publication or deployment.

```bash
runeconomy experiment freeze experiment/ --lock agent.py --json
```

Commands that return a result support `--json`. Read the object identified
by `output_schema` (`runeconomy schema` lists the contracts). Command errors
use `runeconomy.cli.error.v1` on stderr.

When connected via `runeconomy mcp`, use its typed tools. Call
`run_experiment` without a token and show the operator the planned attempts,
commands, cost estimate and freeze status. After explicit approval, repeat
the same arguments with the returned `approval_token`. The token is single-use,
expires after 15 minutes and belongs to this server process. A restart, changed
arguments or changed bound files (including manifest, dataset or evaluator)
require a fresh estimate and approval. Preserve this boundary even if shell
execution is available.

For an approved held-out run through the standalone CLI:

```bash
runeconomy experiment run experiment/ --output results/ --require-frozen --json
runeconomy experiment report results/result.json --output report.html --json
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
