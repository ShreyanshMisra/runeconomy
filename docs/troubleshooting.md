# Troubleshooting

RunEconomy errors separate what failed, why it likely failed, whether evidence
may be incomplete, and the next corrective action. Do not treat partial output
or a checkpoint as a final verdict.

Record the namespaced error code when opening an issue. Explanations may improve
between patch releases, while codes such as `runeconomy.workspace.unresolved`
are intended for issue templates and automation.

## `doctor` cannot connect

Confirm that `--base-url` points to the OpenAI-compatible vLLM root, the server
is running, and the machine running RunEconomy can reach it. Increase
`--timeout` only when the endpoint is healthy but slow to return metadata.

## A cold server is still starting

`doctor`, `capture-vllm`, and static `experiment run` fail immediately when the server does not
accept the connection. For a deployment that needs time to load a model, pass
`--startup-timeout SECONDS`. RunEconomy then retries until the server responds
or the time runs out, and prints one notice while it waits. Only connection
failures and startup 502/503 responses are retried. 429 and other terminal
responses remain immediate failures.

## Authentication failed

Set `VLLM_API_KEY` through your usual credential manager and pass its name:

```bash
runeconomy doctor --base-url "$VLLM_BASE_URL" --api-key-env VLLM_API_KEY
```

If metadata endpoints use different authorization policy from generation,
correct that policy or use a key permitted to read them.

## Evidence format cannot be detected

Run `runeconomy import --help`, then supply the known `--type`. Do not rewrite
the artifact or claim a stronger source merely to make detection pass.

## Import succeeded but a metric is unknown

| Missing evidence | What it affects | How to obtain it |
| --- | --- | --- |
| Prompt segments | Locating a changing block within a shared prefix | Export message structure or use the [local capture proxy](proxy.md). Exact segment attribution also requires token boundaries that match the full prompt. |
| Workflow dependencies | Identifying the critical path when calls overlap | Declare the actual predecessors with `runeconomy.step(..., depends_on=[...])` or emit `runeconomy.depends_on` in the trace. |
| Cached-token counts | Measuring how much of a prompt the server reused | Check that the serving engine returns cached prompt-token usage. For streams, request usage when the backend requires it. The proxy records the usage the backend supplies. |
| Replayable requests | Sending the same baseline input during a new run | Supply the authorized original request and replay context. Counts and hashes alone cannot recover the request. |

## Experiment workspace is unresolved

Run:

```bash
runeconomy experiment validate experiment/ --json
```

Each issue identifies the artifact, field, execution impact, claim impact, and
next action. A missing model revision leaves the weights unpinned. A missing
tokenizer revision or chat-template digest leaves token boundaries uncertain.
A missing evaluator leaves task success unmeasured. For a static import, use
deployment metadata and the actual evaluator to complete the source evidence
or replay context, then re-import and initialize in a fresh directory.
Validate that workspace against the saved doctor report. Executable evaluators
require manual review and explicit `--allow-exec-evaluators` opt-in.

## Backend capabilities are unsupported or unknown

Save a read-only diagnosis, then validate against it:

```bash
runeconomy doctor --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY --json > doctor.json
runeconomy experiment validate experiment/ --doctor-report doctor.json
```

Do not enable development endpoints or reset a shared cache merely to clear a
validation message.

## Experiment execution stopped

Preserve the checkpoint and partial result. Run `doctor` against the same
backend, inspect the reported cause, and resume only if the manifest schedule,
backend identity, and evidence remain valid. A changed model, tokenizer,
template, backend version, or relevant engine setting invalidates comparison.
Use `runeconomy experiment run WORKSPACE --output RESULTS --resume` for either
static or agent-process workspaces. The runner rejects checkpoint records that
do not match the frozen schedule.

## A completed run missed a gate

With `--require-verified`, exit code 3 means the run produced a result.
Open the recorded gate breakdown:

```bash
runeconomy experiment report results/result.json --output report.html --open
```

Read the decisive gate, threshold and observed value. For a completion
failure, inspect attempt statuses for crashes or timeouts. For a task-success
failure, inspect evaluator outcomes and the denominator for each arm. If the
paired confidence bound missed the allowed quality margin, review the sample
size with `experiment plan` before freezing a fresh held-out design.
Successful-pair latency describes only the pairs that passed in both arms.

## An agent-process experiment reports invalid evidence

Inspect the private artifacts directory and attempt status files. Verify that
each agent wrote a complete trace and status, identity fields match, and actual
dependencies use `runeconomy.depends_on`. Do not infer a DAG from span nesting
or delete failed attempts to improve the apparent latency.

## Prepare an issue without attaching private evidence

Create a local support bundle from the workspace:

```bash
runeconomy support-bundle experiment/ \
  --output runeconomy-support.zip
```

If saved JSON reports are available, pass them explicitly with
`--doctor-report` and `--validation-report`. RunEconomy will not collect them by
contacting a backend. The bundle excludes raw traces, results, prompts,
responses, evaluator data, endpoints, credentials, paths, and arbitrary files.

If bundle creation fails, pass the workspace directory as input. Choose a
new `.zip` file path in an existing output directory.

Extract and inspect every JSON file before sharing. Automated privacy controls
reduce risk but cannot guarantee that an artifact is safe for every context.

## The agent status integration is rejected

Initialize `Attempt.from_environment()` only in a runner-specific wrapper or
the `RUNECONOMY_STATUS_PATH` branch of a shared entry point. Do not create
the status JSON manually. Confirm that all runner paths are absolute, distinct,
and inside `RUNECONOMY_ATTEMPT_DIR`, and that no status file already exists.
Call one terminal method exactly once. Keep `stage` and `code` as fixed labels,
not exception messages or application data.

## Experiment planning is blocked

Run `runeconomy experiment validate WORKSPACE` first. Planning does not repair
execution errors. For a runnable workspace, inspect each
`runeconomy.plan.*` issue. Do not weaken a gate after seeing held-out results.
On an unfrozen runnable workspace, preview the evidence-backed design change
with `experiment patch --dry-run`, apply it with a reason, and plan again.
Use calibration to estimate paired latency variability, declare a fixed or
outcome-independent stopping rule, and keep calibration separate from the
operator-declared held-out workload.
