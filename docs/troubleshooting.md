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

`capture-vllm` and `experiment run` fail immediately when the server does not
accept the connection. For a deployment that needs time to load a model, pass
`--startup-timeout SECONDS`. RunEconomy then retries until the server responds
or the time runs out, and prints one notice while it waits.

## Authentication failed

Keep the key outside shell history and manifests:

```bash
export VLLM_API_KEY='replace-me'
runeconomy doctor --base-url "$VLLM_BASE_URL" --api-key-env VLLM_API_KEY
```

If metadata endpoints use different authorization policy from generation,
correct that policy or use a key permitted to read them.

## Evidence format cannot be detected

Run `runeconomy import --help`, then supply the known `--type`. Do not rewrite
the artifact or claim a stronger source merely to make detection pass.

## Experiment workspace is unresolved

Run:

```bash
runeconomy experiment validate experiment/ --json
```

Each issue identifies the artifact, field, execution impact, claim impact, and
next action. Fill fields only from real evidence. Executable evaluators require
manual review and explicit `--allow-exec-evaluators` opt-in.

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

Extract and inspect every JSON file before sharing. Automated privacy controls
reduce risk but cannot guarantee that an artifact is safe for every context.

## The agent status integration is rejected

Use `Attempt.from_environment()` at the real agent entry point. Do not create
the status JSON manually. Confirm that all runner paths are absolute, distinct,
and inside `RUNECONOMY_ATTEMPT_DIR`, and that no status file already exists.
Call one terminal method exactly once. Keep `stage` and `code` as fixed labels,
not exception messages or application data.

## Experiment planning is blocked

Run `runeconomy experiment validate WORKSPACE` first. Planning does not repair
execution errors. For a runnable workspace, inspect each
`runeconomy.plan.*` issue. Do not weaken a gate after seeing held-out results.
Use calibration to estimate paired latency variability, declare a fixed or
outcome-independent stopping rule, and keep calibration separate from the
operator-declared held-out workload.
