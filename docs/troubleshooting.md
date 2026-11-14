# Troubleshooting

RunEconomy errors separate what failed, why it likely failed, whether evidence
may be incomplete, and the next corrective action. Do not treat partial output
or a checkpoint as a final verdict.

## `doctor` cannot connect

Confirm that `--base-url` points to the OpenAI-compatible vLLM root, the server
is running, and the machine running RunEconomy can reach it. Increase
`--timeout` only when the endpoint is healthy but slow to return metadata.

## Authentication failed

Keep the key outside shell history and manifests:

```bash
export VLLM_API_KEY='replace-me'
runeconomy doctor --base-url "$VLLM_BASE_URL" --api-key-env VLLM_API_KEY
```

If metadata endpoints use different authorization policy from generation,
correct that policy or use a key permitted to read them.

## Experiment manifest is invalid

Run:

```bash
runeconomy experiment validate experiment.json
```

Correct the named field. An unresolved skeleton must have all unresolved fields
filled from real evidence before it is runnable. Executable evaluators require
manual review and explicit `--allow-exec-evaluators` opt-in.

## Experiment execution stopped

Preserve the checkpoint and partial result. Run `doctor` against the same
backend, inspect the reported cause, and resume only if the manifest schedule,
backend identity, and evidence remain valid. A changed model, tokenizer,
template, backend version, or relevant engine setting invalidates comparison.

## `compare` reports invalid evidence

Inspect the private artifacts directory and attempt status files. Verify that
each agent wrote a complete trace and status, identity fields match, and actual
dependencies use `runeconomy.depends_on`. Do not infer a DAG from span nesting
or delete failed attempts to improve the apparent latency.
