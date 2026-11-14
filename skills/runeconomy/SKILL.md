---
name: runeconomy
description: Diagnose inference waste in vLLM-backed agent workflows and run quality-gated baseline/candidate experiments with RunEconomy. Use when inspecting compatibility, importing traces, preparing an experiment, comparing agents, or interpreting RunEconomy verdicts.
---

# RunEconomy

Use RunEconomy as a verification layer between agent traces and inference
infrastructure. The goal is not to maximize an isolated metric; it is to learn
whether a change improves successful-task performance.

## Safety boundary

- Begin read-only. Inspect the repository, deployment configuration, existing
  traces, and `runeconomy --help` before changing an agent or server.
- Never print or commit API keys, prompts, model output, evaluator answers, raw
  traces, private endpoints, or experiment artifacts containing them.
- `runeconomy doctor` is read-only. It may make metadata and non-inference
  render probes; it never generates text or calls cache-reset endpoints.
- Do not enable development endpoints, reset shared caches, start paid compute,
  publish results, or alter production infrastructure without explicit approval.
- Treat executable evaluators and agent commands as local code execution. Review
  them before opting in.

## 1. Establish the environment

Install the version requested by the user. For the current alpha:

```bash
python -m pip install --pre runeconomy==0.2.0a2
runeconomy --help
runeconomy demo --open
```

If `--open` is unavailable, generate the file and report its path. The bundled
proof is an `invalid` result: attractive cache and latency measurements were
rejected because task-quality evidence did not pass the frozen gates.

For vLLM, keep the key in an environment variable and diagnose compatibility:

```bash
export VLLM_API_KEY='...'
runeconomy doctor --base-url "$VLLM_BASE_URL" --api-key-env VLLM_API_KEY
```

The operator must supply the OpenAI-compatible server root. If the base URL is
not available, stop and ask for it; do not guess an endpoint from repository
history, logs, or provider metadata.

Do not paste the key into arguments, manifests, reports, or chat. Interpret the
doctor exit code as follows:

- `0`: at least one safe experiment mode is supported.
- `1`: diagnosis completed, but no safe mode is established.
- `2`: the probe could not complete because of configuration, authentication,
  connection, timeout, or malformed-response failure.

Follow the reported “What to do next” action. Optional cache reset, cache salt,
server timing, or cache telemetry must not be described as universal blockers.

## 2. Choose evidence acquisition

Prefer existing OpenTelemetry/OpenInference traces when they preserve real
workflow timing and explicit dependencies. Otherwise use the narrowest importer
that matches available evidence:

```bash
runeconomy import-openinference otlp.json --output trace.json
runeconomy import-vllm-jsonl captures.jsonl --output trace.json \
  --trace-id TRACE --model MODEL
runeconomy import-prefixwise-jsonl ledger.jsonl --output findings.json
```

Use instrumentation when imports lack the workflow relationships required by
the question. Do not invent token boundaries, parent-child dependencies, cached
tokens, timing, or identity fields. Unknown must remain unknown.

For real-agent comparison, every attempt must export durable OpenTelemetry or
OpenInference JSON to `RUNECONOMY_TRACE_PATH` and a versioned attempt status to
`RUNECONOMY_STATUS_PATH`. Preserve variant-local state. Add explicit
`runeconomy.depends_on` relationships for actual execution dependencies; do not
infer a workflow DAG merely from span nesting.

## 3. Diagnose and initialize

```bash
runeconomy analyze trace.json --format json --output analysis.json
runeconomy experiment init --from-finding analysis.json \
  --finding-id FINDING_ID --output experiment.json
runeconomy experiment validate experiment.json
```

Inspect the finding's evidence level, attribution quality, affected requests,
critical-path status, required backend capabilities, proposed causal mechanism,
and predicted direction. An unresolved skeleton is not runnable: fill only
fields supported by evidence, then validate again.

Before execution, predeclare and review:

- workload split and task evaluator;
- baseline and candidate intervention;
- primary end-to-end metric and secondary mechanism metrics;
- absolute task-success and paired non-inferiority gates;
- cache isolation and trial ordering;
- model, revision, tokenizer, template, backend, and relevant engine identity;
- sample size or stopping rule and cost ceiling.

## 4. Execute the right comparison

Use static `experiment` for controlled request manifests:

```bash
runeconomy experiment run experiment.json --output result.json
runeconomy experiment report result.json --output report.html
```

Use `compare` for complete baseline and candidate agent processes:

```bash
runeconomy compare \
  --baseline "python agent_v1.py" \
  --candidate "python agent_v2.py" \
  --dataset tasks.json \
  --evaluator evaluator.py \
  --output comparison.json
```

Use randomized or alternating paired order. Keep failures in the result; never
allow failed or timed-out attempts to disappear from latency accounting. Classify
task failure, model-output failure, application failure, infrastructure failure,
timeout, evaluator failure, and invalid trace separately. Preserve checkpoints
and partial traces when interrupted.

## 5. Interpret the verdict

Read gates in their declared order and report the exact status:

- `verified`: quality gates passed and the declared performance criterion passed.
- `regression`: the candidate violated a declared quality or performance guardrail.
- `inconclusive`: evidence was valid but insufficient to establish the claim.
- `invalid`: the experiment or baseline did not establish valid evidence for the
  comparison.

Only call a change an improvement when the verdict is `verified`. Cache hits,
token reductions, TTFT, or successful-pair latency may be useful mechanism
measurements, but remain conditional when quality gates fail. State which pairs
entered each continuous metric and whether missing fields were unknown.

## Completion report

Report:

1. Evidence source and privacy boundary.
2. Backend readiness and unsupported or unknown capabilities.
3. Frozen intervention, workload, metrics, gates, identities, and hashes.
4. Attempt counts and failure classifications.
5. Verdict and decisive gate.
6. Conditional mechanism measurements, clearly labelled.
7. Remaining uncertainty and the next safe action.
