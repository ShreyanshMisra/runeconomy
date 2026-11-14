# Getting started

RunEconomy identifies inference waste on an agent workflow's critical path and
tests whether a proposed change improves successful-task performance. It is an
alpha command-line tool for engineers using self-hosted vLLM.

## Install the private-tester alpha

Use an isolated Python 3.11 or newer environment:

```bash
python -m pip install --pre runeconomy==0.2.0a2
runeconomy --help
```

## Run the bundled proof

```bash
runeconomy demo --open
```

The default report is deliberately not a success story. A candidate improved
cache reuse, server TTFT, and successful-pair workflow latency, but RunEconomy
returned `invalid` because the frozen task-quality gates did not pass. Read the
[complete public proof](public-proof.md) before using its measurements.

## Diagnose a vLLM server

Keep authentication in an environment variable:

```bash
export VLLM_API_KEY='replace-me'
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

`doctor` is read-only. It uses metadata and non-inference render probes and
never generates text or invokes cache-reset endpoints. Exit codes are:

- `0`: at least one safe experiment mode is supported.
- `1`: diagnosis completed, but no safe mode was established.
- `2`: diagnosis failed because of configuration, authentication, connection,
  timeout, or malformed-response problems.

Use `--json` or `--format json` for automation. See
[vLLM compatibility](doctor.md) for the
capability and safety model.

The complete command and exit-code contract is in the
[CLI reference](cli-reference.md).

## Import or instrument evidence

Choose the importer that matches evidence you actually possess:

```bash
runeconomy import-openinference otlp.json --output trace.json
runeconomy import-vllm-jsonl captures.jsonl --output trace.json \
  --trace-id TRACE --model MODEL
runeconomy import-prefixwise-jsonl ledger.jsonl --output findings.json
```

Then analyze a normalized trace:

```bash
runeconomy analyze trace.json --format json --output analysis.json
```

Importers preserve unknown values rather than inventing prompts, token arrays,
boundaries, timing, or dependencies. See the [trace format](trace-format.md).

## Initialize and validate an experiment

```bash
runeconomy experiment init \
  --from-finding analysis.json \
  --finding-id FINDING_ID \
  --output experiment.json

runeconomy experiment validate experiment.json
```

Some findings can produce runnable manifests. Others produce an unresolved
skeleton because RunEconomy cannot safely synthesize the application change.
Resolve every named field, freeze the quality and performance gates, and
validate again. See the [experiment format](experiment-format.md).

Run static request experiments with `runeconomy experiment run`. Use
`runeconomy compare` only when executing complete baseline and candidate agent
processes; see [closed-loop comparison](closed-loop-comparison.md).

## Interpret the result

Read task-quality gates before conditional performance measurements:

- `verified`: declared quality and performance criteria passed.
- `regression`: a declared candidate guardrail failed.
- `inconclusive`: evidence was valid but insufficient to establish the claim.
- `invalid`: the experiment or baseline did not establish valid evidence.

Never convert a favorable cache, token, TTFT, or successful-pair measurement
into a verified improvement when the verdict is not `verified`.
