# RunEconomy

RunEconomy finds inference waste in multi-step agent workflows and verifies whether proposed changes improve successful-task performance. Its controlled before-and-after experiment workflow is called NegativeSplit.

So far, measurements are limited to pinned, narrow workloads. The bundled demo
is intentionally classified as inconclusive after a methodology review: its
latency and token effects are measured, but its tool allowlist was not selected
on a held-out workload and fifteen agreeing task-outcome pairs cannot prove a
zero-regression margin. Treat it as an auditable example, not a general
benchmark or verified product claim.

The project focuses on critical-path workflow experiments rather than trying to replace prompt-cache collectors or general observability systems. It can consume normalized evidence from those layers and test whether an application or serving intervention improves successful-workflow performance.

The current implementation analyzes exact prompt token sequences, models vLLM's full-block prefix-cache eligibility, attributes divergence only where segment evidence supports it, and separates potential reuse from observed cached tokens. It also imports structural OpenInference traces and Prefixwise ledger findings without inventing missing prompt tokens, captures server-rendered prompts, and runs controlled paired A/B replays against a trusted vLLM benchmark server.

## Installation

Install RunEconomy from PyPI:

```bash
pip install runeconomy
```

RunEconomy requires Python 3.11 or newer. Trace analysis and the bundled demo
run locally without a GPU or network connection.

## Run the CPU-only demo

The bundled demo uses sanitized aggregate evidence from the measured repository-agent experiment. It makes no network or GPU calls.

```bash
runeconomy demo --open
```

This writes `runeconomy-demo.html`, a self-contained NegativeSplit report with
the diagnosis, baseline/candidate comparison, continuous-metric and exact
task-success intervals, quality gates, provenance, and claim boundary.
Generated output and raw trials are not embedded.

## Check a vLLM deployment

Inspect a normal vLLM deployment without generating text or clearing its
cache:

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

The report distinguishes advertised support from capabilities that remain
unknown until a generation is observed. It also states which experiment modes
are safe. A cache salt prevents reuse across salt namespaces; it does not evict
cache blocks or eliminate capacity and eviction effects.

Both Responses and Chat Completions requests are supported. Chat Completions
uses its render endpoint when available and falls back to the standard
chat-form `/tokenize` endpoint for exact prompt token IDs:

```bash
runeconomy capture-vllm chat-request.json \
  --api-dialect chat_completions \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

## Run the current proof from a checkout

The commands below use repository fixtures and tests; they are not included in
an installed wheel.

```bash
uv run runeconomy analyze examples/traces/repository-unused-tools.json
uv run pytest -q
```

Validate the example experiment manifest locally (no GPU call is made):

```bash
uv run runeconomy experiment validate \
  examples/experiments/timestamp-placement.json
```

`experiment` is the primary command. `negative-split` remains a compatible
alias, while NegativeSplit remains the name of the paired methodology and
report.

## Turn a finding into an experiment

Analysis reports contain stable finding IDs, affected requests, evidence
quality, critical-path membership, impact estimates, optimization targets,
required capabilities, and structured experiment templates:

```bash
runeconomy analyze trace.json --format json --output analysis.json
runeconomy experiment init \
  --from-finding analysis.json \
  --finding-id finding-0123456789abcdef \
  --output experiment.json
```

Normalized traces intentionally omit replayable request bodies, tool
definitions, datasets, and evaluators. An ordinary finding therefore produces
an explicit unresolved skeleton listing the missing evidence. RunEconomy emits
a runnable manifest only when a safe template has every required input and the
normal manifest validator accepts the result. It does not substitute guessed
requests or thresholds.

See the [trace format](docs/trace-format.md). The
[experiment format](docs/experiment-format.md) documents the vLLM flags,
reproducibility controls, and result semantics.

Experiment manifests can execute local Python or commands when those evaluator
kinds are enabled. Review a manifest like a script; the CLI requires an
explicit `--allow-exec-evaluators` flag before using them.

## Compare real agent workflows

Run a paired, seeded comparison of two complete agent processes:

```bash
runeconomy compare \
  --baseline "python agent_v1.py" \
  --candidate "python agent_v2.py" \
  --dataset tasks.json \
  --evaluator evaluator.py \
  --output comparison.json \
  --repetitions 3
```

Each process owns one task's full workflow state. RunEconomy measures process
wall time, imports the agent's OpenInference/OpenTelemetry trace, evaluates the
final output, and pairs only matching task/repetition attempts. Failed tasks,
timeouts, infrastructure failures, invalid traces, and evaluator failures stay
separate and cannot be silently pooled into successful-task latency.

See the [closed-loop comparison contract](docs/closed-loop-comparison.md) for
the dataset, environment variables, evaluator protocol, explicit DAG evidence,
statistics, artifacts, and privacy boundary.

Exact-token captures can be converted into a normalized trace without manually
writing the trace schema:

```bash
uv run runeconomy import-vllm-jsonl captures.jsonl \
  --output trace.json \
  --trace-id coding-agent-001 \
  --workload coding-agent \
  --model Qwen/Qwen2.5-1.5B-Instruct
uv run runeconomy analyze trace.json
```

Structured OpenInference/OpenTelemetry exports can be normalized even when
they contain token counts but not rendered token IDs. Prefixwise JSONL exports
become structured findings that can seed an honest unresolved experiment
skeleton:

```bash
uv run runeconomy import-openinference otlp-trace.json \
  --output trace.json \
  --workload coding-agent

uv run runeconomy import-prefixwise-jsonl prefixwise.jsonl \
  --output findings.json
uv run runeconomy experiment init \
  --from-finding findings.json \
  --finding-id FINDING_ID \
  --output experiment-skeleton.json
```

Exact token boundaries, block-level attribution, structural attribution, and
unknown attribution are distinct in the trace format. RunEconomy does not
reconstruct prompt text or token arrays from Prefixwise hashes and counts.

## Modal validation

The private Modal deployment is defined in
[`deploy/modal_vllm.py`](deploy/modal_vllm.py). It pins the model and serving
stack, preloads model weights on CPU, verifies required vLLM capabilities
before GPU allocation, uses one A10 replica, and persists Hugging Face, vLLM,
and FlashInfer caches in separate v2 volumes.

Modal commands require a repository checkout with the locked development
dependencies installed by `uv sync --locked`.

Credentials belong only in the ignored `.env.local` file. A validation window
must end by explicitly stopping the app:

```bash
uv run modal app stop -y runeconomy-vllm
uv run modal container list --json
```

The opt-in `Real vLLM integration` GitHub workflow only contacts an
already-running endpoint configured in the protected `vllm-integration`
environment. It never provisions GPU resources and runs only when manually
dispatched.
