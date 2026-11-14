# RunEconomy

RunEconomy identifies inference waste on the critical path of an agent workflow
and tests whether a change improves successful-task performance. Its controlled
before-and-after methodology and report are called NegativeSplit.

The latest frozen real-agent proof is **invalid**, not verified. The candidate
increased cache reuse, reduced TTFT, and had a favorable latency interval among
task pairs where both variants succeeded. However, the baseline's absolute
task-success lower bound and the paired task-success non-inferiority bound
missed their predeclared gates. RunEconomy therefore declined to verify the
change. The 40-task workload has been consumed and cannot be rerun or retuned
as fresh held-out evidence.

The bundled CPU-only demo currently remains a separate, historical static
request experiment. It is intentionally `inconclusive` after methodology
review. Treat both results as narrow, pinned evidence—not general benchmarks.

The project focuses on critical-path workflow experiments rather than trying to
replace prompt-cache collectors or general observability systems. It consumes
normalized evidence from those layers and tests application or serving
interventions under explicit quality and performance gates.

The current implementation analyzes exact prompt token sequences, models
vLLM's full-block prefix-cache eligibility, attributes divergence only where
segment evidence supports it, and separates potential reuse from observed
cached tokens. It imports structural OpenInference traces and Prefixwise ledger
findings without inventing missing evidence, captures server-rendered prompts,
and runs controlled paired comparisons.

## Choose the right workflow

- `runeconomy experiment` runs static request-level baseline/candidate
  manifests against vLLM. `negative-split` is its backward-compatible alias.
- `runeconomy compare` executes complete baseline and candidate agent
  processes with variant-local state, real tools, an evaluator, and imported
  OpenTelemetry/OpenInference traces.
- Compatibility trials validate the integration contract. Calibration tasks
  are used for development. Neither is held-out performance evidence.
- A frozen held-out proof is executed once after the agent, evaluator, gates,
  workload, environment, and sample size are fixed. Failed gates remain failed;
  observed tasks are never recycled as fresh evidence.

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

## Validate static experiments from a checkout

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
