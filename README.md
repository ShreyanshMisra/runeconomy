# RunEconomy

RunEconomy finds inference waste in multi-step agent workflows and verifies whether proposed changes improve successful-task performance. Its controlled before-and-after experiment workflow is called NegativeSplit.

So far, we've validated on pinned, narrow workloads. Treat the bundled demo as
evidence for its recorded workload and environment, not as a general benchmark.

The project focuses on critical-path workflow experiments rather than trying to replace prompt-cache collectors or general observability systems. It can consume normalized evidence from those layers and test whether an application or serving intervention improves successful-workflow performance.

The current implementation analyzes exact prompt token sequences, models vLLM's full-block prefix-cache eligibility, attributes divergence to semantic prompt segments, and separates potential reuse from observed cached tokens. It also captures server-rendered prompts and runs controlled paired A/B replays against a trusted vLLM benchmark server.

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
runeconomy demo
```

This writes `runeconomy-demo.html`, a self-contained NegativeSplit report with the diagnosis, baseline/candidate comparison, confidence interval, quality gates, provenance, and claim boundary. Generated output and raw trials are not embedded.

## Run the current proof from a checkout

The commands below use repository fixtures and tests; they are not included in
an installed wheel.

```bash
uv run runeconomy analyze examples/traces/repository-unused-tools.json
uv run pytest -q
```

Validate the example experiment manifest locally (no GPU call is made):

```bash
uv run runeconomy negative-split validate \
  examples/experiments/timestamp-placement.json
```

See the [trace format](docs/trace-format.md). The
[experiment format](docs/experiment-format.md) documents the vLLM flags,
reproducibility controls, and result semantics.

Experiment manifests can execute local Python or commands when those evaluator
kinds are enabled. Review a manifest like a script; the CLI requires an
explicit `--allow-exec-evaluators` flag before using them.

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
