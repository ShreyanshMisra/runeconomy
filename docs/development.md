# Development and validation

Install the locked development environment from a repository checkout:

```bash
uv sync --locked
uv run pytest -q
```

Repository fixtures are not installed in the wheel. They can be used for
local, CPU-only validation:

```bash
uv run runeconomy analyze examples/traces/repository-unused-tools.json
uv run runeconomy experiment validate \
  examples/experiments/timestamp-placement.json
```

## Modal validation

The private Modal deployment is defined in
[`deploy/modal_vllm.py`](../deploy/modal_vllm.py). It pins the model and serving
stack, preloads model weights on CPU, verifies required vLLM capabilities
before GPU allocation, uses one A10 replica, and persists Hugging Face, vLLM,
and FlashInfer caches in separate v2 volumes.

Modal commands require the locked development dependencies. Credentials belong
only in the ignored `.env.local` file. Paid resources require explicit approval.
A validation window must end by explicitly stopping the app:

```bash
uv run modal app stop -y runeconomy-vllm
uv run modal container list --json
```

The opt-in `Real vLLM integration` GitHub workflow only contacts an
already-running endpoint configured in the protected `vllm-integration`
environment. It never provisions GPU resources and runs only when manually
dispatched.
