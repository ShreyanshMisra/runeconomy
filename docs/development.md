# Development and validation

Install the locked development environment from a repository checkout:

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
```

Repository fixtures are not installed in the wheel. They can be used for
local, CPU-only validation:

```bash
uv run runeconomy analyze examples/traces/repository-unused-tools.json
uv run runeconomy experiment validate \
  examples/experiments/timestamp-placement.json
```

## Documentation site

The site under [`site/`](../site/) renders these Markdown sources:

```bash
cd site
npm ci
npm run build
npm run check
```

The alpha version on the home page comes from `pyproject.toml`. The stable
version and release-note dates come from the GitHub release list at build
time, with repository values as the fallback when GitHub cannot be reached.
Set `RUNECONOMY_SKIP_RELEASE_LOOKUP=1` for a build without network access.

Article thumbnails are 1280 by 800 PNG files in `site/assets/thumbnails/`,
rendered from the SVG sources in `site/assets/thumbnail-sources/`. Render the
PNG again after editing a source.

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
