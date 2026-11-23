# Modal validation windows

The private vLLM Server declares `min_containers=0`. This is the normal safe
state: there is no warm GPU, and an authenticated request may incur a cold
start. A validation window temporarily pins one warm container, verifies that
change, waits for `/health`, runs exactly one declared validation command, and
restores zero in a `finally` block.

No paid validation is required for local 429, 502, 503, timeout, malformed
response, or resume testing. Use the local mock coverage for those paths.

## Before a paid window

1. Record the experiment, purpose, spending ceiling, commit, model, GPU, vLLM
   version, intervention, and workload in the compute ledger.
2. Check the current Modal price for the selected resources and calculate the
   worst-case cost from the Server startup timeout plus validation timeout.
3. Deploy the reviewed definition. Deployment alone keeps zero warm GPUs:

   ```bash
   uv run modal deploy deploy/modal_vllm.py
   ```

4. Confirm the App rather than assuming its Server name:

   ```bash
   uv run modal app info runeconomy-vllm --json
   ```

The validation helper repeats that discovery, requires exactly one deployed
Server, and operates on the returned Server ID. It does not hard-code
`VLLMServer` or another generated Server identifier.

## Run a window

The health URL remains private. Do not commit it or include it in a support
bundle.

```bash
uv run python scripts/modal_validation_window.py \
  --app runeconomy-vllm \
  --health-url "$VLLM_BASE_URL" \
  -- \
  uv run runeconomy doctor \
    --base-url "$VLLM_BASE_URL" \
    --api-key-env VLLM_API_KEY \
    --startup-timeout 600
```

The helper performs these transitions:

1. Resolve the deployed App and its sole Server ID.
2. Set `min_containers=1` and read the autoscaler settings back.
3. Poll the authenticated health endpoint with `modal curl`.
4. Run the command after `--` without a shell.
5. Set `min_containers=0` in `finally` and read the setting back again.

A validation failure, health timeout, or keyboard interruption still executes
the closing transition. Treat a restoration error as an operational incident.
do not assume the Server is at zero.

## Verify and record closure

After every live operation, inspect the App and containers, then record actual
cost and artifact location in the ledger:

```bash
uv run modal app info runeconomy-vllm --json
uv run modal container list --json
```

The autoscaler must report zero warm containers. The container list may show a
container during the configured scale-down window. Recheck until it is empty
before declaring the window closed.

## Emergency shutdown

Prefer restoring the Server minimum to zero. If the helper cannot do that:

1. Run `modal app info APP --json` and copy the returned Server ID locally.
2. Run `uv run modal server autoscaler update SERVER_ID --min-containers 0`.
3. Verify with `uv run modal server autoscaler settings SERVER_ID --json`.
4. If the override still cannot be restored, use
   `uv run modal app stop -y APP` as the manual emergency shutdown.
5. Confirm `uv run modal container list --json` is empty and record the event.

Stopping the App is a fallback because it is a stronger lifecycle operation
than a temporary autoscaler override.
