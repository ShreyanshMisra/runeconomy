# Check what your vLLM server can safely support.

`runeconomy doctor` performs a read-only compatibility diagnosis before you run
an experiment. It tells you which capabilities are supported, unsupported or
still unknown, then recommends the safest available experiment mode.

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

## What doctor reads

`doctor` reads advertised API routes, model metadata and safe rendering or
tokenization capabilities. When a server advertises a render path and identifies
a model, it may perform a non-generation render probe.

It does not:

- generate text
- execute an agent workflow
- reset a cache
- call a destructive development endpoint
- print the configured host, path, query, embedded credentials or API-key value

Generic capability paths such as `/openapi.json` may appear as evidence. Use
`--format json` or `--json` for structured output.

## Read capability states

| State | Meaning |
| --- | --- |
| Supported | Direct metadata or a safe probe established the capability. |
| Unsupported | Available metadata established that it is absent or disabled. |
| Unknown | A safe read-only probe could not establish it. |

Unknown is not zero. It is also not automatically a blocker. Cache telemetry
and server timing often remain unknown until a real request is captured.

## Sanitized human output

```text
RunEconomy doctor
Readiness: Ready with limitations

Capability                 State         Next action
Chat Completions           Supported     No action required
Exact tokenization         Supported     No action required
Cache telemetry            Unknown       Capture one measured request
Physical cache reset       Unsupported   Use warm-cache or salt isolation

Safe modes: warm-cache, salt-isolated
```

The example is illustrative. It does not describe a real endpoint.

## Choose a safe experiment mode

- `warm-cache` measures ordinary reuse without clearing server state.
- `salt-isolated` uses separate cache namespaces when `cache_salt` is supported.
  A salt does not physically evict blocks or remove capacity effects.
- `physical-reset` may be reported when a development endpoint is advertised.
  It remains manual and private-server-only. `doctor` never invokes it.

Physical cache reset should not be enabled on a shared server merely to satisfy
a benchmark.

## Exit behavior

| Code | Meaning |
| --- | --- |
| `0` | Diagnosis completed and at least one safe mode is supported. |
| `1` | Diagnosis completed, but no safe mode was established. |
| `2` | Diagnosis could not complete. |

Exit `2` includes invalid URLs, missing API-key variables, authentication
failures, connection failures, timeouts and malformed metadata responses. Human
and JSON output use the same failure categories and next actions.

## Correct common problems

- Authentication: set the key in an environment variable and pass its name
  through `--api-key-env`.
- Connection: verify the OpenAI-compatible server root and network access.
- Timeout: increase `--timeout` only after confirming the server is healthy.
- Missing API dialect: expose `/v1/chat/completions` or `/v1/responses`.
- Missing exact-token path: expose `/tokenize` or supported rendering.
- Missing reset: use warm-cache or salt isolation. Do not expose a development
  reset endpoint only for RunEconomy.

See [Troubleshooting](troubleshooting.md) for setup failures and the
[CLI reference](cli-reference.md) for complete JSON and exit-code behavior.
