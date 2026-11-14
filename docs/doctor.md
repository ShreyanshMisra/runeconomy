# vLLM compatibility diagnosis

`runeconomy doctor` inspects whether a vLLM deployment supports RunEconomy's
static experiment runner. It is intentionally read-only: it performs GET
metadata requests and, when advertised and a model is known, a non-inference
render probe. It never sends a generation request or invokes cache reset.

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

Endpoint paths, query strings, embedded credentials, and API-key values are not
included in the output. JSON output is available with `--format json` or the
shorter `--json` alias.

## Capability states

- `supported`: direct metadata or a safe probe established the capability.
- `unsupported`: available metadata established that it is absent or disabled.
- `unknown`: a safe read-only probe could not establish it.

Unknown is not zero and is not automatically a blocker. Cache telemetry and
per-request server timing usually remain unknown until a real smoke capture is
performed. Warm-cache execution can still be safe when generation and exact
tokenization are established.

## Experiment modes

- `warm-cache` measures normal reuse without clearing server state.
- `salt-isolated` uses unique cache namespaces when `cache_salt` is supported.
  It prevents cross-trial prefix reuse but does not physically evict blocks or
  eliminate capacity effects.
- `physical-reset` is reported when the development endpoint is advertised, but
  is marked manual/private-only. `doctor` never invokes it.

The report gives an exact next action for every blocker or unknown readiness
state. Optional mechanism telemetry is distinguished from capabilities required
to execute a valid static experiment.

## Exit behavior

| Code | Meaning |
| --- | --- |
| `0` | Diagnosis completed and at least one safe mode is supported. |
| `1` | Diagnosis completed, but no safe mode was established. |
| `2` | Diagnosis could not complete. |

Exit `2` includes invalid URLs, missing API-key variables, authentication
failures, connection failures, timeouts, and malformed metadata responses.
These failures emit the same category, safe explanation, evidence-completeness
flag, and next action in human and JSON modes.

## Common corrections

- Authentication: set the key in an environment variable and use
  `--api-key-env NAME`; never put the secret directly in a command or manifest.
- Connection: verify the OpenAI-compatible server root and network access.
- Timeout: increase `--timeout` only after confirming the server is healthy.
- No API dialect: expose `/v1/chat/completions` or `/v1/responses`.
- No exact-token path: expose `/tokenize` or start vLLM with scale-out rendering
  enabled.
- Missing reset: use warm-cache or salt-isolated mode; do not expose a
  development reset endpoint merely to satisfy a benchmark.
