# Check what your vLLM server can safely support.

`runeconomy doctor` performs a read-only compatibility diagnosis before you run
an experiment. It tells you which capabilities are supported, unsupported or
still unknown, then recommends the safest available experiment mode.

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY \
  --startup-timeout 600
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

## Read the output

The first line reports overall readiness:

| Readiness | Meaning |
| --- | --- |
| `READY` | At least one safe experiment mode is supported. |
| `UNKNOWN` | Metadata was too limited to establish a safe mode. |
| `BLOCKED` | No safe experiment mode is available. |

This abridged example comes from a local test server that advertises Chat
Completions and `/tokenize` only. It does not describe a real deployment.

```text
RunEconomy doctor · READY
Safe modes available: warm-cache.
Target details: redacted
Backend: vLLM 0.33.0
Models: example/model

Capability                  Status      Evidence
--------------------------  ----------  ----------------------------------------
Responses API               unsupported absent from /openapi.json
Chat Completions API        supported   advertised by /openapi.json
Responses render            unsupported absent from /openapi.json
Chat Completions render     unsupported absent from /openapi.json
Tokenize                    supported   advertised by /openapi.json
Cache telemetry             unknown     requires a completed generation response
Per-request timing          unknown     requires a completed generation response
Cache reset                 unsupported absent from /openapi.json
Cache salt                  unsupported cache_salt is absent from the OpenAPI schema

Experiment modes:
  warm-cache: supported (safe)
  salt-isolated: unsupported
  physical-reset: unsupported

What to do next
  1. Run one non-production smoke capture, then inspect whether cached-token and server-timing fields are present.
  Run with --json for guidance on every capability and identity field.
```

The full output also explains each experiment mode and reminds you which
identity fields to pin in the experiment manifest. The human output lists only
the actions that change what you can run. `--json` carries a next action for
every capability and identity field.

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
- Cold start: set `--startup-timeout` to a bounded window. Doctor reuses the
  same 502/503 and connection retry policy as capture and static experiment
  execution, while 429 and other terminal responses fail immediately.
- Request timeout: increase `--timeout` only after confirming the server is healthy.
- Missing API dialect: expose `/v1/chat/completions` or `/v1/responses`.
- Missing exact-token path: expose `/tokenize` or supported rendering.
- Missing reset: use warm-cache or salt isolation. Do not expose a development
  reset endpoint only for RunEconomy.

See [Troubleshooting](troubleshooting.md) for setup failures and the
[CLI reference](cli-reference.md) for complete JSON and exit-code behavior.
