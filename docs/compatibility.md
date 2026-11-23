# Live vLLM compatibility

Last validated: 2026-11-23

This is a small, dated compatibility contract. Every row below was exercised
against a live, authenticated Modal Server. It is not a claim that unlisted
versions, models, GPUs, or deployment shapes are supported.

## Tested combinations

| Date | RunEconomy commit | vLLM | Model | GPU | API surface | Evidence fields | Limitations | Result | Evidence digest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-23 | `1e6e502` | 0.33.0 | Qwen/Qwen2.5-1.5B-Instruct | A10G | Chat and Responses, stream and non-stream, render, tokenize, reset | Cached tokens, server TTFT, backend, model, template, GPU | The final passing probe reused a warm container. Cold-start time comes from the immediately preceding identical pinned deployment attempt. Client TTFT includes Modal's experimental authenticated curl path. No throughput or performance-superiority claim was tested. | Passed | `sha256:637e2288f4d3…` |
| 2026-11-23 | `632a7b4` | 0.34.0 | Qwen/Qwen2.5-1.5B-Instruct | A10G | Chat and Responses, stream and non-stream, render, tokenize, reset | Cached tokens, server TTFT, backend, model, template, GPU | Client TTFT includes Modal's experimental authenticated curl path. No throughput or performance-superiority claim was tested. | Passed | `sha256:fe9a4d993c4a…` |
| 2026-11-23 | `33b3fc3` | 0.34.0 | Qwen/Qwen2.5-14B-Instruct | L40S | Chat and Responses, stream and non-stream, render, tokenize, reset | Cached tokens, server TTFT, backend, model, template, GPU | Client TTFT includes Modal's experimental authenticated curl path. The first-use FlashInfer sampling module extended cold start. No throughput or performance-superiority claim was tested. | Passed | `sha256:c2f5aea19de6…` |

The sanitized [dated evidence record](compatibility-data/2026-11-23.json) pins the
full model revisions, dependency versions, RunEconomy deployment and driver
commits, raw local-evidence digests, checkpoint digests, evidence grades, and
limitations.

## What passed

Each row passed authenticated endpoint discovery, Server ID discovery, Chat
Completions and Responses in streaming and non-streaming modes, exact prompt
rendering, `/tokenize`, prefix-cache reset, repeated-prefix cached-token
telemetry, per-request server TTFT, stable backend identity, and checkpoint
resume after a planned interruption.

The cold starts observed in these deployments were 52.4 seconds for vLLM
0.33.0 with the 1.5B model, 160.6 seconds for vLLM 0.34.0 with the 1.5B model,
and 309.3 seconds for vLLM 0.34.0 with the 14B model. The 14B first start
included a first-use FlashInfer compilation.

## Evidence boundary

Model name, vLLM version, GPU class, cache telemetry, and server TTFT were
directly observed. Model revision and chat-template digest were reliably
derived from the pinned deployment and pinned model snapshot because the vLLM
HTTP API does not emit the model revision. Client TTFT includes Modal's
experimental authenticated curl path and is not used for a performance claim.

These were compatibility smokes, not throughput studies or optimization
experiments. A passing row means the listed RunEconomy evidence path worked for
that exact combination on 2026-11-23. It does not establish latency superiority,
quality preservation under an engine change, production capacity, or support
for a different patch release.

## Cost and reruns

The exact Modal billing report for Phase 3 was `$0.58168989`, including GPU,
CPU, memory, and network egress. The matrix is intentionally small enough to
rerun on demand or on a monthly release-validation cadence. Monthly execution
is an operational choice, not a product requirement.
