# Capture inference through the local proxy

`runeconomy proxy` captures requests from an OpenAI-compatible client by
changing its base URL. The proxy requires RunEconomy 0.3.0 or newer.
For an existing local vLLM server, run:

```bash
runeconomy proxy --upstream http://127.0.0.1:8000 --record session.jsonl
```

The proxy listens on loopback and prints its client base URL. In another
terminal, point the existing agent at it:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8767/v1 python agent.py
```

Use your client's existing base-URL setting if it uses another variable.
Keep its normal model and credentials. The upstream URL can include a
trailing `/v1`. Use `--port` to choose another port, or `--port 0` to
allocate one and use the URL printed on startup.

For prefix-divergence diagnosis, capture at least two comparable requests in
one workflow. Keep the stable instructions and model fixed and exercise the
real variation you want to diagnose, such as an early timestamp in a system
message. Use vLLM with prefix caching and response usage that includes cached
prompt tokens. For streams, the client must already request usage for servers
that require it. The proxy forwards the client's request options unchanged.

After the agent finishes, stop the proxy with Ctrl-C and wait for the capture
summary. SIGTERM also drains the session. Inspect reported gaps before relying
on the capture. Each session starts with a new file.

## Import and validate a replay

```bash
runeconomy import session.jsonl --output evidence.json
```

The importer detects the versioned proxy session and creates normalized
traces, prefix-divergence findings where supported, and a capability
assessment. When multiple workflows were captured, use `--trace-id` to
select one or initialize from a specific finding. The import output lists
finding IDs and any known replay blockers.

A static replay needs the captured request and serving evidence plus explicit
operator inputs. Supply those in a local replay context and import again:

```bash
runeconomy import session.jsonl --type proxy \
  --replay-context replay-context.json --output replay-evidence.json
runeconomy experiment init --from-evidence replay-evidence.json \
  --finding-id FINDING_ID --output experiment/
runeconomy experiment validate experiment/
```

Replace `FINDING_ID` with the ID reported by import. The following
`replay-context.json` illustrates a small validation trial for an agent
whose correct answer is exactly `OK`. Replace the deployment values,
evaluator and design with your own before running an experiment.

```json
{
  "experiment_template_id": "prompt.move_dynamic_metadata",
  "candidate_strategy": "move_volatile_messages_after_system_block",
  "backend": {
    "model_revision": "<pinned model revision>",
    "tokenizer_revision": "<pinned tokenizer revision>",
    "chat_template_digest": "<chat template SHA-256 digest>",
    "hardware": "<serving hardware>",
    "engine_config": {"enable_prefix_caching": true}
  },
  "execution": {
    "repetitions": 2,
    "order": "alternating",
    "cache_treatment": "warm",
    "require_evaluation": true
  },
  "acceptance": {
    "primary_metric": "total_latency_ms",
    "min_task_success_rate": 0.8,
    "max_task_success_regression": 0.05,
    "min_completed_rate": 1.0
  },
  "evaluator": {"kind": "regex", "pattern": "OK", "full_match": true}
}
```

Supply any required backend fields the capture could not observe, such as
version or block size, and `backend.api_key_env` if replay needs
authentication. Supply `sampling` if the captured request omitted explicit
settings. Declared fields must agree with observed values. You can also
provide a separate `--proxy-identity` JSON object to annotate the normalized
trace with `model_revision`, `backend_version`, `tokenizer`,
`tokenizer_revision`, or `chat_template_digest`.

The example strategy applies when a finding locates changing values in a
leading system or developer message. It replaces the observed volatile value
with a per-trial placeholder and moves the whole message behind the stable
system block. Review the generated baseline and candidate. Other findings may
need an explicit `candidate_request` or `candidate_tool_names`. If redaction
changed the request or body capture was disabled, provide an explicit
`baseline_request`. Stateful requests that refer to prior server-side
responses cannot be replayed statically. A runnable workspace validates the
experiment inputs. A verifier verdict requires newly executed trials.

## Grouping and timing

For a standalone session, `X-RunEconomy-Workflow-Id` takes precedence over
the W3C `traceparent` trace ID. Without either header, the importer starts a
new workflow after a 10-second gap with no request in flight. Override this
with `import --idle-gap-ms`. If two requests without either correlation header
overlap in flight, RunEconomy marks the idle-gap group as ambiguous, warns at
import, records unresolved workflow-grouping evidence, and blocks runnable
workspace generation. Changing the idle gap cannot resolve an overlap.

Give every concurrent agent run a unique workflow ID and attach the same ID
to every inference request from that run. For the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8767/v1",
    default_headers={"X-RunEconomy-Workflow-Id": workflow_id},
)
```

For a direct HTTP call, add the equivalent header. Keep the client's usual
credential header as well. The proxy forwards it upstream and never records
it.

```bash
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-RunEconomy-Workflow-Id: agent-run-42" \
  --data @request.json
```

If the application already propagates W3C Trace Context, keep one 32-hex-digit
trace ID per workflow and send a valid `traceparent` header on every request.
Concurrent workflows must use different trace IDs. A workflow ID is usually
simpler. It must start with a letter or digit, may then contain letters,
digits and `_.:/@+=-`, and is limited to 128 characters. Do not reuse an ID
across unrelated concurrent agent runs.

`X-RunEconomy-Span-Id` identifies a request span.
`X-RunEconomy-Depends-On` supplies a comma-separated list or JSON array
of prior span IDs. Use `none` or `[]` to assert a root span.
The importer never turns request timing, order or W3C parentage into
dependencies. A complete dependency graph requires explicit assertions.

The proxy records Chat Completions, Responses and legacy Completions
exchanges. The static replay runner supports Chat Completions and Responses.
Streaming bytes are forwarded immediately, including partial SSE events.
The recorder parses its copy after the exchange. Streaming TTFT is measured
at the proxy from receipt of the full request to the first output-bearing
event. Inter-event latency summarizes output-bearing SSE events, which may
each carry several tokens. TPOT additionally requires a reported completion
token count. Unreported metrics and unresolved prompt token IDs stay unknown.
Exact message boundaries require a backend-rendered prefix that matches the
corresponding full prompt token prefix.

Token resolution runs in the background by default. Choose
`--token-resolution on_close` to defer auxiliary backend requests until the
capture stops, or `--token-resolution off` to disable them. Chat resolution
uses the vLLM render endpoint or its chat tokenize fallback. Responses
resolution requires the Responses render endpoint. The versioned invariants
and record fields are maintained with the implementation in
[the wire contract](../src/runeconomy/proxy/contract.py) and
[the session schema](../src/runeconomy/data/proxy-session-0.1.schema.json).

## Capture one agent-process attempt

In an agent-process workspace, wrap the baseline and candidate commands
individually and set each wrapper's serving upstream:

```bash
runeconomy proxy --upstream http://127.0.0.1:8000 --attempt -- python agent.py
```

The wrapper needs the runner's attempt environment. It uses an ephemeral
loopback port, preserves the child exit code, writes the capture session in the
attempt directory and exports an OpenInference trace for the runner. An
existing agent-written trace is preserved and captured evidence is kept
alongside it. The agent or its existing adapter still owns the task output and
terminal status through [the Attempt helper](closed-loop-comparison.md).
Use explicit dependency assertions or the agent's own tracing when the
comparison needs a complete DAG.

## Handle captured data

Full capture can contain prompts, tool arguments, generated content, exact
token IDs, routing identifiers and upstream metadata. Keep session files,
imported evidence and replay workspaces private. The proxy persists only
allowlisted headers and masks recognized credentials in recorded bodies.
Add `--redact package.module:function` to apply a trusted local function
with signature `hook(part, value) -> value`. The hook receives
`request_body` or `response_body` and runs after built-in redaction.

`--payload-capture digest_only` omits bodies. Token IDs can still expose
prompt content when resolution is enabled, even with body redaction or
digest-only capture. Use digest-only mode together with
`--token-resolution off` when bodies and token arrays must stay out of the
session. Inspect retained metadata before sharing any files. RunEconomy does
not upload captures.

## Check against local vLLM

The opt-in integration test sends two Chat Completions requests from a Python
client, expects a timestamp divergence, generates a static replay workspace
and validates it within ten minutes. It requires local vLLM with prefix
caching, cached-token usage and compatible prompt rendering. Point it at a
served model and a copy of the replay context above filled with that
deployment's identity.

```bash
RUNECONOMY_PROXY_LIVE_UPSTREAM=http://127.0.0.1:8000 \
VLLM_MODEL=YOUR_SERVED_MODEL \
RUNECONOMY_PROXY_REPLAY_CONTEXT=replay-context.json \
  uv run python -m unittest tests.test_proxy.ProxyTests.test_live_vllm_prefix_finding_and_replay -v
```

Set `VLLM_API_KEY` in the environment if the server requires it. The test
makes two inference calls and auxiliary read-only token and identity
requests. It validates the generated workspace without running a benchmark.
