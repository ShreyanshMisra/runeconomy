# RunEconomy 0.2.0a1 alpha guide

`0.2.0a1` is an invited-tester release. It is not a production service or a
general benchmark suite. RunEconomy remains a local CLI and Python package.

## Install the prerelease

After the alpha is published:

```bash
python -m pip install --pre runeconomy==0.2.0a1
runeconomy --help
runeconomy demo --open
```

The default demo is the aggregate-only frozen rejection proof. Use
`runeconomy demo --case historical` to render the earlier static-request case.

## What is included

- `doctor` performs a read-only capability probe against a vLLM server and
  reports API dialects, tokenization/render support, timing and cache
  telemetry, cache reset/salt support, identities that can be detected, and
  safe experiment modes.
- Chat Completions and Responses are both supported. Exact-token capture uses
  supported render or tokenize capabilities when available.
- Static `experiment` manifests support observed, salted, and development-mode
  cache isolation. A salt isolates cache namespaces but does not physically
  clear blocks or eliminate capacity and eviction effects.
- Analysis findings have stable IDs, evidence/confidence fields, critical-path
  context, impact estimates, optimization targets, capability requirements,
  and structured experiment templates.
- `experiment init --from-finding` creates a runnable manifest only when the
  required evidence is available. Otherwise it creates an explicit unresolved
  skeleton instead of guessing requests, evaluators, or thresholds.
- `import-openinference` preserves structural OpenTelemetry/OpenInference
  evidence without inventing exact token boundaries or workflow dependencies.
- `import-prefixwise-jsonl` imports the conclusions and evidence available in a
  Prefixwise export without reconstructing prompts or tokens from hashes.
- `compare` executes real baseline and candidate agent processes with
  variant-local state, task-level pairing, deterministic evaluation, explicit
  failure classes, imported traces, DAG reconstruction, and joint quality and
  latency gates.
- The attempt-status protocol records versioned completion and failure state
  at `RUNECONOMY_STATUS_PATH`, allowing interrupted attempts to retain durable
  classification evidence without putting generated content in status files.
- The flagship demo shows why a candidate with favorable cache and latency
  measurements can still receive an `invalid` verdict.

## Migration from 0.1.0

There are no intentional Python API removals in this alpha. Existing static
experiment manifests and the `negative-split` command alias remain supported.

Behavioral changes to review:

1. `experiment` is the primary name for static request experiments;
   `negative-split` is a compatibility alias. NegativeSplit remains the method
   and report name.
2. `compare` is reserved for executing complete baseline and candidate agent
   processes. It is not an alias for static request replay.
3. `runeconomy demo` now renders the frozen quality-gated rejection. Select
   `--case historical` for the 0.1.0-era bundled report.
4. Verdicts can be `verified`, `regression`, `inconclusive`, or `invalid`.
   Missing or incomplete evidence is not coerced to zero merely to produce a
   verdict.
5. Real-agent integrations should emit the documented OpenTelemetry or
   OpenInference JSON trace and, where supported, the versioned attempt-status
   document. Review the environment-variable contract in
   [closed-loop-comparison.md](closed-loop-comparison.md).

## Alpha limitations

- The vLLM integration surface varies by deployment. Run `doctor` before an
  experiment and treat unknown capabilities as unknown.
- Development-mode cache reset is suitable only for private benchmark servers.
- A unique cache salt provides logical isolation, not a physical reset.
- OpenInference imports may provide structural evidence without exact rendered
  token IDs. Parent-child nesting alone is not treated as an explicit workflow
  dependency.
- Prefixwise exports often contain hashes and counts rather than replayable
  prompts. Experiments generated from them may therefore remain unresolved.
- `compare` requires the agent to export a durable trace file. RunEconomy does
  not yet provide an OTLP network collector, automatic resume, or
  framework-specific tracing SDK.
- The current flagship proof uses one pinned model, backend, hardware class,
  and consumed workload. It is an example of verification behavior, not a
  universal performance claim.

## Privacy boundary

RunEconomy comparison artifacts can contain task inputs, process output,
stderr, evaluator details, and raw traces. Keep the artifacts directory
private unless it has been explicitly reviewed and sanitized.

The bundled public proof contains aggregate counts, intervals, hashes,
provenance, gate results, high-level failure categories, and the claim
boundary. It excludes raw prompts, task answers, model output, stdout/stderr,
credentials, endpoints, local paths, raw traces, and replayable private
workload data.

Credentials should be supplied through environment variables or an ignored
local environment file. Do not place API keys in manifests, traces, reports,
issues, or test fixtures.
