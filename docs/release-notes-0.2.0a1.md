# RunEconomy 0.2.0a1 release-note draft

RunEconomy `0.2.0a1` is an invited-tester alpha focused on one closed loop:
connect inference evidence to a controlled change, execute paired trials, and
gate the verdict on both task success and performance.

Highlights:

- Read-only `doctor` capability diagnostics for normal vLLM deployments.
- Chat Completions and Responses support, with explicit cache-isolation modes.
- Structured findings and `experiment init` manifest or skeleton generation.
- Evidence-graded OpenInference/OpenTelemetry and Prefixwise imports.
- `compare` for real baseline/candidate agents with variant-local state,
  explicit failure classes, DAG/critical-path reconstruction, and paired
  quality and latency statistics.
- A versioned attempt-status protocol for durable partial evidence.
- A new default NegativeSplit demo in which cache reuse, TTFT, and conditional
  successful-task latency improved, but RunEconomy returned `invalid` because
  the frozen task-quality gates failed.

This alpha does not include a hosted dashboard, OTLP collector, automatic
resume, framework-specific SDK, or broad diagnosis expansion. Raw comparison
artifacts may contain sensitive task and process data and should remain private
until explicitly sanitized.

See `docs/alpha-release.md` for installation, migration notes, limitations,
and the complete privacy boundary.
