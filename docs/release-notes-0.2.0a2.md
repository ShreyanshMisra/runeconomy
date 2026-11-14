# RunEconomy 0.2.0a2 release-note draft

RunEconomy `0.2.0a2` is a private-tester alpha focused on trustworthy first use.

## What changed

- The frozen rejection proof is now a deterministic public case study linked
  directly from the README and bundled report.
- `doctor` now gives a clear readiness verdict, capability table, safe
  experiment modes, exact next actions, structured error JSON, and documented
  exit codes without invoking inference or cache reset.
- Common CLI failures explain what failed, why, whether output may be
  incomplete, and the next corrective action.
- A canonical quickstart connects installation, the proof, compatibility
  diagnosis, trace analysis, `experiment init`, and manifest validation.
- The official RunEconomy coding-agent skill documents safe inspection,
  evidence acquisition, experiment design, comparison, verdict semantics, and
  privacy boundaries.
- A small static documentation site can be built locally from repository
  Markdown sources for later Vercel deployment.

## Flagship proof

The candidate improved cache reuse, server TTFT, and successful-pair workflow
latency, but RunEconomy returned `invalid` because the frozen task-quality gates
did not pass. This release does not change or rerun that consumed experiment.

## Alpha limitations

RunEconomy remains a CLI for invited testers. There is no hosted dashboard,
managed OTLP collector, framework-specific SDK, automatic resume, SGLang
runner, or broad diagnosis catalog. Operators must protect raw prompts, traces,
model output, evaluator data, private endpoints, and experiment artifacts.

The documentation site is prepared locally but is not part of this package
release and is not deployed automatically.
