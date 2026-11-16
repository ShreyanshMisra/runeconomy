# Experiment contract

An experiment workspace contains a versioned `workspace.json`, an execution
`manifest.json`, `intervention.md`, and a short `README.md`. Source evidence is
referenced by name and digest rather than copied into the workspace.

Validation statuses are `runnable`, `unresolved`, `invalid`,
`unsupported_backend`, and `claim_evidence_incomplete`.

Each issue identifies the artifact, field, reason, execution impact, claim
impact, and next action. Fix only the identified field. A skeleton with
unresolved replay context or evaluator logic is not runnable.

Before execution, freeze calibration and held-out boundaries, the intervention,
evaluator, metrics, quality gates, sample size or stopping rule, trial order,
cache-isolation mode, environment identity, cost ceiling, and interruption
policy.

Agent processes must export durable trace and status files using
`RUNECONOMY_TRACE_PATH` and `RUNECONOMY_STATUS_PATH`. Classify task failure,
model-output failure, application failure, infrastructure failure, timeout,
evaluator failure, and invalid trace separately. Failed attempts must not vanish
from latency accounting.

The legacy `compare` command may exist in older releases. For new work, use
`runeconomy experiment init --mode agent-process` followed by `experiment run`.
