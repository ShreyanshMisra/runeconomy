# Experiment contract

An experiment workspace contains a versioned `workspace.json`, an execution
`manifest.json`, `intervention.md`, and a short `README.md`. Source evidence is
referenced by name and digest rather than copied into the workspace.

Validation statuses are `runnable`, `unresolved`, `invalid`,
`unsupported_backend`, and `claim_evidence_incomplete`.

Each issue identifies the artifact, field, reason, execution impact, claim
impact, and next action. For an unresolved static workspace, supply the missing replay context or
evaluator in the source inputs, re-import, and initialize a fresh directory.
Patching an unresolved skeleton cannot make it runnable.

Before execution, freeze calibration and held-out boundaries, the intervention,
evaluator, metrics, quality gates, sample size or stopping rule, trial order,
cache-isolation mode, environment identity, cost ceiling, and interruption
policy.

Agent processes must export durable trace and status files using
`RUNECONOMY_TRACE_PATH` and `RUNECONOMY_STATUS_PATH`. Classify task failure,
model-output failure, application failure, infrastructure failure, timeout,
evaluator failure, and invalid trace separately. Failed attempts must not vanish
from latency accounting.

Use `runeconomy.agent.Attempt` only in a runner-specific wrapper or the
`RUNECONOMY_STATUS_PATH` branch of the existing agent entry point. It reads the
runner environment, exposes typed paths and attempt identity, and writes one
atomic schema `0.1` terminal status. Call exactly one of `completed`,
`task_failure`, `model_output_failure`, `application_failure`, or
`infrastructure_failure`. Pass only fixed content-free `stage` and `code`
labels. Never pass an exception message or runtime data. An unhandled exception
is recorded conservatively as `application_failure` and is re-raised. A clean
exit without a declared terminal status is an integration error.

Do not hand-write status JSON. Do not replace existing OpenTelemetry or
OpenInference instrumentation. External RunEconomy timeouts still override an
application status.

`runeconomy experiment plan WORKSPACE` is read-only and separate from
validation. Validation checks execution structure. Planning checks pairing,
quality-gate feasibility, performance metric alignment, sample assumptions,
and calibration versus held-out declarations. A planning state is not an
experiment verdict. Run it before consuming held-out data or paid compute.

With `runeconomy.track_workflow`, declare actual predecessors using
`runeconomy.step(..., depends_on=[...])`. A worker thread without a workflow
context can fall back to one active tracker. With multiple active trackers,
propagate the intended context or the worker raises `WorkflowTraceError`.

## Change and freeze a reviewed workspace

Preview a design change on an unfrozen runnable workspace with a documented
reason. Use the floor specified for the experiment in place of the example.

```bash
runeconomy experiment patch experiment/ --set acceptance.min_task_success_rate=0.9 \
  --reason "use the declared quality floor" --dry-run --json
```

Review the preview, then repeat without `--dry-run` to apply the change.
`experiment patch` validates it and appends the change to `provenance.jsonl`.
Use this command for manifest changes. For a prospective design, choose the
paired method before execution (`paired_exact` is the default,
`paired_score_exact` is available), declare how `completion_policy` treats
shared failures, and pin a `cost_model` when gating on cost. Review the
planner’s sample size, power and assumptions before freezing.

```bash
runeconomy experiment plan experiment/ --json
runeconomy experiment freeze experiment/ --lock agent.py --json
runeconomy experiment run experiment/ --output results/ --require-frozen --json
```

Repeat `--lock FILE` for additional agent or runner code. The freeze pins
workspace files and the agent-process dataset and evaluator. A changed frozen
design needs a new workspace. Obtain execution approval before the run.

The legacy `compare` command may exist in older releases. For new work, use
`runeconomy experiment init --mode agent-process` followed by `experiment run`.
