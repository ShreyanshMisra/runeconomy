# Complete agent-process experiments

The `agent-process` experiment mode executes real baseline and candidate agent
processes. One
process receives one task and owns its complete workflow state, so later agent
steps can consume earlier model outputs, tool results, retrieved documents, and
files exactly as the agent normally implements them.

It uses the same initialization, validation, execution, and report workflow as
static request experiments. Agent-process mode includes process and tool time,
variant-local state, task evaluation, and imported workflow telemetry.

```bash
runeconomy experiment init --mode agent-process \
  --baseline "python agent_v1.py" \
  --candidate "python agent_v2.py" \
  --dataset tasks.json \
  --evaluator evaluator.py \
  --output experiment/ \
  --repetitions 3 \
  --order randomized \
  --seed 2027

runeconomy experiment validate experiment/
runeconomy experiment plan experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

`runeconomy compare` remains a compatibility wrapper for existing automation.
It emits a migration notice. New integrations should use the workspace workflow.

Agent-process manifest schema `0.2` adds the explicit optimization target,
latency semantics, and latency margin. Schema `0.1` remains readable and maps
to successful-workflow latency superiority with a zero-millisecond margin. No
legacy artifact is silently reinterpreted as an engine-change experiment.

This command executes local code. Review the agent and evaluator commands
before running them. RunEconomy does not invoke a shell: command strings are
split into argument arrays, so pipes, redirects, substitutions, and shell
built-ins are not interpreted.

## Experiment stages

- A **compatibility trial** checks whether an agent can satisfy the process,
  status, trace, dependency, identity, and evaluator contract. It is not a
  benchmark.
- A **calibration workload** is available for prompt, agent, evaluator, and
  infrastructure development. It cannot later become held-out evidence.
- A **frozen held-out proof** fixes code, workload, evaluator, gates, order,
  environment, and sample size before one execution. Once observed, that
  workload cannot be reused as fresh held-out evidence.

The current frozen 40-task proof returned `invalid`. Cache reuse, TTFT, and
successful-pair latency were favorable conditional observations, but the
baseline absolute-success lower bound and paired task-success non-inferiority
bound failed their predeclared gates. It is not a verified improvement and
must not be presented as one.

## Engine-change latency target

Use the explicit engine target for quantization, speculative decoding, vLLM
upgrades, scheduler or kernel changes, parallelism, and other serving changes:

```bash
runeconomy experiment init --mode agent-process \
  --baseline "python agent.py --backend baseline" \
  --candidate "python agent.py --backend candidate" \
  --dataset tasks.json \
  --evaluator evaluator.py \
  --optimization-target engine-change-latency \
  --latency-semantics superiority \
  --output experiment/
```

This target is latency-focused, not quality-blind. The eligible latency
population is task pairs where both variants succeeded. Failed, timed-out,
missing, or unevaluable attempts never enter the favorable latency population.
they remain visible in completion and quality gates. Baseline and candidate
must each have one complete stable backend identity, and those identities must
differ. An identity mutation within either variant invalidates the result.

`superiority` verifies only when the upper bound of the paired 95% Student-t
interval for candidate-minus-baseline wall time is below zero. For a
predeclared non-inferiority question, use `--latency-semantics noninferiority`
and a scientifically justified `--max-latency-regression-ms`. The same upper
bound must be below that margin. A missing interval is inconclusive, not a
pass.

The experiment verifies only the declared workload, evaluator, identities,
hardware, execution order, and gates. It does not establish universal model
quality, production-concurrency behavior, or superiority on failed tasks.

## Dataset

The dataset is either a JSON array or an object with a `tasks` array. Every
task must have a unique non-empty `task_id`. All other fields belong to the
agent and evaluator.

```json
{
  "tasks": [
    {
      "task_id": "issue-001",
      "prompt": "Find and fix the parsing defect.",
      "expected_test": "tests/test_parser.py"
    }
  ]
}
```

## Agent contract

RunEconomy sets these variables for each isolated attempt:

- `RUNECONOMY_TASK_PATH`: absolute path to that attempt's task JSON.
- `RUNECONOMY_TASK_ID`, `RUNECONOMY_VARIANT`, and
  `RUNECONOMY_REPETITION`.
- `RUNECONOMY_SEED`.
- `RUNECONOMY_ATTEMPT_DIR`: private directory for variant-local workflow state.
- `RUNECONOMY_OUTPUT_PATH`: where the agent may write its final output. If it
  does not, captured standard output becomes the final output.
- `RUNECONOMY_TRACE_PATH`: where the agent must write one OpenTelemetry OTLP
  JSON trace after the workflow ends.
- `RUNECONOMY_STATUS_PATH`: where a status-aware agent atomically writes the
  versioned attempt-status record described below.

Use the supported runtime helper at the application's existing entry point:

```python
from runeconomy.agent import Attempt

with Attempt.from_environment() as attempt:
    result = run_agent(attempt.task_path)
    attempt.output_path.write_text(result, encoding="utf-8")
    attempt.completed()
```

`Attempt` validates the runner environment and exposes typed task, output,
trace, status, attempt-directory, seed, variant, repetition, and task-ID
values. It writes exactly one atomic terminal status without overwriting a
preexisting status file. Existing agents may continue using
`write_attempt_status` directly, and legacy agents without a status record
remain supported.

Use fixed content-free labels for repository-classified failures:

```python
attempt.task_failure(stage="evaluation", code="incorrect_answer")
attempt.model_output_failure(stage="planning", code="invalid_structured_output")
attempt.application_failure(stage="execution", code="repository_error")
attempt.infrastructure_failure(stage="model_request", code="upstream_unavailable")
```

An unhandled exception records the conservative code `unhandled_exception` as
an application failure, then the original exception is re-raised. A normal
context exit without a terminal method records an integration failure and
raises `AttemptLifecycleError`. Call `completed()` only after output and trace
writing have succeeded.

### Attempt status protocol

Status schema `0.1` distinguishes `completed`, `task_failure`,
`model_output_failure`, `application_failure`, `infrastructure_failure`, and
`timeout`. Agents predating the protocol remain supported: a zero exit follows
the evaluator path and a nonzero exit remains an infrastructure failure.

A completed record is:

```json
{"schema_version":"0.1","status":"completed"}
```

Every failure record must include content-free provenance:

```json
{
  "schema_version": "0.1",
  "status": "model_output_failure",
  "provenance": {
    "component": "model_output",
    "stage": "planning",
    "code": "invalid_structured_output"
  }
}
```

`component`, `stage`, and `code` are bounded stable labels. Free-form messages,
prompts, generated text, exception strings, URLs, and credentials are not part
of the schema. `runeconomy.attempt_status.write_attempt_status` validates and
atomically writes the record. RunEconomy retains the private file and exposes
only its validated labels and digest in the comparison result.

An external runner timeout overrides any status file. A nonzero exit with a
declared failure preserves that classification. A nonzero exit declaring
`completed`, or a malformed status record, becomes `invalid_status`. Valid
partial traces are imported for failed attempts when present. Model-output,
application, and task failures count as unsuccessful tasks, while true
infrastructure failures and timeouts remain unevaluable and invalidate the
relevant completion gate.

The agent runs once for the whole task. RunEconomy does not emulate steps or
copy baseline state into the candidate. Tools, retrieval, retries, subagents,
and generated intermediate state therefore remain under the real agent's
control and are included in measured process wall time.

The trace importer recognizes OpenInference LLM, tool, and retriever spans. To
establish a complete workflow DAG and calculate critical-path latency, every
retained work span must include a `runeconomy.depends_on` attribute containing
a JSON array of predecessor span IDs. Root spans use `[]`. This is a
RunEconomy extension, not an OpenTelemetry or OpenInference standard. Without
complete declarations, workflow wall time remains measured but critical-path
latency is reported as unavailable.

Optional inference attributes include:

- `runeconomy.prompt_token_ids`: exact rendered prompt tokens.
- `runeconomy.cost_usd` and `runeconomy.gpu_time_ms`.
- OpenInference token counts and standard/provider timing attributes described
  in the normalized trace documentation.

A non-invalid verdict also requires every inference span to identify its model,
model revision, backend, backend version, tokenizer, tokenizer revision, and
chat-template digest. RunEconomy accepts the standard model/provider attributes
where defined and the `runeconomy.model_revision`,
`runeconomy.backend_version`, `runeconomy.tokenizer`,
`runeconomy.tokenizer_revision`, and `runeconomy.chat_template_digest`
extensions. Identity must remain stable within each variant. Baseline and
candidate identities may differ because changing the model or serving stack can
be the declared intervention. Both identities remain visible in the result.

## Evaluator contract

A `.py` evaluator reads one JSON object from standard input and writes one JSON
boolean or result object to standard output:

```python
import json
import sys

request = json.load(sys.stdin)
success = request["output_text"].strip() == request["task"]["expected"]
json.dump({"success": success, "score": float(success), "details": {}}, sys.stdout)
```

The input contains `output_text`, the complete `task`, `task_id`, `variant`,
`repetition`, and `attempt_id`. Evaluator stdout and stderr remain in the local
attempt artifacts. A `.json` file may instead contain any existing RunEconomy
evaluator specification.

## Pairing and verdicts

Before execution, run the separate read-only design check:

```bash
runeconomy experiment plan experiment/
runeconomy experiment plan experiment/ --json
```

`experiment validate` answers whether the workspace is structurally runnable.
`experiment plan` checks claim-design readiness. It reads local files only and
does not start an agent or model, contact a backend, consume tasks, reset a
cache, or modify the workspace. Its states are `blocked`,
`needs_design_input`, `ready_for_calibration`, and `ready_to_freeze`. These are
planning states, not experiment verdicts.

Planning reports unique tasks, repetitions, planned pairs, total attempts,
statistical units, exact quality-gate feasibility, the primary performance
metric, maximum both-successful latency pairs, missing calibration assumptions,
and operator-declared held-out status. Repetitions add task-attempt units but
are not described as additional independent tasks.

The paired-quality section also reports the minimum sample needed to tolerate
zero, one, or two baseline-only outcomes when there are no offsetting
candidate-only outcomes. A 40-pair design with a 10-point margin is explicitly
flagged as fragile because one baseline-only outcome cannot clear the
conservative exact bound. This is the failure mode observed in the frozen
40-task proof. A best-case feasibility check alone is not a sizing argument.

Order is seeded and either randomized or alternating. Pairing uses
`(task_id, repetition)`. Task success uses absolute Clopper-Pearson intervals
and the existing conservative paired exact non-inferiority bound. The default
absolute-success gate requires the lower confidence bound to reach 80%. Change
it only as a predeclared property of the workload.

The primary performance outcome is end-to-end agent process wall time. Its
paired Student-t interval includes only pairs where both baseline and candidate
succeeded. Consequently, a crash, timeout, invalid trace, or failed task can
never improve the candidate's latency sample by disappearing from it.

The planner also translates the completion-rate threshold into an exact
integer failure tolerance. A 100% threshold reports that zero failed or
incomplete attempts are tolerated. Lower thresholds report the exact minimum
completed count and maximum failure count for the planned sample.

Verdicts are:

- `verified`: successful-task latency improved and task success was
  non-inferior.
- `regression`: candidate completion or task quality failed a guardrail, or
  the paired workflow-latency interval establishes that successful tasks are
  slower.
- `inconclusive`: quality passed but uncertainty did not establish both
  non-inferiority and a latency improvement.
- `invalid`: baseline execution, trace evidence, or evaluator evidence could
  not support the comparison.

Decision schema `0.2` preserves this verdict field while separately recording
evidence validity, execution terminality, paired task outcomes, supported
failure origin, and performance eligibility. Concordant paired failures are
reported as common-mode diagnostics without claiming a causal origin. See
[Decision artifact semantics](decision-artifacts.md).

Statuses remain separate per attempt: `completed`, `task_failure`,
`model_output_failure`, `application_failure`, `timeout`,
`infrastructure_failure`, `invalid_status`, `invalid_trace`, and
`evaluator_infrastructure_error`.

Agent-process runs use `attempts.jsonl` as a durable checkpoint. Re-run
`experiment run` with `--resume` after an operator interruption. RunEconomy
revalidates commands, dataset, evaluator, schedule, gates, and every completed
record before continuing. A mismatched or reordered checkpoint is rejected.

## Evidence and sensitive data

The JSON result stores digests (including digested task IDs), metrics, statuses,
uncertainty, provenance, and relative artifact locations, not task contents,
generated output, stderr, or evaluator detail content. Full evaluator details
are stored in the attempt's local
`evaluation.json`. The result contains their digest. The artifacts
directory stores the task, output, stdout, stderr, raw trace, normalized trace,
evaluation, and evaluator streams for auditability. These files may contain prompts,
repository content, secrets printed by an application, or model output. Keep
the directory private and sanitize it before sharing.

The result reports workflow wall time, available critical-path time, inference,
tool and retrieval span time, input/output/cached tokens, queue/prefill/decode
timing, and cost or GPU time when the trace supplies them. Missing telemetry is
null rather than estimated.
