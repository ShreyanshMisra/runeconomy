# Real-agent compatibility trial

This fixture is a methodology and integration-contract trial, not a public
performance benchmark. A minimal agent makes one real planning inference,
reads a real repository excerpt, and makes a second inference that consumes the
plan and tool output. OpenTelemetry SDK spans use OpenInference attributes and
explicit `runeconomy.depends_on` edges.

The baseline places per-attempt metadata before stable instructions. The
candidate moves the same metadata after the stable instructions. No semantic
content is removed. Two calibration tasks are separate from the six frozen
held-out tasks. The held-out run uses three repetitions, giving 18 paired
workflows and 72 inference calls.

The task JSON does not contain answers. `evaluator.py` owns a deterministic,
case-sensitive answer key. Raw task files, generated output, stderr, and OTLP
traces remain in the ignored artifacts directory.

The live command was recorded before execution:

```bash
runeconomy compare \
  --baseline "python examples/compatibility/agent.py --metadata-position prefix" \
  --candidate "python examples/compatibility/agent.py --metadata-position suffix" \
  --dataset examples/compatibility/tasks-heldout.json \
  --evaluator examples/compatibility/evaluator.py \
  --output artifacts/compatibility/result.json \
  --artifacts-dir artifacts/compatibility/run \
  --workdir . \
  --repetitions 3 \
  --order randomized \
  --seed 20261113 \
  --max-task-success-regression 0.20 \
  --min-task-success-rate 0.80 \
  --min-completed-rate 1.0
```

The comparison is valid only when all attempts have complete explicit DAGs and
complete, stable model/backend/tokenizer/template identity. Unknown serving
metrics remain null. The primary metric is paired successful-task process wall
time; cached tokens and TTFT are mechanisms.

The agent writes status schema `0.1` to `RUNECONOMY_STATUS_PATH`. Planning or
answer-format errors are `model_output_failure`; vLLM request failures are
`infrastructure_failure`; other agent exceptions are `application_failure`.
The record contains only stable component, stage, and code labels. Model text
and exception details remain in private attempt artifacts.

## First frozen result

The first held-out execution returned `invalid`, correctly. Five baseline
processes exited before producing complete traces, leaving baseline completion
and explicit-DAG coverage at 13/18 while the candidate completed 18/18.
Absolute task-success and paired non-inferiority gates also failed. RunEconomy
excluded those missing attempts from latency effects rather than allowing
failures to make a variant appear faster.

Among the six pairs where both tasks succeeded, the candidate reused more
cached input and had lower server TTFT but was slower end to end. That is a
mechanism observation, not a valid performance claim, because the overall
trial was invalid and the successful-pair subset was small. Raw traces,
generated output, stderr, endpoint details, and the complete result remain in
ignored private artifacts. This held-out set has now been observed and must not
be retuned or rerun as fresh evidence.

Any reuse of these six tasks after adding the status protocol is engineering
validation only. It can show that previously ambiguous process exits are now
classified and retain partial telemetry; it cannot create a new held-out
performance claim.

## Status-protocol engineering check

The six observed tasks were rerun once for engineering validation after schema
`0.1` was added. All 36 attempts wrote a valid status record, retained a valid
explicit DAG, and preserved complete identity evidence. Five baseline exits
that would previously have appeared as generic process failures were correctly
classified as `model_output_failure`. Three separate vLLM HTTP 503 failures
were classified as `infrastructure_failure`. No attempt produced
`invalid_status`, and no verbatim model output appeared in the shareable
result.

The check also exposed and fixed an aggregation defect: a one-request partial
trace and a two-request completed trace with the same pinned identity had been
mistaken for identity drift because the request counts differed. Identity is
now compared by the observed identity values rather than duplicate-list
length. The corrected engineering verdict remains `invalid` because genuine
infrastructure failures occurred and the workload did not meet the success
gate. No performance claim is attached to this rerun.
