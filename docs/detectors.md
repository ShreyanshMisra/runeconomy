# Detectors and experiment templates

`runeconomy analyze` and `runeconomy import` run every detector. A finding is
a hypothesis attached to an experiment template, never a verdict. The
template's paired experiment confirms or refutes the predicted effect while
the task-quality gates guard the change. Each detector states the evidence it
needs and stays silent when that evidence is unknown.

| Rule | Evidence required | Template | Predicted effect |
| --- | --- | --- | --- |
| `prefix_divergence.*` | exact prompt tokens | `prompt.*`, `history.preserve_prefix`, `retrieval.stabilize_order` | cached tokens increase |
| `cache.unrealized_eligibility` | exact tokens and cache telemetry | `cache.isolated_warm_replay` | cached tokens increase |
| `tools.unused_exposure` | tool names and segments | `tools.scope_allowlist` | input tokens decrease |
| `context.tool_output_carryover` | exact tokens and `tool_results` token boundaries | `context.compact_tool_output` | input tokens decrease |
| `workflow.serialized_independent_steps` | `complete_explicit` dependency edges | `workflow.parallelize_independent_steps` | workflow latency decreases |
| `models.step_overprovisioning` | model size in the model name and per-step output tokens | `models.route_step_to_smaller_model` | workflow latency decreases |
| `output.structured_step_inflation` | output tokens and invoked tools | `output.constrain_structured_step` | output tokens decrease |
| `serving.kv_pressure` | sampled server metrics from a load run (`analyze result.json`) | `serving.reduce_kv_pressure` | workflow latency decreases |

## Workflow waste detectors

**Tool-output carry-over.** This fires when a tool result of at least 256
tokens is re-sent token for token in two or more later requests, with at
least 1,024 carried tokens in total. It reports how many carried tokens the
reported prefix cache did not cover. If cache telemetry is missing, that
figure is unknown.

**Serialized independent steps.** This fires when two steps with no declared
path between them ran back to back and the observed span exceeds the
dependency-DAG critical path by at least 50 ms and 10%. The gap is an upper
bound on what parallelism could save if the declared edges are complete, so
the detector runs only on `complete_explicit` dependency evidence (for
example traces written with `runeconomy.track_workflow`).

**Step over-provisioning.** This fires when steps on a model of at least 7B
parameters produce at most 32 output tokens while other steps in the same
workflow produce longer output. It is a heuristic with low confidence: short
output suggests routing, extraction or formatting, but does not prove the step
is simple. When the model name carries no parameter count, the detector stays
silent.

**Structured-step inflation.** This fires when a step that ended in a tool
call generated at least 256 output tokens. A tool call itself needs few.
Verbose preambles or reasoning tokens add decode time on every such step.

**KV pressure.** This fires when measured server telemetry from a load block
shows preemptions, or a KV cache at or above 95% usage with a waiting queue.
Collect the telemetry with `experiment run --server-metrics` or
`--server-metrics-url`, then run `runeconomy analyze result.json`.

## Templates

These templates change application or serving behavior. RunEconomy
records them in an honest unresolved skeleton when it cannot generate the
candidate. Implement the candidate yourself and run it as an agent-process
experiment with the same dataset and evaluator.

| Template | Candidate change | Quality risk |
| --- | --- | --- |
| `context.compact_tool_output` | Carry earlier tool results forward as a summary or reference | Later steps may need the removed detail |
| `workflow.parallelize_independent_steps` | Run steps without a declared dependency concurrently | An undeclared dependency surfaces as failure or nondeterminism |
| `models.route_step_to_smaller_model` | Route the identified short-output steps to a smaller model | Misrouting or misformatting |
| `output.constrain_structured_step` | Structured output, a tight `max_tokens`, or reasoning disabled on tool-call steps | Truncated or degraded tool calls |
| `serving.reduce_kv_pressure` | Lower `max_num_seqs`, raise the KV budget, use an FP8 KV cache, or shorten `max_model_len` | Engine changes such as an FP8 KV cache can change outputs |

Fixtures for every detector live in `tests/fixtures/detectors/`.

## Validation experiments

Each detector's predicted effect was tested by a paired experiment. The
results, including refutations, are recorded in
[detector validation](detector-validation.md).
