# Proof 2 calibration agent

This calibration-only agent compares one pinned Qwen2.5-14B checkpoint served
as BF16 and vLLM online FP8. Both variants receive identical messages,
generation parameters, tool evidence, and evaluators. Distinct served-model
names truthfully encode the deliberate backend configuration difference while
the source model and tokenizer revision remain identical.

The workload builder creates disjoint pilot, calibration, and draft held-out
task pools. Raw tasks, answer keys, outputs, traces, server logs, and archives
remain under ignored `artifacts/`. The draft held-out pool is not frozen and
must not be executed without a new explicit approval.
