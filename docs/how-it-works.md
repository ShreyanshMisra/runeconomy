# How RunEconomy works

RunEconomy connects application behavior to inference measurements and refuses
to verify a speedup unless task-quality evidence supports the comparison.

## 1. Trace

Instrument a real agent with OpenTelemetry/OpenInference or import evidence from
supported vLLM and Prefixwise exports. Preserve explicit workflow dependencies,
variant-local state, and unknown fields.

## 2. Diagnose

Analyze repeated prompt prefixes and inference behavior in the context of the
workflow. A finding records its evidence quality, affected requests,
critical-path status, optimization target, required capabilities, and possible
experiment template.

## 3. Experiment

Turn a supported finding into a controlled baseline/candidate manifest. If the
application change cannot be generated safely, RunEconomy emits an unresolved
skeleton instead of pretending it is runnable.

## 4. Gate

Execute paired trials with frozen identities, ordering, metrics, task evaluators,
and acceptance gates. Task success and completion are evaluated before
successful-pair latency. Failures remain visible rather than disappearing from
the latency sample.

## 5. Verdict

Return `verified`, `regression`, `inconclusive`, or `invalid`, together with the
decisive gate, provenance, intervals, and conditional mechanism measurements.
The result answers whether the proposed change improved successful-task
performance—not merely whether a cache counter moved.
