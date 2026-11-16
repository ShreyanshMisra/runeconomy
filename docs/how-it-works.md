# How RunEconomy works

RunEconomy connects application behavior to inference measurements and refuses
to verify a speedup unless task-quality evidence supports the comparison.

## 1. Import evidence

Use `runeconomy import` with OpenTelemetry or OpenInference, vLLM capture,
Prefixwise ledger, or Phoenix trace evidence. The resulting capability
assessment explains what is known and what remains unavailable.

## 2. Diagnose

Analyze repeated prompt prefixes and inference behavior in the context of the
workflow. A finding records its evidence quality, affected requests,
critical-path status, optimization target, required capabilities, and possible
experiment template.

## 3. Initialize and validate

Turn a supported finding into a small experiment workspace. Validation reports
whether it is runnable, unresolved, invalid, unsupported by the backend, or
missing evidence for the intended claim.

## 4. Run and gate

Execute paired trials with frozen identities, ordering, metrics, task evaluators,
and acceptance gates. Task success and completion are evaluated before
successful-pair latency. Failures remain visible rather than disappearing from
the latency sample.

## 5. Report the verdict

Return `verified`, `regression`, `inconclusive`, or `invalid`, together with the
decisive gate, provenance, intervals, and conditional mechanism measurements.
The result answers whether the proposed change improved successful-task
performance, not merely whether a cache counter moved.
