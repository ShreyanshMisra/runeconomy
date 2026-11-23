# Proof 2 preregistration proposal

Status: **proposed and unfrozen**

Proposal date: 2026-11-23

Held-out run: **not executed and not authorized**

This document proposes the Proof 2 contract for approval. It is not a frozen
contract. The evaluator and decision gates remain explicitly unfrozen, the
private proposed tasks have never been sent to either backend, and the Modal
runner exposes only the already completed five-task pilot and 40-task
calibration. No held-out run may begin from this document.

## Proposed claim and intervention

For this pinned Qwen2.5-14B repository-agent workload on vLLM 0.34.0 and L40S
GPUs, test whether online FP8 reduces successful paired workflow latency without
more than a five percentage-point task-success regression.

Both profiles use `Qwen/Qwen2.5-14B-Instruct` revision
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, Transformers 5.18.0,
Hugging Face Hub 2.0.0, and chat-template digest
`sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
The baseline uses BF16 weights and activations. The candidate changes only the
backend quantization identity to vLLM online `fp8_per_tensor`. Both servers run
in one Modal container on separate L40S GPUs. Startup checks require the
candidate's Cutlass FP8 scaled-matrix-multiplication kernel and reject any
undeclared fallback.

The claim is intentionally narrow. It does not generalize to other models,
checkpoints, agents, task families, GPU classes, concurrency, throughput,
routing, or production traffic.

## Calibration and sample-size rationale

The final five-task pilot completed successfully for both profiles. The final
40-task calibration also completed 40/40 successful workflows per profile:

- BF16 mean workflow latency: 1,305.125 ms.
- FP8 mean workflow latency: 923.931 ms.
- Paired FP8-minus-BF16 mean: -381.194 ms.
- Paired 95% interval: [-394.132, -368.256] ms.
- Paired standard deviation: 40.072 ms.

The planning effect is deliberately smaller than the observed calibration
effect: -50 ms. At two-sided alpha 0.05, 80% power, and the calibrated paired
standard deviation, the normal approximation requires six latency pairs. The
proposal instead fixes 180 task pairs—90 unique call-expression tasks and 90
unique string-literal tasks—because quality-gate feasibility and balanced
family coverage are more demanding than latency power.

With 180 attempts per profile, the absolute 95% Clopper-Pearson lower bound
must be at least 0.90, requiring at least 170 successes. The paired exact
candidate-minus-baseline lower bound must be at least -0.05. In the special
case of zero candidate-only successes, the design can tolerate at most three
baseline-only successes; zero, one, and two such discordances require at least
72, 110, and 142 pairs respectively. These are planning properties, not
promises about the eventual outcome.

The proposed private dataset is a deterministic balanced selection from the
untouched 240-task draft pool built at repository commit
`0921c71d9d2dab9b84decd6824e210266b1f9126`. Its dataset digest is
`sha256:3fe04b2d9bd499871a85ddbd4b65b7c39ff88bf72f0571f1323378ddbfb08b94`;
the exact-answer evaluator digest is
`sha256:8de9593491d146f7d81126f911165d85ae9353b13f7fa9f9065a05325d089810`.
The task text and answer map remain private.

## Proposed analysis contract

Each unique task receives exactly one attempt from each profile in randomized,
task-paired order with seed 20261123. There is no replacement.

The primary estimand is candidate-minus-baseline `workflow_latency_ms` among
task pairs where both variants pass the deterministic evaluator. Superiority
requires the upper bound of the paired Student-t 95% confidence interval to be
below zero. Failed tasks are excluded only from that conditional latency
estimand; they remain in both quality gates and the 100% completion gate.

Absolute success uses a two-sided 95% Clopper-Pearson interval with lower bound
at least 0.90. Paired quality uses exact marginal binomial bounds with
Bonferroni coverage and a five percentage-point noninferiority margin. Missing
evidence is invalid or unknown, never zero. A hash mismatch, missing or
duplicate task, incomplete DAG, identity mismatch, evaluator mismatch,
unexpected retry, replacement, partial rerun, or second execution invalidates
the affected claim.

If approved and frozen, the held-out sample stops after exactly 180 pairs. The
run will not be extended after a favorable result, rerun after a negative or
invalid result, partially rerun, or repaired by replacing tasks. Any
infrastructure failure therefore produces an invalid held-out result rather
than permission for another attempt.

## Reporting commitment

The result package will prepare the applicable narrative regardless of outcome:

1. **Verified:** latency superiority and both quality gates pass; report the
   narrow claim, intervals, failures, cost, and complete provenance.
2. **Rejected or invalid:** a quality, completion, identity, or evidence gate
   fails; report the failed gate and do not claim a speedup.
3. **Mechanism-only or inconclusive:** quality passes but the conditional
   latency interval does not establish superiority; report measurements without
   a deployment recommendation.

External publication remains a separate approval boundary even after analysis.

## Cost, provenance, and approval boundary

All pilot and calibration iterations cost $2.00556691 in total, including
L40S, CPU, memory, and network charges. That is below the $5 calibration stop
limit. The final pilot archive digest is
`sha256:8af23d85eb918e215551e6c0e7e1ab85b9100515ec12aa2a1e2cb98027237057`;
the calibration archive digest is
`sha256:3b00f9514e12876c5a8c2ea2eeeec0a2f01da4c8c03529e236516c91878b120c`.
Zero held-out compute has been used.

Explicit approval is required before either freeze declaration changes, before
any held-out runner is enabled or invoked, and again before external
publication. Until then, the contract stays proposed and unfrozen.
