# Proof 2 preregistration proposal

Status: **freeze-ready, proposed, and unfrozen**

Proposal date: 2026-11-23

Held-out run: **not executed and not authorized**

This document proposes the Proof 2 contract for approval. It is not a frozen
contract. The evaluator and decision gates remain explicitly unfrozen, the
private proposed tasks have never been sent to either backend, and the Modal
runner exposes only the completed five-task pilot, 40-task calibration, and a
digest-locked crossover validation on those same consumed calibration tasks.
No 180-task entrypoint exists, and no held-out run may begin from this document.

## Proposed claim and intervention

For this pinned Qwen2.5-14B repository-agent workload on vLLM 0.34.0 and L40S
GPUs, test whether online FP8 reduces successful paired workflow latency without
more than a five percentage-point task-success regression.

Both profiles use `Qwen/Qwen2.5-14B-Instruct` revision
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, Transformers 5.18.0,
Hugging Face Hub 2.0.0, and chat-template digest
`sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
The baseline uses BF16 weights and activations. The candidate changes only the
backend quantization identity to vLLM online `fp8_per_tensor`. Both profiles run
in one Modal container on two distinct physical L40S GPUs. Startup checks
require the candidate's Cutlass FP8 scaled-matrix-multiplication kernel and
reject any undeclared fallback.

The held-out design removes permanent profile-to-device assignment as a
confound. The 180 pairs are split deterministically into two 90-pair blocks.
Each block contains 45 call-expression and 45 string-literal tasks. In block A,
BF16 uses physical device 0 and FP8 uses device 1. Both servers are then stopped
and block B starts fresh servers with BF16 on device 1 and FP8 on device 0.
Two distinct L40S UUIDs and PCI bus IDs must be observed before block A; those
raw hardware identifiers remain private.

The claim is intentionally narrow. It does not generalize to other models,
checkpoints, agents, task families, GPU classes, concurrency, throughput,
routing, or production traffic.

## Calibration and sample-size rationale

The final five-task pilot completed successfully for both profiles. The
original 40-task calibration completed 40/40 successful workflows per profile.
The crossover runner was then validated only on those already-consumed 40
tasks, split into two 20-task blocks with 10 tasks from each family per block.
It again completed 40/40 successful workflows per profile:

- BF16 mean workflow latency: 1,307.629 ms.
- FP8 mean workflow latency: 926.723 ms.
- Pooled paired FP8-minus-BF16 mean: -380.907 ms.
- Pooled paired 95% interval: [-393.889, -367.925] ms.
- Paired standard deviation: 40.208 ms.
- Block A descriptive mean: -388.590 ms.
- Block B descriptive mean: -373.224 ms.
- Call-expression descriptive mean: -382.601 ms.
- String-literal descriptive mean: -379.212 ms.

Both validation blocks recorded exactly two fixed schema warm-ups and one
successful prefix-cache reset per profile before any measurement. The device
assignment swapped and two distinct physical L40S devices were verified. The
private validation archive digest is
`sha256:baa1e7af3885d7dcd747ce7166b0757c94bac3f43eba608ca489976063830957`.
This remains consumed calibration evidence, not held-out evidence.

The planning effect is deliberately smaller than the observed calibration
effect: -50 ms. At two-sided alpha 0.05, 80% power, and the crossover-calibrated
paired standard deviation, the normal approximation requires six latency
pairs. The proposal instead fixes 180 task pairs—90 unique call-expression
tasks and 90 unique string-literal tasks—because quality-gate feasibility,
balanced family coverage, and balanced physical-device exposure are more
demanding than latency power.

With 180 attempts per profile, the absolute 95% Clopper-Pearson lower bound
must be at least 0.90, requiring at least 170 successes. The paired exact
candidate-minus-baseline lower bound must be at least -0.05. In the special
case of zero candidate-only successes, the design can tolerate at most three
baseline-only successes; zero, one, and two such discordances require at least
72, 110, and 142 pairs respectively. These are planning properties, not
promises about the eventual outcome.

The proposed private dataset is a deterministic balanced selection from the
untouched 240-task draft pool built at repository commit
`0921c71d9d2dab9b84decd6824e210266b1f9126`. Its crossover-assigned dataset
digest is
`sha256:c945014bc7aed8cd6164b9c12da060a05880bea35a2ed29850e27612aa9924b2`;
the exact-answer evaluator digest is
`sha256:8de9593491d146f7d81126f911165d85ae9353b13f7fa9f9065a05325d089810`.
The task text and answer map remain private.

## Measurement-window procedure

Before each block, both profiles receive the same two fixed structured warm-up
requests—one plan schema and one answer schema—followed by exactly one prefix
cache reset per profile. No measured request occurs before the reset. Block A
servers must fully exit before the fresh, device-swapped block B servers start.

Each unique task receives exactly one attempt from each profile. Block
assignment is deterministic and frozen; task-paired attempt order is randomized
within each block with seed 20261123. There is no task replacement.

## Proposed analysis contract

The primary estimand is candidate-minus-baseline `workflow_latency_ms` among
task pairs where both variants pass the deterministic evaluator. Superiority
requires the upper bound of the paired Student-t 95% confidence interval pooled
across both blocks to be below zero. The pooled test is the sole confirmatory
latency analysis; it has no block or device adjustment. Failed tasks are
excluded only from that conditional latency estimand; they remain in both
quality gates and the 100% completion gate.

Prespecified descriptive sensitivity output reports the paired latency effect
separately for each task family and crossover block, plus successful-workflow
latency for each block/device/profile cell. These outputs have no additional
acceptance threshold, receive no confirmatory interpretation, and cannot
overturn or rescue the pooled primary decision.

Absolute success uses a two-sided 95% Clopper-Pearson interval with lower bound
at least 0.90. Paired quality uses exact marginal binomial bounds with
Bonferroni coverage and a five percentage-point noninferiority margin. Missing
evidence is invalid or unknown, never zero. A hash mismatch, missing or
duplicate task, incomplete DAG, identity mismatch, evaluator mismatch, GPU
identity or assignment mismatch, warm-up or cache-reset failure, unexpected
retry, replacement, partial rerun, or second execution invalidates the affected
claim.

If approved and frozen, the held-out sample stops after exactly 180 pairs. The
run will not be extended after a favorable result, rerun after a negative or
invalid result, partially rerun, or repaired by replacing tasks. Infrastructure
failure therefore produces an invalid held-out result rather than permission
for another attempt.

## Spending ceiling and abort rule

The one-shot held-out allocation has a hard ceiling of **$8**, within the $40
Phase 4 cumulative ceiling. At the 2026-11-23 Modal rates—$0.000542 per L40S
GPU-second, $0.0000131 per CPU-core-second, and $0.00000222 per GiB-second—the
two-L40S, eight-core, 32-GiB, 30-minute hard resource limit projects to
$2.267712 before negligible egress.

No GPU may be allocated if current pricing would put that limit above $8 or if
cumulative Phase 4 spend could exceed $40. The run aborts before block B if
block A consumes 15 minutes. It aborts immediately on dataset/evaluator hash
mismatch, physical-device identity or assignment failure, backend identity
failure, warm-up failure, or cache-reset failure. The 30-minute platform limit
is the final cost backstop. Every abort or platform interruption makes the
held-out result invalid and does not authorize a rerun.

## Reporting commitment

The result package will prepare the applicable narrative regardless of outcome:

1. **Verified:** latency superiority and both quality gates pass; report the
   narrow claim, intervals, failures, cost, and complete provenance.
2. **Rejected or invalid:** a quality, completion, identity, crossover, or
   evidence gate fails; report the failed gate and do not claim a speedup.
3. **Mechanism-only or inconclusive:** quality passes but the conditional
   latency interval does not establish superiority; report measurements without
   a deployment recommendation.

External publication remains a separate approval boundary even after analysis.

## Cost, provenance, and approval boundary

All pilot and initial calibration iterations cost $2.00556691. The consumed-task
crossover validation cost $0.24695705, bringing Phase 4 spend to date to
$2.25252396, including L40S, CPU, memory, and network charges. The final pilot
archive digest is
`sha256:8af23d85eb918e215551e6c0e7e1ab85b9100515ec12aa2a1e2cb98027237057`;
the initial calibration archive digest is
`sha256:3b00f9514e12876c5a8c2ea2eeeec0a2f01da4c8c03529e236516c91878b120c`.
Zero held-out compute has been used.

Explicit approval is required before either freeze declaration changes, before
any held-out runner is enabled or invoked, and again before external
publication. After approval, the immutable freeze commit must be pushed
remotely before the one-shot run. Until then, the contract stays proposed and
unfrozen.
