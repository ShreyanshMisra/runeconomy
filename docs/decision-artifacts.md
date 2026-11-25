# Decision artifact semantics

RunEconomy decision schema `0.2` keeps the legacy `status`, `reason`, and
`checks` fields and adds orthogonal fields for questions that a single verdict
cannot answer safely.

| Field | Question answered |
| --- | --- |
| `evidence_validity` | Are identity, trace, evaluator, and integrity records usable? |
| `execution_terminality` | Did every expected attempt produce a terminal record? |
| `task_outcome` | Which paired tasks succeeded or failed? |
| `failure_origin` | What origin is supported by the sanitized aggregate evidence? |
| `performance_eligibility` | May the performance gates support an interpretation? |
| `decisive_gate` | Which first required gate fixed the verdict, and why? |

These dimensions are intentionally independent. A one-shot execution can be
terminal even when a task fails. Its evidence can be valid while a frozen
completion gate makes its performance measurement ineligible. Conversely, an
apparently successful task result cannot repair invalid trace or evaluator
evidence.

Evidence validity is conservative. It is `valid` only when at least one
required evidence check is present and every required evidence check explicitly
passes. If no required evidence check is recorded, or any required evidence
check is unresolved, evidence validity is `unknown` and performance remains
ineligible. An explicitly failed required evidence check makes it `invalid`.

## Common-mode paired failures

`common_mode_paired_failures` reports the count and rate of pairs where both
variants failed. This is a diagnostic, not a causal reclassification. A
concordant failure is consistent with a shared task, evaluator, environment, or
execution cause, but sanitized aggregate counts cannot distinguish those
possibilities. The `failure_origin.attribution` field therefore remains
`causal_origin_unresolved` unless the recorded gates localize the origin.

Discordant outcomes use paired direction explicitly:

- `baseline_only` means the baseline succeeded and the candidate failed.
- `candidate_only` means the candidate succeeded and the baseline failed.

## Identity assurance

A contract that preregisters identity policy `0.1` gets a derived
`identity_assurance` block. It records each variant's identity grade
(`operator_verified` or `provider_declared`), the claim scope
(`pinned_configuration` or `observed_service_window`), the supported claim,
the claim limits, and a per-variant provider fingerprint summary (counts only,
a drift indicator that never affects validity). Contracts without a policy
have no such block and keep their legacy identity semantics. See
[serving backends and identity grades](backends.md#identity-grades).

## Decisive gates

The decisive gate is the first failed required check in recorded gate order.
It includes the gate name and category, references to the exact recorded
observation and threshold, and an explanation of the consequence. Optional
diagnostic checks never become decisive.

A failed evidence, completion, or task-outcome gate makes performance evidence
ineligible. A failed performance gate means the declared performance criterion
was not established. Passing later checks cannot rescue an earlier decisive
failure.

## Schema compatibility

Decisions without a version and decisions marked `0.1` are treated as legacy
artifacts. `enrich_decision_artifact` preserves their legacy verdict fields and
adds schema `0.2` structure. Applying the transformation to an existing `0.2`
decision is idempotent. Unknown future versions are rejected rather than
silently guessed.

The result-envelope versions do not change: decision schema `0.2` is an
additive nested contract. Existing consumers may continue reading
`decision.status`, `decision.reason`, and `decision.checks`. New consumers can
validate `runeconomy/data/decision-0.2.schema.json`.

## Planner completion warnings

The read-only planner converts `min_completed_rate` into exact integer counts
for the frozen sample. It reports, per variant:

- planned attempts;
- minimum completed attempts; and
- maximum failed or incomplete attempts tolerated.

For example, a threshold of `1.0` with 180 attempts tolerates exactly zero
failed or incomplete attempts. This warning is emitted before execution so the
operator sees the contract's actual failure tolerance rather than only a
decimal rate.

## Historical records

Schema `0.2` applies to future generated decisions. It does not mutate,
reinterpret, or backfill frozen historical outcomes. In particular, the final
Proof 2 verdict remains permanently invalid under its immutable contract.
