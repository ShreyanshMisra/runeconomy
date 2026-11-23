# RunEconomy 0.2.2 release notes

RunEconomy `0.2.2` extends the evidence-to-verdict workflow without changing
the privacy boundary: raw tasks, traces, model output, evaluator answers,
credentials, endpoints, and private experiment artifacts remain local.

## Engine-change verification

Agent-process comparisons can now declare `engine_change_latency` as the
optimization target. The baseline and candidate may intentionally use
different backend identities when that difference is the frozen intervention.
RunEconomy requires one complete, stable identity per variant and verifies that
the identities are distinct before an engine-change latency claim is eligible.

Latency remains conditional on pairs where both variants pass the evaluator.
Absolute task-success, paired non-inferiority, completion, trace integrity, and
environment identity remain independent gates. A faster conditional latency
measurement cannot rescue a failed quality or completion gate.

## Startup reliability and compatibility

The live validation harness distinguishes retryable startup and cold-start
responses from terminal failures. Bounded retry behavior covers transient
startup 502 and 503 responses while authentication, rate-limit, and other
terminal responses remain immediately actionable.

The dated 2026-11-23 compatibility matrix records the pinned vLLM combinations
that were exercised across the supported APIs and evidence fields. It is dated
compatibility evidence, not a claim that every vLLM deployment is equivalent.

## Evidence quality and resumable experiments

Evidence fields carry quality grades such as directly observed, reliably
derived, weakly inferred, and unresolved. Missing evidence remains unresolved
instead of being converted to zero or treated as observed.

Agent-process experiments persist a deterministic attempt checkpoint and can
resume after interruption. Resume revalidates commands, data, evaluator,
schedule, gates, and completed records before continuing. It does not permit a
new attempt after a frozen one-shot contract has reached a terminal result.

## Decision schema 0.2

Decision schema `0.2` preserves the legacy `status`, `reason`, and `checks`
fields and separately records:

- evidence validity
- execution terminality
- paired task outcome
- supported failure origin
- performance eligibility

Evidence validity is conservative. It is `valid` only when at least one
required evidence check is present and every required evidence check explicitly
passes. Missing or unresolved required evidence produces `unknown` and remains
performance-ineligible. A failed required evidence check produces `invalid`.

Common-mode diagnostics report concordant paired failures without claiming a
causal origin that aggregate evidence cannot establish. The decisive-gate
record identifies the first failed required gate, its exact observation and
threshold references, and the consequence for performance eligibility.

Unversioned and schema `0.1` decisions remain readable. Enrichment is additive
and idempotent, while unknown future decision versions are rejected rather than
guessed.

## Exact planning consequences

The read-only planner translates completion-rate thresholds into exact integer
requirements. It reports planned attempts, the minimum completed attempts per
variant, and the precise number of failed or incomplete attempts tolerated. A
100% completion threshold therefore warns that the frozen design tolerates
exactly zero failures.

## Proof 2 final outcome

Proof 2 compared BF16 and online FP8 under a frozen 180-pair balanced crossover
contract. Its single authorized held-out execution completed 179 of 180
attempts for each variant. Because the contract required 100% completion, the
result is permanently `invalid`.

The successful-pair latency gate was favorable, but it is not eligible to
support a verified FP8 performance claim. No rerun, post-hoc reclassification,
or new performance claim is permitted. The tracked Proof 2 outcome contains
privacy-safe aggregates only and remains separate from the immutable freeze
record.

## Installation

Install the current release with an exact pin:

```bash
python -m pip install runeconomy==0.2.2
```

The official coding-agent skill remains version-neutral. Tester-facing release
instructions and current-release examples use exact package pins.
