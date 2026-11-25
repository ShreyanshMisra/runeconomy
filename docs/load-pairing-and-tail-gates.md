# Load, pairing, and tail-metric gates

Status: design note · Contract versions: static manifest `0.4`, agent-process manifest `0.4`

This note is the normative spec for three things: how RunEconomy runs a paired
experiment under concurrent load, how it pairs variants so that load stays
balanced, and which statistics gate tail latency and unit economics. Everything
here is prospective. Manifests at earlier versions keep `arrival_pattern:
"sequential"` and their original gates, byte for byte.

Out of scope here: the Prometheus `/metrics` scraper and the `cost_model`. The
seams they attach to are defined in the "Interfaces for telemetry and cost"
section below.

## 1. What a load experiment estimates

A serving change (FP8 instead of BF16, a new engine release, speculative
decoding, a prompt layout that changes prefix-cache reuse) changes how the
server behaves when it's busy. The estimand is:

> the difference between a deployment that serves **only the baseline** at the
> declared load and a deployment that serves **only the candidate** at the same
> load, on the same frozen task population.

Two consequences follow.

1. **No mixed load on a shared server.** If both variants are in flight on the
   same server at once, each variant's requests shape the other's queueing,
   batching, and KV-cache pressure. A candidate that cuts prefill also speeds
   up the baseline it shares a batch with, which pulls the measured difference
   toward zero (interference, a SUTVA violation). On a shared server,
   variants therefore never overlap in time.
2. **Load is per variant.** `concurrency: 16` means 16 workflows of that
   variant in flight against its serving stack. `rate_per_second: 4` means 4
   workflow arrivals per second for that variant.

## 2. Arrival patterns

`execution.arrival_pattern` selects the pattern and `execution.load` holds its
parameters.

| Pattern | Meaning | Parameters |
|---|---|---|
| `sequential` | One trial at a time in frozen order (legacy, default). | none |
| `closed_loop` | N virtual agents per variant. Each starts its next unit as soon as its previous one finishes. | `concurrency` or `concurrency_levels` + `gate_concurrency` |
| `constant_rps` | Open loop. Arrivals every `1/rate` seconds. | `rate_per_second` or `rate_levels` + `gate_rate`, `max_in_flight` |
| `poisson` | Open loop. Exponential inter-arrival times from a seeded generator. | as `constant_rps` |

A **sweep** lists several levels. Each level is a *stage* that runs the whole
population at that level, stages run in ascending order, and the
preregistered gate level is the only stage that feeds the decision. Other
stages are reported as descriptive, ungated summaries, so a sweep never
multiplies the chance of a false `verified`.

A **unit** is one task (agent-process mode), one case (static mode), or one
whole workflow when `cases_form_workflow` is true. A workflow's steps run in
order inside one virtual agent.

## 3. Pairing designs

`execution.load.pairing` chooses one of two designs. `serving_topology`
says whether the variants share a serving stack.

### 3.1 `variant_blocked` (required for `serving_topology: "shared"`)

Each stage is split into **block pairs**. A block pair covers `block_units`
units of one repetition (by default all units of that repetition) and holds
two blocks, one per variant, over **the same units in the same dispatch
order**. For open-loop patterns it also uses **the same arrival offsets**.

- Only one variant is in flight during a block. Every block ends with a
  barrier, so all of its trials finish before the next block starts. There
  is no cross-variant interference.
- The two blocks of a pair are adjacent in time, so slow drift (thermal
  state, a noisy neighbour, a warming cache) hits both arms nearly equally.
- Block order within a pair is counterbalanced. `order: "alternating"` gives
  AB, BA, AB, ... (an exact ABBA balance of carry-over). `order:
  "randomized"` flips a seeded coin per pair.
- Identical dispatch order and identical arrival offsets are common random
  numbers. The two arms see the same realization of the arrival process, so
  arrival-process noise cancels in the paired difference instead of inflating
  it.

Pairs are `(unit, repetition)` inside the gate stage, as before.

### 3.2 `interleaved` (only for `serving_topology: "isolated"`)

When each variant has its own serving stack (two endpoints, typical of an
agent-process engine comparison), both variants run **at the same time**,
each in its own pool at the declared per-variant load, consuming the same
dispatch order and the same arrival offsets. Paired trials start at nearly
the same instant, which balances shared client-side resources (CPU, network,
tool backends) and time. Each server still sees only its own variant.

`interleaved` on a shared topology is rejected at validation time. Static
mode always targets one `backend.base_url`, so it is always `shared`.

### 3.3 Ramp and tail

A closed-loop block starts at full concurrency, but once its queue is empty,
in-flight work drains below N. With `tail: "drain"` (the default), the last
roughly N units of each block finish under partial load. Both arms of a block
pair drain the same way, so the comparison stays balanced, but the claim then
describes the declared load with a drain tail. With `tail: "sustain"`, idle
virtual agents run **load-filler** trials, which cycle through the block's
units in the same order, until every population trial of the block has
finished. Open-loop patterns keep generating arrivals instead. Fillers:

- are marked `role: "load_filler"` in the artifact store and are never part
  of the population, so no population trial is ever dropped;
- are excluded by role, which is fixed before execution, never by outcome;
- cost real compute, and the planner says so.

The planner warns when `block_units < 4 × concurrency` with `tail: "drain"`,
because then more than a quarter of each block runs under partial load.

### 3.4 Cache state

Under concurrency, `cache_treatment: "cold"` with `physical_reset` is
rejected, because a reset with requests in flight is refused by vLLM and would
race them anyway. `salt_isolated` cold runs and `warm` runs are allowed. Warm
runs reset once before the first block, as in sequential mode, and rely on
counterbalanced block order for carry-over balance.

### 3.5 The arrival schedule is a frozen input

The full schedule (stages, block pairs, block order, per-block dispatch order,
open-loop offsets, and pool assignment) is a pure function of the unit list,
repetitions, seed, order, and load parameters. RunEconomy writes it to the
result (and to `arrival-schedule.json` in agent-process artifacts) with a
SHA-256 digest, and the planner computes the same digest before execution so
the preregistration can freeze it. `execution.load.schedule_sha256` can pin
it. The runner then refuses to execute when the schedule it builds doesn't
match.

### 3.6 Load-delivery integrity

Closed-loop concurrency holds by construction. Open-loop arrivals can fall
behind when in-flight work reaches `max_in_flight`, and then the declared load
was not the load delivered. Every trial records its `dispatch_lag_ms`. On the
gate stage:

- `baseline_arrival_schedule_integrity` is an evidence gate. If the
  baseline's p95 dispatch lag exceeds `max_dispatch_lag_ms`, the declared load
  was never established and the verdict is `invalid`.
- `candidate_offered_load_completion` is a completion gate. If the candidate
  could not absorb the offered load on schedule, that's a real result, and
  the verdict is `regression`.

Each trial also records `in_flight_at_start` and its time-weighted
`mean_in_flight`. The non-required `observed_concurrency_balance` check
reports both arms' medians.

## 4. Gateable metrics

`acceptance.metric_gates` is a list of preregistered gates that are added to
the existing quality, completion, and latency gates. They never replace those
gates.

| Metric | Level | Direction | Definition |
|---|---|---|---|
| `workflow_latency_ms` | workflow | lower is better | Agent-process: process wall time. Static: request latency, or the sum over a workflow's steps. |
| `client_ttft_ms` | request | lower is better | Client-observed time to first output token. |
| `server_ttft_ms` | request | lower is better | Server-reported TTFT. |
| `tpot_ms` | request | lower is better | Client-observed `(total_latency_ms − client_ttft_ms) / (output_tokens − 1)`. Defined only when `output_tokens ≥ 2`. |
| `usd_per_1k_successful_workflows` | variant ratio | lower is better | `1000 × Σ cost_usd / Σ successes` over **all** population attempts of the variant, so failed attempts cost money too. |
| `successful_workflows_per_gpu_hour` | variant ratio | higher is better | Goodput: `Σ successes / (Σ gpu_time_ms / 3.6e6)`. |

Latency metrics take `statistic: "mean" | "p50" | "p90" | "p95" | "p99"`.
Ratio metrics take `statistic: "ratio"`.

### 4.1 Gate rule

Let `D` be the candidate-minus-baseline difference in the metric's "worse"
direction (for goodput, baseline minus candidate). A two-sided 95% interval
`[L, U]` for `D` gates:

- `semantics: "superiority"`: pass iff `U < −margin`;
- `semantics: "noninferiority"`: pass iff `U < margin`;
- regression iff `L > margin` (non-inferiority) or `L > 0` (superiority).

`margin` is in metric units. `relative_margin` instead expresses it as a
fraction of the baseline value, and the interval is then for `M_c / M_b − 1`.
This is the same convention as the existing latency gate, whose rule is the
upper bound of a two-sided 95% interval.

### 4.2 Population

- `population: "successful_pairs"` (the default for latency metrics) uses
  pairs where both variants succeeded. It matches the existing latency gate,
  and both arms describe the same task mixture. Failures are gated by the
  task-success and completion gates, not hidden. But if those gates tolerate
  failures (`min_completed_rate < 1`), a candidate whose slow tail times out
  can look fast.
- `population: "all_attempts_conservative"` uses every population attempt.
  A **candidate** attempt without a measured latency counts as `+∞`, and a
  **baseline** attempt without one is dropped. Both choices push against the
  candidate, so the gate stays valid when timeouts sit in the tail.
- Ratio metrics always use all population attempts.

### 4.3 Methods

**`quantile_order_statistic`** (workflow-level quantiles only; the default for
them). Distribution-free, exact under iid sampling. For quantile `q` and
sorted values `x(1) ≤ … ≤ x(n)`:

- an upper bound at level `1 − a` is `x(k)` for the smallest `k` with
  `P(Bin(n, q) ≤ k − 1) ≥ 1 − a`, and `+∞` if no `k ≤ n` qualifies;
- a lower bound is `x(j)` for the largest `j` with `P(Bin(n, q) ≥ j) ≥ 1 − a`,
  and `−∞` if none qualifies.

The difference interval is `[L_c − U_b, U_c − L_b]` with each of the four
marginal bounds at `a = α/4 = 0.0125`. Bonferroni gives simultaneous coverage
of at least 95% whatever the dependence between arms, so the method stays
valid under pairing, conservatively. Each task contributes exactly `r`
repetitions, which is stratified sampling. By Jensen's inequality, the
empirical CDF then has variance at most that of iid sampling from the task
mixture, so the order-statistic bound is conservative for the finite-task
estimand too. The estimator is the nearest-rank sample quantile used
throughout RunEconomy reports.

**`paired_cluster_bootstrap`** (all statistics, required for request-level
metrics and ratio metrics). Resample **tasks** (clusters) with replacement,
carrying every repetition and both arms of each task together. That keeps the
pairing, the within-task correlation across repetitions, and the
within-workflow correlation between requests. The statistic is recomputed on
each of 4000 seeded replicates, and the 2.5% and 97.5% percentiles form the
interval. Replicates where a ratio is undefined (zero successes or zero
GPU-time) count as `+∞` against the candidate. This is an approximate method:
the percentile bootstrap for a quantile has coverage error that shrinks slowly
with `n`, which is why it carries the larger minimum samples below.

### 4.4 Minimum samples (documented and enforced)

`n` is the number of analyzable observations in the gate population.
Request-level metrics count requests. Clusters are distinct tasks.

| Method | Statistic | Minimum | Reason |
|---|---|---|---|
| `quantile_order_statistic` | `p50` | 7 | smallest `n` with `q^n ≤ α/4`, so a finite upper bound exists |
| | `p90` | 42 | same rule |
| | `p95` | 86 | same rule |
| | `p99` | 437 | same rule |
| `paired_cluster_bootstrap` | quantile `q` | `max(20, ⌈10 / (1 − q)⌉)` observations and 20 clusters | at least 10 observations above the quantile: `p95` → 200, `p99` → 1000 |
| | `mean`, ratio | 20 clusters | below that the bootstrap distribution is too coarse |

Below the minimum, the gate reports `status: "insufficient_sample"`, fails as
a required check, and the verdict is `inconclusive`, never `regression`. The
planner flags a frozen design whose gate-stage pairs are below the minimum as
a blocking error before any compute is spent. For request-level metrics, it
can check only the cluster count, since request counts depend on the data.

### 4.5 Known limitation: dependence under load

Every method above assumes units are independent given the design. Under load,
a transient server stall slows every trial in flight at that moment, which is
positive dependence across nearby trials. That makes iid-based intervals
somewhat too narrow. Blocking means the dependence is mostly *within* an arm's
block, which hurts the variance estimate but doesn't bias the difference. Task
resampling doesn't capture it. Read gate intervals as conditional on the
realized load timeline, and prefer more block pairs over larger blocks once
`block_units ≥ 4 × concurrency`. A time-series-aware method (a block
bootstrap over dispatch windows) is future work.

## 5. Executor architecture

Static and agent-process runs share one pipeline:

```
RunSpec ──▶ ArrivalSchedule ──▶ LoadRunner(worker pool) ──▶ TrialExecutor.execute
  (frozen      (frozen, hashed)     sequential or concurrent      static request / agent process
   manifest)                                                      + evaluator
                                         │
                                         ▼
                                    ArtifactStore (append-only JSONL, fsync, locked)
                                         │
                                         ▼
                                    DecisionEngine (shared statistics, metric gates,
                                                    load integrity, decision schema 0.2)
```

- `runeconomy.execution` builds the schedule, runs it, and defines the
  `TrialExecutor`, `ArtifactStore`, and `LoadObserver` protocols.
- `runeconomy.metric_gates` holds the metric registry, the quantile and
  bootstrap methods, and the load-integrity checks shared by both modes.
- `sequential` runs still take their original code path, byte for byte, so
  Proofs 1 and 2 stay reproducible.

Resume is not supported for concurrent arrival patterns yet. A partially
completed block can't be replayed under the same load, and resuming at block
granularity would discard observed trials, which needs its own transparency
rules.

## 6. Interfaces for telemetry and cost

These seams are for the Prometheus scraper and the cost model. Neither is
implemented here.

- **Server telemetry.** `run_experiment(..., observers=[...])` and
  `run_comparison(..., observers=[...])` accept `LoadObserver` objects.
  `on_block_start(block)` and `on_block_end(block)` run on the scheduler
  thread at block barriers. A mapping returned from `on_block_end` is attached
  to that block's record under `telemetry`. `on_trial_end(trial, record)` can
  attach per-attempt server telemetry by returning a mapping, which is stored
  under `record["server_telemetry"]`.
- **Cost.** `attempt_economics` is a callable `(record) -> {"cost_usd":
  float | None, "gpu_time_ms": float | None, ...}`. It runs on each completed
  population record before decisions, and its result is stored under
  `record["economics"]`. When it's absent, agent-process attempts fall back
  to the summed trace `cost_usd` and `gpu_time_ms`. Unknown values stay
  unknown. If any population attempt of a variant has unknown cost, the ratio
  metric is unknown and its gate can't pass.

## Cost models and server telemetry

### `cost_model` (schema 0.4)

A manifest may declare one versioned cost model. It is part of the manifest
fingerprint and of an `experiment freeze` lock, and it prices every population
attempt (`economics`) before any decision is made. The
`usd_per_1k_successful_workflows` and `successful_workflows_per_gpu_hour`
metric gates read those prices.

```json
"cost_model": {
  "schema_version": "0.1",
  "kind": "gpu_hourly",
  "usd_per_gpu_hour": 1.95,
  "gpus_per_variant": 1,
  "source": "Modal L40S list price, 2026-11-25"
}
```

- `gpu_hourly` charges reserved accelerator time: wall time multiplied by
  `gpus_per_variant`. Under concurrent load, the charge is divided by the
  attempt's time-weighted mean in-flight count. This is a fair-share
  attribution of reserved GPU time, not a kernel-time measurement.
- `per_token` charges `usd_per_1m_input_tokens`,
  `usd_per_1m_cached_input_tokens` (input price when omitted) and
  `usd_per_1m_output_tokens`. When token counts are unknown, the cost is
  unknown. It is never priced as zero.

Every economics record carries `cost_model_sha256`. Static results record the
model under `provenance.cost_model`, and agent-process results record it under
`design.cost_model`.

### Server metrics during load blocks

`experiment run --server-metrics` samples the static backend adapter's
Prometheus endpoint while each load block runs.
`--server-metrics-url TARGET=URL` samples any `/metrics` URL. `TARGET` is
`baseline`, `candidate` (agent-process variants on separate servers) or
`server`. Samples are taken every `--server-metrics-interval` seconds (default
1).

Each block record gets `telemetry.server_metrics`, which records:

- the mean, max and p95 of running and waiting requests and of KV-cache usage
- counter deltas for preemptions and prefix-cache hits and queries
- the in-block prefix-cache hit rate

Each trial gets the latest gauges from the server that served it
(`server_telemetry`). Telemetry is descriptive. It never enters a gate or a
verdict, and a failed scrape is recorded as unavailable rather than stopping
the run.
