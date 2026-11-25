# Detector validation

Each waste detector predicts an effect. This page records the paired
experiment that tests each prediction, with the design fixed before any
measurement. A refutation is reported as a refutation.

## Protocol

- **Agent:** the lab agent (`examples/lab_agent/agent.py`), a repository-lookup
  workflow on `Qwen/Qwen2.5-14B-Instruct` (revision `cf98f3b…`) with vLLM
  0.34.0 on Modal L40S GPUs.
- **Tasks:** the 40-task `detector` split from `scripts/build_phase7_workload.py`.
  It is disjoint from every proof split and from all earlier proofs.
- **Workspaces:** each experiment is an agent-process workspace built only
  through `experiment init` and `experiment patch`
  (`scripts/prepare_phase7.py`).
- **Gates:** declared before running. Task-success floor 0.5, non-inferiority
  margin 10 points, `task_outcome` completion policy for shared failures, and
  a 0.95 completion floor.
- **Status of the evidence:** these are calibration-grade validations, not
  preregistered public claims. The workspaces are not frozen.
- **Detection:** for each detector, the detector is first shown to fire on the
  baseline arm's own traces (`runeconomy analyze` on the normalized traces or
  the result). The experiment then tests the predicted direction of the
  template's primary metric, with the quality gates applied.

| Detector | Baseline arm | Candidate arm | Prediction |
| --- | --- | --- | --- |
| `context.tool_output_carryover` | full search excerpt carried into every later call | compact summary after the first call | input tokens and workflow latency decrease, success non-inferior |
| `workflow.serialized_independent_steps` | two independent verification calls in sequence | the same calls concurrently | workflow latency decreases |
| `models.step_overprovisioning` | every step on 14B | planning step routed to Qwen2.5-1.5B (declared step roles) | workflow latency decreases, success non-inferior |
| `output.structured_step_inflation` | unconstrained planning that reasons before its JSON | schema-constrained planning | output tokens and workflow latency decrease |
| `serving.kv_pressure` | 14B server with a 0.75 GPU-memory budget at 48 concurrent agents | the same model with a 0.92 budget | KV pressure detected on the baseline; p95 workflow latency decreases |

## Results

Executed 2026-11-25 in one Modal window: 27.7 minutes remote, about $2.39 at
published rates. Window archive digest: `sha256:f635caa2…e597`. Every
attempt in every arm completed. All five predictions were **confirmed**:
task success was 100% in both arms and passed non-inferiority, and the
predicted effect passed its paired gate.

| Detector | Fired on baseline traces | Predicted effect | Measured (candidate − baseline) | Outcome |
| --- | --- | --- | --- | --- |
| `context.tool_output_carryover` | 39/40 | input tokens ↓ | mean input tokens 3,363 → 1,034 (−69%); successful-workflow latency −104 ms (paired SD 28 ms) | confirmed |
| `workflow.serialized_independent_steps` | 40/40 | latency ↓ | −385 ms (paired SD 20 ms) | confirmed |
| `models.step_overprovisioning` | 40/40 | latency ↓, success non-inferior | −576 ms (paired SD 290 ms) with the planner step on Qwen2.5-1.5B; success 40/40 in both arms | confirmed |
| `output.structured_step_inflation` | 14/40 | output tokens ↓ | mean output tokens 260 → 28; latency −9.07 s (paired SD 1.46 s) | confirmed |
| `serving.kv_pressure` | yes (baseline server) | p95 latency ↓ | p95 successful-workflow latency 27.7 s → 10.8 s at 48 agents; p95 superiority gate passed | confirmed, see note |

Result digests: carry-over `sha256:7963a791…`, serialization
`sha256:d47661be…`, over-provisioning `sha256:cb5f674e…`, inflation
`sha256:32720133…`, KV pressure `sha256:6c4ff00a…`.

Notes, including what the evidence does not show:

- **Inflation fired on 14 of 40 baselines.** Only those unconstrained plans
  exceeded the 256-token threshold. The effect was measured on all 40 pairs.
- **The KV-pressure trigger was thin.** One preemption was observed on the
  small-budget server, and none on the large-budget server. The larger
  measured difference was prefix-cache eviction: the in-block prefix-cache
  hit rate was 10.4% with the small KV budget against 65.7% with the large
  one. The detector's preemption rule caught the pressure. The eviction
  signal is not yet a detector rule (an open follow-up). The gauge metrics
  (running, waiting, KV usage) were recorded as unknown in this window. The
  cause was a scraper defect, not vLLM 0.34.0: the combined vLLM/SGLang
  mapping let SGLang's source names replace vLLM's for shared keys. It is
  fixed and covered by a regression test (`PrometheusScraperTests`). Gauges
  from this window remain unknown.
- **Over-provisioning is a heuristic.** It flagged every planning step
  (at most 32 output tokens on a 14B model, against a long explained answer).
  It does not show that the routed step is simple in general, only that this
  one was.
- These are calibration-grade validations on 40 tasks (120 attempts per arm
  for KV pressure). They show the detectors point at real, measurable effects
  on this agent and stack. They are not public performance claims.
