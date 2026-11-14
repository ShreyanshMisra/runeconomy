# Public proof: RunEconomy rejected an apparently faster candidate

This is the canonical RunEconomy proof for the frozen
40-task held-out run.
The verdict is **`invalid`**, not verified. Every number below is generated
from the checked-in aggregate artifact; private tasks, answers, model output,
stderr, endpoints, and raw traces are not published.

## What changed

- Baseline: dynamic run metadata before the static system instructions.
- Candidate: the same dynamic run metadata after the static system instructions.
- Only changed property: system-message block order.
- Held constant: agent code, static instructions, dynamic metadata content, model, backend, tools, evaluator, tasks, and acceptance gates.
- Expected mechanism: keep the static instruction prefix stable so the second inference call can reuse more cached prompt tokens.

## What was measured

The frozen workload contained 40 tasks (13 direct-call, 13 docstring, 14 string-literal),
with one baseline and one candidate attempt per task. Each attempt made two
inference calls around one real bounded repository-search tool. Order was
seeded and randomized (`20261114`). The predeclared product outcome
was end-to-end workflow latency, paired only where both variants succeeded.

Environment: `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`, vLLM `0.33.0`, one
A10, Chat Completions, colocated driver.

## Task quality

| Variant | Successes | Point rate | Exact 95% interval | Absolute gate |
|---|---:|---:|---:|---:|
| Baseline | 37/40 | 92.50% | 79.61%–98.43% | lower bound ≥ 80% |
| Candidate | 38/40 | 95.00% | 83.08%–99.39% | lower bound ≥ 80% |

Paired success was
2.50 percentage points; its one-sided
lower bound was -12.55 percentage points, missing
the predeclared
−10.00 percentage points
non-inferiority margin. Outcomes were 36
both-success, 1 baseline-only,
2 candidate-only, and
1 both-failure.

## Decision gates, in order

1. `evidence_integrity` — passed: 80/80 valid traces and 80/80 complete explicit DAGs.
2. `baseline_completion_and_absolute_task_success` — failed: 79.61% lower bound was below 80.00% **(decisive)**.
3. `candidate_completion_and_absolute_task_success` — passed: 83.08% lower bound exceeded 80.00%.
4. `candidate_point_estimate_task_success_guardrail` — passed: candidate point estimate 95.00% exceeded baseline 92.50%.
5. `paired_task_success_noninferiority` — failed independently: -12.55 percentage points was below -10.00 percentage points.
6. `successful_pair_workflow_latency` — conditionally favorable at -62.43 ms but ineligible to verify the change.

The baseline validity gate was decisive. RunEconomy therefore returned
**`invalid`** before the favorable latency result could authorize a verified
performance claim. The paired non-inferiority gate also failed independently.

## Conditional mechanism and latency observations

Among the 36 pairs where both
variants succeeded:

- Cached input tokens: +105.33 tokens; 95% interval +98.65 to +112.02; 36 pairs.
- Server TTFT: -9.08 ms; 95% interval -11.85 to -6.30; 36 pairs.
- Client TTFT: -9.39 ms; 95% interval -12.21 to -6.57; 36 pairs.
- Workflow latency: -62.43 ms; 95% interval -123.86 to -1.01; 36 pairs.
- Critical-path latency: -61.98 ms; 95% interval -123.44 to -0.52; 36 pairs.

These are measured conditional effects, not verified improvements. Failed tasks
remain in the quality analysis and cannot make the candidate appear faster by
disappearing from the verdict.

## What RunEconomy demonstrated

RunEconomy connected an application-level prompt-layout change to cache,
inference, workflow, and task-success evidence; then declined to verify an
attractive latency result because the frozen quality requirements were not
established. That rejection—not a speedup claim—is the product proof.

## What this does not prove

- One pinned model revision, vLLM version, A10 hardware class, and repository-analysis workload were evaluated.
- Execution was sequential and colocated; production concurrency and multi-replica routing were not tested.
- Continuous effects use only the 36 task pairs where both variants succeeded.
- Workflow latency includes process startup and agent execution but excludes evaluator time.
- Prefill, decode, retrieval, cost, and GPU-time effects were unavailable and remain unknown rather than zero.
- The 40-task held-out workload is consumed and cannot be reused as fresh held-out evidence.
- External readers can audit the aggregate arithmetic, gate logic, provenance hashes, and claim boundary, but cannot independently recompute the experiment without the private raw tasks and traces.

It also does not establish universal model behavior, production-concurrency
performance, superiority over another tool, or a verified performance win.

## Audit trail

The machine-readable source is
[`src/runeconomy/data/public-proof-rejection.json`](../src/runeconomy/data/public-proof-rejection.json).
It can be rebuilt only when the ignored private evidence is present:

```bash
uv run python scripts/build_public_proof_artifact.py
```

Frozen hashes:

- Agent: `sha256:e6f42c619f15be192512d234ee21b623a09e69402a62ab16516190c9ea348dcb`
- Comparison: `sha256:d8df549ebcb7d814aa1d70b02e2d9c482830e02d7812a40376d1c8161dd31e8f`
- Driver: `sha256:1a6a589353e5352116a58ee446493fbc83c486f4fc7634a49483ac81b474b1b1`
- Environment Identity: `sha256:a62037eec6d2c8d31743f404018469cb8d42c9359b1fe5a84f1e0053c86638ef`
- Evaluator: `sha256:8f6f0ac64145392fd9f2086705345871628fce3957bae810f5cd551be4b02dba`
- Experiment Design: `sha256:51975b24f8792f6f9ab453fb6a2fcba84ab1247e9f1a230e3ca0b9910564a48e`
- Private Result: `sha256:783c08972d972b2cc7aad610c13d9df934d8f4c573915ad6423ebaf74402aa99`
- Status Protocol: `sha256:b3885985bfa88510d901600b28e3d6341427b2390b3dedb2c16ccd124e68e723`
- Workload: `sha256:85e53d2d2f2ab108b5bc83e27947a67d710ed53326a89467005630dea3bddf61`

The public artifact contains aggregate evidence only. Its sanitizer fails
closed on unexpected top-level fields, endpoints, credentials, local paths,
and raw-evidence keys.
