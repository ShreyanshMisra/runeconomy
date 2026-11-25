# Proofs 3 and 4, and studies

Every number here is recomputed from result files by
`scripts/build_phase7_artifacts.py`. The decision records and reports are in
[`docs/proof-records/`](proof-records/).

| Item | Design | Verdict | Record |
| --- | --- | --- | --- |
| Proof 3 | BF16 vs online FP8, 200 frozen held-out pairs, 16 agents per variant | **verified** (target: verified) | [`proof-3-decision.json`](proof-records/proof-3-decision.json) |
| Proof 4 | Search tool removed, 100 frozen held-out pairs | **regression** (target: regression) | [`proof-4-decision.json`](proof-records/proof-4-decision.json) |
| Study 1 | FP8 vs BF16 at 1, 8 and 32 agents | descriptive | [`study-1.json`](proof-records/study-1.json) |
| Study 2 | Dynamic value before vs after a stable prompt prefix | descriptive | [`study-2.json`](proof-records/study-2.json) |
| Study 3 | Hosted frontier API vs self-hosted open weights | not run: needs a hosted API key | none |

Proofs 1 and 2 are unchanged. They are the history that led to the
completion policy, the exact paired method, and the power planner.

## How the proofs were run

1. **Workload.** `scripts/build_phase7_workload.py` builds the lab agent's
   task splits from a frozen repository snapshot:
   - calibration splits for each proof
   - a detector split
   - a 400-task held-out pool that no model saw before the held-out window

   Answers appear only in the evaluators.
2. **Calibration window.** 40 tasks per proof and the five detector
   validations ([detector validation](detector-validation.md)).
   `scripts/analyze_phase7.py calibration-record` derives each design's
   planning assumptions from the calibration results alone.
3. **Design and freeze.** `scripts/prepare_phase7.py heldout` builds each
   proof workspace using only `experiment init`, `experiment patch` and
   `experiment freeze`, with no hand-edited JSON. The planner sized both
   proofs at 80% power. Proof 4 was powered to *detect* its predicted harm
   (`detection_target: regression`). Both workspaces reached
   `ready_to_freeze` and were frozen before any held-out task ran.
4. **Execution.** `deploy/modal_phase7.py` ran each frozen workspace once
   with `--require-frozen`, on 2× L40S under vLLM 0.34.0, sampling server
   `/metrics` during every load block. Each result embeds its intact freeze
   record digest.

## Proof 3: verified

FP8 lowered mean successful-workflow latency by 499 ms (95% CI −515 to
−483 ms, about 27%). p95 fell from 2,254 to 1,625 ms, and cost per 1,000
successful workflows fell from $0.065 to $0.048. Task success was 200/200 in
both arms, and the non-inferiority lower bound was −1.49 points against a
5-point margin. Boundaries: one model, one GPU type, one synthetic agent,
closed-loop load, and list GPU prices.

## Proof 4: regression

Removing the search tool cut input tokens from 470 to 206 per workflow and
halved p50 latency, and every attempt completed. Task success fell from 100%
to 54% (95% CI 43.7%–64.0% against an 80% floor). That floor was the decisive
gate, and the faster latency is ineligible for a claim. A request benchmark
would have shipped this change.

## Study 1: FP8 under 1, 8 and 32 agents

| Agents | BF16 p50 / p95 (ms) | FP8 p50 / p95 (ms) | Paired mean difference | Relative |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,297 / 1,650 | 910 / 1,149 | −392 ms | −29.6% |
| 8 | 1,546 / 1,920 | 1,049 / 1,290 | −490 ms | −31.5% |
| 32 | 2,312 / 2,879 | 1,603 / 2,054 | −719 ms | −31.0% |

The absolute saving grows with load, and the relative saving stays near 30%.
Task success was 100% at every level. The 100 tasks per level were unused
held-out pool tasks. The study was not preregistered, so it shows a pattern on
this workload and makes no claim.

## Study 2: a dynamic value at the start of the prefix

48 routing requests share one ~1.8K-token system prompt. The baseline puts a
per-request ID and timestamp first. The candidate moves them after the stable
prefix. Everything else is identical: model, server, sampling, 8 clients and
5 repetitions (240 requests per variant).

| Measure | Dynamic value first | Dynamic value after prefix |
| --- | ---: | ---: |
| Mean cached prompt tokens | 1.3 of 1,793.5 | 1,741.6 of 1,793.5 |
| Server TTFT p50 / p95 | 935 / 1,029 ms | 96 / 318 ms |
| Client TTFT p50 / p95 | 1,548 / 1,940 ms | 139 / 373 ms |
| Task success | 100% | 100% |
| Cost per 1,000 successful requests | $0.151 | $0.022 |

The static prefix-cache prediction was +1,760 cached tokens per request, and
+1,740 were observed (98.9%). A single dynamic token at the front of a long
system prompt defeats vLLM's prefix cache for the whole prompt.

**An earlier execution was superseded, and it is not hidden.** Study 2 first
ran in the held-out window and returned `inconclusive` because of a
measurement defect:

- The runner launched vLLM without `--enable-prompt-tokens-details` and
  `--enable-per-request-metrics`, so cached tokens and server TTFT were
  unmeasured.
- 144 requests per variant was below the p95 gate's 200-observation minimum.

Both were fixed before the rerun:

- The runner now passes both flags, checks that vLLM accepts every launch flag
  before starting a GPU, and refuses to proceed if warm-up responses lack
  those fields.
- The planner now counts request-level gate samples exactly for static
  replays, so it blocks a design like the first one.

The superseded result's digest is in `study-2.json`.

## Defects found and fixed during these runs

- **Server gauges recorded as unknown.** The vLLM/SGLang scraper mapping let
  SGLang's metric names replace vLLM's, so running, waiting and KV-usage
  gauges were never read from vLLM. Counters were unaffected. This is fixed,
  covered by a regression test, and confirmed on vLLM 0.34.0 in the Study 2
  rerun. The Proof 3 and Study 1 results carry prefix-cache and preemption
  counters but unknown gauges.
- **Workspaces embedded absolute paths.** `experiment init` wrote absolute
  paths for inputs outside the workspace, including the local home directory.
  New workspaces reference such inputs relatively. The frozen proof
  workspaces predate the fix and keep their paths, because changing them would
  break their freeze digests.

## Reproducing

The reproducibility bundle (`phase-7-reproducibility.tar.gz`, built by
`scripts/build_phase7_artifacts.py`; not attached to this repository) contains:

- the frozen and calibration workspaces (manifests, preregistration records,
  provenance chains)
- the task splits and evaluators
- the lab agent and the repository snapshot
- the runner and every script
- the calibration record, the four result files and the decision records

Both frozen workspaces verify `intact` after extraction elsewhere:

```bash
runeconomy experiment validate bundles/heldout/workspaces/p3-heldout
```

To re-execute, use `deploy/modal_phase7.py` with the bundle. That gives new
evidence on the same design. It does not repeat the one-shot proof.
