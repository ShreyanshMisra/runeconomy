# The FP8 near-miss: every quality and latency gate passed, but the verdict was invalid

Proof 2 compared BF16 and online FP8 on one self-hosted vLLM agent workload.
Every identity, trace-integrity, task-quality and latency gate passed. On the
successful pairs, FP8 measured 391 ms lower mean workflow latency. The final
verdict was still `invalid`, so that number is descriptive and supports no
performance claim.

The cause was one shared application failure. Both variants failed the same
task, which left each at 179 completed attempts out of 180. The contract we
froze before the run required 100% completion, which meant it tolerated
zero failures. We chose that threshold, and we're holding ourselves to it.

This is a methodology case study. It is not evidence that FP8 is safe or faster
for any deployment, including this one.

## The decision we actually tested

The question wasn't only whether FP8 reduced request time. It was whether FP8
made **successful end-to-end agent workflows** faster while preserving task
quality, completion, trace integrity and serving identity.

The claim was deliberately narrow: the pinned `Qwen/Qwen2.5-14B-Instruct`
checkpoint, one repository-agent workload with two task families, vLLM 0.34.0
and L40S GPUs. It said nothing about other models,
agents, task families, GPU classes, concurrency, throughput or production
traffic.

Before execution, we froze the claim, task and evaluator digests, 180-pair
sample, balanced two-block crossover, seed, device assignment, warm-up and
cache-reset procedure, statistical gates, spending ceiling, abort rules and
no-rerun rule. The held-out contract was executed exactly once.

## What passed

Across the 180 paired tasks:

- the intentional backend-identity difference passed
- the trace and environment-identity gate passed
- both absolute task-success gates passed at 179/180
- paired task-success non-inferiority passed
- the successful-pair workflow-latency gate passed

Among the 179 pairs where both variants succeeded, the descriptive
FP8-minus-BF16 workflow-latency mean was **-391.080 ms**, with a paired 95%
interval of **[-398.606, -383.554] ms**.

## Why the verdict was `invalid`

Both variants failed the same task, and both failures were classified as
application failures, not infrastructure failures. Neither variant failed a
task the other completed. That one shared application failure left BF16 and
FP8 at 179 completed attempts out of 180. The frozen contract required 100%
completion, so at 180 attempts it allowed zero failed or incomplete attempts
per variant.

Both completion gates therefore failed. Completion is a required gate, and a
failed required gate can't be offset by gates that passed.

The latency difference was measured under the frozen protocol, but it isn't
eligible to support a verified FP8 performance claim. The verdict is
permanently **`invalid`**. No rerun, post-hoc reclassification or FP8
deployment recommendation is permitted.

## What changed afterward, prospectively

The result exposed a modeling problem for future decisions. Evidence validity,
execution terminality, paired task outcome, failure origin and performance
eligibility are different questions. RunEconomy decision schema 0.2 now records
them separately, adds common-mode paired-failure diagnostics, names the exact
decisive gate and shows the failure tolerance a completion threshold implies.

Those changes apply only to new contracts. They don't alter or reinterpret the
historical Proof 2 verdict, and we haven't rerun Proof 2 under them.

## Why this matters for agent workloads

Agent runs fail in ways request benchmarks often hide. A model can return an
unusable answer, a tool can fail, a trace can be incomplete, or both variants
can fail the same task. A credible comparison keeps those outcomes in the
population instead of letting missing work make the surviving requests look
faster.

RunEconomy reports one of four verdicts from the frozen contract:
`verified`, `regression`, `inconclusive` or `invalid`.

[Inspect the aggregate outcome](proof-2-outcome.md) or
<a href="/demo/">open the hosted Proof 1 report</a>.
