# Your serving benchmark got faster. Did your agent?

**Verify that a serving change or model migration makes your agent cheaper or
faster without breaking it.**

<section class="verdict-preview" aria-labelledby="proof-verdicts">
  <p class="eyebrow">Recorded verdicts</p>
  <h2 id="proof-verdicts">When does a faster workflow count as a win?</h2>
  <div class="verdict-grid">
    <article>
      <p class="proof-label">Proof 3 · Online FP8</p>
      <p class="verdict-value">verified</p>
      <p>With 16 concurrent agents per variant, FP8 cut mean workflow latency by 499 ms (27%). All 200 tasks succeeded in both arms and every frozen gate passed.</p>
      <a href="/proofs/#proof-3">Read Proof 3</a>
    </article>
    <article>
      <p class="proof-label">Proof 4 · Search tool removed</p>
      <p class="verdict-value">regression</p>
      <p>Removing search halved median latency. All attempts completed, but task success fell from 100% to 54%, below the 80% floor.</p>
      <a href="/proofs/#proof-4">Read Proof 4</a>
    </article>
    <article>
      <p class="proof-label">Proof 1</p>
      <p class="verdict-value">invalid</p>
      <p>The candidate measured 62 ms faster on 36 both-successful pairs, but the baseline success lower bound was 79.61% against an 80% floor. Paired non-inferiority also failed.</p>
      <a href="/proof/">Read Proof 1</a>
    </article>
    <article>
      <p class="proof-label">Proof 2 near-miss</p>
      <p class="verdict-value">invalid</p>
      <p>Every quality and latency gate passed. FP8 measured 391 ms faster, but one shared application failure broke the zero-tolerance completion gate we chose.</p>
      <a href="/articles/proof-2-fp8-near-miss/">Read Proof 2</a>
    </article>
  </div>
  <p class="report-link"><a href="/demo/">Open the real bundled report</a></p>
</section>

RunEconomy joins paired workflow measurements to frozen task-quality,
completion and identity gates, then returns `verified`, `regression`,
`inconclusive` or `invalid`.

The alpha supports local, operator-run experiments, sequentially or at fixed
concurrency, for controlled private testing. [Read all current boundaries](/alpha/).

## The two decisions

1. **Change self-hosted serving.** Compare BF16 with FP8, AWQ or NVFP4, upgrade
   an engine, or enable speculative decoding.
2. **Move from a hosted API to self-hosted open weights.** Compare the complete
   agent workflow, including the evaluator, rather than request latency alone.

<!-- INSTALL -->

## Integration paths

**Local proxy capture.** Point your OpenAI-compatible client at the loopback
URL printed by `runeconomy proxy` to record a workflow. Import the private
session file and check which requests and metrics it captured. [Capture guide](/proxy/).

**Static replay.** Compare baseline and candidate requests using captured
traffic or imported traces. The read-only `doctor` probe and static replay
support vLLM, SGLang and OpenAI-compatible endpoints. Hosted model identity
is provider-declared.

**Agent-process experiments.** Run both complete agent commands against the
same tasks when a tool change, retry or model migration can alter later calls.

[Follow the experiment workflow](/start/) or
[give a coding agent the RunEconomy contract](/agents/).

## Where RunEconomy fits

These categories answer adjacent questions. They are not designed for the same
frozen, paired decision RunEconomy makes.

| Category | Examples | Designed to answer | Not designed for |
| --- | --- | --- | --- |
| Throughput benchmarkers | `vllm bench serve`, GuideLLM, AIPerf | How the serving system performs at the request and load level | Whether a serving change preserves success for a specific multi-step agent workflow |
| Eval platforms | Braintrust, Promptfoo, Inspect AI | Whether model or application outputs pass evaluators | A paired serving experiment with frozen backend identity, completion and performance gates |
| Observability tools | Langfuse, Phoenix, LangSmith, Prefixwise | What happened in production traces and where time or tokens went | A preregistered baseline-versus-candidate verdict |

RunEconomy imports or complements evidence from these layers. It does not
replace load testing, evaluators or observability.
