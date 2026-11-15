# RunEconomy Experiment Report

Qwen2.5-7B-Instruct · vLLM 0.33.0 · A10 · 40 paired tasks · 80 attempts

## Intervention Tested

vLLM can reuse cached prompt tokens while a new prompt matches a previous prompt from the beginning. In the baseline, changing metadata appeared first, so the shared prefix ended immediately. The candidate moved that metadata after the stable instructions.

- **Baseline:** Changing run metadata appeared before stable system instructions.
- **Candidate:** The same changing metadata moved after stable system instructions.
- **Why this might help:** A longer stable prefix may allow more prompt tokens to be reused from cache.
- **Control:** The prompt content stayed the same. Only the block order changed.

## Results

### Cached input tokens

**+105**

More prompt tokens were served from cache.

### Server time to first token

**9 ms lower**

The first token arrived sooner at the server.

### Workflow latency

**62 ms lower**

Successful task pairs completed sooner.

**Important:** Measured across the 36 task pairs where both variants succeeded. These measurements are not a verified improvement until the quality requirements pass.

- **Baseline:** 37 of 40 tasks succeeded.
- **Candidate:** 38 of 40 tasks succeeded.
- **Both succeeded:** 36 pairs
- **Baseline only succeeded:** 1 pair
- **Candidate only succeeded:** 2 pairs
- **Both failed:** 1 pair

### Cache reuse by agent call

| Agent call | Mean cached input tokens across both-successful pairs |
|---|---|
| Planning call | Baseline 0.00, candidate 108.89, difference 108.89 cached tokens |
| Final answer call | Baseline 196.00, candidate 192.44, difference -3.56 cached tokens |

## Verdict

### Result not verified

The candidate showed better cache reuse and lower measured latency, but this experiment did not establish the task-quality evidence required for a verified performance claim.

Verdict: Invalid

Invalid does not mean the candidate was slower. It means the attractive measurements were not eligible to become a verified improvement or regression claim.

### Was the baseline workload reliable enough?

| Measure | Recorded result |
|---|---|
| Required lower confidence bound | 80.00% |
| Observed baseline lower bound | 79.61% |
| Result | Failed by 0.39 percentage points |

The baseline succeeded on 37 of 40 tasks. With only 40 tasks, the lower confidence bound remained slightly below the required threshold.

### Did the candidate preserve task quality?

| Measure | Recorded result |
|---|---|
| Maximum allowed regression | 10.00 percentage points |
| Observed lower-bound difference | negative 12.55 percentage points |
| Result | Failed by 2.55 percentage points |

The candidate's observed success rate was higher, but the uncertainty was too wide to rule out a regression larger than the allowed margin.

### Decision path

1. **Evidence usable?** Passed
1. **Baseline reliable enough?** Failed
1. **Performance claim:** Not eligible for verification

- Candidate absolute quality: Passed
- Paired quality preservation: Failed
- Conditional workflow latency: Favorable but not eligible

**Recommended next step:** Do not present this candidate as a verified latency improvement. Establish a more reliable baseline and test the intervention on a new, sufficiently powered held-out workload. The current 40-task workload has been consumed and cannot be reused as fresh evidence.

## Experiment Details

Each task gave the agent a code clue and a bounded set of repository files. The agent planned a search, used a repository-search tool and identified the enclosing Python function or class.

| Experiment fact | Recorded value |
|---|---|
| Task families | 13 direct-call lookup tasks, 13 docstring lookup tasks and 14 unique-string lookup tasks |
| Paired workload | 40 baseline and candidate pairs, 80 total attempts |
| Workflow | 2 model calls and 1 repository-search tool call per attempt |
| Environment | Qwen2.5-7B-Instruct, vLLM 0.33.0, Modal A10 |
| Execution | Chat Completions, colocated driver and sequential execution |

**Controls:** The model, backend, hardware, agent, tasks, evaluator and acceptance rules were chosen and locked before the held-out workload was run.

## Technical Details

<details>
<summary>Acceptance checks</summary>

| Field | Recorded value |
|---|---|
| trace_and_environment_identity | Passed. Threshold: every attempt has an explicit complete DAG and each variant has complete, stable environment identity. Observed: Baseline: complete DAG count 40, complete DAG rate 1.0, environment identity complete yes, environment identity stable yes, valid trace count 40. Candidate: complete DAG count 40, complete DAG rate 1.0, environment identity complete yes, environment identity stable yes, valid trace count 40 |
| baseline_minimum_task_success | Failed. Threshold: 0.8. Observed: 0.7961352512671009 to 0.9842578201489585 |
| candidate_minimum_task_success | Passed. Threshold: 0.8. Observed: 0.8308031360405823 to 0.9938863534006492 |
| task_success_noninferiority | Failed. Threshold: -0.1. Observed: candidate minus baseline mean 0.025, confidence interval -0.14498205538583642 to 0.18780457959853636, confidence level 0.95, method exact marginal binomial bounds with Bonferroni coverage, one sided lower bound -0.12547221188341495, outcome counts (baseline only 1, both failure 1, both success 36, candidate only 2), paired count 40 |
| baseline_completion | Passed. Threshold: 1.0. Observed: 1.0 |
| candidate_completion | Passed. Threshold: 1.0. Observed: 1.0 |
| successful_task_workflow_latency_improvement | Passed. Threshold: upper bound of paired 95% Student-t interval < 0. Observed: available yes, candidate minus baseline mean -62.43227297222221, confidence interval -123.85694821653173 to -1.0075977279126818, confidence level 0.95, method paired Student-t interval, paired count 36 |
| successful_task_workflow_latency_non_regression | Passed. Threshold: lower bound of paired 95% Student-t interval <= 0. Observed: available yes, candidate minus baseline mean -62.43227297222221, confidence interval -123.85694821653173 to -1.0075977279126818, confidence level 0.95, method paired Student-t interval, paired count 36 |

</details>

<details>
<summary>Statistical details</summary>

| Field | Recorded value |
|---|---|
| Cached Input Tokens | Mean candidate minus baseline: 105.333. 95% interval: 98.648 to 112.018. Pairs: 36 |
| Client Ttft Ms | Mean candidate minus baseline: -9.386. 95% interval: -12.206 to -6.566. Pairs: 36 |
| Cost Usd | Mean candidate minus baseline: Unknown. 95% interval: Unknown. Pairs: 0 |
| Critical Path Ms | Mean candidate minus baseline: -61.980. 95% interval: -123.439 to -0.522. Pairs: 36 |
| Decode Ms | Mean candidate minus baseline: Unknown. 95% interval: Unknown. Pairs: 0 |
| Gpu Time Ms | Mean candidate minus baseline: Unknown. 95% interval: Unknown. Pairs: 0 |
| Inference Span Ms | Mean candidate minus baseline: -61.914. 95% interval: -123.346 to -0.483. Pairs: 36 |
| Input Tokens | Mean candidate minus baseline: -3.444. 95% interval: -5.311 to -1.578. Pairs: 36 |
| Output Tokens | Mean candidate minus baseline: -1.556. 95% interval: -3.419 to 0.308. Pairs: 36 |
| Prefill Ms | Mean candidate minus baseline: Unknown. 95% interval: Unknown. Pairs: 0 |
| Queue Ms | Mean candidate minus baseline: -0.040. 95% interval: -0.334 to 0.254. Pairs: 36 |
| Retrieval Span Ms | Mean candidate minus baseline: Unknown. 95% interval: Unknown. Pairs: 0 |
| Server Ttft Ms | Mean candidate minus baseline: -9.075. 95% interval: -11.846 to -6.304. Pairs: 36 |
| Tool Span Ms | Mean candidate minus baseline: -0.066. 95% interval: -0.199 to 0.067. Pairs: 36 |
| Workflow Latency Ms | Mean candidate minus baseline: -62.432. 95% interval: -123.857 to -1.008. Pairs: 36 |
| Paired task success | Mean candidate minus baseline: 2.50 percentage points. 95% interval: -14.50 percentage points to 18.78 percentage points. One-sided lower bound: -12.55 percentage points. Method: exact marginal binomial bounds with Bonferroni coverage |
| Planning call | Baseline mean: 0.00. Candidate mean: 108.89. Difference: 108.89 cached tokens. Pairs: 36 |
| Final answer call | Baseline mean: 196.00. Candidate mean: 192.44. Difference: -3.56 cached tokens. Pairs: 36 |

</details>

<details>
<summary>Environment and provenance</summary>

| Field | Recorded value |
|---|---|
| Api Dialect | chat_completions |
| Backend | vllm |
| Backend Version | 0.33.0 |
| Chat Template Digest | sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f |
| Driver Location | same Modal container as vLLM |
| Engine Configuration | {"block_size": 16, "gpu_memory_utilization": 0.85, "max_model_len": 8192, "max_num_seqs": 8} |
| Hardware | A10 |
| Model | Qwen/Qwen2.5-7B-Instruct |
| Model Revision | a09a35458c702b33eeacc393d103063234e8bc28 |
| Tokenizer | Qwen/Qwen2.5-7B-Instruct |
| Tokenizer Revision | a09a35458c702b33eeacc393d103063234e8bc28 |
| Agent hash | sha256:e6f42c619f15be192512d234ee21b623a09e69402a62ab16516190c9ea348dcb |
| Comparison hash | sha256:d8df549ebcb7d814aa1d70b02e2d9c482830e02d7812a40376d1c8161dd31e8f |
| Driver hash | sha256:1a6a589353e5352116a58ee446493fbc83c486f4fc7634a49483ac81b474b1b1 |
| Environment Identity hash | sha256:a62037eec6d2c8d31743f404018469cb8d42c9359b1fe5a84f1e0053c86638ef |
| Evaluator hash | sha256:8f6f0ac64145392fd9f2086705345871628fce3957bae810f5cd551be4b02dba |
| Experiment Design hash | sha256:51975b24f8792f6f9ab453fb6a2fcba84ab1247e9f1a230e3ca0b9910564a48e |
| Private Result hash | sha256:783c08972d972b2cc7aad610c13d9df934d8f4c573915ad6423ebaf74402aa99 |
| Status Protocol hash | sha256:b3885985bfa88510d901600b28e3d6341427b2390b3dedb2c16ccd124e68e723 |
| Workload hash | sha256:85e53d2d2f2ab108b5bc83e27947a67d710ed53326a89467005630dea3bddf61 |

</details>

<details>
<summary>Evidence integrity</summary>

| Field | Recorded value |
|---|---|
| Attempt Count | 80 |
| Attempt Status Schema Versions | ['0.1'] |
| Complete Dag Count | 80 |
| Environment Identity Complete | True |
| Environment Identity Stable | True |
| Generated Content Included | False |
| Raw Attempts Included | False |
| Valid Trace Count | 80 |
| Evaluator Failure | 0 |
| Invalid Trace | 0 |
| Process Or Infrastructure Failure | 0 |
| Task Quality Failure | 5 |
| Timeout | 0 |

</details>

<details>
<summary>Limitations</summary>

| Field | Recorded value |
|---|---|
| Supported claim | RunEconomy rejected an apparently faster, more cache-efficient candidate because the frozen successful-task quality requirements were not established. |
| Claims not supported | a verified performance improvement, universal model behavior, production-concurrency performance, superiority over another tool |
| Limitations | One pinned model revision, vLLM version, A10 hardware class, and repository-analysis workload were evaluated. Execution was sequential and colocated. Production concurrency and multi-replica routing were not tested. Continuous effects use only the 36 task pairs where both variants succeeded. Workflow latency includes process startup and agent execution but excludes evaluator time. Prefill, decode, retrieval, cost, and GPU-time effects were unavailable and remain unknown rather than zero. The 40-task held-out workload is consumed and cannot be reused as fresh held-out evidence. External readers can audit the aggregate arithmetic, gate logic, provenance hashes, and claim boundary, but cannot independently recompute the experiment without the private raw tasks and traces. |
| Artifact source | src/runeconomy/data/public-proof-rejection.json |

</details>
