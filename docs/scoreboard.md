# Verified decision cards

Each card restates one frozen RunEconomy decision record. Proof cards
were preregistered and executed once. Study cards are descriptive and
carry no verdict. Every number links back to its record and digest.
Latency is measured on successful workflows only, so a regression card
can show lower latency for the attempts that still succeeded.

| Card | Intervention | Verdict | Task success (base → cand.) | p95 successful-workflow latency (ms) | $ per 1k successful |
| --- | --- | --- | --- | --- | --- |
| [Proof 3](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/proof-records/proof-3-decision.json) | BF16 weights versus vLLM online fp8_per_tensor quantization (one GPU per variant) | **verified** | 100% → 100% | 2,254 → 1,625 | 0.0647 → 0.0481 |
| [Proof 4](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/proof-records/proof-4-decision.json) | Remove the repository-search tool and answer directly (same model and server) | **regression** | 100% → 54% | 1,642 → 670 | not gated |

## Proof 3: verified

- **Intervention:** BF16 weights versus vLLM online fp8_per_tensor quantization (one GPU per variant)
- **Environment:** vllm 0.34.0 on NVIDIA L40S
- **Workload:** 200 pairs, closed_loop at concurrency 16
- **Decisive gate:** `all_required_gates`. task success was non-inferior and successful-task latency superiority was established
- **Record:** [`docs/proof-records/proof-3-decision.json`](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/proof-records/proof-3-decision.json) `sha256:b1eed0ec96a63d32c8b717d97553c73f33d170346fd92a6e208c38d2a8a2fd07`

Claim boundary:

- One model (Qwen/Qwen2.5-14B-Instruct at a pinned revision), one GPU type (NVIDIA L40S), and vLLM 0.34.0.
- One synthetic multi-step repository-lookup agent on a fixed repository snapshot, not production traffic.
- 16 closed-loop agents per variant, each variant on its own GPU (isolated serving topology).
- Cost uses the Modal L40S list price with fair-share attribution by in-flight time, it is not an invoice.
- Both arms succeeded on every task, so non-inferiority is shown against a 5-point margin, not equality of quality on harder work.

## Proof 4: regression

- **Intervention:** Remove the repository-search tool and answer directly (same model and server)
- **Environment:** vllm 0.34.0 on NVIDIA L40S
- **Workload:** 100 pairs, sequential at concurrency 1
- **Decisive gate:** `candidate_minimum_task_success`. Required task_outcome gate 'candidate_minimum_task_success' failed: observed [0.43741157986647183,0.6401566470205877], threshold 0.8. Performance evidence is ineligible for a claim.
- **Record:** [`docs/proof-records/proof-4-decision.json`](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/proof-records/proof-4-decision.json) `sha256:59ac25c5d43e4c2a79beb96ab701599890c0aadd65ee01303b47b5d56fa842ca`

Claim boundary:

- One model (Qwen/Qwen2.5-14B-Instruct at a pinned revision) on one vLLM 0.34.0 BF16 server.
- One synthetic multi-step repository-lookup agent whose answers need repository evidence.
- The regression shows that RunEconomy blocks a change that request benchmarks favor, it is not a claim about tool pruning in general.

## Descriptive studies

- [FP8 vs BF16 on a multi-step agent under load](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/proof-records/study-1.json): descriptive (not preregistered, the gate verdict at the top level is reported, not claimed).
- [What a dynamic value at the start of a prompt prefix costs in vLLM prefix-cache reuse](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/proof-records/study-2.json): descriptive (not preregistered).
