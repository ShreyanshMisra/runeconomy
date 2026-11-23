# Proof 2 calibration design

Status: calibration contract only; not frozen for held-out execution
Selection date: 2026-11-23
Source branch: `proof-2`

## Intervention selected before model evaluation

Proof 2 compares the same pinned `Qwen/Qwen2.5-14B-Instruct` checkpoint on
the same vLLM 0.34.0 stack and L40S hardware:

- Baseline: BF16 weights and activations.
- Candidate: vLLM online `fp8_per_tensor` quantization, which applies FP8
  E4M3 per-tensor weights and dynamic activations at load time.

The source model revision, tokenizer revision, chat template, request content,
agent code, task order, generation parameters, tool, evaluator, and GPU class
are controlled. The served-model identity differs deliberately so RunEconomy
can establish the declared backend-identity intervention. Strict quantization
mode makes startup fail instead of silently falling back to a different
activation scheme.

This intervention was selected because quantization is a common deployment
decision with a real latency-versus-task-quality question. Online quantization
uses the exact Phase 3 model revision instead of introducing a separately
published quantized checkpoint. vLLM documents FP8 W8A8 support on Ada GPUs,
the architecture used by L40S. The supporting references are the
[vLLM quantization guide](https://docs.vllm.ai/en/v0.34.0/features/quantization/)
and [online quantization contract](https://docs.vllm.ai/en/v0.34.0/features/quantization/online/).

Standard speculative decoding was not selected because vLLM's standard
rejection sampler is designed to preserve the target distribution; that makes
it a weaker fit for the requested quality tradeoff. AWQ and third-party FP8
checkpoints were not selected because they would add a checkpoint-production
and revision confound. A vLLM version upgrade was already covered by the Phase
3 compatibility work and would be less specific than a precision intervention.

## Calibration workload

The agent performs two schema-constrained inference calls around one real,
bounded repository-search tool. The first call plans an exact search. The tool
reads five candidate source paths and derives the enclosing identifier from
the Python AST. The second call answers from that evidence. A deterministic
evaluator requires the exact identifier.

Two disjoint task families are constructed mechanically from the repository
snapshot:

1. unique string-literal ownership;
2. unique call-expression ownership.

The builder excludes queries from the consumed first public proof, audits one
static answer per task, and creates disjoint five-task pilot, 40-task
calibration, and 240-task draft held-out pools. The draft pool is preparation
input only: it is not frozen, has not been sent to either backend, and cannot
be executed by the calibration-only Modal entrypoints.

## Calibration decisions

The five-task pilot may repair infrastructure or evidence-capture mechanics.
The 40-task calibration estimates baseline reliability, candidate reliability,
paired task-success disagreement, paired successful-workflow latency effect,
and paired latency dispersion. Calibration may inform the proposed sample size
and scientifically meaningful margin, but it cannot become held-out evidence.

The eventual held-out contract must use a fixed sample selected from the
untouched draft pool, randomized task-paired order, deterministic evaluators,
and exactly one execution. Contract freeze and held-out execution require a
new explicit approval.

## Cost and stop controls

At the current published Modal rate of $1.95 per L40S GPU-hour, two L40S GPUs
cost $3.90 per GPU-hour. The five-task pilot is budgeted for at most 12 minutes
($0.78 GPU) and the 40-task calibration for at most 20 minutes ($1.30 GPU), for
an expected GPU ceiling of $2.08 plus small CPU, memory, storage, and network
charges. The operational calibration stop limit is $5, below the Phase 4
ceiling. No paid held-out compute is authorized by this document.
