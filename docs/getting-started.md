# Start with RunEconomy

Move from a clean installation to a quality-gated experiment. The bundled demo
works offline. A real experiment requires your evidence and a compatible
backend.

## 1. Install and inspect the proof

RunEconomy requires Python 3.11 or newer.

```bash
python -m pip install --pre runeconomy==0.2.0a4
runeconomy --version
runeconomy demo --open
```

The demo renders sanitized aggregate evidence. It makes no network or GPU calls
and does not rerun the consumed proof workload.

## 2. Check vLLM safely

Keep authentication in an environment variable:

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY \
  --json > doctor.json
```

`doctor` is read-only. It does not generate text or reset caches. It reports
supported, unsupported, and unknown capabilities, plus the safe experiment
modes and exact next action.

## 3. Import evidence

The unified importer detects supported evidence formats and records what each
artifact can support:

```bash
runeconomy import otlp.json --output evidence.json
```

Use `--type` only when detection is ambiguous. Supported inputs include
OpenInference or OpenTelemetry OTLP JSON, Phoenix JSON traces, exact-token vLLM
capture JSONL, and Prefixwise ledger JSONL. The importer does not invent prompt
tokens, dependencies, task outcomes, or replay requests.

## 4. Initialize and validate a workspace

```bash
runeconomy experiment init \
  --from-evidence evidence.json \
  --output experiment/

runeconomy experiment validate experiment/ \
  --doctor-report doctor.json
```

The workspace records the source digest, selected finding, intervention,
experiment mode, gates, metrics, identity, and unresolved fields. If several
findings are viable, select one with `--finding-id`. If evidence is incomplete,
validation explains the exact field and safe correction. An unresolved
workspace is not runnable.

## 5. Run and report

Review the generated commands, evaluator, quality gates, cost ceiling, and
private artifact locations before execution.

```bash
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json \
  --output report.html \
  --open
```

For a new real-agent experiment, initialize the same workspace family directly:

```bash
runeconomy experiment init \
  --mode agent-process \
  --baseline "python agent_v1.py" \
  --candidate "python agent_v2.py" \
  --dataset tasks.json \
  --evaluator evaluator.py \
  --output experiment/
```

The older `compare` command remains a compatibility wrapper. New integrations
should use `experiment init`, `validate`, `run`, and `report` for both modes.

## 6. Read the verdict

- `verified` means declared quality and performance criteria passed.
- `regression` means a declared candidate guardrail failed.
- `inconclusive` means usable evidence was insufficient for the claim.
- `invalid` means the experiment could not support an improvement or regression
  claim. It does not mean the candidate was slower.

Cache reuse, token reductions, TTFT, and latency remain conditional observations
unless the declared quality and performance gates make the claim eligible.

Continue with [Evidence formats](evidence-formats.md), the
[experiment contract](experiment-format.md), [connector support](connectors.md),
or [Troubleshooting](troubleshooting.md).
