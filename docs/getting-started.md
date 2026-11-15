# Start with RunEconomy

Move from a clean installation to a quality-gated experiment. The bundled demo
works offline. A real experiment requires your own evidence and a compatible
vLLM server.

## 1. Install

RunEconomy requires Python 3.11 or newer. This guide describes the `0.2.0a3`
alpha. It is a prerelease, so request it explicitly:

```bash
python -m pip install --pre runeconomy==0.2.0a3
runeconomy --version
```

## 2. Run the bundled demo

```bash
runeconomy demo --open
```

The demo opens a self-contained experiment report. It uses sanitized aggregate
evidence and makes no network or GPU calls. It does not rerun the proof
experiment.

The report deliberately shows a rejected result. Cache reuse and measured
latency looked better, but the frozen task-quality requirements did not pass.
Read the [complete public proof](public-proof.md) before quoting its conditional
measurements.

## 3. Check your server with doctor

Keep authentication in an environment variable:

```bash
export VLLM_API_KEY='replace-me'
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

`doctor` is read-only. It checks metadata and safe render capabilities. It does
not generate text or reset caches. Continue with the
[vLLM compatibility guide](doctor.md) when a capability is unsupported or
unknown.

## 4. Import or instrument evidence

Use the importer that matches evidence you actually possess:

```bash
runeconomy import-openinference otlp.json --output trace.json
runeconomy import-vllm-jsonl captures.jsonl --output trace.json \
  --trace-id TRACE --model MODEL
runeconomy import-ledger-jsonl ledger.jsonl --output findings.json
```

Then analyze a normalized trace:

```bash
runeconomy analyze trace.json --format json --output analysis.json
```

Importers preserve unknown values instead of inventing prompts, token arrays,
timing or dependencies. The [trace format](trace-format.md) explains the
evidence boundary.

## 5. Initialize an experiment

```bash
runeconomy experiment init \
  --from-finding analysis.json \
  --finding-id FINDING_ID \
  --output experiment.json
```

Some findings produce runnable manifests. Others produce an unresolved
skeleton because RunEconomy cannot safely invent the application change,
workload or evaluator. Resolve only fields supported by real evidence.

## 6. Validate before execution

```bash
runeconomy experiment validate experiment.json
```

Review the evaluator, task-quality gates, performance metric, cache treatment,
model identity and cost ceiling before running anything. See the
[experiment format](experiment-format.md) for the complete contract.

## 7. Run and report

```bash
runeconomy experiment run experiment.json --output result.json
runeconomy experiment report result.json --output report.html
```

Use `runeconomy compare` only when executing complete baseline and candidate
agent processes. The [closed-loop comparison guide](closed-loop-comparison.md)
documents that integration.

## 8. Read the verdict

- `verified` means the declared quality and performance criteria passed.
- `regression` means a declared candidate guardrail failed.
- `inconclusive` means the evidence was usable but insufficient for the claim.
- `invalid` means the experiment or baseline did not establish valid evidence.

A favorable cache, token or latency measurement is not a verified improvement
unless the declared gates pass.

## When something fails

Start with [Troubleshooting](troubleshooting.md), then use the
[CLI reference](cli-reference.md) for exit codes and machine-readable output.
Review the [alpha boundaries](alpha-release.md) before relying on RunEconomy in
an unattended workflow.
