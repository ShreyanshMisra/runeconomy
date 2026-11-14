# RunEconomy

RunEconomy identifies inference waste on the critical path of an agent workflow
and tests whether a change improves successful-task performance. It is an alpha
CLI for engineers running multi-step agents on self-hosted vLLM.

RunEconomy sits between agent observability and inference infrastructure:

```text
trace → diagnosis → controlled experiment → quality-gated verdict
```

It reports `verified`, `regression`, `inconclusive`, or `invalid`. Favorable
cache, token, TTFT, or latency measurements are not called improvements unless
the declared task-quality and performance gates pass.

## The flagship result is an honest rejection

In the frozen 40-task proof, the candidate gained about 105 cached tokens,
reduced server TTFT by about 9.08 ms, and reduced workflow latency by about
62.43 ms among the 36 pairs where both variants succeeded. RunEconomy still
returned **`invalid`**: the baseline task-success lower bound was 79.61%, below
the frozen 80% gate, and paired success non-inferiority also failed.

All 80 attempts completed with valid traces. The quality gate—not
infrastructure—rejected the apparent speedup. The workload is consumed and
cannot be reused or retuned as fresh held-out evidence.

[Read the complete public proof](docs/public-proof.md) ·
[Inspect the aggregate evidence](src/runeconomy/data/public-proof-rejection.json)

## Install and see it in five minutes

RunEconomy requires Python 3.11 or newer. The bundled proof is CPU-only and
makes no network or GPU calls.

```bash
python -m pip install --pre runeconomy==0.2.0a2
runeconomy demo --open
```

Then diagnose a vLLM deployment without generating text or resetting caches:

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

`doctor` reports supported, unsupported, and unknown capabilities; safe
experiment modes; and an exact next action. Use `--json` for automation.

[Follow the complete quickstart](docs/getting-started.md) ·
[Read the compatibility guide](docs/doctor.md) ·
[CLI reference](docs/cli-reference.md) ·
[Troubleshoot setup](docs/troubleshooting.md)

## Choose the right experiment

- `runeconomy experiment` runs controlled static request manifests against
  vLLM. `negative-split` remains a backward-compatible alias; NegativeSplit is
  the methodology and report name.
- `runeconomy compare` executes complete baseline and candidate agent processes
  with variant-local state, explicit dependencies, real tool time, task-level
  pairing, and durable OpenTelemetry/OpenInference evidence.
- Importers accept exact-token vLLM captures, structural
  OpenInference/OpenTelemetry traces, and evidence present in Prefixwise JSONL
  exports without inventing missing prompts, tokens, timing, or dependencies.

[Static experiment format](docs/experiment-format.md) ·
[Closed-loop comparison](docs/closed-loop-comparison.md) ·
[Trace format](docs/trace-format.md) ·
[How RunEconomy works](docs/how-it-works.md)

## Give your coding agent the same contract

The canonical [RunEconomy agent skill](skills/runeconomy/SKILL.md) teaches an
agent how to inspect compatibility, acquire evidence, prepare an experiment,
preserve workflow state, interpret verdicts, protect private material, and stop
before destructive or external operations.

[Install the skill for Codex, Claude Code, or adapt it for Cursor](docs/agent-skill.md)

## Alpha boundaries

RunEconomy is ready for narrow private testing, not unattended production use.
It does not provide a hosted dashboard, managed OTLP collector,
framework-specific SDK, automatic resume, SGLang runner, or broad diagnosis
catalog. Raw prompts, traces, evaluator data, and experiment artifacts remain
the operator's responsibility.

[Alpha and migration notes](docs/alpha-release.md) ·
[Development and validation](docs/development.md) ·
[PyPI](https://pypi.org/project/runeconomy/) ·
[GitHub](https://github.com/ShreyanshMisra/runeconomy)

The local static documentation site is generated from these version-controlled
sources under [`site/`](site/). It is prepared for Vercel but is not deployed.
