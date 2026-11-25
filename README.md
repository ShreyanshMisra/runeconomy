# RunEconomy

RunEconomy identifies inference waste on the critical path of an agent workflow
and tests whether a change improves successful-task performance. It is an alpha
CLI for engineers running multi-step agents on self-hosted vLLM or SGLang, or
comparing a hosted OpenAI-compatible API against self-hosted serving.

RunEconomy sits between agent observability and inference infrastructure:

```text
import evidence → initialize → validate → run → report
```

It reports `verified`, `regression`, `inconclusive`, or `invalid`. Favorable
cache, token, TTFT, or latency measurements are not called improvements unless
the declared task-quality and performance gates pass.

## The flagship result is an honest invalid verdict

In the frozen 40-task proof, the candidate gained about 105 cached tokens,
reduced server TTFT by about 9.08 ms, and reduced workflow latency by about
62.43 ms among the 36 pairs where both variants succeeded. RunEconomy still
returned **`invalid`**: the baseline task-success lower bound was 79.61%, below
the frozen 80% gate, and paired success non-inferiority also failed.

All 80 attempts completed with valid traces. The quality gate, not
infrastructure, made the apparent speedup ineligible for verification. The
workload is consumed and cannot be reused or retuned as fresh held-out evidence.

[Read the complete public proof](https://runeconomy.vercel.app/proof/) ·
[Inspect the aggregate evidence](https://github.com/ShreyanshMisra/runeconomy/blob/main/src/runeconomy/data/public-proof-rejection.json)

## Install and see it in five minutes

RunEconomy requires Python 3.11 or newer. The tutorial is CPU-only and makes no
network or GPU calls.

```bash
python -m pip install runeconomy==0.3.0
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

Follow the generated README in another terminal. The tutorial is deterministic
and local. It validates product plumbing, not real model performance. It uses
RunEconomy-enriched OpenInference evidence. Ordinary traces may support
diagnosis while still producing an unresolved workspace.

Inspect the frozen public proof separately with `runeconomy demo --open`.

Then diagnose a vLLM deployment without generating text or resetting caches:

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY
```

`doctor` reports supported, unsupported, and unknown capabilities, safe
experiment modes, and an exact next action. Use `--json` for automation.

When a tester needs help, create a local privacy-minimized support archive:

```bash
runeconomy support-bundle experiment/ --output runeconomy-support.zip
```

RunEconomy does not upload the archive or copy raw traces, prompts, results,
credentials, endpoints, or arbitrary workspace files. Inspect every file before
sharing it.

Read the [privacy and support-bundle contract](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/privacy.md) before attaching
an archive to an issue.

[Follow the complete quickstart](https://runeconomy.vercel.app/start/) ·
[Read the compatibility guide](https://runeconomy.vercel.app/doctor/) ·
[CLI reference](https://runeconomy.vercel.app/cli/) ·
[Troubleshoot setup](https://runeconomy.vercel.app/troubleshooting/)

## Move from evidence to a verdict

```bash
runeconomy import otlp.json --output evidence.json
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/
runeconomy experiment plan experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

For real agents, the official coding-agent skill integrates the existing entry
point with `runeconomy.agent.Attempt`. The helper implements only the status
contract. It does not replace the application's tracing or agent framework.

The importer explains which diagnoses, replay modes, and claims the evidence can
support. An incomplete import creates an unresolved workspace with corrective
actions instead of invented prompts, tokens, dependencies, or evaluators.

`experiment` supports both static request comparisons and complete baseline and
candidate agent processes. `compare` remains a compatibility wrapper for older
agent-process integrations.

[Evidence formats](https://runeconomy.vercel.app/evidence/) ·
[Experiment contract](https://runeconomy.vercel.app/experiments/) ·
[Connector support](https://runeconomy.vercel.app/connectors/) ·
[How RunEconomy works](https://runeconomy.vercel.app/articles/how-runeconomy-works/)

## Give your coding agent the same contract

The canonical [RunEconomy agent skill](https://github.com/ShreyanshMisra/runeconomy/tree/main/skills/runeconomy) teaches an
agent how to inspect compatibility, acquire evidence, prepare an experiment,
preserve workflow state, interpret verdicts, protect private material, and stop
before destructive or external operations.

```bash
npx skills add ShreyanshMisra/runeconomy --skill runeconomy --all
```

[Install the skill for Codex, Claude Code, or adapt it for Cursor](https://runeconomy.vercel.app/agents/)

Agents and CI can also drive the whole workflow without editing JSON:

- `runeconomy mcp` serves typed MCP tools, and runs execute only after an
  approved estimate ([MCP server](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/mcp.md)).
- `experiment patch` and `experiment freeze` change and lock a workspace.
- Every command's `--json` output follows a versioned contract
  ([JSON contracts](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/json-contracts-patch-freeze.md)).
- The GitHub Action posts a verdict table on pull requests
  ([GitHub Action](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/github-action.md)).

## Alpha boundaries

RunEconomy is ready for narrow private testing, not unattended production use.
It does not provide a hosted dashboard, a managed OTLP collector, or a
framework-specific SDK. Raw prompts, traces, evaluator data, and experiment artifacts remain
the operator's responsibility.

[Alpha and migration notes](https://runeconomy.vercel.app/alpha/) ·
[Development and validation](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/development.md) ·
[Documentation](https://runeconomy.vercel.app/) ·
[PyPI](https://pypi.org/project/runeconomy/) ·
[GitHub](https://github.com/ShreyanshMisra/runeconomy)

The [documentation site](https://runeconomy.vercel.app/) is generated from these version-controlled
sources under [`site/`](https://github.com/ShreyanshMisra/runeconomy/tree/main/site).
