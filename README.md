# RunEconomy

<!-- mcp-name: io.github.ShreyanshMisra/runeconomy -->

**Your serving benchmark got faster. Did your agent still finish the task?**

RunEconomy is a local alpha CLI for inference engineers. Capture an agent’s
OpenAI-compatible traffic with `runeconomy proxy` or import existing traces
to see where time goes. Traces with explicit dependencies also show which
calls sit on the workflow’s critical path. Then run the same tasks
on a baseline and a candidate to test a change to the serving stack or agent.
The read-only `doctor` probe and static replay support vLLM, SGLang and
OpenAI-compatible endpoints. Hosted model identity is provider-declared.

Each experiment reports `verified`, `regression`, `inconclusive` or `invalid`.
To verify a speedup, the run must clear the declared identity, completion,
task-quality and performance checks.

## What a faster workflow can mean

| Frozen experiment | Workflow latency | Verdict and deciding evidence |
| --- | --- | --- |
| Proof 3 · BF16 to online FP8, 16 concurrent agents per variant | Mean fell 499 ms (27%) | **`verified`** · All 200 tasks succeeded in both arms and the latency gate passed on this workload. |
| Proof 4 · Remove the search tool | Median fell by half | **`regression`** · All attempts completed, but success fell from 100/100 to 54/100, below the 80% floor. |
| Proof 1 · Move dynamic metadata behind stable instructions | Mean fell 62.43 ms on 36 pairs where both arms succeeded | **`invalid`** · 37/40 baseline and 38/40 candidate tasks passed. The baseline lower bound was 79.61% against an 80% floor, and paired non-inferiority also failed. |
| Proof 2 · BF16 to FP8 | Mean fell 391 ms on 179 successful pairs | **`invalid`** · One shared application failure left each arm at 179/180 completions. The frozen contract required 180/180. |

In Proof 1, all 80 attempts completed with valid traces. On the 36 successful
pairs, the candidate also gained 105 cached tokens and cut server time to first
token by 9.08 ms. Retuning on those results would reuse a consumed held-out
workload. A fresh claim needs fresh tasks.

[Read Proof 1](https://runeconomy.vercel.app/proof/) ·
[Read Proof 2](https://runeconomy.vercel.app/articles/proof-2-fp8-near-miss/) ·
[Read Proofs 3 and 4 and the load studies](https://runeconomy.vercel.app/proofs/) ·
[Inspect the Proof 1 aggregate evidence](https://github.com/ShreyanshMisra/runeconomy/blob/main/src/runeconomy/data/public-proof-rejection.json)

## Try the workflow in five minutes

With Python 3.11 or newer, start in an empty directory. The tutorial runs
synthetic requests against a deterministic localhost server and needs no GPU
or API key.

```bash
python -m pip install runeconomy==0.3.0
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

Leave the server running. In a second terminal **in the same parent directory**,
run the commands from the generated tutorial README:

```bash
runeconomy import runeconomy-tutorial/evidence.json --output evidence.json
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

The tutorial report opens with a `verified` verdict for the mock trials. To inspect the
frozen GPU result from Proof 1, run `runeconomy demo --open`.

## Run on your own deployment

Start in a fresh private directory outside version control. Set
`VLLM_BASE_URL` to your vLLM server root and, for authenticated deployments,
keep the credential in `VLLM_API_KEY`. Save the read-only probe for validation:

```bash
runeconomy doctor --base-url "$VLLM_BASE_URL" --api-key-env VLLM_API_KEY --json > doctor.json
```

**Choose what to compare.** For a serving or prompt-layout change you can test
with captured requests, use static replay. For changes to tools, retries or
multi-turn decisions, run both complete agents with `--mode agent-process`.
The [quickstart](https://runeconomy.vercel.app/start/) shows that setup.

For static replay, start the proxy in your private directory:

```bash
runeconomy proxy --upstream "$VLLM_BASE_URL" --record session.jsonl
```

Point your agent’s client base URL at the loopback URL printed by the proxy,
run representative tasks, then stop the proxy with Ctrl-C. The capture can
contain prompts and model output. Review its gap counters before import.
Supply your serving identity, evaluator and replay design in
`replay-context.json` as shown in the [capture guide](https://runeconomy.vercel.app/proxy/).

```bash
runeconomy import session.jsonl --type proxy --replay-context replay-context.json --output evidence.json
runeconomy experiment init --from-evidence evidence.json --output experiment/
runeconomy experiment validate experiment/ --doctor-report doctor.json
runeconomy experiment plan experiment/
```

Already have OpenInference/OTLP, Phoenix, vLLM or Prefixwise evidence?
Start with `runeconomy import SOURCE --output evidence.json`. An *unresolved
workspace* means import succeeded but a required input, such as a pinned
model revision or task evaluator, is missing. Supply the missing evidence
and initialize a fresh workspace. Once validation passes and you have reviewed
the plan, execute the comparison:

```bash
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

[Evidence formats](https://runeconomy.vercel.app/evidence/) ·
[Experiment contract](https://runeconomy.vercel.app/experiments/) ·
[Connector support](https://runeconomy.vercel.app/connectors/) ·
[CLI reference](https://runeconomy.vercel.app/cli/)

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
- Each command that returns a result can emit `--json` output that follows a versioned contract
  ([JSON contracts](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/json-contracts-patch-freeze.md)).
- The GitHub Action posts a verdict table on pull requests
  ([GitHub Action](https://github.com/ShreyanshMisra/runeconomy/blob/main/docs/github-action.md)).

## Troubleshooting and bug reports

Record the displayed `runeconomy.*` error code. To collect diagnostics locally:

```bash
runeconomy support-bundle experiment/ --output runeconomy-support.zip
```

Inspect the archive before sharing it. Keep raw captures and experiment
artifacts private. [Troubleshooting](https://runeconomy.vercel.app/troubleshooting/) ·
[Privacy and support bundles](https://runeconomy.vercel.app/privacy/)

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
