# Start with RunEconomy

Move from a clean installation to a quality-gated experiment. The installed
tutorial works offline. A real experiment requires your evidence and a
compatible backend.

## 1. Install

RunEconomy requires Python 3.11 or newer.

```bash
python -m pip install runeconomy==0.3.0
runeconomy --version
```

## 2. Try the complete workflow locally

```bash
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

The first command creates a sanitized, versioned directory. The second starts a
deterministic server on `127.0.0.1:8765`. It is not vLLM and does not run a
model. It exercises the CLI using only localhost, without credentials, a GPU,
or paid compute.

Leave the server running. Open a second terminal in the directory containing
`runeconomy-tutorial/` and follow `runeconomy-tutorial/README.md`. It gives the complete
`import`, `experiment init`, `validate`, `run`, and `report` sequence. The final
report is generated from newly executed controlled mock trials. It is not a
relabelled historical trace or a real performance benchmark.

The tutorial uses a RunEconomy-enriched OpenInference fixture with explicit
prompt tokens, prompt segments, workflow dependencies, and experiment context.
Ordinary OpenTelemetry or OpenInference evidence may support diagnosis while
still producing an unresolved workspace. This means import succeeded, but
execution needs a missing input such as a model revision or a task evaluator.

To inspect the bundled frozen proof with an `invalid` verdict, run:

```bash
runeconomy demo --open
```

The proof demo makes no network or GPU calls and does not rerun the consumed
workload. Read the [complete public proof](public-proof.md) before quoting its
conditional measurements.

## 3. Check a real vLLM deployment safely

Start in a new private directory outside version control. Set `VLLM_BASE_URL`
to the server root and keep its credential in `VLLM_API_KEY` if required.

```bash
runeconomy doctor \
  --base-url "$VLLM_BASE_URL" \
  --api-key-env VLLM_API_KEY \
  --json > doctor.json
```

`doctor` is read-only. It does not generate text or reset caches. It reports
supported, unsupported, and unknown capabilities, plus the safe experiment
modes and exact next action.

For SGLang, add `--backend sglang` and use its server root. For a hosted
OpenAI-compatible endpoint, use `--backend openai_compat`, its API base URL
and the name of its credential variable. [Serving backends](backends.md).

## 4. Choose your experiment mode

- **Static replay** runs captured requests against baseline and candidate
  serving configurations. Choose it for questions such as how prompt layout
  affects prefix caching. Follow the static replay path below.
- **Agent-process** runs each agent command from start to finish. Choose it
  when changing tools, retry logic or the model can change what the agent
  does next. Go directly to the agent-process path below.

### Static replay from captured or exported requests

If you have OpenInference/OTLP JSON, a Phoenix export, vLLM capture JSONL or a
Prefixwise ledger, run `runeconomy import SOURCE --output evidence.json`.
Read the capability assessment to see whether the artifact supports replay.
To capture new requests from an OpenAI-compatible client:

```bash
runeconomy proxy --upstream "$VLLM_BASE_URL" --record session.jsonl
```

In another terminal, run the agent with its base URL set to the loopback URL
the proxy prints. Stop the proxy with Ctrl-C after the last request and check
the session summary for gaps. Full capture stores prompts and responses
locally. Prepare a private `replay-context.json` with your evaluator,
pinned serving identity and replay design. [Capture guide and example context](proxy.md).

```bash
runeconomy import session.jsonl --type proxy --replay-context replay-context.json --output evidence.json
```

Initialize from the imported bundle. If it contains several findings, pass
the `--finding-id` reported for the one you want to test.

```bash
runeconomy experiment init --from-evidence evidence.json --output experiment/
```

### Agent-process with your existing entry points

Integrate `runeconomy.agent.Attempt` in a runner-only wrapper or guard it
with `RUNECONOMY_STATUS_PATH` in a shared entry point. Test both paths with a
local stub. Keep your existing tracing, or record steps and their explicit
dependencies with `runeconomy.track_workflow` and `runeconomy.step`.
Use your own commands, dataset and evaluator in this example.
[Agent contract](closed-loop-comparison.md) · [Coding-agent integration guide](agent-skill.md).

```bash
runeconomy experiment init --mode agent-process \
  --baseline "python agent_v1.py" --candidate "python agent_v2.py" \
  --dataset tasks.json --evaluator evaluator.py --output experiment/
```

## 5. Validate the inputs and review the plan

```bash
runeconomy experiment validate experiment/ --doctor-report doctor.json
runeconomy experiment plan experiment/
```

Validation identifies missing inputs. Static replay also checks the saved
doctor report for the required API dialect. For agent-process mode, use
`runeconomy experiment validate experiment/` without the report. If a
static workspace is unresolved, use the reported fields to complete the source
evidence or replay context, re-import and initialize a new directory. On a
runnable workspace, use `experiment patch` to record design changes.
Planning estimates sample size and cost before execution. Review the commands,
evaluator, gates, private output locations and any design issues it reports.

## 6. Run and open the report

```bash
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json --output report.html --open
```

For a held-out test, freeze the reviewed design before running and add
`--require-frozen` to the run command. [Patch and freeze guide](json-contracts-patch-freeze.md).

## 7. Read the verdict

- `verified` means declared quality and performance criteria passed.
- `regression` means a declared candidate guardrail failed.
- `inconclusive` means usable evidence was insufficient for the claim.
- `invalid` means the experiment could not support an improvement or regression
  claim. It does not mean the candidate was slower.

Cache reuse, token reductions, TTFT, and latency remain conditional observations
unless the declared quality and performance gates make the claim eligible.

## 8. Prepare a privacy-minimized support artifact

If setup or execution fails, record the displayed `runeconomy.*` error code and
create a local support bundle:

```bash
runeconomy support-bundle experiment/ \
  --output runeconomy-support.zip
```

The command does not upload anything or contact the backend. Inspect every file
inside the archive before attaching it to an issue. Do not attach raw traces,
results, prompts, generated output, evaluator data, or credentials.

Read the complete [privacy and support-bundle contract](privacy.md).

Continue with [Evidence formats](evidence-formats.md), the
[experiment contract](experiment-format.md), [connector support](connectors.md),
or [Troubleshooting](troubleshooting.md).
