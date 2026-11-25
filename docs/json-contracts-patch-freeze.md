# JSON contracts, patch and freeze

Coding agents and CI pipelines can drive RunEconomy end to end with CLI flags
alone: change a workspace with `experiment patch`, lock it with `experiment
freeze`, and read every result through a versioned `--json` contract.

```bash
runeconomy experiment init --mode agent-process ... --output experiment/ --json
runeconomy experiment patch experiment/ \
  --set 'design.planning={"schema_version":"0.1", ...}' \
  --reason "declare the preregistered design" --json
runeconomy experiment plan experiment/ --json
runeconomy experiment freeze experiment/ --lock agent.py --json
runeconomy experiment run experiment/ --output results/ --require-frozen --json
runeconomy experiment report results/result.json --output report.html --json
```

## `experiment patch`

`runeconomy experiment patch WORKSPACE` changes `manifest.json` in a workspace
created by `experiment init`.

| Flag | Effect |
| --- | --- |
| `--set PATH=JSON` | Set a field to a JSON value: `acceptance.min_task_success_rate=0.9`, `execution.order='"alternating"'`. |
| `--set-string PATH=TEXT` | Set a field to a literal string without JSON quoting. |
| `--unset PATH` | Remove a field. |
| `--patch-file FILE` | Apply `{"patch_version": "0.1", "operations": [{"op": "set", "path": ..., "value": ...}]}` first. `replace`/`add` mean `set` and `remove` means `unset`. |
| `--repin` | Re-pin agent-process `dataset.sha256` and `evaluator.sha256` to the current files. |
| `--dry-run` | Validate and report the change without writing. |
| `--reason TEXT` | Record why the change was made. |

Paths accept `a.b[0].c`, `a.b.0.c` or a JSON Pointer `/a/b/0/c`. `set`
creates missing objects and can append at the end of an array.

The patch runs on a copy of the manifest. The result goes through the same
validator as `experiment validate` and `experiment run`, and nothing is
written unless it passes. Then:

- `manifest.json` is replaced atomically.
- One entry is appended to `provenance.jsonl`. The entry records each
  operation with its previous and new value, the manifest digests before and
  after, the RunEconomy version, the reason, and `previous_entry_sha256`.
  Those digests chain the log, and a broken chain refuses further patches
  (`runeconomy.provenance.broken`).
- If `manifest.json` changed outside `experiment patch` since the last entry,
  the new entry sets `untracked_change_before: true`.
- Changing an agent-process `dataset.path` or `evaluator.path` re-pins its
  digest automatically. The re-pin appears as a `derived_digest` operation.

Patches never:

- change `artifact_type` or `mode`.
- make an unresolved workspace runnable. Supply evidence-backed values to the
  source evidence and run `experiment init` again.
- touch a frozen workspace (`runeconomy.workspace.frozen`).

New semantics still need their prospective schema version. Setting
`acceptance.task_success_method` requires `schema_version` `0.3` in the same
patch.

## `experiment freeze`

`runeconomy experiment freeze WORKSPACE` refuses unless:

1. `experiment validate` reports the workspace runnable, and
2. `experiment plan` reports `ready_to_freeze`. With
   `--allow-incomplete-design`, `needs_design_input` and
   `ready_for_calibration` are accepted and recorded as
   `design_completeness: incomplete`. A `blocked` design is always refused.

It then writes `preregistration.json` (artifact
`runeconomy.preregistration_record`, schema `0.1`), which locks by SHA-256:

- `manifest.json`, `workspace.json` and `intervention.md`.
- Agent-process `dataset` and `evaluator` files, plus every `--lock FILE`
  (agent code, runners, allocators).
- The acceptance gates and the execution design. Both are copied verbatim and
  digested canonically.
- The planner's filled preregistration: claim, population, gates, method, n,
  tolerance, cost estimate and stopping rule. The planner's `unknown_fields`
  are kept, never filled in.
- The head of the provenance chain. A `freeze` event is appended to it.

Inputs that affect the run but cannot be digest-locked are listed under
`unlocked_inputs`. Examples are code an agent command launches that you did
not name with `--lock`, Python or command evaluators resolved at run time, and
the serving stack behind a static `backend.base_url`.

After freezing:

- `experiment patch` refuses the workspace. A changed design needs a new
  workspace and a new freeze.
- `experiment validate` reports any digest drift as `invalid` with
  `runeconomy.freeze.drift`. Its JSON `freeze.status` is `not_frozen`,
  `intact` or `drifted`.
- `experiment run` refuses a drifted workspace before executing anything.
  `--require-frozen` also refuses a workspace that has never been frozen
  (`runeconomy.freeze.required`). The run's JSON, `result.json`
  (`preregistration`) and the rendered report all name the record digest the
  run started under.
- Freezing again returns `already_frozen` when every lock is intact.

## `--json` output contracts

Every command that finishes with a result accepts `--json`. In JSON mode:

- stdout holds exactly one JSON object. Its `output_schema` names the JSON
  Schema that describes it, for example `runeconomy.cli.experiment-run.v1`.
- On failure stdout is empty, and stderr holds one
  `runeconomy.cli.error.v1` object. That object carries a stable
  `error.code`, a `next_action`, and for patch and freeze a content-free
  `error.details` (reason, planner state, issue codes or mismatched locks).
- Progress and warnings go to stderr.

`runeconomy schema` lists the contracts and `runeconomy schema NAME` prints one
schema. The files ship in `runeconomy/data/cli-schemas/`.

| Contract | Commands | Exit codes |
| --- | --- | --- |
| `analyze.v1` | `analyze` | 0 |
| `import.v1` | `import` | 0 |
| `convert.v1` | `import-vllm-jsonl`, `import-openinference`, `import-ledger-jsonl`, `import-prefixwise-jsonl` | 0 |
| `capture-vllm.v1` | `capture-vllm` | 0 |
| `demo.v1` | `demo` | 0 |
| `tutorial-init.v1` | `tutorial init` | 0 |
| `support-bundle.v1` | `support-bundle` | 0 |
| `doctor.v1` | `doctor` | 0 ready, 1 not ready |
| `compare.v1` | `compare` | 0, or 3 with `--require-verified` and no `verified` verdict |
| `experiment-init.v1` | `experiment init` | 0 |
| `experiment-validate.v1` | `experiment validate` | 0 runnable, 1 not runnable, 2 invalid |
| `experiment-plan.v1`, `experiment-plan-examples.v1` | `experiment plan` | 0 ready, 1 needs design input, 2 blocked |
| `experiment-run.v1` | `experiment run` | 0, or 3 with `--require-verified` and no `verified` verdict |
| `experiment-report.v1` | `experiment report` | 0 |
| `experiment-patch.v1` | `experiment patch` | 0 |
| `experiment-freeze.v1` | `experiment freeze` | 0 |
| `proxy.v1` | `proxy --record` (after the session is flushed) or `proxy --attempt` | 0 complete, 1 capture gaps; the child's code for `--attempt` |
| `schema-list.v1` | `schema` | 0 |
| `error.v1` | any command, on stderr | 2 |

`tutorial serve` is a long-running server and has no `--json` result. In
`proxy --json`, the readiness line (`OPENAI_BASE_URL=...`) goes to stderr so
stdout carries only the final summary.

### Compatibility

- Additive changes keep the version, such as a new optional key. Consumers
  must ignore keys they do not recognize.
- Breaking changes bump the version: removing or renaming a required key,
  changing its type, or changing what a value means. The previous schema file
  stays in the package.
- Domain payloads keep their own version fields (`report_version`,
  `validation_version`, the plan's `schema_version`, doctor's
  `schema_version`). The CLI contract pins only the keys its schema lists as
  required.
- Existing JSON outputs kept their shape. The only addition is the
  `output_schema` key, plus `freeze` on `experiment validate` and
  `evidence_bundle`/`next_command` on `import`.
