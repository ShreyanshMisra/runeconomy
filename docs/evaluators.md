# Evaluators

An evaluator turns one attempt's output into `success`, `score` and
`details`. Static manifests declare it per case (`cases[].evaluator`). An
agent-process workspace uses one evaluator file: a Python script or a JSON
spec using any kind below. An evaluator that cannot reach a judgment raises
an infrastructure error. The attempt is then unevaluable and counts against
completion, never as a silent pass or fail.

Kinds that run local code or contact a network judge (`python`, `command`,
`pytest`, `braintrust`, `inspect_ai`, `llm_judge`, and `promptfoo` with `python`
assertions) are *executable*. A static run accepts them only with
`--allow-exec-evaluators`, and `experiment freeze` lists them under
`unlocked_inputs` because the code they reach is not digest-locked.

## Built-in kinds

| Kind | Judges |
| --- | --- |
| `exact`, `regex`, `json_schema` | The output text |
| `python`, `command` | A trusted local callable or command |
| `all` | Every child must pass |
| `pytest` | The repository's tests in the attempt's working copy |
| `tool_trajectory` | The attempt's sequence of tool calls |
| `llm_judge` | A calibrated, identity-pinned LLM judge |
| `promptfoo` | Promptfoo `assert` entries, unchanged |
| `braintrust` | A Braintrust/autoevals scorer, unchanged |
| `inspect_ai` | An Inspect AI scorer, unchanged |

### `pytest`

```json
{"kind": "pytest", "workdir": "repo", "args": ["-q", "tests/"], "tests_field": "tests", "timeout_seconds": 600}
```

The tests run in `attempt_dir/workdir`, the agent's own variant-local working
copy. Exit 0 is success and exit 1 is task failure. Any other pytest exit is
an infrastructure error: interrupted, internal error, usage error, or no tests
collected. `tests_field` selects per-task node IDs. The score is the passed
fraction.

### `tool_trajectory`

```json
{"kind": "tool_trajectory", "source": "trace", "expected_field": "expected_tools",
 "match": "in_order", "forbidden": ["delete_file"], "max_calls": 6}
```

`source: trace` reads the attempt's TOOL spans in start order. `output_json`
reads `{"tool_calls": [...]}` or a list of names from the output. `match` is
one of:

- `exact`: the same sequence
- `in_order`: expected is a subsequence of the actual calls
- `unordered`: the same multiset
- `superset`: every expected tool was called

### `llm_judge`

```json
{"kind": "llm_judge", "mode": "rubric", "rubric": "Score 1 when the patch fixes the bug without unrelated edits.",
 "judge": {"model": "judge-model-2026-01", "base_url_env": "JUDGE_BASE_URL", "api_key_env": "JUDGE_API_KEY"},
 "pass_threshold": 0.5,
 "calibration": {"path": "judge-calibration.json", "sha256": "sha256:...", "min_kappa": 0.6}}
```

The judge refuses to run unless its calibration record exists and meets all
of these conditions:

- it matches its pinned digest
- it was made with the same judge model and rubric
- its recorded Cohen's kappa against human labels reaches `min_kappa`

The record is produced by `runeconomy.evaluator_kinds.calibrate_judge` from at
least 20 human-labeled examples. Each judgment:

- runs at temperature 0 on an OpenAI-compatible endpoint
- fails as infrastructure when the endpoint serves a different model
- records the reported model and a digest of the provider fingerprint

`mode` is `rubric` or `reference`. Reference mode compares against the task's
`expected` field. Pairwise judging is not offered: a preference between two
outputs is not the per-attempt pass/fail that RunEconomy's quality gates
need.

### `promptfoo`

```json
{"kind": "promptfoo", "assert": [
  {"type": "icontains", "value": "{{expected}}"},
  {"type": "not-contains", "value": "error"},
  {"type": "python", "value": "file://checks/answer.py"}]}
```

The `assert` list is the same one used in a Promptfoo config (JSON form).
Task fields are the template variables, and a `file://` Python assertion's
`get_assert(output, context)` runs unchanged. Supported types:

- `equals`, `contains`, `icontains`, `contains-any`, `contains-all`,
  `icontains-any`, `icontains-all`, `regex`, `starts-with`, `is-json` (with an
  optional schema), `contains-json` and `python`
- `not-` forms of every type

Types that do not map onto a deterministic per-attempt pass/fail are rejected
rather than approximated. These include `llm-rubric`, `similar`, `cost` and
`latency`.

### `braintrust`

```json
{"kind": "braintrust", "scorer": "autoevals:Levenshtein", "threshold": 0.8}
```

The scorer is called as `scorer(input=..., output=..., expected=...)` with
the task's `input` and `expected` fields, exactly as Braintrust calls it. It
can be synchronous or async and can return:

- a number
- an `autoevals.Score`
- a dict with `score`
- a list of scores, of which the minimum counts

A `None` score (abstention) makes the attempt unevaluable.

### `inspect_ai`

```json
{"kind": "inspect_ai", "scorer": "inspect_ai.scorer:includes", "scorer_args": {"ignore_case": true},
 "target_field": "target", "threshold": 1.0}
```

Install Inspect AI alongside RunEconomy (`pip install inspect-ai`). The
scorer factory is called with `scorer_args`, then its scorer runs once per
attempt on a single-sample `TaskState`: the task's `input` as the user
message, the attempt output as the assistant message and model output, and
the task's `target` (a string or list of strings) as the `Target`.

`Score.value` becomes a number:

- `true`/`false` and numbers are used as they are
- strings map through `value_map`, by default Inspect's `C` = 1, `P` = 0.5,
  `I` = 0 and `N` = 0
- a dict value needs `value_key` to pick one entry

The attempt passes when that number meets `threshold`. The default of 1.0
means only a fully correct score passes, so `PARTIAL` fails unless you lower
it. A missing target, a missing package, an exception, or a value that does
not map makes the attempt unevaluable.

Model-graded Inspect scorers (`model_graded_qa`, `model_graded_fact`) call a
model through Inspect's own configuration. RunEconomy does not pin or
calibrate that judge. To gate on an LLM judgment, use `llm_judge`, which
requires a calibration record.
