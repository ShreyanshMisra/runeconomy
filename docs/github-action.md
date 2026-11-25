# GitHub Action

The RunEconomy Action runs an experiment workspace on a pull request and posts
one verdict comment, updated in place on later pushes. Copy
[`examples/github-action/runeconomy-verify.yml`](../examples/github-action/runeconomy-verify.yml)
into `.github/workflows/` and limit `paths` to the prompts, tools and serving
configuration whose changes should be verified.

```yaml
- uses: ShreyanshMisra/runeconomy@v0.3.0
  with:
    workspace: experiment
    mode: frozen            # or: compatibility
    require-verified: "true"
```

| Input | Default | Meaning |
| --- | --- | --- |
| `workspace` | required | Workspace directory created by `experiment init` |
| `mode` | `frozen` | `frozen` runs with `--require-frozen` and reports the verdict. `compatibility` runs an unfrozen trial of the agent contract and makes no verdict claim. |
| `require-verified` | `false` | Fail the job unless a frozen run is `verified` |
| `comment` | `true` | Create or update the marked verdict comment on the pull request |
| `allow-exec-evaluators` | `false` | Allow python and command evaluators |
| `runeconomy-version` | `runeconomy==0.3.0` | Package specifier; `.` installs the checked-out repository |

The Action has three outputs:

- `verdict`
- `verified`, which is true only for a frozen run whose verdict is `verified`
- `result`, the path to `result.json`

The verdict table is also written to the job summary.

## Safety

- Run credentialed experiments only on trusted branches and reviewed agent code.
  Agent processes run local commands. A freeze checks for drift since registration.
  The example skips fork pull requests.
- A frozen workspace is immutable. A pull request that edits the frozen
  manifest, dataset, evaluator or a `--lock` file fails with
  `runeconomy.freeze.drift` before anything runs. To change a design, add a
  new workspace with a new freeze in that pull request.
- Inputs reach shell steps only through environment variables, never through
  template expansion inside scripts.
- The comment needs `pull-requests: write`. Nothing else is written to the
  repository. Serving credentials come from repository secrets, and RunEconomy
  reads only the variable that the manifest names.
- Running a held-out experiment on every push consumes it. Use `compatibility`
  mode for routine pull requests, and reserve `frozen` runs for changes you
  intend to claim.

`runeconomy ci-summary RESULT [--mode frozen|compatibility] [--github-comment]`
is the command behind the comment, so other CI systems can render the same
table.
