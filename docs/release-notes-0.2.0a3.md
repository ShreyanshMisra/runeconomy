# RunEconomy 0.2.0a3 release notes

RunEconomy `0.2.0a3` is a prerelease focused on report clarity and one
consistent experiment workflow.

## What changed

- Experiment reports begin with the tested intervention, then show results,
  the gated verdict, experiment details and collapsed technical evidence.
- The flagship proof separates favorable conditional measurements from the
  `invalid` verdict that prevented a verified performance claim.
- Reports show task outcomes, quality-gate shortfalls, the decision path,
  frozen provenance and a sanitized cache-reuse breakdown by agent call.
- HTML and Markdown reports share one evidence-derived view model across
  verified, regression, inconclusive and invalid outcomes.
- `runeconomy experiment` is the sole static-experiment command.
- `doctor` lists only the next actions that change what you can run. The
  complete per-capability guidance remains in `--json`.
- A connection that drops or stalls in the middle of a response now produces
  the structured CLI error instead of a traceback.
- The documentation site now presents a concise onboarding path, read-only
  compatibility guidance and the canonical coding-agent skill.

## Why it matters

A fast-looking metric should not become a performance claim before task quality
passes. The new report order makes the intervention and measurements easy to
understand while keeping the failed gates and claim boundary explicit.

## Installation

`0.2.0a3` is a prerelease, so request it explicitly:

```bash
python -m pip install --pre runeconomy==0.2.0a3
```

## Compatibility and migration

The former prerelease command alias is removed. Use:

```text
runeconomy experiment init
runeconomy experiment validate
runeconomy experiment run
runeconomy experiment report
```

`capture-vllm` and `experiment run` now fail immediately when the server
cannot be reached. Pass `--startup-timeout SECONDS` to keep retrying while a
cold server starts.

The ledger importer is now `runeconomy import-ledger-jsonl`. The per-trial
request placeholder is now `{{RUNECONOMY_TRIAL_ID}}`, and manifests that use
the earlier placeholder name still run.

Existing static experiment manifests remain supported. The frozen proof was not
rerun or retuned.

## Known limitations

RunEconomy remains local alpha software. It does not provide a hosted dashboard,
managed collector, framework-specific SDK, automatic resume, SGLang runner or
broad diagnosis catalog.

Protect raw prompts, traces, evaluator data, generated output, credentials,
private endpoints and unsanitized result artifacts.

## Documentation

- [Start with RunEconomy](getting-started.md)
- [Check vLLM compatibility](doctor.md)
- [Read the public proof](public-proof.md)
- [Use the coding-agent skill](agent-skill.md)
