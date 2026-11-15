# RunEconomy 0.2.0a3 release-note draft

RunEconomy `0.2.0a3` is an invited-tester alpha focused on report clarity and
one consistent experiment workflow.

## What changed

- Experiment reports now begin with the tested intervention, then show results,
  the gated verdict, experiment details and collapsed technical evidence.
- The flagship proof clearly separates favorable conditional measurements from
  the `invalid` verdict that prevented a verified performance claim.
- The report shows task outcomes, quality-gate shortfalls, the decision path,
  frozen provenance and a sanitized cache-reuse breakdown by agent call.
- HTML and Markdown reports now share one evidence-derived view model across
  verified, regression, inconclusive and invalid outcomes.
- `runeconomy experiment` is now the sole static-experiment command. The former
  prerelease alias and methodology branding were removed.
- The README, documentation site and official agent skill use the same commands
  and verdict language as generated reports.

## Frozen proof

The experiment was not rerun or retuned. The bundled proof retains its exact
`invalid` verdict, acceptance thresholds, results, confidence intervals and
frozen hashes. Its 40 held-out tasks remain consumed.

## Compatibility note

This prerelease intentionally removes the former static-experiment command
alias. Use `runeconomy experiment init`, `validate`, `run` and `report`.

RunEconomy remains local alpha software. Protect raw prompts, traces, evaluator
data, generated output, credentials, private endpoints and unsanitized result
artifacts.
