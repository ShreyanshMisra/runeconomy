# Troubleshooting

Record the stable `runeconomy.*` error code shown by the CLI. Preserve that code
separately from the human explanation because the wording may improve between
releases.

## Import cannot detect the source

Run `runeconomy import --help`, choose the known `--type`, and retry. Do not
rewrite the source merely to satisfy detection.

## Workspace is unresolved

Run `runeconomy experiment validate WORKSPACE`. Follow each issue's next action.
For static replay, complete the source inputs, re-import, and initialize in a
new directory. For a proxy session, use `--replay-context` to supply the design. Use `experiment patch`
for subsequent design changes to the runnable workspace.

## Capture has gaps

Inspect the summary and the final `session_close` counters locally.
`observations_dropped` and `record_errors` indicate missing exchanges.
`token_resolution_incomplete_at_close` means exact prompt tokens may be
missing. Re-import to check which analyses the remaining evidence supports
and recapture if the relevant calls are incomplete.

## Plan is blocked

Check validation first, then the issue codes and corrective actions in
`experiment plan WORKSPACE --json`. Preview a design fix with
`experiment patch WORKSPACE --dry-run`, supply the reviewed operations and
reason, apply the patch, and plan again. A changed frozen design requires a
new workspace.

## Backend is unsupported

Save a doctor report and validate with it. Use only modes supported by the API
dialect, tokenization or render path, isolation mode, and timing evidence.

## Claim evidence is incomplete

Execution may be useful for engineering diagnosis, but it cannot support the
requested public claim. Add provenance before execution or narrow the claim.

## An attempt was interrupted

Keep the partial trace, status file, and classification. Never convert missing
output into a successful or zero-latency attempt.

## Version mismatch

Run `runeconomy --version` and use documentation from the matching release or
Git tag. Ask before upgrading the installed package.

## Preparing a maintainer report

Create a local support bundle from the affected workspace:

```bash
runeconomy support-bundle WORKSPACE --output runeconomy-support.zip
```

Add `--doctor-report` or `--validation-report` only for saved reports already on
disk. The command never probes a server. Inspect the archive before sharing it.
Never attach raw traces, private results, prompts, model output, evaluator
answers, credentials, or endpoint details. Bundle creation does not authorize
upload or public disclosure.
