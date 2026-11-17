# Troubleshooting

## Import cannot detect the source

Run `runeconomy import --help`, choose the known `--type`, and retry. Do not
rewrite the source merely to satisfy detection.

## Workspace is unresolved

Run `runeconomy experiment validate WORKSPACE`. Follow each issue's next action.
Do not fabricate missing replay requests, dependencies, evaluator logic, or
identity.

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
