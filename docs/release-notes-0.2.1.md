# RunEconomy 0.2.1 release notes

RunEconomy `0.2.1` adds privacy-safe tester support to the complete
evidence-to-verdict workflow. It remains alpha-stage software for controlled
experiments and private testing.

## Support bundles

Create a local diagnostic archive from an experiment workspace:

```bash
runeconomy support-bundle experiment/ \
  --output runeconomy-support.zip
```

Optional `--doctor-report` and `--validation-report` arguments add only
allowlisted states from reports that already exist locally. The command never
contacts a backend, runs an experiment, enables telemetry, or uploads the
archive.

The versioned `0.1` bundle contains:

- `manifest.json`
- `environment.json`
- `workspace.json`
- `capabilities.json`
- `validation.json`
- `errors.json`

Raw traces, imported evidence, prompts, messages, generated output, evaluator
data, replay requests, token arrays, credentials, endpoints, IP addresses,
checkpoints, experiment results, source files, absolute paths, usernames, and
arbitrary workspace files are excluded by construction. Symlinks are never
followed.

RunEconomy validates the archive before reporting success. Automated privacy
controls reduce risk but cannot guarantee anonymity. Inspect every file before
sharing the bundle.

## Stable error codes

The public evidence-to-report workflow now exposes stable, namespaced error
codes such as `runeconomy.evidence.invalid`,
`runeconomy.workspace.unresolved`, and
`runeconomy.backend.authentication`.

Human output retains explanations and next actions. JSON error output exposes
the same code separately from mutable prose. Corrective validation issues also
carry codes without changing their existing statuses or exit behavior.

## Compatibility

The existing CLI commands, importer-specific compatibility commands, `compare`
wrapper, evidence bundle schema, workspace schema, attempt-status protocol,
experiment schemas, result readability, and verdict logic remain supported.

The support-bundle schema and error-code registry are additive. No existing
artifact is reinterpreted, and no missing evidence becomes known.

## Known boundaries

- A support bundle is diagnostic metadata, not experiment evidence or a
  verification result.
- Missing saved doctor or validation reports remain explicitly unavailable.
- The command does not diagnose a live backend automatically.
- The bundled tutorial remains a deterministic product-plumbing check, not
  real-vLLM evidence or a performance benchmark.
- The frozen public proof remains unchanged and retains its `invalid` verdict.

## Installation

```bash
python -m pip install runeconomy==0.2.1
```

RunEconomy remains alpha-stage software. Keep raw experiment artifacts private
and review every support archive before attaching it to an issue.
