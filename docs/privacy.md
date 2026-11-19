# Privacy and support bundles

The support-bundle command keeps diagnostic collection local. It does not
enable telemetry, upload traces, or send support bundles anywhere.

`runeconomy.agent.Attempt` writes only status schema `0.1`, a terminal state,
and fixed content-free component, stage, and code labels. It has no field for
prompts, generated output, exception messages, URLs, or credentials. Do not
derive labels from runtime text.

`runeconomy experiment plan` reads local design metadata and emits counts,
booleans, safe categories, confidence calculations, and stable issue codes. It
does not echo task IDs, task contents, commands, paths, hypothesis text, claim
boundaries, endpoint data, or artifact names.

## Create a support bundle

```bash
runeconomy support-bundle experiment/ \
  --output runeconomy-support.zip
```

To include sanitized states from reports you already saved locally:

```bash
runeconomy support-bundle experiment/ \
  --doctor-report doctor.json \
  --validation-report validation.json \
  --output runeconomy-support.zip
```

The command does not run `doctor`, contact a backend, execute an experiment, or
upload the archive. It refuses to overwrite an existing archive.

The archive contains only allowlisted environment categories, workspace schema
and status, artifact-role presence, content digests, capability states,
validation issue codes, generalized field paths, and stable error codes.

It excludes raw evidence, traces, prompts, messages, generated output, evaluator
inputs or answers, replay requests, token arrays, credentials, endpoint URLs,
IP addresses, checkpoints, results, source files, absolute paths, usernames,
and arbitrary workspace files. Symlinks are never followed.

This is privacy minimization, not a guarantee of anonymity. Always extract and
inspect every JSON file before attaching the archive to an issue. Creating a
bundle does not authorize uploading or publicly sharing it.

## Determinism

The archive uses fixed member names, ordering, permissions, and timestamps. It
does not include a creation time. Repeated generation is byte-for-byte
deterministic when the inputs and environment categories are unchanged.

The RunEconomy and Python versions, operating-system family, architecture,
installation type, and input digests intentionally vary across installations.

## Stable error codes

Public workflow errors use namespaced codes such as:

```text
runeconomy.evidence.invalid
runeconomy.workspace.unresolved
runeconomy.backend.authentication
runeconomy.checkpoint.mismatch
runeconomy.report.generation
```

Human explanations and next actions may improve over time. The code is the
stable value to include in an issue or automation. JSON output exposes the code
separately from mutable prose.
