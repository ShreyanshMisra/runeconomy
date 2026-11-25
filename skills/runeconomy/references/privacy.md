# Privacy and infrastructure safety

Keep prompts, retrieved context, generated output, task answers, evaluator data,
raw traces, credentials, private endpoints, local paths, unsanitized stderr, and
replayable private workloads private by default.

Inspect reports before sharing them. Prefer deterministic aggregate artifacts
with provenance hashes and claim boundaries.

## Proxy recordings

Record in a private directory outside the repository, or a directory already
ignored by Git. The default session can contain raw prompts, tool inputs and
outputs, completions and exact token IDs. Keep captures out of commits,
stdout, chat and issue attachments. Use the client’s normal credential
environment variable. The proxy forwards authorization upstream and excludes
credential headers from the recording. For `doctor`, pass only the variable
name with `--api-key-env`.

For body scrubbing, supply a trusted local hook with
`--redact package.module:function`. A redacted body can still have recoverable
prompt token IDs. When both bodies and token arrays must be omitted, use
`--payload-capture digest_only --token-resolution off`. Review retained
metadata privately.

## Support bundles

When diagnostic context is required, create a privacy-minimized archive
locally:

```bash
runeconomy support-bundle WORKSPACE --output runeconomy-support.zip
```

The command reads only allowlisted workspace roles and saved reports supplied
explicitly. It does not contact a backend, enable telemetry, upload the archive,
or copy raw evidence. Automated minimization cannot guarantee anonymity.
Extract and inspect every JSON member before sharing. Creating the bundle does
not authorize an agent to upload it or publish an issue.

Do not start paid resources, reset shared caches, enable unsafe development
endpoints, rotate credentials, modify production routing, push code, publish a
package, create a release, or deploy a site without explicit approval.

A per-request cache salt may isolate logical cache namespaces, but it does not
prove physical cache eviction or reset. Describe the actual isolation mode and
its limitations.
