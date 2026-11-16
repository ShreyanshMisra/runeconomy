# Privacy and infrastructure safety

Keep prompts, retrieved context, generated output, task answers, evaluator data,
raw traces, credentials, private endpoints, local paths, unsanitized stderr, and
replayable private workloads private by default.

Inspect reports before sharing them. Prefer deterministic aggregate artifacts
with provenance hashes and claim boundaries.

Do not start paid resources, reset shared caches, enable unsafe development
endpoints, rotate credentials, modify production routing, push code, publish a
package, create a release, or deploy a site without explicit approval.

A per-request cache salt may isolate logical cache namespaces, but it does not
prove physical cache eviction or reset. Describe the actual isolation mode and
its limitations.
