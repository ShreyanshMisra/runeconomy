# RunEconomy 0.2.0a2 release notes

RunEconomy `0.2.0a2` was published as a GitHub prerelease on November 14, 2026.
It focused on trustworthy first use for invited testers.

## What changed

- The frozen rejection proof became a deterministic public case study linked
  from the README and bundled report.
- `doctor` gained a clear readiness verdict, capability table, safe modes,
  structured JSON errors and documented exit codes.
- Common CLI failures began explaining what failed, whether evidence could be
  incomplete and what to do next.
- The quickstart connected installation, the proof, compatibility diagnosis,
  trace analysis, experiment initialization and validation.
- The official coding-agent skill documented safe evidence acquisition,
  experiment design, verdict semantics and privacy boundaries.
- A small static documentation site began rendering repository Markdown.

## Why it matters

The release reduced setup ambiguity without weakening experiment validity.
Unsupported and unknown capabilities remained distinct, and the CLI kept
private evidence outside shareable reports.

## Installation

`0.2.0a2` is a historical GitHub prerelease. It is not the version currently
served by production PyPI. Inspect the
[tagged source](https://github.com/ShreyanshMisra/runeconomy/releases/tag/v0.2.0a2)
rather than assuming `pip install runeconomy` installs this release.

## Compatibility and migration

The release added `doctor`, Chat Completions support, structured findings,
OpenInference and Prefixwise import paths, closed-loop `compare`, and the
versioned attempt-status protocol. The current documentation may describe
later prerelease changes.

## Known limitations

RunEconomy remained a local CLI for invited testers. It did not include a hosted
dashboard, managed OTLP collector, framework-specific SDK, automatic resume,
SGLang runner or broad diagnosis catalog.

Raw prompts, traces, model output, evaluator data, private endpoints and
experiment artifacts remained the operator's responsibility.

## Documentation

- [Start with RunEconomy](getting-started.md)
- [Check vLLM compatibility](doctor.md)
- [Read the public proof](public-proof.md)
- [Review alpha boundaries](alpha-release.md)
