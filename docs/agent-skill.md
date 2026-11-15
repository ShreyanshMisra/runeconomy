# Agents

Most users will have a coding agent help install, configure and operate
RunEconomy. The official RunEconomy skill gives that agent the same safety,
privacy and verdict contract documented in this repository.

The canonical source is
[`skills/runeconomy/SKILL.md`](../skills/runeconomy/SKILL.md). The website copy
and download actions are generated from that exact file during the site build.
The repository copy is the source of truth.

## What the skill teaches

- inspect a repository and deployment without changing infrastructure
- install the requested RunEconomy version
- run and interpret the read-only `doctor` command
- choose between instrumentation and supported trace imports
- initialize, validate and execute a controlled experiment
- preserve variant-local state and explicit workflow dependencies
- read task-quality gates before conditional latency measurements
- distinguish verified, regression, inconclusive and invalid verdicts

The agent is told not to call a change an improvement unless the verdict is
`verified`.

## Safety and privacy boundaries

The skill tells an agent to protect prompts, generated output, evaluator data,
raw traces, credentials, private endpoints and unsanitized result artifacts. It
also requires explicit approval before paid compute, cache reset, development
endpoint changes, publication or production infrastructure changes.

The compatible upcoming package version is `0.2.0a3`. That package has not been
published yet.

## Codex

Copy the complete skill directory to a project-scoped path:

```text
.agents/skills/runeconomy/SKILL.md
```

For user scope, use `~/.agents/skills/runeconomy/SKILL.md`. Start a new session
after copying the directory.

## Claude Code

Claude Code uses a compatible `SKILL.md` folder shape. Copy the complete
directory to:

```text
.claude/skills/runeconomy/SKILL.md
```

Use `~/.claude/skills/runeconomy/SKILL.md` for user scope. Review the skill
before allowing it to execute commands.

## Cursor

Cursor commonly uses project rules under `.cursor/rules` or a root `AGENTS.md`.
Adapt the canonical skill into a project rule and preserve its safety
boundaries. Cursor support is not universal `SKILL.md` discovery, so do not
describe this adaptation as universal native skill support.

## Keep one canonical source

Replace the installed directory from a newer tagged release instead of editing
multiple copies. The repository file remains authoritative for every tool.
