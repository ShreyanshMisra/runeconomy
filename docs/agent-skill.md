# Agents

The official RunEconomy skill gives a coding agent the same evidence, privacy,
authorization, and verdict contract documented in this repository. The
canonical source is the complete
[`skills/runeconomy`](../skills/runeconomy/) directory.

## Install

From the project where the agent will operate, run:

```bash
npx skills add ShreyanshMisra/runeconomy --skill runeconomy --all
```

The Agent Skills installer places the portable skill for each selected agent.
Review the paths it prints. For Codex and Claude Code, confirm that each
installed `SKILL.md` resolves to the same canonical skill directory and includes
the `references/` files.

The current Agent Skills CLI requires Node.js 22.20 or newer. If `npx` reports
an engine or `node:util` error, update Node before retrying the same command.

If an agent does not rediscover skills during the current session, tell it to
read the newly installed `SKILL.md` completely before continuing. Cursor skill
discovery can vary by version, so use the installer's reported result rather
than assuming a universal path.

## Bootstrap prompt

```text
Install the official RunEconomy skill for this project by running:
npx skills add ShreyanshMisra/runeconomy --skill runeconomy --all

Review the installed files, read the RunEconomy SKILL.md completely, and follow it for this task. Begin read-only. Inspect the repository and available inference evidence, identify what is missing for a valid experiment, and propose the exact import, initialization, validation, run, and report commands. Do not start paid compute, reset caches, expose private traces, or modify production infrastructure without my explicit approval.
```

## Version contract

The skill does not hardcode a package version. It tells the agent to inspect
`runeconomy --version`, use documentation from the matching release or tag, and
ask before upgrading. Versioned and latest skill copies on the documentation
site are generated from this repository source. Their SHA-256 checksum is shown
on this page after the site build.

## What the skill teaches

- begin read-only and diagnose vLLM safely
- choose the unified evidence importer and respect capability grades
- create and validate an experiment workspace
- choose static request or agent-process mode
- preserve variant-local state and explicit workflow dependencies
- read task-quality gates before mechanism metrics
- distinguish verified, regression, inconclusive, and invalid
- protect prompts, traces, answers, credentials, and raw artifacts
- stop before paid, destructive, production, or publication actions

The repository copy is the source of truth. Replace the installed directory
from a tagged release rather than maintaining hand-edited copies.
