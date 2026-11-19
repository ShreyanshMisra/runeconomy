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

Locate the repository's real agent entry point and existing tracing. Explain the smallest integration before editing. Use `from runeconomy.agent import Attempt` instead of implementing the attempt-status JSON manually. Preserve normal application execution, exit behavior, and existing tracing. Map only failure classes the repository can establish truthfully, add focused local tests, and run a free compatibility trial before any benchmark. Initialize the agent-process workspace, run `experiment validate`, then run the read-only `experiment plan` check. Stop before paid compute, backend changes, held-out execution, publication, or deployment, and report exactly what still requires my approval.
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
- integrate a real agent with the minimal supported `Attempt` helper
- assess design readiness with the read-only `experiment plan` command
- choose static request or agent-process mode
- preserve variant-local state and explicit workflow dependencies
- read task-quality gates before mechanism metrics
- distinguish verified, regression, inconclusive, and invalid
- record stable error codes and create a local support bundle when needed
- protect prompts, traces, answers, credentials, and raw artifacts
- stop before paid, destructive, production, or publication actions

The skill requires an agent to inspect a support bundle before sharing it. It
also states that bundle creation never authorizes upload or public disclosure.

## One-paste integration flow

1. Paste the bootstrap prompt above into Codex or Claude Code.
2. The coding agent installs and reads the complete skill and focused references.
3. It begins read-only, finds the real agent entry point and preserves existing tracing.
4. It proposes and adds the minimal `Attempt.from_environment()` integration.
5. It initializes an `agent-process` workspace and runs `experiment validate`.
6. It runs `experiment plan` without executing the agent or contacting a backend.
7. It explains whether the design is blocked, needs input, is ready for calibration, or is ready to freeze.
8. It stops before any paid or held-out execution and requests the required approval.

The coding agent cannot infer truthful success labels, failure classes,
dependencies, or held-out status without repository evidence. Unknowns must
remain explicit.

The repository copy is the source of truth. Replace the installed directory
from a tagged release rather than maintaining hand-edited copies.
