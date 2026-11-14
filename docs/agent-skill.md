# RunEconomy agent skill

The canonical RunEconomy coding-agent skill is
[`skills/runeconomy/SKILL.md`](../skills/runeconomy/SKILL.md). It teaches an
agent to diagnose a vLLM environment, choose an evidence path, initialize and
validate an experiment, preserve workflow state and dependencies, and interpret
quality-gated verdicts without overstating conditional measurements.

Always install or copy the complete `skills/runeconomy` directory from a tagged
RunEconomy release. The repository copy is the source of truth.

## Codex

For a project-scoped installation, copy the directory to:

```text
.agents/skills/runeconomy/SKILL.md
```

For a user-scoped installation, use:

```text
~/.agents/skills/runeconomy/SKILL.md
```

Start a new Codex session after copying it. These paths follow the current
[Codex skill installation guidance](https://developers.openai.com/api/reference/cli#skills).

## Claude Code

Claude Code uses the same `SKILL.md` folder shape. Copy the directory to:

```text
.claude/skills/runeconomy/SKILL.md
```

or `~/.claude/skills/runeconomy/SKILL.md` for user scope. Start a new session
after installation. Do not assume that a Codex plugin manifest or other
tool-specific metadata is portable; the canonical artifact here is the skill
directory itself.

## Cursor

Cursor's documented portable mechanism is project rules under `.cursor/rules`
or a root `AGENTS.md`, not universal `SKILL.md` discovery. Copy the body of the
canonical skill into a project rule, preserve its safety boundaries, and link
back to the repository source. Do not describe that adaptation as a native
RunEconomy skill installation.

See [Cursor Rules](https://docs.cursor.com/context/rules-for-ai) for its current
rule format.

## Updating

Replace the installed directory from a newer tagged release rather than editing
multiple divergent copies. Review the diff before allowing an agent to execute
commands from an updated skill.

## Forward validation

The `0.2.0a2` release candidate was copied into a clean temporary repository
under the Codex project-scoped path and followed without repository-internal
instructions. The agent path successfully reached installation verification,
the bundled proof, a read-only mocked compatibility diagnosis, trace analysis,
an unresolved experiment skeleton, and validation of a known manifest.

Two operator-provided prerequisites are intentionally not inferred: a real
vLLM base URL for compatibility diagnosis and real trace/finding evidence for
experiment initialization. The skill now tells an agent to stop rather than
guess either value. No additional agent-specific SDK was required.
