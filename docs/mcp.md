# MCP server

`runeconomy mcp` serves RunEconomy as Model Context Protocol tools over stdio.
A coding agent can then take a repository from evidence to a frozen, executed
and reported experiment without editing JSON.

```bash
# Claude Code
claude mcp add runeconomy -- runeconomy mcp

# Any client that launches stdio servers (Cursor, Claude Desktop, ...)
{
  "mcpServers": {
    "runeconomy": { "command": "runeconomy", "args": ["mcp"] }
  }
}
```

The server uses only the Python standard library. Each tool runs the matching
CLI command with `--json` in a subprocess and returns that command's versioned
output contract as `structuredContent` (see
[JSON contracts](json-contracts-patch-freeze.md)). MCP and the CLI therefore
behave identically. Agent and evaluator processes can never write into the
protocol stream.

## Tools

| Tool | Command | Read-only |
| --- | --- | --- |
| `doctor` | `runeconomy doctor` | yes (probes metadata only) |
| `import_evidence` | `runeconomy import` | writes the evidence bundle |
| `analyze` | `runeconomy analyze` | yes |
| `init_experiment` | `runeconomy experiment init` | creates a workspace |
| `validate_experiment` | `runeconomy experiment validate` | yes |
| `patch_experiment` | `runeconomy experiment patch` | writes with provenance |
| `freeze_experiment` | `runeconomy experiment freeze` | irreversible for the workspace |
| `plan_experiment` | `runeconomy experiment plan` | yes |
| `run_experiment` | `runeconomy experiment run` | **requires approval** |
| `render_report` | `runeconomy experiment report` | writes the report |
| `inspect_result` | reads a result JSON | yes |

Tool annotations mark read-only, destructive and open-world behavior so
clients can apply their own confirmation policy.

## The approval boundary

`run_experiment` executes agent commands and contacts serving backends, so it
can spend money. Its first call never executes anything. It returns:

- the planner state, planned pairs and total attempts
- the declared cost estimate and the manifest's `cost_model`
- the exact local commands that will run
- whether the workspace is frozen and intact
- a single-use `approval_token`

The agent must show this to the human and call again with the token only
after explicit approval. Each estimate issues a fresh random token. The server binds it to:

- the exact tool arguments
- the current digests of `manifest.json`, `workspace.json`,
  `preregistration.json`, the dataset and the evaluator

Changing any of them invalidates the token, as does reuse or the 15-minute
expiry. Tokens live only in the server process and are never written to disk.

## Zero-JSON-edit path

1. `init_experiment` (agent-process, or `from_evidence` after
   `import_evidence`)
2. `patch_experiment` with `set` for `design.planning` and any gate changes
3. `freeze_experiment` with `lock` naming the agent code
4. `plan_experiment` to review power and the preregistration
5. `run_experiment` with `require_frozen: true`, which returns an estimate;
   after approval, call it again with the token
6. `render_report` and `inspect_result`

`tests/test_mcp_server.py` exercises this sequence end to end over JSON-RPC.
