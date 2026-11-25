"""Versioned ``--json`` output contracts for every RunEconomy command.

Each command that finishes with a result prints exactly one JSON object on
stdout in ``--json`` mode. That object carries ``output_schema``, the ``$id``
of the JSON Schema that describes it, for example
``runeconomy.cli.experiment-validate.v1``. Failures print one object with the
``runeconomy.cli.error.v1`` contract on stderr and leave stdout empty.

Compatibility rules:

* Additive changes (a new optional key, a new enum value documented as
  open) keep the version. Consumers must ignore keys they do not know.
* Removing or renaming a required key, changing its type, or changing the
  meaning of a value is breaking and bumps the version. The previous schema
  file stays in the package.
* Domain payloads embedded in a contract (an analysis report, an experiment
  plan, a doctor capability report) keep their own ``*_version`` fields; the
  CLI contract pins only the envelope keys listed as required.

The schemas live in ``runeconomy/data/cli-schemas`` and can be printed with
``runeconomy schema NAME``.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Mapping


SCHEMA_DIRECTORY = "cli-schemas"
ID_PREFIX = "runeconomy.cli."

# contract name -> current major version
CONTRACTS: dict[str, int] = {
    "analyze": 1,
    "import": 1,
    "convert": 1,
    "capture-vllm": 1,
    "demo": 1,
    "tutorial-init": 1,
    "support-bundle": 1,
    "doctor": 1,
    "compare": 1,
    "experiment-init": 1,
    "experiment-validate": 1,
    "experiment-plan": 1,
    "experiment-plan-examples": 1,
    "experiment-run": 1,
    "experiment-report": 1,
    "experiment-patch": 1,
    "experiment-freeze": 1,
    "proxy": 1,
    "ci-summary": 1,
    "schema-list": 1,
    "error": 1,
}

# command path (as typed) -> contracts its --json mode can print on stdout
COMMAND_CONTRACTS: dict[str, tuple[str, ...]] = {
    "analyze": ("analyze",),
    "import": ("import",),
    "import-vllm-jsonl": ("convert",),
    "import-openinference": ("convert",),
    "import-ledger-jsonl": ("convert",),
    "import-prefixwise-jsonl": ("convert",),
    "capture-vllm": ("capture-vllm",),
    "demo": ("demo",),
    "tutorial init": ("tutorial-init",),
    "support-bundle": ("support-bundle",),
    "doctor": ("doctor",),
    "compare": ("compare",),
    "experiment init": ("experiment-init",),
    "experiment validate": ("experiment-validate",),
    "experiment plan": ("experiment-plan", "experiment-plan-examples"),
    "experiment run": ("experiment-run",),
    "experiment report": ("experiment-report",),
    "experiment patch": ("experiment-patch",),
    "experiment freeze": ("experiment-freeze",),
    "schema": ("schema-list",),
    "proxy": ("proxy",),
    "ci-summary": ("ci-summary",),
}

# Commands without a --json result, and why.
EXEMPT_COMMANDS: dict[str, str] = {
    "tutorial serve": "long-running local server with no completion result",
    "mcp": "long-running stdio MCP server; each tool returns the --json contract of the command it runs",
}


def contract_id(name: str) -> str:
    if name not in CONTRACTS:
        raise KeyError(f"unknown CLI contract {name!r}")
    return f"{ID_PREFIX}{name}.v{CONTRACTS[name]}"


def stamp(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``payload`` with its ``output_schema`` contract identifier."""

    return {**dict(payload), "output_schema": contract_id(name)}


def render(name: str, payload: Mapping[str, Any]) -> str:
    return json.dumps(stamp(name, payload), indent=2, sort_keys=True)


def schema_filename(name: str, version: int | None = None) -> str:
    return f"{name}.v{CONTRACTS[name] if version is None else version}.schema.json"


def load_schema(name: str, version: int | None = None) -> dict[str, Any]:
    if name not in CONTRACTS:
        raise KeyError(f"unknown CLI contract {name!r}")
    resource = files("runeconomy").joinpath("data", SCHEMA_DIRECTORY, schema_filename(name, version))
    return json.loads(resource.read_text(encoding="utf-8"))


def list_contracts() -> dict[str, Any]:
    commands: dict[str, list[str]] = {}
    for command, names in COMMAND_CONTRACTS.items():
        for name in names:
            commands.setdefault(name, []).append(command)
    return {
        "contracts": [
            {
                "name": name,
                "version": version,
                "id": contract_id(name),
                "commands": sorted(commands.get(name, [])),
                "stream": "stderr" if name == "error" else "stdout",
            }
            for name, version in sorted(CONTRACTS.items())
        ],
        "exempt_commands": [
            {"command": command, "reason": reason} for command, reason in sorted(EXEMPT_COMMANDS.items())
        ],
    }
