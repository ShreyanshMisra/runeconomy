"""Render the dated compatibility page from sanitized pinned evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
DEFAULT_DATA = ROOT / "docs/compatibility-data/2026-11-23.json"
DEFAULT_PAGE = ROOT / "docs/compatibility.md"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "0.1":
        raise ValueError("unsupported compatibility evidence schema")
    combinations = payload.get("combinations")
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("compatibility evidence must contain combinations")
    required = set(payload.get("required_checks", []))
    for row in combinations:
        if row.get("status") != "passed":
            raise ValueError("only passing live rows may be published")
        surfaces = row.get("api_surfaces")
        if not isinstance(surfaces, dict) or not all(surfaces.values()):
            raise ValueError("published rows require every declared API surface")
        observed = set(surfaces) | {
            "authenticated_endpoint",
            "server_id_discovered",
            "backend_version",
            "model_identity",
            "cached_tokens_reported",
            "cached_tokens_positive_on_repeat",
            "server_ttft_reported",
            "checkpoint_resume",
        }
        if required - observed:
            raise ValueError("published row omits required checks")
    return payload


def render(payload: dict[str, Any]) -> str:
    date = payload["date"]
    rows = []
    for row in payload["combinations"]:
        api = "Chat and Responses, stream and non-stream, render, tokenize, reset"
        evidence = "Cached tokens, server TTFT, backend, model, template, GPU"
        limitations = " ".join(row["limitations"])
        rows.append(
            "| "
            + " | ".join(
                [
                    date,
                    f"`{row['runeconomy_deployment_commit'][:7]}`",
                    row["vllm_version"],
                    row["model"],
                    row["gpu"],
                    api,
                    evidence,
                    limitations,
                    "Passed",
                    f"`{row['raw_evidence_sha256'][:19]}…`",
                ]
            )
            + " |"
        )
    cost = payload["phase_cost_usd"]["actual"]
    return f"""# Live vLLM compatibility

Last validated: {date}

This is a small, dated compatibility contract. Every row below was exercised
against a live, authenticated Modal Server. It is not a claim that unlisted
versions, models, GPUs, or deployment shapes are supported.

## Tested combinations

| Date | RunEconomy commit | vLLM | Model | GPU | API surface | Evidence fields | Limitations | Result | Evidence digest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

The sanitized [dated evidence record](compatibility-data/{date}.json) pins the
full model revisions, dependency versions, RunEconomy deployment and driver
commits, raw local-evidence digests, checkpoint digests, evidence grades, and
limitations.

## What passed

Each row passed authenticated endpoint discovery, Server ID discovery, Chat
Completions and Responses in streaming and non-streaming modes, exact prompt
rendering, `/tokenize`, prefix-cache reset, repeated-prefix cached-token
telemetry, per-request server TTFT, stable backend identity, and checkpoint
resume after a planned interruption.

The cold starts observed in these deployments were 52.4 seconds for vLLM
0.33.0 with the 1.5B model, 160.6 seconds for vLLM 0.34.0 with the 1.5B model,
and 309.3 seconds for vLLM 0.34.0 with the 14B model. The 14B first start
included a first-use FlashInfer compilation.

## Evidence boundary

Model name, vLLM version, GPU class, cache telemetry, and server TTFT were
directly observed. Model revision and chat-template digest were reliably
derived from the pinned deployment and pinned model snapshot because the vLLM
HTTP API does not emit the model revision. Client TTFT includes Modal's
experimental authenticated curl path and is not used for a performance claim.

These were compatibility smokes, not throughput studies or optimization
experiments. A passing row means the listed RunEconomy evidence path worked for
that exact combination on {date}. It does not establish latency superiority,
quality preservation under an engine change, production capacity, or support
for a different patch release.

## Cost and reruns

The exact Modal billing report for Phase 3 was `${cost:.8f}`, including GPU,
CPU, memory, and network egress. The matrix is intentionally small enough to
rerun on demand or on a monthly release-validation cadence. Monthly execution
is an operational choice, not a product requirement.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render(load(args.data))
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("compatibility page differs from pinned evidence")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
