from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTCOME_PATH = ROOT / "docs" / "proof-2-outcome.json"
FREEZE_PATH = ROOT / "docs" / "proof-2-freeze-record.json"

EXPECTED_FREEZE_SHA256 = (
    "61cd37d718d4a54c40857b9a6335ea7f8fa4e6ae5b96a9e1e69aae763aa014ec"
)
EXPECTED_LOCKED_PROJECTION_SHA256 = (
    "ff9202d3d35a862742038b20901abd8caac91e1a7bd14aedc5e1e9f4904eeba0"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _locked_projection(outcome: dict[str, Any]) -> dict[str, Any]:
    gate_results = {
        gate["name"]: gate["passed"] for gate in outcome["gates"]
    }
    return {
        "freeze": outcome["freeze"],
        "verdict": outcome["verdict"],
        "execution": {
            key: outcome["execution"][key]
            for key in (
                "invocation_count",
                "one_shot_ledger_state",
                "pair_count",
                "block_pair_counts",
                "seed",
                "private_raw_archive_sha256",
            )
        },
        "gate_results": gate_results,
    }


def _projection_digest(outcome: dict[str, Any]) -> str:
    payload = json.dumps(
        _locked_projection(outcome), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _all_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_proof_2_final_verdict_and_provenance_are_semantically_locked() -> None:
    outcome = _load(OUTCOME_PATH)
    freeze = _load(FREEZE_PATH)

    assert hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest() == EXPECTED_FREEZE_SHA256
    assert outcome["schema_version"] == "0.1"
    assert outcome["freeze"]["record_sha256"] == f"sha256:{EXPECTED_FREEZE_SHA256}"
    assert outcome["freeze"]["contract_sha256"] == freeze["artifacts"]["contract_sha256"]
    assert outcome["freeze"]["dataset_sha256"] == freeze["artifacts"]["dataset_sha256"]
    assert outcome["freeze"]["evaluator_sha256"] == freeze["artifacts"]["evaluator_sha256"]
    assert outcome["verdict"] == {
        "status": "invalid",
        "final": True,
        "decisive_gate": "completion",
        "reason": "Both variants completed 179 of 180 attempts, below the frozen 100% completion threshold.",
        "performance_claim_eligible": False,
        "rerun_permitted": False,
        "post_hoc_reclassification_permitted": False,
        "new_performance_claim_permitted": False,
    }
    assert _projection_digest(outcome) == EXPECTED_LOCKED_PROJECTION_SHA256
    assert outcome["integrity"]["locked_projection_sha256"] == (
        f"sha256:{EXPECTED_LOCKED_PROJECTION_SHA256}"
    )


def test_proof_2_outcome_contains_only_sanitized_aggregate_structures() -> None:
    outcome = _load(OUTCOME_PATH)
    forbidden_keys = {
        "attempts",
        "answer_map",
        "endpoint",
        "gpu_inventory",
        "model_output",
        "physical_gpu_id",
        "prompt",
        "task_id",
        "task_text",
        "trace",
    }

    assert not (_all_keys(outcome) & forbidden_keys)
    assert outcome["privacy"] == {
        "contains_task_text": False,
        "contains_answer_map": False,
        "contains_model_outputs": False,
        "contains_traces": False,
        "contains_physical_gpu_identifiers": False,
        "raw_artifact_committed": False,
    }
    assert outcome["publication_status"] == "aggregate_record_only"


def test_proof_2_human_rendering_preserves_final_boundary() -> None:
    rendered = (ROOT / "docs" / "proof-2-outcome.md").read_text()

    assert "final and invalid under the frozen contract" in rendered
    assert "No rerun" in rendered
    assert "not eligible to support a verified performance claim" in rendered
    assert "not promotional material" in rendered
