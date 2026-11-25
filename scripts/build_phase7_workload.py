"""Construct disjoint Phase 6/7 task splits and a frozen repository snapshot.

The tasks are the Proof 2 repository-lookup family (which function or class
encloses an exact string literal or call expression), built mechanically from
a snapshot of this repository's Python sources. The snapshot is copied into
the output directory and digested, so the agent's search tool reads exactly
the files the tasks were built from, independent of later code changes.

Every query used by an earlier public proof or Proof 2 split (including the
consumed Proof 2 held-out pool) is excluded. The splits are disjoint:

* ``detector``           40 tasks for the Phase 6 detector validations
* ``proof3-calibration``  40 tasks
* ``proof4-calibration``  40 tasks
* ``heldout-pool``       untouched; the held-out samples are drawn from it only
  after calibration has fixed their sizes
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261125
SPLITS = (("detector", 40), ("proof3-calibration", 40), ("proof4-calibration", 40))
HELDOUT_POOL = 400
TASK_PREFIX = "p7"

_spec = importlib.util.spec_from_file_location("proof2_builder", ROOT / "scripts" / "build_proof_2_workload.py")
assert _spec is not None and _spec.loader is not None
proof2 = importlib.util.module_from_spec(_spec)
sys.modules["proof2_builder"] = proof2
_spec.loader.exec_module(proof2)


def _consumed_queries() -> set[str]:
    """Every question and phrase any earlier proof split sent to a model."""

    queries: set[str] = set()
    paths = list((ROOT / "artifacts").glob("public-proof-preparation/*-tasks.json"))
    paths += list((ROOT / "artifacts").glob("proof-2-preparation*/*-tasks.json"))
    paths += list((ROOT / "artifacts").glob("proof-2-*-extracted/**/tasks.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for task in payload.get("tasks", []) if isinstance(payload, dict) else []:
            question = task.get("question") if isinstance(task, dict) else None
            if not isinstance(question, str):
                continue
            queries.add(question)
            for marker in ("exact string literal ", "exact call expression "):
                if marker in question and question.endswith("?"):
                    try:
                        value = ast.literal_eval(question.split(marker, 1)[1][:-1])
                    except (SyntaxError, ValueError):
                        continue
                    if isinstance(value, str):
                        queries.add(value)
    return queries | proof2._previous_queries()


def _snapshot(files: list[Path], output: Path) -> dict[str, object]:
    snapshot = output / "repo-snapshot"
    digests: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digests[relative] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    (snapshot / "SNAPSHOT.json").write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"file_count": len(digests), "snapshot_sha256": "sha256:" + hashlib.sha256(manifest).hexdigest()}


def _write(output: Path, name: str, candidates: list, files: list[Path], answer_pool: list[str], commit: str) -> dict[str, object]:
    tasks, answers = [], {}
    for candidate in candidates:
        task, answer = proof2._task(candidate, files, commit, answer_pool)
        task["task_id"] = f"{TASK_PREFIX}-{candidate.family}-{candidate.fingerprint[:12]}"
        # The answer lives only in the evaluator; the agent reads the task file.
        tasks.append(task)
        answers[task["task_id"]] = answer
    (output / f"{name}-tasks.json").write_text(json.dumps({"tasks": tasks}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluator = f'''"""Deterministic evaluator for the Phase 7 {name} split."""
import json
import sys

ANSWERS = {json.dumps(answers, indent=2, sort_keys=True)}

request = json.load(sys.stdin)
task_id = request.get("task_id")
if task_id not in ANSWERS:
    raise ValueError("unknown Phase 7 {name} task")
observed = str(request.get("output_text", "")).strip()
success = observed == ANSWERS[task_id]
json.dump({{"success": success, "score": float(success), "details": {{}}}}, sys.stdout)
'''
    (output / f"{name}-evaluator.py").write_text(evaluator, encoding="utf-8")
    return {
        "task_count": len(tasks),
        "family_counts": dict(Counter(task["family"] for task in tasks)),
        "tasks_sha256": proof2._digest(output / f"{name}-tasks.json"),
        "evaluator_sha256": proof2._digest(output / f"{name}-evaluator.py"),
    }


def build(output: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    files = proof2._source_files()
    excluded = _consumed_queries()
    candidates = [
        candidate
        for candidate in proof2._collect_candidates(files)
        if candidate.query not in excluded and proof2._question(candidate) not in excluded
    ]
    pools = {family: [c for c in candidates if c.family == family] for family in ("call_expression", "string_literal")}
    rng = random.Random(SEED)
    for pool in pools.values():
        rng.shuffle(pool)
    selected: dict[str, list] = {name: proof2._balanced_take(pools, count) for name, count in SPLITS}
    heldout_count = min(HELDOUT_POOL, 2 * min(len(pool) for pool in pools.values()))
    selected["heldout-pool"] = proof2._balanced_take(pools, heldout_count)
    combined = [candidate for split in selected.values() for candidate in split]
    proof2._assert_static_audit(combined, candidates)
    commit = proof2._git_commit()
    answer_pool = sorted({candidate.answer for candidate in candidates})
    snapshot = _snapshot(files, output)
    record = {
        "schema_version": "0.1",
        "status": "splits_built_heldout_untouched",
        "repository_commit_parent": commit,
        "repository_snapshot": snapshot,
        "seed": SEED,
        "consumed_queries_excluded": len(excluded),
        "eligible_candidate_count": len(candidates),
        "splits": {
            name: _write(output, name, split, files, answer_pool, commit) for name, split in selected.items()
        },
        "separation": "all splits are disjoint; no query was used by any earlier proof",
    }
    (output / "construction-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "phase-7-preparation")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
