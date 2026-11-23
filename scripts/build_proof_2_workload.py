"""Construct disjoint pilot, calibration, and draft-held-out Proof 2 tasks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    ROOT / "src" / "runeconomy",
    ROOT / "scripts",
    ROOT / "deploy",
    ROOT / "examples",
)
SEED = 20261123
PILOT_COUNT = 5
CALIBRATION_COUNT = 40
DRAFT_HELDOUT_COUNT = 240


@dataclass(frozen=True)
class Candidate:
    family: str
    query: str
    path: str
    answer: str

    @property
    def fingerprint(self) -> str:
        value = f"{self.family}\0{self.query}\0{self.path}\0{self.answer}"
        return hashlib.sha256(value.encode()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_files() -> list[Path]:
    return sorted(
        path
        for source_root in SOURCE_ROOTS
        for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "build_proof_2_workload.py"
    )


def _owner(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return current.name
    return None


def _safe_literal(value: str) -> bool:
    return (
        18 <= len(value) <= 120
        and "\n" not in value
        and "\r" not in value
        and not value.startswith(("http://", "https://", "sha256:"))
        and not any(character in value for character in ("{", "}"))
        and sum(character.isalpha() for character in value) >= 8
    )


def _collect_candidates(files: Iterable[Path]) -> list[Candidate]:
    provisional: list[Candidate] = []
    sources: dict[str, str] = {}
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = source
        for node in ast.walk(tree):
            answer = _owner(node, parents)
            if answer is None:
                continue
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _safe_literal(node.value)
            ):
                provisional.append(
                    Candidate("string_literal", node.value, relative, answer)
                )
            elif isinstance(node, ast.Call):
                segment = ast.get_source_segment(source, node)
                if (
                    isinstance(segment, str)
                    and 18 <= len(segment) <= 140
                    and "\n" not in segment
                    and "\r" not in segment
                    and not segment.startswith(("print(", "assert "))
                ):
                    provisional.append(
                        Candidate("call_expression", segment, relative, answer)
                    )

    counts = Counter(candidate.query for candidate in provisional)
    return sorted(
        (
            candidate
            for candidate in provisional
            if counts[candidate.query] == 1
            and len(candidate.answer) <= 24
            and not candidate.answer.startswith("_")
            and len(candidate.query) <= 64
            and sum(source.count(candidate.query) for source in sources.values()) == 1
        ),
        key=lambda candidate: candidate.fingerprint,
    )


def _previous_queries() -> set[str]:
    queries: set[str] = set()
    paths = list(
        (ROOT / "artifacts" / "public-proof-preparation").glob("*-tasks.json")
    )
    paths.extend(
        path
        for preparation in (ROOT / "artifacts").glob("proof-2-preparation*")
        for path in preparation.glob("*-tasks.json")
        if path.name != "draft-heldout-tasks.json"
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for task in payload.get("tasks", []):
            question = task.get("question") if isinstance(task, dict) else None
            if isinstance(question, str):
                queries.add(question)
                for marker in (
                    "exact docstring phrase ",
                    "exact unique string literal ",
                    "unique call ",
                    "exact string literal ",
                    "exact call expression ",
                ):
                    if marker in question and question.endswith("?"):
                        encoded = question.split(marker, 1)[1][:-1]
                        try:
                            value = ast.literal_eval(encoded)
                        except (SyntaxError, ValueError):
                            continue
                        if isinstance(value, str):
                            queries.add(value)
    return queries


def _balanced_take(
    pools: dict[str, list[Candidate]], count: int
) -> list[Candidate]:
    families = sorted(pools)
    selected: list[Candidate] = []
    family_index = 0
    while len(selected) < count:
        family = families[family_index % len(families)]
        family_index += 1
        if pools[family]:
            selected.append(pools[family].pop())
        elif not any(pools.values()):
            raise RuntimeError(f"only constructed {len(selected)} of {count} tasks")
    return selected


def _question(candidate: Candidate) -> str:
    marker = (
        "exact string literal"
        if candidate.family == "string_literal"
        else "exact call expression"
    )
    return f"Which Python function or class encloses the {marker} {candidate.query!r}?"


def _task(
    candidate: Candidate,
    files: list[Path],
    commit: str,
    answer_pool: list[str],
) -> tuple[dict[str, object], str]:
    rng = random.Random(int(candidate.fingerprint[:16], 16))
    relative_files = [path.relative_to(ROOT).as_posix() for path in files]
    distractors = [path for path in relative_files if path != candidate.path]
    rng.shuffle(distractors)
    paths = [candidate.path, *distractors[:4]]
    rng.shuffle(paths)
    other_answers = [answer for answer in answer_pool if answer != candidate.answer]
    rng.shuffle(other_answers)
    answer_choices = [candidate.answer, *other_answers[:3]]
    rng.shuffle(answer_choices)
    task_id = f"proof2-{candidate.family}-{candidate.fingerprint[:12]}"
    return (
        {
            "task_id": task_id,
            "family": candidate.family,
            "question": _question(candidate),
            "candidate_paths": paths,
            "answer_choices": answer_choices,
            "repository_commit": commit,
        },
        candidate.answer,
    )


def _evaluator_source(name: str, answers: dict[str, str]) -> str:
    encoded = json.dumps(answers, indent=2, sort_keys=True)
    return f'''"""Deterministic evaluator for the Proof 2 {name} split."""
import json
import sys

ANSWERS = {encoded}

request = json.load(sys.stdin)
task_id = request.get("task_id")
if task_id not in ANSWERS:
    raise ValueError("unknown Proof 2 {name} task")
observed = str(request.get("output_text", "")).strip()
success = observed == ANSWERS[task_id]
json.dump({{"success": success, "score": float(success), "details": {{}}}}, sys.stdout)
'''


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(
    output: Path,
    name: str,
    candidates: list[Candidate],
    files: list[Path],
    commit: str,
    answer_pool: list[str],
) -> dict[str, object]:
    tasks_and_answers = [
        _task(candidate, files, commit, answer_pool) for candidate in candidates
    ]
    tasks = [item[0] for item in tasks_and_answers]
    answers = {task["task_id"]: answer for task, answer in tasks_and_answers}
    tasks_path = output / f"{name}-tasks.json"
    evaluator_path = output / f"{name}-evaluator.py"
    tasks_path.write_text(
        json.dumps({"tasks": tasks}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluator_path.write_text(_evaluator_source(name, answers), encoding="utf-8")
    return {
        "task_count": len(tasks),
        "family_counts": dict(Counter(task["family"] for task in tasks)),
        "tasks_sha256": _digest(tasks_path),
        "evaluator_sha256": _digest(evaluator_path),
    }


def _assert_static_audit(
    split: list[Candidate], all_candidates: list[Candidate]
) -> None:
    query_counts = Counter(candidate.query for candidate in all_candidates)
    if any(query_counts[candidate.query] != 1 for candidate in split):
        raise RuntimeError("split contains a non-unique repository query")
    fingerprints = [candidate.fingerprint for candidate in split]
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("split contains duplicate task fingerprints")


def build(output: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    files = _source_files()
    previous = _previous_queries()
    candidates = [
        candidate
        for candidate in _collect_candidates(files)
        if candidate.query not in previous and _question(candidate) not in previous
    ]
    pools = {
        family: [candidate for candidate in candidates if candidate.family == family]
        for family in ("call_expression", "string_literal")
    }
    rng = random.Random(SEED)
    for pool in pools.values():
        rng.shuffle(pool)

    pilot = _balanced_take(pools, PILOT_COUNT)
    calibration = _balanced_take(pools, CALIBRATION_COUNT)
    draft_heldout = _balanced_take(pools, DRAFT_HELDOUT_COUNT)
    combined = [*pilot, *calibration, *draft_heldout]
    _assert_static_audit(combined, candidates)
    commit = _git_commit()
    answer_pool = sorted({candidate.answer for candidate in candidates})
    record = {
        "schema_version": "0.1",
        "status": "draft_heldout_not_frozen",
        "repository_commit": commit,
        "seed": SEED,
        "source_file_count": len(files),
        "prior_workload_queries_excluded": len(previous),
        "eligible_candidate_count": len(candidates),
        "splits": {
            "pilot": _write_split(
                output, "pilot", pilot, files, commit, answer_pool
            ),
            "calibration": _write_split(
                output, "calibration", calibration, files, commit, answer_pool
            ),
            "draft_heldout_pool": _write_split(
                output,
                "draft-heldout",
                draft_heldout,
                files,
                commit,
                answer_pool,
            ),
        },
        "separation": {
            "pilot_calibration_overlap": False,
            "pilot_draft_heldout_overlap": False,
            "calibration_draft_heldout_overlap": False,
            "prior_consumed_proof_queries_excluded": True,
        },
    }
    record_path = output / "construction-record.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "proof-2-preparation",
    )
    args = parser.parse_args()
    record = build(args.output.resolve())
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
