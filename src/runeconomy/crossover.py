"""Deterministic balanced block allocation for crossover experiments."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def balanced_two_block_split(
    tasks: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    family_names: tuple[str, str],
    block_names: tuple[str, str] = ("block_a", "block_b"),
) -> dict[str, list[dict[str, Any]]]:
    """Split two equally sized, even strata into deterministic balanced blocks."""

    if len(set(block_names)) != 2:
        raise ValueError("crossover block names must be distinct")
    if len(set(family_names)) != 2:
        raise ValueError("crossover family names must be distinct")
    normalized = [dict(task) for task in tasks]
    task_ids = [task.get("task_id") for task in normalized]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("every crossover task must have a non-empty task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("crossover task IDs must be unique")

    families = Counter(task.get("family") for task in normalized)
    if set(families) != set(family_names):
        raise ValueError("crossover tasks do not match the declared task families")
    if len(set(families.values())) != 1:
        raise ValueError("crossover task families must have equal counts")
    if any(count % 2 for count in families.values()):
        raise ValueError("each crossover task family count must be even")

    blocks: dict[str, list[dict[str, Any]]] = {
        block_names[0]: [],
        block_names[1]: [],
    }
    for family in family_names:
        family_tasks = [task for task in normalized if task["family"] == family]
        ranked = sorted(
            family_tasks,
            key=lambda task: hashlib.sha256(
                f"{seed}:{task['task_id']}".encode()
            ).hexdigest(),
        )
        midpoint = len(ranked) // 2
        blocks[block_names[0]].extend(ranked[:midpoint])
        blocks[block_names[1]].extend(ranked[midpoint:])

    for block_tasks in blocks.values():
        block_tasks.sort(key=lambda task: task["task_id"])
    return blocks
