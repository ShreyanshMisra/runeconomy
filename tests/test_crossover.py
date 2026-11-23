from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from runeconomy.crossover import balanced_two_block_split


ROOT = Path(__file__).resolve().parents[1]


def _tasks(per_family: int) -> list[dict[str, str]]:
    return [
        {"task_id": f"{family}-{index:03d}", "family": family}
        for family in ("call_expression", "string_literal")
        for index in range(per_family)
    ]


class BalancedCrossoverTests(unittest.TestCase):
    def test_180_tasks_split_into_balanced_disjoint_90_task_blocks(self) -> None:
        tasks = _tasks(90)
        first = balanced_two_block_split(
            tasks,
            seed=20261123,
            family_names=("call_expression", "string_literal"),
        )
        second = balanced_two_block_split(
            list(reversed(tasks)),
            seed=20261123,
            family_names=("call_expression", "string_literal"),
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"block_a", "block_b"})
        for block in first.values():
            self.assertEqual(len(block), 90)
            self.assertEqual(
                Counter(task["family"] for task in block),
                {"call_expression": 45, "string_literal": 45},
            )
        ids = [
            {task["task_id"] for task in block_tasks} for block_tasks in first.values()
        ]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual(ids[0] | ids[1], {task["task_id"] for task in tasks})

    def test_split_rejects_unbalanced_odd_and_duplicate_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal counts"):
            balanced_two_block_split(
                _tasks(4)[:-1],
                seed=1,
                family_names=("call_expression", "string_literal"),
            )
        with self.assertRaisesRegex(ValueError, "even"):
            balanced_two_block_split(
                _tasks(3),
                seed=1,
                family_names=("call_expression", "string_literal"),
            )
        duplicate = _tasks(4)
        duplicate[-1]["task_id"] = duplicate[0]["task_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            balanced_two_block_split(
                duplicate,
                seed=1,
                family_names=("call_expression", "string_literal"),
            )

    def test_validation_runner_remains_consumed_calibration_only(self) -> None:
        source = (ROOT / "deploy" / "modal_proof_2.py").read_text(encoding="utf-8")
        self.assertIn("CONSUMED_CALIBRATION_TASKS_SHA256", source)
        self.assertIn("CONSUMED_CALIBRATION_EVALUATOR_SHA256", source)
        self.assertIn("def run_crossover_validation(", source)
        self.assertIn("def crossover_validation(", source)
        self.assertIn("CROSSOVER_DEVICE_ASSIGNMENTS", source)
        self.assertIn("client.reset_prefix_cache()", source)
        self.assertIn("def heldout(", source)
        self.assertIn("FROZEN_HELDOUT_TASKS_SHA256", source)
        self.assertIn("FROZEN_HELDOUT_EVALUATOR_SHA256", source)
        self.assertIn("heldout_run_ledger.put", source)


if __name__ == "__main__":
    unittest.main()
