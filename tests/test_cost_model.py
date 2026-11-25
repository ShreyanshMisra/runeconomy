from __future__ import annotations

import unittest

from runeconomy.cost_model import CostModelError, parse_cost_model


class CostModelTests(unittest.TestCase):
    def test_gpu_hourly_sequential_and_fair_share_under_load(self) -> None:
        model = parse_cost_model({"schema_version": "0.1", "kind": "gpu_hourly", "usd_per_gpu_hour": 3.6})
        sequential = model({"workflow_latency_ms": 1000.0})
        self.assertAlmostEqual(sequential["cost_usd"], 0.001)
        self.assertEqual(sequential["attribution"], "reserved_gpu_time")
        loaded = model({"load": {"started_ms": 0.0, "ended_ms": 4000.0, "mean_in_flight": 4.0}})
        self.assertAlmostEqual(loaded["gpu_time_ms"], 1000.0)
        self.assertEqual(loaded["attribution"], "fair_share_reserved_gpu_time")
        self.assertTrue(loaded["cost_model_sha256"].startswith("sha256:"))

    def test_per_token_unknown_tokens_stay_unknown(self) -> None:
        model = parse_cost_model(
            {"schema_version": "0.1", "kind": "per_token", "usd_per_1m_input_tokens": 1.0,
             "usd_per_1m_cached_input_tokens": 0.1, "usd_per_1m_output_tokens": 4.0}
        )
        priced = model({"metrics": {"input_tokens": 1000, "cached_input_tokens": 500, "output_tokens": 100}})
        self.assertAlmostEqual(priced["cost_usd"], (500 * 1.0 + 500 * 0.1 + 100 * 4.0) / 1e6)
        self.assertIsNone(model({"metrics": {}})["cost_usd"])

    def test_invalid_declarations_are_rejected(self) -> None:
        for value in (
            {"schema_version": "0.2", "kind": "gpu_hourly", "usd_per_gpu_hour": 1},
            {"schema_version": "0.1", "kind": "gpu_hourly", "usd_per_gpu_hour": 0},
            {"schema_version": "0.1", "kind": "per_token", "usd_per_1m_input_tokens": 1},
            {"schema_version": "0.1", "kind": "gpu_hourly", "usd_per_gpu_hour": 1, "extra": 1},
        ):
            with self.subTest(value=value), self.assertRaises(CostModelError):
                parse_cost_model(value)


if __name__ == "__main__":
    unittest.main()
