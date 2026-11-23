from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.build_compatibility_page import load, render


ROOT = Path(__file__).parents[1]
DATA = ROOT / "docs/compatibility-data/2026-11-23.json"
PAGE = ROOT / "docs/compatibility.md"


class CompatibilityPageTests(unittest.TestCase):
    def test_checked_in_page_is_generated_from_sanitized_evidence(self) -> None:
        payload = load(DATA)
        self.assertEqual(PAGE.read_text(encoding="utf-8"), render(payload))
        self.assertEqual(len(payload["combinations"]), 3)
        self.assertEqual(
            {row["vllm_version"] for row in payload["combinations"]},
            {"0.33.0", "0.34.0"},
        )
        self.assertTrue(
            any("14B" in row["model"] for row in payload["combinations"])
        )
        self.assertTrue(
            all(row["status"] == "passed" for row in payload["combinations"])
        )

    def test_public_evidence_has_no_private_endpoint_or_identifier(self) -> None:
        rendered = json.dumps(json.loads(DATA.read_text()), sort_keys=True).lower()
        for marker in (
            "modal.direct",
            "server_url",
            '"server_id":',
            '"object_id":',
            "modal-authorization",
            "/users/",
        ):
            self.assertNotIn(marker, rendered)


if __name__ == "__main__":
    unittest.main()
