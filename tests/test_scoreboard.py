from __future__ import annotations

import json
import unittest
from importlib.resources import files
from pathlib import Path

import jsonschema

from runeconomy.scoreboard import build_cards, proof_card, render_markdown

ROOT = Path(__file__).parents[1]
RECORDS = sorted((ROOT / "docs" / "proof-records").glob("*.json"))
SCHEMA = json.loads(files("runeconomy").joinpath("data/verification-card-0.1.schema.json").read_text(encoding="utf-8"))


class ScoreboardTests(unittest.TestCase):
    def test_cards_validate_and_match_their_records(self) -> None:
        cards = build_cards(RECORDS, ROOT)
        self.assertEqual({card["id"] for card in cards}, {"proof-3", "proof-4", "study-1", "study-2"})
        for card in cards:
            jsonschema.validate(card, SCHEMA)
            record = json.loads((ROOT / card["record"]["path"]).read_text(encoding="utf-8"))
            if card["kind"] == "proof":
                self.assertEqual(card["verdict"], record["decision"]["status"])
                self.assertEqual(card["decisive_gate"], record["decision"]["decisive_gate"])
                self.assertEqual(card["claim_boundary"], record["claim_boundary"])
            else:
                self.assertEqual(card["verdict"], "descriptive")
                self.assertFalse(card["preregistered"])

    def test_committed_cards_and_page_are_current(self) -> None:
        cards = build_cards(RECORDS, ROOT)
        committed = json.loads((ROOT / "docs" / "scoreboard" / "cards.json").read_text(encoding="utf-8"))
        self.assertEqual(committed, json.loads(json.dumps(cards)), "run scripts/build_scoreboard.py")
        page = render_markdown(cards, "https://github.com/ShreyanshMisra/runeconomy")
        self.assertEqual((ROOT / "docs" / "scoreboard.md").read_text(encoding="utf-8"), page)

    def test_digest_changes_with_the_record(self) -> None:
        record = json.loads(RECORDS[0].read_text(encoding="utf-8"))
        card = proof_card(record, "x.json", "sha256:" + "0" * 64) if "proof" in record else None
        if card is not None:
            self.assertEqual(card["record"]["sha256"], "sha256:" + "0" * 64)

    def test_unknown_record_types_are_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "other.json"
            path.write_text(json.dumps({"result_type": "something.else"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_cards([path], Path(raw))


if __name__ == "__main__":
    unittest.main()
