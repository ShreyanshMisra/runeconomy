"""Build verification cards and the scoreboard page from published decision records.

    uv run python scripts/build_scoreboard.py

Writes docs/scoreboard/cards.json and docs/scoreboard.md. The site renders the
page only when RUNECONOMY_PUBLISH_SCOREBOARD=1 is set at build time.
"""

from __future__ import annotations

import json
from pathlib import Path

from runeconomy.scoreboard import build_cards, render_markdown

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "docs" / "proof-records"
REPOSITORY = "https://github.com/ShreyanshMisra/runeconomy"


def main() -> None:
    paths = sorted(RECORDS.glob("*.json"))
    cards = build_cards(paths, ROOT)
    output = ROOT / "docs" / "scoreboard"
    output.mkdir(exist_ok=True)
    (output / "cards.json").write_text(json.dumps(cards, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "docs" / "scoreboard.md").write_text(render_markdown(cards, REPOSITORY), encoding="utf-8")
    print(f"wrote {len(cards)} cards")


if __name__ == "__main__":
    main()
