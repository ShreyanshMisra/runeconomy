from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills" / "runeconomy" / "SKILL.md"


class AgentSkillTests(unittest.TestCase):
    def test_skill_is_canonical_and_version_aligned(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        with (ROOT / "pyproject.toml").open("rb") as handle:
            version = tomllib.load(handle)["project"]["version"]
        self.assertIn("name: runeconomy", text)
        self.assertIn(f"runeconomy=={version}", text)
        self.assertRegex(text, r"description: .+")
        self.assertNotRegex(text, r"TODO|PLACEHOLDER|example\.com")

    def test_skill_preserves_safety_and_verdict_contract(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        for required in (
            "runeconomy doctor",
            "runeconomy experiment init",
            "runeconomy experiment validate",
            "runeconomy compare",
            "RUNECONOMY_TRACE_PATH",
            "RUNECONOMY_STATUS_PATH",
            "runeconomy.depends_on",
            "verified",
            "regression",
            "inconclusive",
            "invalid",
            "Do not enable development endpoints",
            "Never print or commit API keys",
        ):
            self.assertIn(required, text)
        self.assertIn("Only call a change an improvement", text)

    def test_installation_guide_keeps_repository_source_canonical(self) -> None:
        text = (ROOT / "docs" / "agent-skill.md").read_text(encoding="utf-8")
        self.assertIn("repository copy is the source of truth", text)
        self.assertIn(".agents/skills/runeconomy/SKILL.md", text)
        self.assertIn(".claude/skills/runeconomy/SKILL.md", text)
        self.assertIn(".cursor/rules", text)
        self.assertIn("not universal `SKILL.md` discovery", text)


if __name__ == "__main__":
    unittest.main()
