from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SKILL_DIR = ROOT / "skills" / "runeconomy"
SKILL = SKILL_DIR / "SKILL.md"
EVALS = ROOT / "tests" / "fixtures" / "skill-evals.json"


class AgentSkillTests(unittest.TestCase):
    def test_skill_is_canonical_and_version_neutral(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        self.assertIn("name: runeconomy", text)
        self.assertRegex(text, r"description: .+")
        self.assertNotRegex(text, r"runeconomy==\d")
        self.assertNotRegex(text, r"TODO|PLACEHOLDER|example\.com")
        self.assertIn("runeconomy --version", text)
        self.assertIn("Never upgrade RunEconomy without asking first", text)

    def test_progressive_references_are_complete(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        for name in (
            "evidence-imports.md",
            "experiment-contract.md",
            "verdicts.md",
            "privacy.md",
            "troubleshooting.md",
        ):
            self.assertIn(f"references/{name}", text)
            self.assertTrue((SKILL_DIR / "references" / name).is_file())

    def test_skill_preserves_workflow_safety_and_verdict_contract(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [SKILL, *sorted((SKILL_DIR / "references").glob("*.md"))]
        )
        for required in (
            "runeconomy import SOURCE",
            "runeconomy experiment init",
            "runeconomy experiment validate",
            "--mode agent-process",
            "RUNECONOMY_TRACE_PATH",
            "RUNECONOMY_STATUS_PATH",
            "runeconomy.depends_on",
            "verified",
            "regression",
            "inconclusive",
            "invalid",
            "Only call a change an improvement",
            "Do not start paid resources",
        ):
            self.assertIn(required, combined)

    def test_installation_guide_uses_portable_installer(self) -> None:
        text = (ROOT / "docs" / "agent-skill.md").read_text(encoding="utf-8")
        self.assertIn("repository source", text)
        self.assertIn(
            "npx skills add ShreyanshMisra/runeconomy --skill runeconomy --all",
            text,
        )
        self.assertIn("read the newly installed `SKILL.md` completely", text)
        self.assertNotIn("The skill targets RunEconomy", text)

    def test_evaluation_cases_cover_required_agent_decisions(self) -> None:
        cases = json.loads(EVALS.read_text(encoding="utf-8"))
        observed = {case["expected_behavior"] for case in cases}
        required = {
            "choose_importer",
            "refuse_fabrication",
            "protect_private_evidence",
            "explain_unresolved",
            "choose_agent_process",
            "gate_before_mechanism",
            "interpret_all_verdicts",
            "request_paid_compute_approval",
            "complete_golden_path",
        }
        self.assertEqual(observed, required)
        for case in cases:
            self.assertRegex(case["prompt"], r"\S")
            self.assertRegex(case["observable"], r"\S")
            self.assertIsNone(re.search(r"api[_ -]?key\s*[:=]\s*\S+", case["prompt"], re.I))


if __name__ == "__main__":
    unittest.main()
