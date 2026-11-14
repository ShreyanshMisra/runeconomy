from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class DocumentationTests(unittest.TestCase):
    def test_current_alpha_is_consistent_across_onboarding_sources(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            version = tomllib.load(handle)["project"]["version"]
        for relative in (
            "README.md",
            "docs/alpha-release.md",
            "docs/getting-started.md",
            "skills/runeconomy/SKILL.md",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(f"runeconomy=={version}", text, relative)

    def test_local_markdown_links_resolve(self) -> None:
        sources = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
        for source in sources:
            text = source.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (source.parent / target).resolve()
                self.assertTrue(resolved.exists(), f"{source}: missing {target}")

    def test_site_sources_are_repository_documents(self) -> None:
        build = (ROOT / "site" / "build.mjs").read_text(encoding="utf-8")
        sources = re.findall(r'source: "([^"]+)"', build)
        self.assertGreaterEqual(len(sources), 10)
        for source in sources:
            self.assertTrue((ROOT / source).is_file(), source)
        self.assertIn("docs/public-proof.md", sources)
        self.assertIn("docs/agent-skill.md", sources)

    def test_frozen_claim_boundary_remains_prominent(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        proof = (ROOT / "docs" / "public-proof.md").read_text(encoding="utf-8")
        for text in (readme, proof):
            self.assertIn("invalid", text.lower())
            self.assertIn("79.61%", text)
            self.assertIn("80", text)
            self.assertIn("consumed", text)


if __name__ == "__main__":
    unittest.main()
