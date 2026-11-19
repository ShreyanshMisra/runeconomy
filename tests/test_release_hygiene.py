from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def test_public_manifests_use_non_routable_endpoint(self) -> None:
        for path in sorted((ROOT / "examples" / "experiments").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["backend"]["base_url"],
                "https://vllm.example.invalid",
                path,
            )
            self.assertEqual(payload["backend"]["api_key_env"], "VLLM_API_KEY", path)

    def test_public_sources_contain_no_private_modal_endpoint_or_local_path(
        self,
    ) -> None:
        roots = [
            ROOT / ".env.example",
            ROOT / ".github",
            ROOT / "README.md",
            ROOT / "deploy",
            ROOT / "docs",
            ROOT / "examples",
            ROOT / "pyproject.toml",
            ROOT / "scripts",
            ROOT / "site",
            ROOT / "skills",
            ROOT / "src",
            ROOT / "tests",
            ROOT / "vercel.json",
        ]
        files: list[Path] = []
        for root in roots:
            files.extend([root] if root.is_file() else root.rglob("*"))
        private_modal_suffix = "." + "modal.direct"
        local_home_prefix = "/" + "Users/"
        modal_proxy_variable = "MODAL_" + "PROXY_TOKEN"
        for path in files:
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or "node_modules" in path.parts
                or (ROOT / "site" / "dist") in path.parents
                or (ROOT / "docs" / "internal") in path.parents
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn(private_modal_suffix, text, path)
            self.assertNotIn(local_home_prefix, text, path)
            self.assertNotIn(modal_proxy_variable, text, path)

    def test_local_environment_file_is_ignored(self) -> None:
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", ".env.local"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)

    def test_internal_documents_are_ignored_as_one_directory(self) -> None:
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", "docs/internal/release-checklist.md"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)

    def test_distribution_configuration_excludes_private_material(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            configuration = tomllib.load(handle)
        sdist = configuration["tool"]["hatch"]["build"]["targets"]["sdist"]
        self.assertIn("/docs/internal", sdist["exclude"])
        self.assertIn(
            "/examples/experiments/repository-agent-tool-pruning-heldout.json",
            sdist["exclude"],
        )
        self.assertIn(
            "/examples/fixtures/recorded-cli-build-parser.txt",
            sdist["exclude"],
        )
        self.assertIn("/skills", sdist["include"])
        self.assertNotIn("/.env.local", sdist["include"])

    def test_public_governance_documents_are_not_prematurely_published(self) -> None:
        for name in ("CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md"):
            self.assertFalse((ROOT / name).exists(), name)

    def test_package_metadata_points_to_public_repository(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["name"], "runeconomy")
        self.assertEqual(project["version"], "0.2.1")
        self.assertIn(
            "Development Status :: 3 - Alpha",
            project["classifiers"],
        )
        self.assertEqual(
            project["urls"]["Repository"],
            "https://github.com/ShreyanshMisra/runeconomy",
        )

    def test_publish_workflows_use_oidc_not_repository_secrets(self) -> None:
        workflow_root = ROOT / ".github" / "workflows"
        for name in ("publish.yml", "publish-testpypi.yml"):
            text = (workflow_root / name).read_text(encoding="utf-8")
            self.assertIn("id-token: write", text)
            self.assertNotIn("secrets.", text)
            self.assertNotIn("password:", text)

    def test_support_issue_template_requires_privacy_confirmation(self) -> None:
        template = (
            ROOT / ".github" / "ISSUE_TEMPLATE" / "support.yml"
        ).read_text(encoding="utf-8")
        for marker in (
            "Stable error code",
            "never uploaded automatically",
            "I inspected every file I am attaching",
            "I did not attach raw traces",
            "did not authorize uploading",
        ):
            self.assertIn(marker, template)

    def test_removed_experiment_brand_is_absent_from_active_tracked_files(self) -> None:
        frozen = {
            Path("examples/fixtures/recorded-cli-build-parser.txt"),
            Path("examples/experiments/repository-agent-tool-pruning-heldout.json"),
        }
        tracked = (
            subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .split("\0")
        )
        pattern = re.compile(
            "negative" + r"[ _-]?split|negative" + "split",
            re.IGNORECASE,
        )
        for relative_text in tracked:
            if not relative_text:
                continue
            relative = Path(relative_text)
            if not (ROOT / relative).is_file():
                continue
            if (
                relative in frozen
                or (ROOT / "docs" / "internal") in (ROOT / relative).parents
            ):
                continue
            text = (ROOT / relative).read_text(encoding="utf-8", errors="ignore")
            self.assertIsNone(pattern.search(text), relative)

    def test_frozen_parser_evidence_remains_byte_identical(self) -> None:
        expected = {
            "examples/fixtures/recorded-cli-build-parser.txt": "2e53bcb76255840483deb6398032b55158af2cbfeca37034e233329e843ca5e0",
            "examples/experiments/repository-agent-tool-pruning-heldout.json": "3842cc39f61b374b75b0d91e6bede9543f0ba54acfd1d9ef280875894eb201ec",
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                digest,
                relative,
            )


if __name__ == "__main__":
    unittest.main()
