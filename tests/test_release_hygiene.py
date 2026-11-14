from __future__ import annotations

import json
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

    def test_public_sources_contain_no_private_modal_endpoint_or_local_path(self) -> None:
        roots = [
            ROOT / ".env.example",
            ROOT / ".github",
            ROOT / "README.md",
            ROOT / "deploy",
            ROOT / "docs",
            ROOT / "examples",
            ROOT / "pyproject.toml",
            ROOT / "scripts",
            ROOT / "src",
            ROOT / "tests",
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
        self.assertNotIn("/.env.local", sdist["include"])

    def test_public_governance_documents_are_not_prematurely_published(self) -> None:
        for name in ("CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md"):
            self.assertFalse((ROOT / name).exists(), name)

    def test_package_metadata_points_to_public_repository(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["name"], "runeconomy")
        self.assertEqual(project["version"], "0.1.0")
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


if __name__ == "__main__":
    unittest.main()
