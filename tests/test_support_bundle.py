from __future__ import annotations

import json
import os
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy.cli import main
from runeconomy.support_bundle import (
    SUPPORT_BUNDLE_FILES,
    SupportBundleError,
    create_support_bundle,
    validate_support_bundle,
)


SENTINELS = (
    "PRIVATE_PROMPT_SENTINEL",
    "PRIVATE_RESPONSE_SENTINEL",
    "PRIVATE_ANSWER_SENTINEL",
    "PRIVATE_TOKEN_918273",
    "PRIVATE_USERNAME_SENTINEL",
    "sk-private-api-key-sentinel",
    "Authorization: Bearer private-bearer-sentinel",
    "https://private.example.test/secret?token=private",
    "/" + "Users/private-user/SecretProject",
    "203.0.113.77",
    "PRIVATE_COOKIE_SENTINEL",
    "PRIVATE_ENV_VALUE_SENTINEL",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _workspace(root: Path, *, status: str = "runnable", version: str = "0.1") -> Path:
    workspace = root / "experiment"
    workspace.mkdir()
    descriptor = {
        "artifact_type": "runeconomy.experiment_workspace",
        "workspace_version": version,
        "generation_status": status,
        "mode": "static_request",
        "source_evidence": {
            "name": "SecretProject-customer-trace.json",
            "sha256": "sha256:" + "a" * 64,
            "kind": "openinference",
        },
        "selected_finding": {
            "title": SENTINELS[0],
            "finding_id": "customer-private-id",
        },
        "capability_assessment": {
            "assessment_version": "0.1",
            "runnable_experiment": status == "runnable",
            "recommended_mode": "static_request",
            "next_action": SENTINELS[1],
            "capabilities": {
                "basic_diagnosis": {
                    "status": "available",
                    "evidence": SENTINELS[0],
                },
                "exact_token_cache_analysis": {
                    "status": "partial",
                    "next_action": SENTINELS[3],
                },
            },
        },
        "unresolved_fields": (
            [
                {
                    "path": "backend.model_revision",
                    "reason": SENTINELS[2],
                    "private_value": SENTINELS[5],
                }
            ]
            if status == "unresolved"
            else []
        ),
        "files": {
            "manifest": "manifest.json",
            "intervention": "intervention.md",
            "readme": "README.md",
        },
    }
    manifest = {
        "artifact_type": "runeconomy.experiment_manifest",
        "schema_version": "0.3",
        "backend": {
            "base_url": SENTINELS[7],
            "api_key": SENTINELS[5],
            "cookies": SENTINELS[10],
            "environment": {"PRIVATE_ENV": SENTINELS[11]},
        },
        "baseline_request": {
            "messages": [{"role": "user", "content": SENTINELS[0]}],
            "prompt_token_ids": [918273, 918274],
        },
        "candidate_request": {"generated_output": SENTINELS[1]},
        "evaluator": {"expected": SENTINELS[2]},
    }
    _write_json(workspace / "workspace.json", descriptor)
    _write_json(workspace / "manifest.json", manifest)
    (workspace / "intervention.md").write_text(SENTINELS[0], encoding="utf-8")
    (workspace / "README.md").write_text(SENTINELS[6], encoding="utf-8")
    return workspace


def _documents(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: json.loads(archive.read(name).decode("utf-8"))
            for name in archive.namelist()
        }


class SupportBundleTests(unittest.TestCase):
    def test_valid_workspace_produces_versioned_allowlisted_archive(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = _workspace(root)
            doctor = root / "doctor.json"
            validation = root / "validation.json"
            _write_json(
                doctor,
                {
                    "schema_version": "0.2",
                    "backend": "vllm",
                    "backend_version": "0.33.0",
                    "model_ids": ["private/customer-model"],
                    "chat_completions": {
                        "status": "supported",
                        "evidence": SENTINELS[0],
                    },
                    "tokenize": {"status": "unknown", "next_action": SENTINELS[7]},
                    "readiness": {"status": "ready", "summary": SENTINELS[1]},
                },
            )
            _write_json(
                validation,
                {
                    "validation_version": "0.1",
                    "status": "claim_evidence_incomplete",
                    "runnable": True,
                    "mode": "static_request",
                    "experiment_id": "private-customer-experiment",
                    "issues": [
                        {
                            "code": "runeconomy.claim.incomplete",
                            "artifact_path": SENTINELS[8],
                            "field_path": "acceptance.require_environment_identity",
                            "reason": SENTINELS[0],
                            "execution_blocking": False,
                            "claim_weakening": True,
                            "next_action": SENTINELS[7],
                            "command": SENTINELS[6],
                        }
                    ],
                },
            )
            output = root / "support.zip"
            summary = create_support_bundle(
                workspace,
                output,
                doctor_report=doctor,
                validation_report=validation,
            )
            validate_support_bundle(output)
            self.assertEqual(summary["schema_version"], "0.1")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(tuple(archive.namelist()), SUPPORT_BUNDLE_FILES)
                text = "\n".join(
                    [*archive.namelist(), *(archive.read(name).decode() for name in archive.namelist())]
                )
            for sentinel in SENTINELS:
                self.assertNotIn(sentinel, text)
            self.assertNotIn("918273", text)
            documents = _documents(output)
            self.assertEqual(documents["manifest.json"]["schema_version"], "0.1")
            self.assertFalse(
                documents["manifest.json"]["determinism"]["creation_timestamp_included"]
            )
            self.assertEqual(
                documents["validation.json"]["issues"][0]["code"],
                "runeconomy.claim.incomplete",
            )
            self.assertEqual(
                documents["capabilities.json"]["saved_backend_assessment"][
                    "capabilities"
                ]["chat_completions"]["status"],
                "supported",
            )

    def test_archive_is_deterministic_on_one_environment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = _workspace(root)
            first = root / "first.zip"
            second = root / "second.zip"
            create_support_bundle(workspace, first)
            create_support_bundle(workspace, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_unresolved_workspace_and_missing_reports_are_honest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "support.zip"
            create_support_bundle(_workspace(root, status="unresolved"), output)
            documents = _documents(output)
            self.assertEqual(documents["workspace.json"]["status"], "unresolved")
            self.assertEqual(
                documents["capabilities.json"]["saved_backend_assessment"][
                    "availability"
                ],
                "unavailable",
            )
            self.assertEqual(documents["validation.json"]["availability"], "unavailable")
            self.assertEqual(
                documents["errors.json"]["errors"][0]["code"],
                "runeconomy.workspace.unresolved",
            )
            unresolved = documents["workspace.json"]["unresolved_fields"][0]
            self.assertEqual(unresolved["field_path"], "backend.model_revision")
            self.assertNotIn("reason", unresolved)

    def test_malformed_and_unsupported_workspace_still_produce_safe_bundle(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "malformed"
            workspace.mkdir()
            (workspace / "workspace.json").write_text("{not json", encoding="utf-8")
            (workspace / "manifest.json").write_text(SENTINELS[0], encoding="utf-8")
            output = root / "malformed.zip"
            create_support_bundle(workspace, output)
            documents = _documents(output)
            self.assertEqual(documents["workspace.json"]["status"], "invalid")
            self.assertTrue(documents["errors.json"]["errors"])

            unsupported = _workspace(root, version="99")
            unsupported_output = root / "unsupported.zip"
            create_support_bundle(unsupported, unsupported_output)
            self.assertEqual(
                _documents(unsupported_output)["workspace.json"]["status"],
                "unsupported_schema",
            )

    def test_existing_output_is_not_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = _workspace(root)
            output = root / "support.zip"
            output.write_bytes(b"keep-me")
            with self.assertRaisesRegex(SupportBundleError, "already exists"):
                create_support_bundle(workspace, output)
            self.assertEqual(output.read_bytes(), b"keep-me")

    def test_symlink_roles_fail_and_unexpected_entries_are_ignored(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "outside.json"
            target.write_text(SENTINELS[0], encoding="utf-8")
            unsafe = root / "unsafe"
            unsafe.mkdir()
            os.symlink(target, unsafe / "workspace.json")
            (unsafe / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(SupportBundleError, "must not be a symlink"):
                create_support_bundle(unsafe, root / "unsafe.zip")

            workspace = _workspace(root)
            (workspace / "arbitrary-private-file.txt").write_text(
                SENTINELS[1], encoding="utf-8"
            )
            os.symlink(target, workspace / "unexpected-link")
            output = root / "safe.zip"
            create_support_bundle(workspace, output)
            documents = _documents(output)
            self.assertEqual(
                documents["workspace.json"]["unexpected_entries"],
                {"count": 2, "handling": "ignored", "symlink_count": 1},
            )
            text = json.dumps(documents)
            self.assertNotIn("arbitrary-private-file", text)
            self.assertNotIn(SENTINELS[1], text)

    def test_malformed_optional_reports_are_recorded_without_payloads(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = _workspace(root)
            doctor = root / "doctor.json"
            validation = root / "validation.json"
            doctor.write_text(SENTINELS[5] + "{", encoding="utf-8")
            validation.write_text(SENTINELS[6] + "{", encoding="utf-8")
            output = root / "support.zip"
            create_support_bundle(
                workspace,
                output,
                doctor_report=doctor,
                validation_report=validation,
            )
            documents = _documents(output)
            self.assertEqual(
                documents["capabilities.json"]["saved_backend_assessment"][
                    "availability"
                ],
                "invalid",
            )
            self.assertEqual(documents["validation.json"]["availability"], "invalid")
            text = json.dumps(documents)
            self.assertNotIn(SENTINELS[5], text)
            self.assertNotIn(SENTINELS[6], text)

    def test_input_derived_labels_are_reduced_to_known_safe_values(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = _workspace(root)
            descriptor = json.loads((workspace / "workspace.json").read_text())
            descriptor["mode"] = SENTINELS[4]
            descriptor["generation_status"] = SENTINELS[4]
            descriptor["capability_assessment"]["assessment_version"] = SENTINELS[4]
            descriptor["capability_assessment"]["recommended_mode"] = SENTINELS[4]
            _write_json(workspace / "workspace.json", descriptor)
            doctor = root / "doctor.json"
            _write_json(
                doctor,
                {
                    "backend": "vllm",
                    "backend_version": SENTINELS[4],
                    "readiness": {"status": SENTINELS[4]},
                },
            )
            output = root / "support.zip"
            create_support_bundle(workspace, output, doctor_report=doctor)
            documents = _documents(output)
            text = json.dumps(documents)
            self.assertNotIn(SENTINELS[4], text)
            self.assertEqual(documents["workspace.json"]["mode"], "unknown")
            self.assertEqual(documents["workspace.json"]["status"], "invalid")
            doctor_section = documents["capabilities.json"][
                "saved_backend_assessment"
            ]
            self.assertEqual(doctor_section["backend_version"], "unknown")
            self.assertEqual(doctor_section["readiness_status"], "unknown")

    def test_human_and_json_errors_use_same_code_and_remove_exception_payload(self) -> None:
        error = SupportBundleError(
            f"failure {SENTINELS[0]} {SENTINELS[7]}",
            error_code="runeconomy.support.unsafe_input",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = _workspace(root)
            human = StringIO()
            with (
                patch("runeconomy.cli.create_support_bundle", side_effect=error),
                redirect_stderr(human),
            ):
                self.assertEqual(
                    main(
                        [
                            "support-bundle",
                            str(workspace),
                            "--output",
                            str(root / "human.zip"),
                        ]
                    ),
                    2,
                )
            machine = StringIO()
            with (
                patch("runeconomy.cli.create_support_bundle", side_effect=error),
                redirect_stderr(machine),
            ):
                self.assertEqual(
                    main(
                        [
                            "support-bundle",
                            str(workspace),
                            "--output",
                            str(root / "json.zip"),
                            "--json",
                        ]
                    ),
                    2,
                )
            payload = json.loads(machine.getvalue())
            code = "runeconomy.support.unsafe_input"
            self.assertIn(f"Code: {code}", human.getvalue())
            self.assertEqual(payload["error"]["code"], code)
            for sentinel in (SENTINELS[0], SENTINELS[7]):
                self.assertNotIn(sentinel, human.getvalue())
                self.assertNotIn(sentinel, machine.getvalue())

    def test_cli_success_lists_contents_and_privacy_warning(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = _workspace(root)
            output = root / "support.zip"
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "support-bundle",
                            str(workspace),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            rendered = stdout.getvalue()
            for name in SUPPORT_BUNDLE_FILES:
                self.assertIn(name, rendered)
            self.assertIn("did not upload", rendered)
            self.assertIn("Inspect every file", rendered)


if __name__ == "__main__":
    unittest.main()
