from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import jsonschema

from runeconomy.attempt_status import (
    ATTEMPT_STATUSES,
    AttemptStatus,
    AttemptStatusError,
    write_attempt_status,
)


class AttemptStatusTests(unittest.TestCase):
    def test_checked_in_schema_accepts_every_documented_status(self) -> None:
        schema_path = (
            Path(__file__).parents[1]
            / "src"
            / "runeconomy"
            / "data"
            / "attempt-status-0.1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        for status in ATTEMPT_STATUSES:
            payload: dict[str, object] = {
                "schema_version": "0.1",
                "status": status,
            }
            if status != "completed":
                payload["provenance"] = {
                    "component": "agent",
                    "stage": "workflow",
                    "code": "declared_failure",
                }
            jsonschema.validate(payload, schema)
            AttemptStatus.from_dict(payload)

    def test_writes_and_loads_completed_status_atomically(self) -> None:
        with TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "status.json"
            write_attempt_status(path, "completed")
            status = AttemptStatus.from_path(path)

        self.assertEqual(status.status, "completed")
        self.assertIsNone(status.provenance)

    def test_failure_requires_content_free_provenance(self) -> None:
        with TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "status.json"
            write_attempt_status(
                path,
                "model_output_failure",
                component="model_output",
                stage="planning",
                code="invalid_structured_output",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], "0.1")
        self.assertNotIn("message", payload)
        self.assertEqual(payload["provenance"]["stage"], "planning")

    def test_rejects_free_form_or_extra_status_content(self) -> None:
        with self.assertRaises(AttemptStatusError):
            AttemptStatus.from_dict(
                {
                    "schema_version": "0.1",
                    "status": "application_failure",
                    "provenance": {
                        "component": "application",
                        "stage": "planning response contained secret text",
                        "code": "failed",
                    },
                }
            )
        with self.assertRaises(AttemptStatusError):
            AttemptStatus.from_dict(
                {
                    "schema_version": "0.1",
                    "status": "model_output_failure",
                    "provenance": {
                        "component": "model_output",
                        "stage": "planning",
                        "code": "invalid",
                    },
                    "generated_text": "do not publish",
                }
            )

    def test_rejects_completed_status_on_failure_provenance(self) -> None:
        with self.assertRaises(AttemptStatusError):
            AttemptStatus.from_dict(
                {
                    "schema_version": "0.1",
                    "status": "completed",
                    "provenance": {
                        "component": "agent",
                        "stage": "finish",
                        "code": "unexpected",
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
