from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.comparison import _summarize
from runeconomy.decision_artifacts import enrich_decision_artifact
from runeconomy.identity import (
    IdentityPolicy,
    IdentityPolicyError,
    assess_request_identity,
    identity_assurance_record,
    parse_identity_policy,
    parse_undisclosed,
    variant_identity_summary,
)
from runeconomy.importers import import_openinference
from runeconomy.models import TraceValidationError


PINNED = {
    "backend": "vllm",
    "backend_version": "0.11.0",
    "model": "org/model",
    "model_revision": "abc123",
    "tokenizer": "org/model",
    "tokenizer_revision": "abc123",
    "chat_template_digest": "sha256:" + "1" * 64,
}
HOSTED = {
    "backend": "openai",
    "backend_version": None,
    "model": "model-2026-01-01",
    "model_revision": "model-2026-01-01",
    "tokenizer": None,
    "tokenizer_revision": None,
    "chat_template_digest": None,
}
HOSTED_UNDISCLOSED = [
    "backend_version",
    "chat_template_digest",
    "tokenizer",
    "tokenizer_revision",
]


class IdentityPolicyTests(unittest.TestCase):
    def test_policy_parsing_is_strict_and_none_means_legacy(self) -> None:
        self.assertIsNone(parse_identity_policy(None))
        policy = parse_identity_policy(
            {"schema_version": "0.1", "minimum_grade": "provider_declared"}
        )
        self.assertTrue(policy.accepts("operator_verified"))
        self.assertTrue(policy.accepts("provider_declared"))
        strict = IdentityPolicy()
        self.assertTrue(strict.accepts("operator_verified"))
        self.assertFalse(strict.accepts("provider_declared"))
        for bad in (
            [],
            {"schema_version": "0.2", "minimum_grade": "provider_declared"},
            {"schema_version": "0.1", "minimum_grade": "unknown"},
            {"schema_version": "0.1", "minimum_grade": "provider_declared", "x": 1},
        ):
            with self.assertRaises(IdentityPolicyError):
                parse_identity_policy(bad)

    def test_undisclosed_fields_cannot_name_backend_or_model(self) -> None:
        self.assertEqual(
            parse_undisclosed('["tokenizer", "backend_version"]'),
            ["backend_version", "tokenizer"],
        )
        for bad in ('["model"]', '["backend"]', "not json", '[1]'):
            with self.assertRaises(IdentityPolicyError):
                parse_undisclosed(bad)


class RequestAssessmentTests(unittest.TestCase):
    def test_pinned_identity_is_complete_operator_verified(self) -> None:
        result = assess_request_identity(PINNED, grade=None, undisclosed=None)
        self.assertEqual(result["grade"], "operator_verified")
        self.assertTrue(result["complete"])

    def test_declared_undisclosed_fields_complete_only_when_provider_declared(
        self,
    ) -> None:
        provider = assess_request_identity(
            HOSTED, grade="provider_declared", undisclosed=HOSTED_UNDISCLOSED
        )
        self.assertTrue(provider["complete"])
        self.assertEqual(provider["undisclosed_fields"], [
            "backend_version",
            "tokenizer",
            "tokenizer_revision",
            "chat_template_digest",
        ])
        operator = assess_request_identity(
            HOSTED, grade="operator_verified", undisclosed=HOSTED_UNDISCLOSED
        )
        self.assertFalse(operator["complete"])
        self.assertEqual(len(operator["missing_fields"]), 4)

    def test_undeclared_missing_field_stays_incomplete(self) -> None:
        result = assess_request_identity(
            HOSTED, grade="provider_declared", undisclosed=["tokenizer"]
        )
        self.assertFalse(result["complete"])
        self.assertIn("chat_template_digest", result["missing_fields"])

    def test_placeholder_strings_are_not_identity_values(self) -> None:
        identity = dict(PINNED, tokenizer="provider-managed", backend_version="Undisclosed")
        result = assess_request_identity(identity, grade=None, undisclosed=None)
        self.assertFalse(result["complete"])
        self.assertEqual(
            sorted(result["placeholder_fields"]), ["backend_version", "tokenizer"]
        )

    def test_declaring_a_present_field_undisclosed_is_contradictory(self) -> None:
        result = assess_request_identity(
            PINNED, grade="provider_declared", undisclosed=["tokenizer"]
        )
        self.assertFalse(result["complete"])
        self.assertEqual(result["contradicted_undisclosed_fields"], ["tokenizer"])


class VariantSummaryTests(unittest.TestCase):
    def test_weakest_grade_and_fingerprint_drift_never_affect_validity(self) -> None:
        policy = parse_identity_policy(
            {"schema_version": "0.1", "minimum_grade": "provider_declared"}
        )
        assessments = [
            assess_request_identity(PINNED, grade=None, undisclosed=None),
            assess_request_identity(
                HOSTED, grade="provider_declared", undisclosed=HOSTED_UNDISCLOSED
            ),
        ]
        summary = variant_identity_summary(
            policy, assessments, ["sha256:a", "sha256:b"]
        )
        self.assertEqual(summary["grade"], "provider_declared")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["provider_fingerprint"]["distinct_count"], 2)
        self.assertTrue(summary["provider_fingerprint"]["changed_during_run"])
        self.assertFalse(summary["provider_fingerprint"]["affects_validity"])

        record = identity_assurance_record(
            policy, {"baseline": summary, "candidate": summary}
        )
        self.assertEqual(record["claim_scope"], "observed_service_window")
        self.assertIn(
            "identity.provider_fingerprint_changed",
            [limit["code"] for limit in record["claim_limits"]],
        )

    def test_operator_verified_scope_is_pinned_configuration(self) -> None:
        policy = IdentityPolicy()
        summary = variant_identity_summary(
            policy,
            [assess_request_identity(PINNED, grade=None, undisclosed=None)],
            [None],
        )
        record = identity_assurance_record(
            policy, {"baseline": summary, "candidate": summary}
        )
        self.assertEqual(record["claim_scope"], "pinned_configuration")
        self.assertEqual(record["claim_limits"], [])


def _attempt(variant: str, identity: dict, declaration: dict | None) -> dict:
    metrics = {
        "environment_identity": [identity],
        "environment_identity_complete": all(
            value not in {None, "", "unknown"} for value in identity.values()
        ),
        "dependency_evidence": "complete_explicit",
    }
    if declaration is not None:
        metrics["identity_declarations"] = [declaration]
    return {
        "variant": variant,
        "task_id": "task-1",
        "repetition": 0,
        "status": "completed",
        "task_success": True,
        "workflow_latency_ms": 10.0,
        "metrics": metrics,
    }


class SummaryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attempts = [
            _attempt(
                "baseline",
                HOSTED,
                {
                    "grade": "provider_declared",
                    "undisclosed_fields": HOSTED_UNDISCLOSED,
                    "provider_fingerprint_sha256": "sha256:f",
                },
            ),
            _attempt("candidate", PINNED, None),
        ]

    def test_without_policy_summary_is_legacy(self) -> None:
        summary = _summarize(self.attempts)
        self.assertNotIn("identity_assurance", summary)
        evidence = summary["variants"]["baseline"]["trace_evidence"]
        self.assertFalse(evidence["environment_identity_complete"])
        self.assertNotIn("identity_grade", evidence)
        decision = enrich_decision_artifact(
            {"status": "invalid", "reason": "x", "checks": []}, summary
        )
        self.assertNotIn("identity_assurance", decision)

    def test_policy_grades_identity_and_records_assurance(self) -> None:
        policy = parse_identity_policy(
            {"schema_version": "0.1", "minimum_grade": "provider_declared"}
        )
        summary = _summarize(self.attempts, identity_policy=policy)
        baseline = summary["variants"]["baseline"]["trace_evidence"]
        self.assertTrue(baseline["environment_identity_complete"])
        self.assertEqual(baseline["identity_grade"], "provider_declared")
        decision = enrich_decision_artifact(
            {"status": "verified", "reason": "x", "checks": []}, summary
        )
        assurance = decision["identity_assurance"]
        self.assertEqual(assurance["claim_scope"], "observed_service_window")
        self.assertIn("provider-declared service", assurance["supported_claim"])
        again = enrich_decision_artifact(decision, summary)
        self.assertEqual(again["identity_assurance"], assurance)


class ImporterTests(unittest.TestCase):
    def _trace(self, directory: Path, attributes: dict) -> Path:
        span = {
            "traceId": "a" * 32,
            "spanId": "b" * 16,
            "name": "llm",
            "startTimeUnixNano": "1000000",
            "endTimeUnixNano": "2000000",
            "attributes": [
                {"key": key, "value": {"stringValue": value}}
                for key, value in {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "model-2026-01-01",
                    "llm.provider": "openai",
                    **attributes,
                }.items()
            ],
        }
        path = directory / "trace.json"
        path.write_text(
            json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}),
            encoding="utf-8",
        )
        return path

    def test_legacy_trace_has_no_declaration_keys(self) -> None:
        with TemporaryDirectory() as raw:
            normalized = import_openinference(self._trace(Path(raw), {}))
        request = normalized["requests"][0]
        for key in (
            "identity_grade",
            "identity_undisclosed_fields",
            "provider_fingerprint_sha256",
        ):
            self.assertNotIn(key, request)

    def test_declarations_are_imported_and_fingerprint_is_digested(self) -> None:
        with TemporaryDirectory() as raw:
            normalized = import_openinference(
                self._trace(
                    Path(raw),
                    {
                        "runeconomy.identity_grade": "provider_declared",
                        "runeconomy.identity_undisclosed": '["tokenizer"]',
                        "runeconomy.provider_system_fingerprint": "fp_secretish",
                    },
                )
            )
        request = normalized["requests"][0]
        self.assertEqual(request["identity_grade"], "provider_declared")
        self.assertEqual(request["identity_undisclosed_fields"], ["tokenizer"])
        self.assertTrue(request["provider_fingerprint_sha256"].startswith("sha256:"))
        self.assertNotIn("fp_secretish", json.dumps(normalized))
        self.assertEqual(
            normalized["evidence_quality"]["fields"]["backend_identity"]["grade"],
            "weakly_inferred",
        )

    def test_malformed_grade_is_a_trace_error(self) -> None:
        with TemporaryDirectory() as raw:
            path = self._trace(Path(raw), {"runeconomy.identity_grade": "trusted"})
            with self.assertRaises(TraceValidationError):
                import_openinference(path)


if __name__ == "__main__":
    unittest.main()
