from __future__ import annotations

import unittest

from runeconomy.attempt_status import AttemptStatusError
from runeconomy.backends import BackendProbeError
from runeconomy.errors import classify_exception
from runeconomy.evaluators import EvaluatorInfrastructureError
from runeconomy.experiments import ExperimentValidationError
from runeconomy.models import TraceValidationError
from runeconomy.tutorial import TutorialError
from runeconomy.vllm import VLLMError, VLLMHTTPError


class StableErrorCodeTests(unittest.TestCase):
    def test_public_workflow_failures_have_namespaced_codes(self) -> None:
        cases = (
            (
                TraceValidationError("cannot detect private evidence type"),
                "import",
                "runeconomy.evidence.ambiguous",
            ),
            (
                TraceValidationError("unsupported evidence type"),
                "import",
                "runeconomy.evidence.unsupported",
            ),
            (
                ExperimentValidationError("workspace is unresolved"),
                "experiment.validate",
                "runeconomy.workspace.unresolved",
            ),
            (
                ExperimentValidationError("checkpoint does not match manifest"),
                "experiment.run",
                "runeconomy.checkpoint.mismatch",
            ),
            (
                ExperimentValidationError("environment identity drift detected"),
                "experiment.run",
                "runeconomy.identity.drift",
            ),
            (
                ExperimentValidationError("cannot render experiment report"),
                "experiment.report",
                "runeconomy.report.generation",
            ),
            (
                BackendProbeError("private", category="authentication"),
                "doctor",
                "runeconomy.backend.authentication",
            ),
            (
                BackendProbeError("private", category="connection"),
                "doctor",
                "runeconomy.backend.connection",
            ),
            (
                BackendProbeError("private", category="timeout"),
                "doctor",
                "runeconomy.backend.timeout",
            ),
            (
                VLLMHTTPError("/private", 401, "secret"),
                "experiment.run",
                "runeconomy.backend.authentication",
            ),
            (
                VLLMError("private endpoint"),
                "experiment.run",
                "runeconomy.backend.connection",
            ),
            (
                EvaluatorInfrastructureError("private evaluator output"),
                "experiment.run",
                "runeconomy.evaluator.infrastructure",
            ),
            (
                AttemptStatusError("private status payload"),
                "experiment.run",
                "runeconomy.attempt.status_invalid",
            ),
            (
                TutorialError("private tutorial path"),
                "tutorial.init",
                "runeconomy.tutorial.invalid",
            ),
        )
        for error, stage, expected in cases:
            with self.subTest(error=type(error).__name__, stage=stage):
                descriptor = classify_exception(error, stage=stage)
                self.assertEqual(descriptor.code, expected)
                self.assertTrue(descriptor.code.startswith("runeconomy."))
                self.assertTrue(descriptor.summary)
                self.assertTrue(descriptor.next_action_code.startswith("runeconomy.action."))


if __name__ == "__main__":
    unittest.main()
