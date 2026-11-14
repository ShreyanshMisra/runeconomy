from __future__ import annotations

import unittest

from runeconomy.experiments import (
    AcceptanceConfig,
    BackendConfig,
    ExecutionConfig,
    ExperimentCase,
    ExperimentManifest,
)
from runeconomy.models import CacheConfig, InferenceRequest, RequestMetrics


class ModelCompatibilityTests(unittest.TestCase):
    def test_inference_request_preserves_legacy_positional_argument_order(self) -> None:
        request = InferenceRequest(
            "request-1",
            1,
            "model",
            "vllm",
            (1, 2, 3),
            "model-revision",
            "backend-version",
            "tokenizer",
            "tokenizer-revision",
            "sha256:template",
            (),
            ("read",),
            ("read",),
            CacheConfig(),
            RequestMetrics(input_tokens=3),
        )

        self.assertEqual(request.model_revision, "model-revision")
        self.assertEqual(request.backend_version, "backend-version")
        self.assertEqual(request.prompt_token_count, None)
        self.assertEqual(request.token_evidence_quality, "exact")

    def test_experiment_models_preserve_legacy_positional_argument_order(self) -> None:
        backend = BackendConfig(
            "https://example.invalid",
            None,
            "model",
            "model-revision",
            "tokenizer-revision",
            "sha256:template",
            "0.1",
            "cpu",
            16,
            {"enable_prefix_caching": True},
        )
        execution = ExecutionConfig(
            1, 0, "alternating", 0, "warm", "sequential", False, True, False
        )
        acceptance = AcceptanceConfig(
            "total_latency_ms", "cache_reuse", 0.0, 1.0, 0.0, 0.0, True
        )
        case = ExperimentCase("case", {}, {}, None, None, None)
        manifest = ExperimentManifest(
            "0.1", "experiment", backend, {}, execution, acceptance, {}, (case,)
        )

        self.assertEqual(backend.api_dialect, "responses")
        self.assertEqual(execution.cache_isolation, "shared")
        self.assertEqual(acceptance.task_success_method, "paired_exact")
        self.assertIsNone(case.workflow_id)
        self.assertEqual(manifest.design, {})


if __name__ == "__main__":
    unittest.main()
