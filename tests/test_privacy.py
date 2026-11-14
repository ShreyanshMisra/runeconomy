from __future__ import annotations

import unittest

from runeconomy.privacy import assert_safe_for_export, sensitive_export_paths


class PrivacyTests(unittest.TestCase):
    def test_finds_nested_secret_keys_bearer_values_and_url_queries(self) -> None:
        value = {
            "metadata": {"api_key": "x"},
            "header": "Authorization: Bearer abc.def",
            "endpoint": "https://user:pass@example.test/private?token=x",
        }
        paths = sensitive_export_paths(value)
        self.assertIn("$.metadata.api_key", paths)
        self.assertIn("$.header", paths)
        self.assertIn("$.endpoint", paths)
        with self.assertRaisesRegex(ValueError, "sensitive URL parameters"):
            assert_safe_for_export(value)

    def test_allows_sanitized_measurement_payload(self) -> None:
        assert_safe_for_export(
            {
                "model": "example/model",
                "latency_ms": 12.5,
                "endpoint": "local",
                "api_key_env": "VLLM_API_KEY",
            }
        )
