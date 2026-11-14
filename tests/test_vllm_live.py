"""Opt-in integration test against a real, already-running vLLM server."""

from __future__ import annotations

import os
import unittest

from runeconomy.backends import discover_vllm_capabilities
from runeconomy.vllm import VLLMClient


@unittest.skipUnless(os.environ.get("VLLM_BASE_URL"), "VLLM_BASE_URL is not set")
class LiveVLLMTests(unittest.TestCase):
    def test_chat_capture_against_pinned_backend(self) -> None:
        base_url = os.environ["VLLM_BASE_URL"]
        api_key = os.environ.get("VLLM_API_KEY")
        capabilities = discover_vllm_capabilities(base_url, api_key=api_key)
        expected_version = os.environ.get("VLLM_EXPECTED_VERSION")
        if expected_version:
            self.assertEqual(capabilities.backend_version, expected_version)
        self.assertEqual(capabilities.chat_completions.status, "supported")
        self.assertTrue(capabilities.model_ids)

        client = VLLMClient(
            base_url,
            api_key=api_key,
            api_dialect="chat_completions",
            timeout_seconds=120,
        )
        captured = client.capture(
            {
                "model": capabilities.model_ids[0],
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "temperature": 0.0,
                "max_tokens": 2,
            }
        )
        self.assertGreater(len(captured.prompt_token_ids), 0)
        self.assertGreater(captured.input_tokens or 0, 0)
        self.assertIn(captured.status, {"completed", "incomplete"})


if __name__ == "__main__":
    unittest.main()
