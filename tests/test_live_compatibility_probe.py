from __future__ import annotations

import io
import unittest

from scripts.live_compatibility_probe import _sse_events, _stream_checks, _token_ids


class _StreamingClient:
    model = "example/model"

    def stream(self, path, payload):
        if path == "/v1/chat/completions":
            return 20.0, 10.0, [{"choices": [{"delta": {"content": "OK"}}]}]
        return 30.0, 12.0, [
            {"type": "response.output_text.delta", "delta": "OK"},
            {"type": "response.incomplete"},
        ]


class LiveCompatibilityProbeTests(unittest.TestCase):
    def test_token_ids_accept_current_renderer_shapes(self) -> None:
        self.assertEqual(_token_ids({"token_ids": [1, 2]}), (1, 2))
        self.assertEqual(
            _token_ids({"engine_input": {"prompt_token_ids": [3, 4]}}),
            (3, 4),
        )
        with self.assertRaises(RuntimeError):
            _token_ids({"token_ids": [True]})

    def test_sse_parser_ignores_done_and_keeps_terminal_event(self) -> None:
        events = list(
            _sse_events(
                iter(
                    io.StringIO(
                        'data: {"type":"response.output_text.delta"}\n\n'
                        'data: {"type":"response.incomplete"}\n\n'
                        "data: [DONE]\n\n"
                    ).readlines()
                )
            )
        )
        self.assertEqual(
            [event[1]["type"] for event in events],
            ["response.output_text.delta", "response.incomplete"],
        )

    def test_incomplete_is_a_valid_responses_stream_terminal_event(self) -> None:
        checks = _stream_checks(_StreamingClient())
        self.assertTrue(checks["chat_completions_stream"])
        self.assertTrue(checks["responses_stream"])


if __name__ == "__main__":
    unittest.main()
