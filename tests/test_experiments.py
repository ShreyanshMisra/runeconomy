from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runeconomy.experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    _apply_request_transform,
    run_experiment,
)
from runeconomy.vllm import CapturedResponse


def _manifest() -> dict:
    return {
        "schema_version": "0.1",
        "experiment_id": "timestamp-placement",
        "backend": {
            "kind": "vllm",
            "base_url": "http://127.0.0.1:8000",
            "model": "example/model",
            "model_revision": "model-sha",
            "tokenizer_revision": "tokenizer-sha",
            "chat_template_digest": "sha256:template",
            "backend_version": "0.12.0",
            "hardware": "NVIDIA L4 24GB",
            "block_size": 16,
            "engine_config": {
                "enable_prefix_caching": True,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
            },
        },
        "sampling": {"temperature": 0.0, "max_output_tokens": 64, "seed": 7},
        "execution": {
            "repetitions": 2,
            "warmup_repetitions": 1,
            "order": "alternating",
            "seed": 7,
            "cache_treatment": "warm",
            "arrival_pattern": "sequential",
            "require_evaluation": True,
        },
        "cases": [
            {
                "case_id": "case-1",
                "evaluator": {"kind": "exact", "expected": "PASS"},
                "variants": {
                    "baseline": {"request": {"input": "baseline {{AID_TRIAL_ID}}"}},
                    "candidate": {"request": {"input": "candidate {{AID_TRIAL_ID}}"}},
                },
            }
        ],
    }


def flaky_rubric(output: str, context: dict) -> bool:
    if context["variant"] == "candidate" and context["repetition"] == 1:
        raise RuntimeError("rubric service unavailable")
    return output == "PASS"


class _FakeClient:
    def __init__(self) -> None:
        self.resets = 0
        self.requests: list[dict] = []

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        self.resets += 1

    def capture(self, request_body: dict) -> CapturedResponse:
        self.requests.append(dict(request_body))
        candidate = request_body["input"].startswith("candidate")
        sequence = len(self.requests)
        prompt_tokens = (
            tuple(range(48)) + (10_000 + sequence,)
            if candidate
            else (20_000 + sequence,) + tuple(range(1, 49))
        )
        return CapturedResponse(
            prompt_token_ids=prompt_tokens,
            input_tokens=100,
            output_tokens=10,
            cached_tokens=20 if candidate else 10,
            client_ttft_ms=80.0 if candidate else 100.0,
            total_latency_ms=180.0 if candidate else 200.0,
            server_ttft_ms=70.0 if candidate else 90.0,
            server_queue_ms=1.0,
            generation_time_ms=100.0,
            mean_itl_ms=10.0,
            tokens_per_second=100.0,
            response_id=None,
            status="completed",
            output_digest="candidate" if candidate else "baseline",
            output_text="PASS",
        )


class _TokenReductionClient(_FakeClient):
    def capture(self, request_body: dict) -> CapturedResponse:
        captured = super().capture(request_body)
        lean = "candidate" in request_body["input"]
        return CapturedResponse(
            **{
                **captured.__dict__,
                "input_tokens": 50 if lean else 100,
                "cached_tokens": 0,
                "server_ttft_ms": 25.0 if lean else 50.0,
                "client_ttft_ms": 35.0 if lean else 70.0,
                "total_latency_ms": 75.0 if lean else 125.0,
            }
        )


class _InterruptedClient(_FakeClient):
    def capture(self, request_body: dict) -> CapturedResponse:
        if len(self.requests) == 2:
            raise RuntimeError("simulated transport interruption")
        return super().capture(request_body)


class ExperimentTests(unittest.TestCase):
    def test_warm_experiment_alternates_and_builds_paired_effects(self) -> None:
        manifest = ExperimentManifest.from_dict(_manifest())
        client = _FakeClient()
        result = run_experiment(manifest, client)

        self.assertEqual(client.resets, 1)
        self.assertEqual(len(result["trials"]), 6)
        measured = [trial for trial in result["trials"] if not trial["warmup"]]
        self.assertEqual(
            [trial["variant"] for trial in measured],
            ["baseline", "candidate", "candidate", "baseline"],
        )
        self.assertTrue(
            all(request["model"] == "example/model" for request in client.requests)
        )
        self.assertNotIn("{{AID_TRIAL_ID}}", client.requests[0]["input"])
        self.assertEqual(
            client.requests[2]["input"].removeprefix("baseline "),
            client.requests[3]["input"].removeprefix("candidate "),
        )
        effect = result["summary"]["effects"]["client_ttft_ms"]
        self.assertEqual(effect["paired_count"], 2)
        self.assertEqual(effect["candidate_minus_baseline_mean"], -20.0)
        self.assertEqual(effect["bootstrap_95_ci"], [-20.0, -20.0])
        self.assertEqual(
            result["summary"]["variants"]["baseline"]["outcomes"]["completed_rate"],
            1.0,
        )
        self.assertEqual(
            result["summary"]["paired_output_digest"]["exact_match_rate"], 0.0
        )
        self.assertEqual(result["decision"]["status"], "verified")
        self.assertGreater(
            result["summary"]["prediction_vs_observation"][
                "predicted_candidate_minus_baseline_cached_tokens"
            ],
            0,
        )

    def test_cold_experiment_resets_before_every_trial(self) -> None:
        payload = _manifest()
        payload["execution"]["cache_treatment"] = "cold"
        manifest = ExperimentManifest.from_dict(payload)
        client = _FakeClient()
        result = run_experiment(manifest, client)
        self.assertEqual(client.resets, len(result["trials"]))

    def test_stateful_responses_requests_are_rejected(self) -> None:
        payload = _manifest()
        payload["cases"][0]["variants"]["baseline"]["request"][
            "previous_response_id"
        ] = "resp_123"
        with self.assertRaisesRegex(ExperimentValidationError, "stateful fields"):
            ExperimentManifest.from_dict(payload)

    def test_tool_order_transforms_preserve_catalog_and_are_deterministic(self) -> None:
        tools = [
            {"type": "function", "name": name, "parameters": {"type": "object"}}
            for name in ("zebra", "alpha", "middle", "bravo", "omega")
        ]
        request = {"input": "test", "tools": tools}

        first = _apply_request_transform(request, {"kind": "shuffle_tools"}, 17)
        second = _apply_request_transform(request, {"kind": "shuffle_tools"}, 17)
        canonical = _apply_request_transform(first, {"kind": "canonicalize_tools"}, 999)

        self.assertEqual(first, second)
        self.assertNotEqual(first["tools"], request["tools"])
        self.assertEqual(
            [tool["name"] for tool in canonical["tools"]],
            ["alpha", "bravo", "middle", "omega", "zebra"],
        )
        self.assertCountEqual(first["tools"], canonical["tools"])
        self.assertEqual(
            [tool["name"] for tool in request["tools"]],
            ["zebra", "alpha", "middle", "bravo", "omega"],
        )

    def test_tool_transform_requires_tools(self) -> None:
        payload = _manifest()
        payload["cases"][0]["variants"]["baseline"]["request_transform"] = {
            "kind": "shuffle_tools"
        }
        with self.assertRaisesRegex(
            ExperimentValidationError, "requires request.tools"
        ):
            ExperimentManifest.from_dict(payload)

    def test_unknown_placeholder_is_rejected_during_manifest_validation(self) -> None:
        payload = _manifest()
        payload["cases"][0]["variants"]["baseline"]["request"]["input"] = (
            "{{UNKNOWN_FIXTURE}}"
        )
        with self.assertRaisesRegex(ExperimentValidationError, "UNKNOWN_FIXTURE"):
            ExperimentManifest.from_dict(payload)

    def test_evaluator_infrastructure_error_is_recorded_as_unknown_quality(
        self,
    ) -> None:
        payload = _manifest()
        payload["cases"][0]["evaluator"] = {
            "kind": "command",
            "argv": ["not-a-real-runeconomy-command"],
        }
        result = run_experiment(ExperimentManifest.from_dict(payload), _FakeClient())
        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertEqual(len(result["evaluator_errors"]), len(result["trials"]))
        self.assertTrue(
            all(trial["task_success"] is None for trial in result["trials"])
        )

    def test_one_evaluator_error_invalidates_an_otherwise_passing_run(self) -> None:
        payload = _manifest()
        payload["cases"][0]["evaluator"] = {
            "kind": "python",
            "callable": "tests.test_experiments:flaky_rubric",
        }
        result = run_experiment(ExperimentManifest.from_dict(payload), _FakeClient())
        self.assertEqual(len(result["evaluator_errors"]), 1)
        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertEqual(result["decision"]["checks"][0]["name"], "evaluator_integrity")

    def test_checkpoint_preserves_trials_and_resume_replays_warm_prefix(self) -> None:
        manifest = ExperimentManifest.from_dict(_manifest())
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "experiment.trials.jsonl"
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_experiment(
                    manifest,
                    _InterruptedClient(),
                    checkpoint_path=checkpoint,
                )
            self.assertEqual(
                len(checkpoint.read_text(encoding="utf-8").splitlines()), 3
            )

            resumed_client = _FakeClient()
            result = run_experiment(
                manifest,
                resumed_client,
                checkpoint_path=checkpoint,
                resume=True,
            )
            self.assertEqual(len(result["trials"]), 6)
            self.assertEqual(len(resumed_client.requests), 6)
            self.assertEqual(
                len(checkpoint.read_text(encoding="utf-8").splitlines()), 7
            )

    def test_sequential_cases_are_aggregated_for_input_token_decision(self) -> None:
        payload = _manifest()
        payload["execution"].update(
            {
                "warmup_repetitions": 0,
                "cache_treatment": "cold",
                "cases_form_workflow": True,
            }
        )
        payload["acceptance"] = {
            "primary_metric": "server_ttft_ms",
            "optimization_target": "input_tokens",
            "min_input_token_reduction": 90,
            "require_latency_improvement": True,
        }
        second_case = {
            **payload["cases"][0],
            "case_id": "case-2",
            "variants": {
                "baseline": {"request": {"input": "baseline step two"}},
                "candidate": {"request": {"input": "candidate step two"}},
            },
        }
        payload["cases"].append(second_case)

        result = run_experiment(
            ExperimentManifest.from_dict(payload), _TokenReductionClient()
        )

        workflow = result["summary"]["workflow"]
        self.assertEqual(workflow["step_count"], 2)
        self.assertEqual(
            workflow["effects"]["input_tokens"]["candidate_minus_baseline_mean"],
            -100.0,
        )
        self.assertEqual(
            workflow["effects"]["server_ttft_ms"]["candidate_minus_baseline_mean"],
            -50.0,
        )
        self.assertEqual(result["decision"]["status"], "verified")


if __name__ == "__main__":
    unittest.main()
