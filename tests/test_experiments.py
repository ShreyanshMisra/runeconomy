from __future__ import annotations

import json
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
        "acceptance": {"task_success_method": "point_estimate"},
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


class _NoResetClient(_FakeClient):
    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        raise AssertionError("salt-isolated execution must not call cache reset")


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


class _FailingTaskClient(_FakeClient):
    def capture(self, request_body: dict) -> CapturedResponse:
        captured = super().capture(request_body)
        return CapturedResponse(**{**captured.__dict__, "output_text": "FAIL"})


class _InterruptedClient(_FakeClient):
    def capture(self, request_body: dict) -> CapturedResponse:
        if len(self.requests) == 2:
            raise RuntimeError("simulated transport interruption")
        return super().capture(request_body)


class _IdentityClient(_FakeClient):
    def __init__(self, identities: list[dict]) -> None:
        super().__init__()
        self.identities = identities
        self.identity_reads = 0

    def environment_identity(self) -> dict:
        identity = self.identities[min(self.identity_reads, len(self.identities) - 1)]
        self.identity_reads += 1
        return identity


class ExperimentTests(unittest.TestCase):
    def test_schema_03_defaults_absolute_success_floor(self) -> None:
        payload = _manifest()
        payload["schema_version"] = "0.3"

        manifest = ExperimentManifest.from_dict(payload)

        self.assertEqual(manifest.acceptance.min_task_success_rate, 0.80)

    def test_paired_success_gate_is_not_degenerate_when_all_pairs_pass(self) -> None:
        payload = _manifest()
        payload["execution"].update(
            {"repetitions": 15, "warmup_repetitions": 0}
        )
        payload["acceptance"].update(
            {
                "task_success_method": "paired_exact",
                "max_task_success_regression": 0.0,
            }
        )

        result = run_experiment(ExperimentManifest.from_dict(payload), _FakeClient())

        self.assertEqual(result["result_schema_version"], "0.3")
        outcomes = result["summary"]["variants"]["candidate"]["outcomes"]
        paired = result["summary"]["effects"]["task_success"]
        self.assertEqual(outcomes["task_success_rate"], 1.0)
        self.assertLess(outcomes["task_success_95_ci"][0], 1.0)
        self.assertLess(paired["one_sided_lower_bound"], 0.0)
        self.assertEqual(paired["outcome_counts"]["both_success"], 15)
        self.assertEqual(result["decision"]["status"], "inconclusive")

    def test_paired_success_gate_can_verify_a_predeclared_margin(self) -> None:
        payload = _manifest()
        payload["execution"].update(
            {"repetitions": 40, "warmup_repetitions": 0}
        )
        payload["acceptance"].update(
            {
                "task_success_method": "paired_exact",
                "max_task_success_regression": 0.10,
                "min_task_success_rate": 0.80,
            }
        )

        result = run_experiment(ExperimentManifest.from_dict(payload), _FakeClient())

        lower = result["summary"]["effects"]["task_success"][
            "one_sided_lower_bound"
        ]
        self.assertGreater(lower, -0.10)
        self.assertEqual(result["decision"]["status"], "verified")

    def test_absolute_success_floor_invalidates_weak_baseline(self) -> None:
        payload = _manifest()
        payload["execution"].update(
            {"repetitions": 40, "warmup_repetitions": 0}
        )
        payload["acceptance"].update(
            {
                "task_success_method": "paired_exact",
                "max_task_success_regression": 0.10,
                "min_task_success_rate": 0.80,
            }
        )

        result = run_experiment(
            ExperimentManifest.from_dict(payload), _FailingTaskClient()
        )

        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertIn("baseline", result["decision"]["reason"])
        absolute = result["decision"]["checks"][0]
        self.assertEqual(absolute["name"], "minimum_task_success_rate")
        self.assertFalse(absolute["passed"])

    def test_required_environment_identity_invalidates_drift(self) -> None:
        payload = _manifest()
        payload["acceptance"].update(
            {
                "task_success_method": "point_estimate",
                "require_environment_identity": True,
            }
        )
        expected = {
            "model": "example/model",
            "model_revision": "model-sha",
            "tokenizer_revision": "tokenizer-sha",
            "chat_template_digest": "sha256:template",
            "backend_version": "0.12.0",
            "engine_config": {
                "enable_prefix_caching": True,
                "tensor_parallel_size": 1,
                "dtype": "bfloat16",
            },
        }
        changed = {**expected, "model_revision": "different-model-sha"}

        result = run_experiment(
            ExperimentManifest.from_dict(payload),
            _IdentityClient([expected, changed]),
        )

        self.assertEqual(result["decision"]["status"], "invalid")
        self.assertEqual(result["decision"]["checks"][0]["name"], "environment_identity")
        self.assertIn(
            "model_revision",
            result["provenance"]["environment_identity"]["drift"],
        )

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

    def test_predeclared_prediction_is_compared_with_paired_interval(self) -> None:
        payload = _manifest()
        payload["design"] = {
            "hypothesis": {
                "causal_mechanism": "candidate reduces prefill work",
                "prediction": {
                    "metric": "client_ttft_ms",
                    "direction": "decrease",
                },
            }
        }
        result = run_experiment(ExperimentManifest.from_dict(payload), _FakeClient())
        assessment = result["summary"]["predeclared_prediction"]
        self.assertEqual(assessment["status"], "supported")
        self.assertEqual(assessment["confidence_interval"], [-20.0, -20.0])

    def test_salt_isolation_uses_unique_salts_without_claiming_reset(self) -> None:
        payload = _manifest()
        payload["execution"].update(
            {"cache_treatment": "cold", "cache_isolation": "salt_isolated"}
        )
        manifest = ExperimentManifest.from_dict(payload)
        client = _NoResetClient()
        result = run_experiment(manifest, client)

        salts = [request["cache_salt"] for request in client.requests]
        self.assertEqual(len(salts), len(set(salts)))
        self.assertTrue(all(len(salt) >= 43 for salt in salts))
        self.assertNotIn("cache_salt", json.dumps(result))
        self.assertEqual(result["provenance"]["cache_isolation"], "salt_isolated")

    def test_cache_isolation_semantics_are_validated(self) -> None:
        payload = _manifest()
        payload["execution"]["cache_isolation"] = "salt_isolated"
        with self.assertRaisesRegex(ExperimentValidationError, "warm cache treatment"):
            ExperimentManifest.from_dict(payload)

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
            "task_success_method": "point_estimate",
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

    def test_distinct_workflow_units_are_paired_independently(self) -> None:
        payload = _manifest()
        payload["execution"].update(
            {
                "warmup_repetitions": 0,
                "cache_treatment": "cold",
                "cases_form_workflow": True,
            }
        )
        template = payload["cases"][0]
        payload["cases"] = []
        for workflow_id in ("task-a", "task-b"):
            for step in (1, 2):
                payload["cases"].append(
                    {
                        "case_id": f"{workflow_id}-step-{step}",
                        "workflow_id": workflow_id,
                        "evaluator": template["evaluator"],
                        "variants": {
                            "baseline": {
                                "request": {
                                    "input": f"baseline {workflow_id} step {step}"
                                }
                            },
                            "candidate": {
                                "request": {
                                    "input": f"candidate {workflow_id} step {step}"
                                }
                            },
                        },
                    }
                )

        manifest = ExperimentManifest.from_dict(payload)
        result = run_experiment(manifest, _FakeClient())

        workflow = result["summary"]["workflow"]
        self.assertEqual(workflow["workflow_count"], 2)
        self.assertEqual(workflow["step_count"], 2)
        self.assertEqual(workflow["step_counts"], {"task-a": 2, "task-b": 2})
        self.assertEqual(
            workflow["effects"]["total_latency_ms"]["paired_count"], 4
        )
        self.assertEqual(
            workflow["effects"]["task_success"]["outcome_counts"]["both_success"],
            4,
        )

    def test_partial_workflow_grouping_is_rejected(self) -> None:
        payload = _manifest()
        payload["execution"]["cases_form_workflow"] = True
        payload["cases"][0]["workflow_id"] = "task-a"
        second = {
            **payload["cases"][0],
            "case_id": "case-2",
        }
        second.pop("workflow_id")
        payload["cases"].append(second)

        with self.assertRaisesRegex(ExperimentValidationError, "every case"):
            ExperimentManifest.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
