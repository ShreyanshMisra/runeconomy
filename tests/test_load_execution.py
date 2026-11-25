from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runeconomy import metric_gates
from runeconomy.comparison import run_comparison
from runeconomy.execution import (
    LoadRunner,
    LoadSpec,
    LoadSpecError,
    ScheduledTrial,
    build_arrival_schedule,
)
from runeconomy.experiments import (
    ExperimentManifest,
    ExperimentValidationError,
    run_experiment,
    static_arrival_schedule,
)
from runeconomy.planning import plan_experiment
from runeconomy.vllm import CapturedResponse
from runeconomy.workspaces import (
    agent_run_arguments,
    create_agent_workspace,
    resolve_agent_manifest,
)


def _spec(pattern: str = "closed_loop", **value) -> LoadSpec:
    payload = {"serving_topology": "shared", **value}
    if pattern == "closed_loop":
        if "concurrency_levels" not in payload:
            payload.setdefault("concurrency", 4)
    elif "rate_levels" not in payload:
        payload.setdefault("rate_per_second", 50.0)
    return LoadSpec.from_value(pattern, payload)


class LoadSpecTests(unittest.TestCase):
    def test_round_trip_and_canonical_form(self) -> None:
        for pattern, value in (
            ("closed_loop", {"concurrency_levels": [16, 4], "gate_concurrency": 16}),
            ("poisson", {"rate_per_second": 2.5, "max_in_flight": 8}),
            ("constant_rps", {"rate_levels": [1, 2], "gate_rate": 2, "tail": "sustain"}),
        ):
            spec = _spec(pattern, **value)
            self.assertEqual(LoadSpec.from_value(pattern, spec.to_dict()), spec)
        sweep = _spec(concurrency_levels=[16, 4], gate_concurrency=16)
        self.assertEqual(sweep.levels, (4, 16))
        self.assertEqual(sweep.gate_stage, 1)

    def test_invalid_declarations_are_rejected(self) -> None:
        cases = [
            ({"concurrency_levels": [4, 16]}, "gate_concurrency"),
            ({"concurrency_levels": [4, 16], "gate_concurrency": 8}, "one of"),
            ({"concurrency": 4, "rate_per_second": 2}, "unsupported"),
            ({"concurrency": 0}, ">= 1"),
            ({"concurrency": 4, "pairing": "interleaved"}, "isolated"),
            ({"concurrency": 4, "block_units": 0}, "block_units"),
            ({"concurrency": 4, "schedule_sha256": "abc"}, "sha256"),
        ]
        for value, message in cases:
            with self.subTest(value=value), self.assertRaisesRegex(LoadSpecError, message):
                LoadSpec.from_value("closed_loop", {"serving_topology": "shared", **value})
        with self.assertRaisesRegex(LoadSpecError, "serving_topology"):
            LoadSpec.from_value("closed_loop", {"concurrency": 4})
        with self.assertRaisesRegex(LoadSpecError, "'shared'"):
            LoadSpec.from_value(
                "closed_loop",
                {"concurrency": 4, "serving_topology": "isolated"},
                forced_topology="shared",
            )


class ScheduleTests(unittest.TestCase):
    def test_variant_blocked_pairs_share_order_and_offsets(self) -> None:
        load = _spec("poisson", rate_per_second=20.0, block_units=5)
        schedule = build_arrival_schedule(
            load, [f"u{i}" for i in range(10)], repetitions=2, seed=9, order="alternating"
        )
        self.assertEqual(len(schedule.blocks), 8)
        pairs: dict[int, list] = {}
        for block in schedule.blocks:
            self.assertEqual(len(block.variants), 1)
            pairs.setdefault(block.block_pair, []).append(block)
        arm_orders = []
        for pair, (first, second) in sorted(pairs.items()):
            self.assertNotEqual(first.variants, second.variants)
            def key(trial):
                return (trial.unit, trial.repetition, trial.offset_ms)

            self.assertEqual([key(t) for t in first.trials], [key(t) for t in second.trials])
            arm_orders.append(first.variants[0])
        self.assertEqual(arm_orders, ["baseline", "candidate", "baseline", "candidate"])
        counts = Counter(
            (trial.unit, trial.repetition, trial.variant) for trial in schedule.trials
        )
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(len(counts), 10 * 2 * 2)

    def test_schedule_is_deterministic_and_seed_sensitive(self) -> None:
        load = _spec()
        units = [f"u{i}" for i in range(12)]
        first = build_arrival_schedule(load, units, repetitions=1, seed=1, order="randomized")
        again = build_arrival_schedule(load, units, repetitions=1, seed=1, order="randomized")
        other = build_arrival_schedule(load, units, repetitions=1, seed=2, order="randomized")
        self.assertEqual(first.digest(), again.digest())
        self.assertNotEqual(first.digest(), other.digest())

    def test_interleaved_runs_both_pools_in_one_block(self) -> None:
        load = LoadSpec.from_value(
            "closed_loop",
            {"concurrency": 2, "pairing": "interleaved", "serving_topology": "isolated"},
        )
        schedule = build_arrival_schedule(
            load, ["a", "b", "c"], repetitions=2, seed=0, order="alternating"
        )
        (block,) = schedule.blocks
        self.assertEqual(block.variants, ("baseline", "candidate"))
        def order(variant):
            return [(t.unit, t.repetition) for t in block.pool(variant)]

        self.assertEqual(order("baseline"), order("candidate"))

    def test_load_that_cannot_be_reached_is_rejected(self) -> None:
        with self.assertRaisesRegex(LoadSpecError, "never be reached"):
            build_arrival_schedule(_spec(concurrency=8), ["a", "b"], repetitions=2, seed=0, order="alternating")
        with self.assertRaisesRegex(LoadSpecError, "block of"):
            build_arrival_schedule(
                _spec(concurrency=4, block_units=6),
                [f"u{i}" for i in range(8)],
                repetitions=1,
                seed=0,
                order="alternating",
            )


class _RecordingExecutor:
    def __init__(self, duration: float = 0.01) -> None:
        self.duration = duration
        self.lock = threading.Lock()
        self.in_flight: Counter = Counter()
        self.peak: Counter = Counter()
        self.mixed = False
        self.roles: Counter = Counter()

    def execute(self, trial: ScheduledTrial, *, role: str, filler_index):
        with self.lock:
            self.in_flight[trial.variant] += 1
            self.peak[trial.variant] = max(self.peak[trial.variant], self.in_flight[trial.variant])
            if self.in_flight["baseline"] and self.in_flight["candidate"]:
                self.mixed = True
            self.roles[role] += 1
        time.sleep(self.duration * (1 + trial.unit % 3))
        with self.lock:
            self.in_flight[trial.variant] -= 1
        return {"variant": trial.variant, "unit": trial.unit}


class RunnerTests(unittest.TestCase):
    def test_closed_loop_reaches_declared_concurrency_without_mixing_variants(self) -> None:
        load = _spec(concurrency=6, tail="sustain")
        schedule = build_arrival_schedule(
            load, [f"u{i}" for i in range(24)], repetitions=1, seed=3, order="alternating"
        )
        executor = _RecordingExecutor()
        result = LoadRunner(schedule, executor).run()
        self.assertEqual(executor.peak["baseline"], 6)
        self.assertEqual(executor.peak["candidate"], 6)
        self.assertFalse(executor.mixed)
        self.assertEqual(len(result.records), 48)
        self.assertGreater(len(result.fillers), 0)
        self.assertEqual(
            [record["load"]["schedule_index"] for record in result.records], list(range(48))
        )
        mean_in_flight = [record["load"]["mean_in_flight"] for record in result.records]
        # Sustain keeps every population trial at (nearly) full concurrency.
        self.assertGreater(min(mean_in_flight), 4.5)

    def test_drain_tail_runs_no_fillers(self) -> None:
        schedule = build_arrival_schedule(
            _spec(concurrency=4), [f"u{i}" for i in range(8)], repetitions=1, seed=3, order="alternating"
        )
        result = LoadRunner(schedule, _RecordingExecutor(0.002)).run()
        self.assertEqual(result.fillers, [])

    def test_interleaved_pools_run_concurrently(self) -> None:
        load = LoadSpec.from_value(
            "closed_loop",
            {"concurrency": 3, "pairing": "interleaved", "serving_topology": "isolated"},
        )
        schedule = build_arrival_schedule(
            load, [f"u{i}" for i in range(9)], repetitions=1, seed=3, order="alternating"
        )
        executor = _RecordingExecutor()
        LoadRunner(schedule, executor).run()
        self.assertTrue(executor.mixed)
        self.assertEqual(executor.peak["baseline"], 3)

    def test_open_loop_records_dispatch_lag(self) -> None:
        load = _spec("constant_rps", rate_per_second=200.0, max_in_flight=2)
        schedule = build_arrival_schedule(
            load, [f"u{i}" for i in range(10)], repetitions=1, seed=0, order="alternating"
        )
        result = LoadRunner(schedule, _RecordingExecutor(0.02)).run()
        lags = [record["load"]["dispatch_lag_ms"] for record in result.records]
        self.assertTrue(all(lag is not None and lag >= 0 for lag in lags))
        # Two slots cannot absorb 200 arrivals/s of 20-60 ms work.
        self.assertGreater(max(lags), 20.0)

    def test_open_loop_sustain_fillers_stop_with_the_population(self) -> None:
        load = _spec("poisson", rate_per_second=100.0, tail="sustain", max_in_flight=4)
        schedule = build_arrival_schedule(
            load, [f"u{i}" for i in range(12)], repetitions=1, seed=5, order="alternating"
        )
        executor = _RecordingExecutor(0.01)
        result = LoadRunner(schedule, executor).run()
        self.assertEqual(len(result.records), 24)
        for filler in result.fillers:
            block_end = max(
                record["load"]["ended_ms"]
                for record in result.records
                if record["load"]["block"] == filler["load"]["block"]
            )
            self.assertLessEqual(filler["load"]["started_ms"], block_end)

    def test_executor_errors_propagate_after_the_barrier(self) -> None:
        class Failing(_RecordingExecutor):
            def execute(self, trial, *, role, filler_index):
                if trial.index == 3:
                    raise RuntimeError("transport failed")
                return super().execute(trial, role=role, filler_index=filler_index)

        schedule = build_arrival_schedule(
            _spec(concurrency=2), [f"u{i}" for i in range(6)], repetitions=1, seed=0, order="alternating"
        )
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
            LoadRunner(schedule, Failing(0.001)).run()

    def test_observers_attach_block_and_trial_telemetry(self) -> None:
        class Observer:
            def __init__(self) -> None:
                self.started = 0

            def on_block_start(self, block):
                self.started += 1

            def on_block_end(self, block):
                return {"kv_cache_usage_perc_max": 0.5}

            def on_trial_end(self, trial, record):
                return {"num_requests_waiting": 0}

        observer = Observer()
        schedule = build_arrival_schedule(
            _spec(concurrency=2), ["a", "b"], repetitions=1, seed=0, order="alternating"
        )
        result = LoadRunner(schedule, _RecordingExecutor(0.001), observers=[observer]).run()
        self.assertEqual(observer.started, 2)
        self.assertEqual(result.blocks[0]["telemetry"], {"kv_cache_usage_perc_max": 0.5})
        self.assertEqual(result.records[0]["server_telemetry"], {"num_requests_waiting": 0})


def _static_manifest(*, cases: int = 120, concurrency: int = 16) -> dict:
    return {
        "schema_version": "0.4",
        "experiment_id": "fp8-under-load",
        "backend": {
            "kind": "vllm",
            "base_url": "http://127.0.0.1:8000",
            "model": "example/model",
            "model_revision": "model-sha",
            "tokenizer_revision": "tokenizer-sha",
            "chat_template_digest": "sha256:template",
            "backend_version": "0.12.0",
            "hardware": "NVIDIA L4 24GB",
            "engine_config": {"enable_prefix_caching": True},
        },
        "sampling": {"temperature": 0.0, "max_output_tokens": 32},
        "execution": {
            "repetitions": 1,
            "order": "alternating",
            "seed": 11,
            "cache_treatment": "warm",
            "arrival_pattern": "closed_loop",
            "load": {"concurrency": concurrency, "tail": "sustain"},
            "require_evaluation": True,
        },
        "acceptance": {
            "optimization_target": "input_tokens",
            "task_success_method": "paired_score_exact",
            "max_task_success_regression": 0.05,
            "min_task_success_rate": 0.8,
            "metric_gates": [
                {
                    "metric": "workflow_latency_ms",
                    "statistic": "p95",
                    "semantics": "noninferiority",
                    "margin": 10,
                },
                {
                    "metric": "usd_per_1k_successful_workflows",
                    "semantics": "noninferiority",
                    "relative_margin": 0.05,
                },
            ],
        },
        "cases": [
            {
                "case_id": f"case-{index:03d}",
                "evaluator": {"kind": "exact", "expected": "PASS"},
                "variants": {
                    "baseline": {"request": {"input": f"baseline {index} {{{{RUNECONOMY_TRIAL_ID}}}}"}},
                    "candidate": {"request": {"input": f"candidate {index} {{{{RUNECONOMY_TRIAL_ID}}}}"}},
                },
            }
            for index in range(cases)
        ],
    }


class _LoadClient:
    """Thread-safe fake vLLM whose latency grows with in-flight requests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.mixed = False
        self.variants: Counter = Counter()
        self.resets = 0

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        self.resets += 1

    def capture(self, request_body: dict) -> CapturedResponse:
        candidate = request_body["input"].startswith("candidate")
        variant = "candidate" if candidate else "baseline"
        with self.lock:
            self.in_flight += 1
            self.variants[variant] += 1
            self.peak = max(self.peak, self.in_flight)
            if self.variants["baseline"] and self.variants["candidate"]:
                self.mixed = True
            load = self.in_flight
        time.sleep(0.005)
        with self.lock:
            self.in_flight -= 1
            self.variants[variant] -= 1
        index = int(request_body["input"].split()[1])
        latency = (60.0 if candidate else 100.0) + load + index % 9
        return CapturedResponse(
            prompt_token_ids=(1, 2, 3, index),
            input_tokens=40 if candidate else 50,
            output_tokens=9,
            cached_tokens=0,
            client_ttft_ms=latency / 4,
            total_latency_ms=latency,
            server_ttft_ms=latency / 5,
            server_queue_ms=float(load),
            generation_time_ms=latency / 2,
            mean_itl_ms=4.0,
            tokens_per_second=100.0,
            response_id=None,
            status="completed",
            output_digest="digest",
            output_text="PASS",
        )


def _economics(trial: dict) -> dict:
    # Stand-in for the cost model: per-token pricing on recorded usage.
    tokens = trial["input_tokens"] + trial["output_tokens"]
    return {"cost_usd": tokens * 1e-6, "gpu_time_ms": trial["total_latency_ms"]}


class StaticLoadExperimentTests(unittest.TestCase):
    def test_success_gate_shape_at_concurrency_16(self) -> None:
        manifest = ExperimentManifest.from_dict(_static_manifest())
        client = _LoadClient()
        with patch.object(metric_gates, "BOOTSTRAP_REPLICATES", 300):
            result = run_experiment(manifest, client, attempt_economics=_economics)
        json.dumps(result, allow_nan=False)
        self.assertEqual(client.peak, 16)
        self.assertFalse(client.mixed, "variant-blocked design must never mix variants")
        self.assertEqual(client.resets, 1)
        decision = result["decision"]
        checks = {check["name"]: check for check in decision["checks"]}
        p95 = checks["p95_workflow_latency_ms_noninferiority"]
        cost = checks["usd_per_1k_successful_workflows_noninferiority"]
        self.assertTrue(p95["passed"], p95)
        self.assertTrue(cost["passed"], cost)
        self.assertEqual(p95["observed"]["method"], "quantile_order_statistic")
        self.assertEqual(decision["status"], "verified", decision["reason"])
        self.assertEqual(decision["schema_version"], "0.2")
        self.assertEqual(result["result_schema_version"], "0.4")
        schedule = result["summary"]["load"]["arrival_schedule"]
        self.assertEqual(schedule["gate_level"], 16)
        self.assertEqual(schedule["sha256"], result["provenance"]["arrival_schedule_sha256"])
        self.assertEqual(schedule["sha256"], static_arrival_schedule(manifest).digest())
        trials = [trial for trial in result["trials"] if not trial["warmup"]]
        self.assertEqual(len(trials), 240)
        self.assertTrue(all("economics" in trial and "load" in trial for trial in trials))

    def test_pinned_schedule_must_match(self) -> None:
        raw = _static_manifest(cases=20, concurrency=4)
        manifest = ExperimentManifest.from_dict(raw)
        raw["execution"]["load"]["schedule_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ExperimentValidationError, "schedule_sha256"):
            run_experiment(ExperimentManifest.from_dict(raw), _LoadClient())
        raw["execution"]["load"]["schedule_sha256"] = static_arrival_schedule(manifest).digest()
        pinned = ExperimentManifest.from_dict(raw)
        self.assertNotEqual(pinned.fingerprint(), manifest.fingerprint())
        self.assertEqual(static_arrival_schedule(pinned).digest(), static_arrival_schedule(manifest).digest())

    def test_concurrent_contracts_are_prospective_and_cache_safe(self) -> None:
        raw = _static_manifest(cases=20, concurrency=4)
        old = deepcopy(raw)
        old["schema_version"] = "0.3"
        old["acceptance"].pop("metric_gates")
        with self.assertRaisesRegex(ExperimentValidationError, "requires schema_version '0.4'"):
            ExperimentManifest.from_dict(old)
        cold = deepcopy(raw)
        cold["execution"].update({"cache_treatment": "cold", "cache_isolation": "physical_reset"})
        with self.assertRaisesRegex(ExperimentValidationError, "physical_reset"):
            ExperimentManifest.from_dict(cold)
        interleaved = deepcopy(raw)
        interleaved["execution"]["load"]["pairing"] = "interleaved"
        with self.assertRaisesRegex(ExperimentValidationError, "isolated"):
            ExperimentManifest.from_dict(interleaved)
        with self.assertRaisesRegex(ExperimentValidationError, "resume"):
            run_experiment(ExperimentManifest.from_dict(raw), _LoadClient(), resume=True)

    def test_resolved_manifest_round_trips_and_legacy_shape_is_unchanged(self) -> None:
        manifest = ExperimentManifest.from_dict(_static_manifest(cases=20, concurrency=4))
        resolved = manifest.to_resolved_dict()
        self.assertEqual(ExperimentManifest.from_dict(resolved).fingerprint(), manifest.fingerprint())
        legacy_path = Path(__file__).parents[1] / "examples" / "experiments" / "timestamp-placement.json"
        legacy = ExperimentManifest.from_path(legacy_path)
        resolved_legacy = legacy.to_resolved_dict()
        self.assertNotIn("load", resolved_legacy["execution"])
        self.assertNotIn("metric_gates", resolved_legacy["acceptance"])

    def test_sweep_gates_only_the_preregistered_level(self) -> None:
        raw = _static_manifest(cases=100, concurrency=4)
        raw["execution"]["load"] = {"concurrency_levels": [2, 8], "gate_concurrency": 8}
        raw["acceptance"]["metric_gates"] = raw["acceptance"]["metric_gates"][:1]
        result = run_experiment(ExperimentManifest.from_dict(raw), _LoadClient())
        load = result["summary"]["load"]
        self.assertEqual(load["concurrency"], 8)
        self.assertEqual([stage["concurrency"] for stage in load["stages"]], [2])
        self.assertTrue(load["stages"][0]["descriptive_only"])
        measured = [trial for trial in result["trials"] if not trial["warmup"]]
        self.assertEqual(len(measured), 400)
        self.assertEqual(
            result["summary"]["variants"]["baseline"]["total_latency_ms"]["count"], 100
        )


AGENT = r'''
import json, os, time
from pathlib import Path
from runeconomy.attempt_status import write_attempt_status

task = json.loads(Path(os.environ["RUNECONOMY_TASK_PATH"]).read_text())
variant = os.environ["RUNECONOMY_VARIANT"]
lock_dir = Path(os.environ["LOAD_PROBE_DIR"])
marker = lock_dir / (os.environ["RUNECONOMY_ATTEMPT_DIR"].replace("/", "_"))
marker.write_text(variant)
time.sleep(0.05)
Path(os.environ["RUNECONOMY_OUTPUT_PATH"]).write_text(task["expected"])
def attr(key, value):
    return {"key": key, "value": {"stringValue": value}}
model = "fixture/candidate" if variant == "candidate" else "fixture/baseline"
span = {
    "traceId": "t-" + os.environ["RUNECONOMY_TASK_ID"] + variant,
    "spanId": "s1",
    "startTimeUnixNano": "1000000000",
    "endTimeUnixNano": "1040000000",
    "attributes": [
        attr("openinference.span.kind", "LLM"),
        attr("llm.model_name", model),
        attr("llm.model_revision", "rev"),
        attr("llm.provider", "vllm"),
        attr("llm.backend_version", "0.1"),
        attr("runeconomy.tokenizer", "tok"),
        attr("runeconomy.tokenizer_revision", "tok-rev"),
        attr("runeconomy.chat_template_digest", "sha256:template"),
        attr("runeconomy.task_id", os.environ["RUNECONOMY_TASK_ID"]),
        attr("runeconomy.depends_on", "[]"),
        {"key": "llm.token_count.prompt", "value": {"intValue": 10}},
        {"key": "llm.token_count.completion", "value": {"intValue": 5}},
        {"key": "runeconomy.client_ttft_ms", "value": {"doubleValue": 8.0}},
        {"key": "runeconomy.cost_usd", "value": {"doubleValue": 0.002 if variant == "candidate" else 0.003}},
    ],
}
Path(os.environ["RUNECONOMY_TRACE_PATH"]).write_text(
    json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]})
)
marker.unlink()
write_attempt_status(os.environ["RUNECONOMY_STATUS_PATH"], "completed")
'''

EVALUATOR = r'''
import json, sys
request = json.load(sys.stdin)
json.dump({"success": request["output_text"] == request["task"]["expected"]}, sys.stdout)
'''


class SequentialMetricGateTests(unittest.TestCase):
    def test_static_sequential_manifest_can_gate_tail_latency(self) -> None:
        raw = _static_manifest(cases=100, concurrency=4)
        raw["execution"]["arrival_pattern"] = "sequential"
        raw["execution"].pop("load")
        raw["acceptance"]["metric_gates"] = raw["acceptance"]["metric_gates"][:1]
        result = run_experiment(ExperimentManifest.from_dict(raw), _LoadClient())
        names = [check["name"] for check in result["decision"]["checks"]]
        self.assertIn("p95_workflow_latency_ms_noninferiority", names)
        self.assertNotIn("observed_concurrency_balance", names)
        self.assertNotIn("load", result["summary"])
        self.assertEqual(result["result_schema_version"], "0.4")
        self.assertEqual(result["decision"]["status"], "verified", result["decision"]["reason"])


class AgentLoadComparisonTests(unittest.TestCase):
    def test_isolated_interleaved_closed_loop_with_cost_gate(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
            (root / "evaluator.py").write_text(EVALUATOR, encoding="utf-8")
            probe = root / "probe"
            probe.mkdir()
            dataset = root / "tasks.json"
            dataset.write_text(
                json.dumps({"tasks": [{"task_id": f"t{i}", "expected": f"ok-{i}"} for i in range(20)]}),
                encoding="utf-8",
            )
            command = f"{sys.executable} agent.py"
            workspace = root / "workspace"
            create_agent_workspace(
                output=workspace,
                baseline_command=command,
                candidate_command=command,
                dataset_path=dataset,
                evaluator_path=root / "evaluator.py",
                optimization_target="engine_change_latency",
                latency_semantics="noninferiority",
                max_latency_regression_ms=1000.0,
                task_success_method="paired_score_exact",
                max_task_success_regression=0.15,
                arrival_pattern="closed_loop",
                load={"concurrency": 4, "pairing": "interleaved", "serving_topology": "isolated"},
                metric_gates=[
                    {
                        "metric": "usd_per_1k_successful_workflows",
                        "semantics": "superiority",
                    },
                    {
                        "metric": "client_ttft_ms",
                        "statistic": "p50",
                        "semantics": "noninferiority",
                        "margin": 1.0,
                    },
                ],
            )
            manifest, _ = resolve_agent_manifest(workspace)
            self.assertEqual(manifest["schema_version"], "0.4")
            arguments = agent_run_arguments(manifest, workspace)
            with patch.dict("os.environ", {"LOAD_PROBE_DIR": str(probe)}), patch.object(
                metric_gates, "BOOTSTRAP_REPLICATES", 300
            ):
                result = run_comparison(
                    **arguments,
                    output_path=root / "result.json",
                    artifacts_dir=root / "artifacts",
                    workdir=root,
                )
            artifacts = root / "artifacts"
            self.assertTrue((artifacts / "arrival-schedule.json").is_file())
            stored = json.loads((artifacts / "arrival-schedule.json").read_text())
        self.assertEqual(
            result["design"]["execution_load"]["arrival_schedule_sha256"],
            result["provenance"]["arrival_schedule_sha256"],
        )
        self.assertEqual(stored["gate_stage"], 0)
        self.assertEqual(len(result["attempts"]), 40)
        concurrency = [attempt["load"]["in_flight_at_start"] for attempt in result["attempts"]]
        self.assertEqual(max(concurrency), 4)
        checks = {check["name"]: check for check in result["decision"]["checks"]}
        cost = checks["usd_per_1k_successful_workflows_superiority"]
        self.assertAlmostEqual(cost["observed"]["baseline"], 3.0)
        self.assertAlmostEqual(cost["observed"]["candidate"], 2.0)
        self.assertTrue(cost["passed"])
        self.assertTrue(checks["p50_client_ttft_ms_noninferiority"]["passed"])
        self.assertEqual(result["decision"]["status"], "verified", result["decision"]["reason"])
        json.dumps(result, allow_nan=False)

    def test_sequential_agent_run_applies_metric_gates(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            (root / "agent.py").write_text(textwrap.dedent(AGENT), encoding="utf-8")
            (root / "evaluator.py").write_text(EVALUATOR, encoding="utf-8")
            probe = root / "probe"
            probe.mkdir()
            dataset = root / "tasks.json"
            dataset.write_text(
                json.dumps([{"task_id": f"t{i}", "expected": "ok"} for i in range(3)]),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"LOAD_PROBE_DIR": str(probe)}):
                result = run_comparison(
                    baseline_command=f"{sys.executable} agent.py",
                    candidate_command=f"{sys.executable} agent.py",
                    dataset_path=dataset,
                    evaluator_path=root / "evaluator.py",
                    output_path=root / "result.json",
                    artifacts_dir=root / "artifacts",
                    workdir=root,
                    metric_gates=[
                        {"metric": "usd_per_1k_successful_workflows", "semantics": "superiority"}
                    ],
                )
        check = result["decision"]["checks"][-1]
        self.assertEqual(check["name"], "usd_per_1k_successful_workflows_superiority")
        # Three tasks are below the 20-cluster bootstrap minimum.
        self.assertEqual(check["observed"]["status"], "insufficient_sample")
        self.assertEqual(
            result["design"]["acceptance"]["metric_gates"][0]["metric"],
            "usd_per_1k_successful_workflows",
        )
        self.assertNotIn("execution_load", result["design"])
        self.assertNotEqual(result["decision"]["status"], "verified")

    def test_agent_load_requires_declared_topology_and_new_schema(self) -> None:
        with TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            dataset = root / "tasks.json"
            dataset.write_text(json.dumps([{"task_id": "a"}]), encoding="utf-8")
            evaluator = root / "evaluator.py"
            evaluator.write_text(EVALUATOR, encoding="utf-8")
            with self.assertRaisesRegex(ExperimentValidationError, "serving_topology"):
                create_agent_workspace(
                    output=root / "w1",
                    baseline_command="true",
                    candidate_command="true",
                    dataset_path=dataset,
                    evaluator_path=evaluator,
                    arrival_pattern="closed_loop",
                    load={"concurrency": 2},
                )
            create_agent_workspace(
                output=root / "w2",
                baseline_command="true",
                candidate_command="true",
                dataset_path=dataset,
                evaluator_path=evaluator,
            )
            manifest_path = root / "w2" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["execution"]["arrival_pattern"] = "poisson"
            manifest["execution"]["load"] = {"rate_per_second": 1, "serving_topology": "shared"}
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentValidationError, "'0.4'"):
                resolve_agent_manifest(root / "w2")


def _planning(*, concurrency: bool) -> dict:
    return {
        "schema_version": "0.1",
        "hypothesis": "FP8 keeps quality and tail latency under load.",
        "intervention": "Serve FP8 weights instead of BF16.",
        "predicted_direction": "decrease",
        "claim": {"level": "request", "primary_metric": "total_latency_ms"},
        "measurement_scope": {
            "tool_time": False,
            "concurrency": concurrency,
            "routing": False,
            "production_behavior": False,
            "live_agent": False,
        },
        "intervention_variables": ["quantization"],
    }


class LoadPlanningTests(unittest.TestCase):
    def _plan(self, raw: dict) -> dict:
        with TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "manifest.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            return plan_experiment(path)

    def test_planner_accepts_concurrency_claim_when_design_measures_it(self) -> None:
        raw = _static_manifest()
        raw["design"] = {"planning": _planning(concurrency=True)}
        report = self._plan(raw)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertNotIn("runeconomy.plan.unsupported_concurrency_claim", codes)
        self.assertNotEqual(report["state"], "blocked", report["issues"])
        self.assertTrue(report["load"]["measures_concurrency"])
        self.assertEqual(report["load"]["gate_level"], 16)
        digest = static_arrival_schedule(ExperimentManifest.from_dict(raw)).digest()
        self.assertEqual(report["load"]["arrival_schedule_sha256"], digest)
        self.assertIn("runeconomy.plan.arrival_schedule_not_pinned", codes)
        population = report["preregistration"]["population"]["arrival_schedule"]
        self.assertEqual(population["sha256"], digest)
        gate_names = [gate["name"] for gate in report["preregistration"]["gates"]]
        self.assertIn("p95_workflow_latency_ms_noninferiority", gate_names)
        self.assertIn("usd_per_1k_successful_workflows_noninferiority", gate_names)
        metric_rows = {row["name"]: row for row in report["performance"]["metric_gates"]}
        self.assertTrue(
            metric_rows["p95_workflow_latency_ms_noninferiority"]["meets_minimum_observations"]
        )

    def test_sequential_design_still_cannot_claim_concurrency(self) -> None:
        raw = _static_manifest()
        raw["execution"]["arrival_pattern"] = "sequential"
        raw["execution"].pop("load")
        raw["design"] = {"planning": _planning(concurrency=True)}
        report = self._plan(raw)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("runeconomy.plan.unsupported_concurrency_claim", codes)
        self.assertFalse(report["load"]["measures_concurrency"])

    def test_tail_gate_below_minimum_blocks_the_design(self) -> None:
        raw = _static_manifest(cases=40, concurrency=4)
        raw["acceptance"]["metric_gates"] = [
            {"metric": "workflow_latency_ms", "statistic": "p99", "semantics": "noninferiority", "margin": 5}
        ]
        report = self._plan(raw)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("runeconomy.plan.metric_gate_sample_below_minimum", codes)
        self.assertEqual(report["state"], "blocked")

    def test_static_request_gate_sample_is_counted_exactly(self) -> None:
        # Isolated static replay issues one request per paired case, so a
        # request-level tail gate's sample is known before running.
        raw = _static_manifest(cases=48, concurrency=8)
        raw["execution"]["repetitions"] = 3
        raw["acceptance"]["metric_gates"] = [
            {"metric": "client_ttft_ms", "statistic": "p95", "semantics": "superiority"}
        ]
        report = self._plan(raw)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("runeconomy.plan.metric_gate_sample_below_minimum", codes)
        self.assertNotIn("runeconomy.plan.metric_gate_request_count_data_dependent", codes)
        row = report["performance"]["metric_gates"][0]
        self.assertEqual(row["planned_observations_upper_bound"], 144)
        self.assertFalse(row["meets_minimum_observations"])
        self.assertEqual(report["state"], "blocked")

        raw["execution"]["repetitions"] = 5
        report = self._plan(raw)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertNotIn("runeconomy.plan.metric_gate_sample_below_minimum", codes)
        self.assertTrue(report["performance"]["metric_gates"][0]["meets_minimum_observations"])

    def test_pinned_schedule_is_a_freeze_check(self) -> None:
        raw = _static_manifest(cases=100, concurrency=8)
        digest = static_arrival_schedule(ExperimentManifest.from_dict(raw)).digest()
        raw["execution"]["load"]["schedule_sha256"] = digest
        report = self._plan(raw)
        self.assertTrue(report["load"]["schedule_pin_matches"])
        self.assertTrue(report["design_declarations"]["freeze_checks"]["arrival_schedule_pinned"])
        raw["execution"]["seed"] = 12
        report = self._plan(raw)
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("runeconomy.plan.arrival_schedule_pin_mismatch", codes)

    def test_sweep_multiplies_planned_attempts(self) -> None:
        raw = _static_manifest(cases=100, concurrency=4)
        raw["execution"]["load"] = {"concurrency_levels": [1, 4, 16], "gate_concurrency": 16}
        report = self._plan(raw)
        self.assertEqual(report["load"]["stages"], 3)
        self.assertEqual(report["load"]["population_attempts_all_stages"], 600)
        self.assertEqual(report["preregistration"]["population"]["planned_pairs"], 100)


if __name__ == "__main__":
    unittest.main()
