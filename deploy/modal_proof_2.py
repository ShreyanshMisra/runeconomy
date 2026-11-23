"""Cost-bounded dual-L40S calibration and crossover validation for Proof 2."""

from __future__ import annotations

import hashlib
import io
import json
import os
import statistics
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, BinaryIO

import modal

from runeconomy.crossover import balanced_two_block_split


APP_NAME = "runeconomy-proof-2-calibration"
SOURCE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
CHAT_TEMPLATE_DIGEST = (
    "sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
VLLM_VERSION = "0.34.0"
TRANSFORMERS_VERSION = "5.18.0"
HUGGINGFACE_HUB_VERSION = "2.0.0"
GPU = "L40S:2"
PORTS = {"baseline": 8000, "candidate": 8001}
SERVED_MODELS = {
    "baseline": "runeconomy-qwen2.5-14b-bf16",
    "candidate": "runeconomy-qwen2.5-14b-fp8-per-tensor",
}
QUANTIZATION = {"baseline": None, "candidate": "fp8_per_tensor"}
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_SEQS = 4
RUN_TIMEOUT_SECONDS = 30 * 60
CROSSOVER_SEED = 20261123
FROZEN_CONTRACT_SHA256 = (
    "sha256:b349073a6f93a9391d628f9abe38d49120cbd0f4b5cc4202f190f924da6a9816"
)
FROZEN_HELDOUT_TASKS_SHA256 = (
    "sha256:c945014bc7aed8cd6164b9c12da060a05880bea35a2ed29850e27612aa9924b2"
)
FROZEN_HELDOUT_EVALUATOR_SHA256 = (
    "sha256:8de9593491d146f7d81126f911165d85ae9353b13f7fa9f9065a05325d089810"
)
FROZEN_HELDOUT_AUTHORIZATION = "AUTHORIZE_PROOF_2_HELDOUT_B349073A6F93"
HELDOUT_HARD_CEILING_USD = 8.0
PHASE_4_CUMULATIVE_CEILING_USD = 40.0
PHASE_4_SPEND_AT_FREEZE_USD = 2.25252396
CONSUMED_CALIBRATION_TASKS_SHA256 = (
    "sha256:313434d48f4a870db387a42217d8a898391f015b082f23dcd64a87830db9abc7"
)
CONSUMED_CALIBRATION_EVALUATOR_SHA256 = (
    "sha256:5906611e49d74211503b1ec97942a1343a4295ebd0aa511f35c76f6d0bf8fa4f"
)
CROSSOVER_DEVICE_ASSIGNMENTS = {
    "block_a": {"baseline": 0, "candidate": 1},
    "block_b": {"baseline": 1, "candidate": 0},
}

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"
REMOTE_REPOSITORY = Path("/root/proof-2-repo")
REMOTE_AGENT = Path("/root/proof-2/agent.py")

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name(
    "agent-inference-doctor-hf-cache", create_if_missing=True, version=2
)
vllm_cache = modal.Volume.from_name(
    "agent-inference-doctor-vllm-cache", create_if_missing=True, version=2
)
flashinfer_cache = modal.Volume.from_name(
    "agent-inference-doctor-flashinfer-cache", create_if_missing=True, version=2
)
heldout_run_ledger = modal.Dict.from_name(
    "runeconomy-proof-2-heldout-ledger", create_if_missing=True
)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"huggingface_hub[hf_xet]=={HUGGINGFACE_HUB_VERSION}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": HF_CACHE_PATH})
)

proof_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        "jsonschema>=4.23,<5",
        "opentelemetry-sdk>=1.38,<2",
        f"huggingface_hub[hf_xet]=={HUGGINGFACE_HUB_VERSION}",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": HF_CACHE_PATH,
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
    .add_local_file("examples/proof_2/agent.py", str(REMOTE_AGENT), copy=True)
    .add_local_dir(
        "src/runeconomy", str(REMOTE_REPOSITORY / "src" / "runeconomy"), copy=True
    )
    .add_local_dir("scripts", str(REMOTE_REPOSITORY / "scripts"), copy=True)
    .add_local_dir("deploy", str(REMOTE_REPOSITORY / "deploy"), copy=True)
    .add_local_dir("examples", str(REMOTE_REPOSITORY / "examples"), copy=True)
    .add_local_python_source("runeconomy")
)


@app.function(
    image=download_image,
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=30 * 60,
    cpu=2,
    memory=4096,
)
def download_model() -> str:
    """Cache the exact shared source checkpoint without allocating a GPU."""

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=SOURCE_MODEL,
        revision=MODEL_REVISION,
        cache_dir=HF_CACHE_PATH,
    )
    hf_cache.commit()
    return path


def vllm_command(profile: str) -> list[str]:
    command = [
        "vllm",
        "serve",
        SOURCE_MODEL,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        SERVED_MODELS[profile],
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--enable-prefix-caching",
        "--disable-log-requests",
        "--port",
        str(PORTS[profile]),
    ]
    if QUANTIZATION[profile] is not None:
        command.extend(["--quantization", str(QUANTIZATION[profile])])
    return command


def _wait_until_healthy(
    profile: str, process: subprocess.Popen[bytes], timeout_seconds: float = 10 * 60
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{PORTS[profile]}/health"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"{profile} vLLM exited during startup ({returncode})")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"{profile} vLLM did not become healthy before timeout")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _chat_template_digest() -> str:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=SOURCE_MODEL,
            revision=MODEL_REVISION,
            cache_dir=HF_CACHE_PATH,
            local_files_only=True,
        )
    )
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError("resolved model revision differs from the pin")
    tokenizer = json.loads(
        (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    template = tokenizer.get("chat_template")
    if not isinstance(template, str):
        raise RuntimeError("tokenizer configuration has no chat template")
    return "sha256:" + hashlib.sha256(template.encode()).hexdigest()


def _warm_and_reset_structured_paths() -> dict[str, Any]:
    from runeconomy.vllm import VLLMClient

    schemas = (
        (
            "proof_2_search_plan",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 240}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        (
            "proof_2_identifier_answer",
            {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        ),
    )
    procedure: dict[str, Any] = {
        "warmup_requests_per_profile": len(schemas),
        "prefix_cache_resets_per_profile": 1,
        "profiles": {},
    }
    for profile in ("baseline", "candidate"):
        client = VLLMClient(
            f"http://127.0.0.1:{PORTS[profile]}",
            api_dialect="chat_completions",
            timeout_seconds=120,
        )
        for name, schema in schemas:
            client.capture(
                {
                    "model": SERVED_MODELS[profile],
                    "messages": [
                        {
                            "role": "user",
                            "content": "Return a JSON object containing the value OK.",
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 16,
                    "cache_salt": "runeconomy-proof-2-warmup",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": name,
                            "strict": True,
                            "schema": schema,
                        },
                    },
                }
            )
        client.reset_prefix_cache()
        procedure["profiles"][profile] = {
            "warmup_response_schemas": [name for name, _ in schemas],
            "prefix_cache_reset": True,
        }
    return procedure


def _assert_backend_profiles(work: Path, block: str = "") -> None:
    prefix = f"{block}-" if block else ""
    baseline = (work / f"{prefix}baseline-vllm.log").read_text(
        encoding="utf-8", errors="replace"
    )
    candidate = (work / f"{prefix}candidate-vllm.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if "quantization=None" not in baseline:
        raise RuntimeError("baseline did not start with unquantized BF16 weights")
    if "quantization=fp8_per_tensor" not in candidate:
        raise RuntimeError("candidate did not start with fp8_per_tensor quantization")
    if "Selected CutlassFP8ScaledMMLinearKernel" not in candidate:
        raise RuntimeError("candidate did not select the expected L40S FP8 kernel")
    if (
        "activations as unquantized" in candidate
        or "quantization fallback" in candidate
    ):
        raise RuntimeError("candidate quantization used an undeclared fallback")


def _private_archive(
    root: Path, metadata_payload: dict[str, Any], *, metadata_name: str
) -> bytes:
    metadata_path = root / metadata_name
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(root)))
    return buffer.getvalue()


def _start_server(
    profile: str, work: Path, *, block: str = "", device_index: int | None = None
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    prefix = f"{block}-" if block else ""
    log = (work / f"{prefix}{profile}-vllm.log").open("wb")
    if device_index is None:
        device_index = 0 if profile == "baseline" else 1
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(device_index),
    }
    process = subprocess.Popen(
        vllm_command(profile),
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_payload_digest(payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return _digest_bytes(encoded)


def _balanced_blocks(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return balanced_two_block_split(
        tasks,
        seed=CROSSOVER_SEED,
        family_names=("call_expression", "string_literal"),
    )


def _frozen_heldout_blocks(
    tasks_payload: dict[str, Any], evaluator_source: str
) -> dict[str, list[dict[str, Any]]]:
    if _canonical_payload_digest(tasks_payload) != FROZEN_HELDOUT_TASKS_SHA256:
        raise ValueError("held-out tasks do not match the frozen dataset digest")
    if _digest_bytes(evaluator_source.encode()) != FROZEN_HELDOUT_EVALUATOR_SHA256:
        raise ValueError("held-out evaluator does not match the frozen digest")
    tasks = tasks_payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 180:
        raise ValueError("frozen held-out execution requires exactly 180 tasks")
    blocks = _balanced_blocks(tasks)
    expected_block_by_id = {
        task["task_id"]: block
        for block, block_tasks in blocks.items()
        for task in block_tasks
    }
    for task in tasks:
        if task.get("crossover_block") != expected_block_by_id[task["task_id"]]:
            raise ValueError("held-out task crossover assignment differs from freeze")
    for block, block_tasks in blocks.items():
        if len(block_tasks) != 90:
            raise ValueError(f"{block} does not contain exactly 90 frozen tasks")
        if Counter(task["family"] for task in block_tasks) != {
            "call_expression": 45,
            "string_literal": 45,
        }:
            raise ValueError(f"{block} is not balanced across the frozen families")
    return blocks


def _projected_resource_cost_usd(
    *,
    l40s_per_gpu_second: float,
    cpu_per_core_second: float,
    memory_per_gib_second: float,
) -> float:
    rates = (
        l40s_per_gpu_second,
        cpu_per_core_second,
        memory_per_gib_second,
    )
    if any(rate <= 0 for rate in rates):
        raise ValueError("current Modal resource rates must be positive")
    return RUN_TIMEOUT_SECONDS * (
        2 * l40s_per_gpu_second + 8 * cpu_per_core_second + 32 * memory_per_gib_second
    )


def _physical_gpu_inventory() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    inventory: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("unexpected nvidia-smi GPU inventory format")
        inventory.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "pci_bus_id": fields[3],
            }
        )
    if len(inventory) != 2 or len({gpu["uuid"] for gpu in inventory}) != 2:
        raise RuntimeError("crossover requires exactly two distinct physical GPUs")
    if any("L40S" not in gpu["name"] for gpu in inventory):
        raise RuntimeError("crossover requires two physical L40S GPUs")
    return sorted(inventory, key=lambda gpu: gpu["index"])


def _annotated_attempts(
    result: dict[str, Any],
    *,
    block: str,
    artifacts: Path,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    device_assignment = CROSSOVER_DEVICE_ASSIGNMENTS[block]
    for original in result["attempts"]:
        attempt = dict(original)
        attempt_dir = artifacts / original["artifacts"]["directory"]
        task = json.loads((attempt_dir / "task.json").read_text(encoding="utf-8"))
        attempt["crossover_block"] = block
        attempt["physical_device_index"] = device_assignment[attempt["variant"]]
        attempt["task_family"] = task["family"]
        attempt["attempt_id"] = f"{block}-{attempt['attempt_id']}"
        artifact_paths = dict(attempt["artifacts"])
        artifact_paths["directory"] = str(
            attempt_dir.relative_to(artifacts.parent.parent)
        )
        attempt["artifacts"] = artifact_paths
        annotated.append(attempt)
    return annotated


def _descriptive_crossover_sensitivity(
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    def effect(subset: list[dict[str, Any]]) -> dict[str, Any]:
        by_key = {
            (attempt["task_id"], attempt["variant"]): attempt for attempt in subset
        }
        differences: list[float] = []
        for task_id in sorted({attempt["task_id"] for attempt in subset}):
            baseline = by_key.get((task_id, "baseline"))
            candidate = by_key.get((task_id, "candidate"))
            if not baseline or not candidate:
                continue
            if not baseline["task_success"] or not candidate["task_success"]:
                continue
            differences.append(
                float(candidate["workflow_latency_ms"])
                - float(baseline["workflow_latency_ms"])
            )
        return {
            "both_successful_pair_count": len(differences),
            "candidate_minus_baseline_mean_ms": (
                sum(differences) / len(differences) if differences else None
            ),
            "candidate_minus_baseline_median_ms": (
                statistics.median(differences) if differences else None
            ),
        }

    block_effects = {
        block: effect(
            [attempt for attempt in attempts if attempt["crossover_block"] == block]
        )
        for block in CROSSOVER_DEVICE_ASSIGNMENTS
    }
    family_effects = {
        family: effect(
            [attempt for attempt in attempts if attempt["task_family"] == family]
        )
        for family in ("call_expression", "string_literal")
    }
    cells: dict[str, Any] = {}
    for block, assignments in CROSSOVER_DEVICE_ASSIGNMENTS.items():
        for profile, device in assignments.items():
            successful = [
                float(attempt["workflow_latency_ms"])
                for attempt in attempts
                if attempt["crossover_block"] == block
                and attempt["variant"] == profile
                and attempt["task_success"]
            ]
            cells[f"{block}:gpu_{device}:{profile}"] = {
                "successful_attempt_count": len(successful),
                "mean_workflow_latency_ms": (
                    sum(successful) / len(successful) if successful else None
                ),
            }
    return {
        "role": "descriptive_sensitivity_only",
        "block_effects": block_effects,
        "family_effects": family_effects,
        "device_profile_cells": cells,
    }


@app.function(
    image=proof_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
        FLASHINFER_CACHE_PATH: flashinfer_cache,
    },
    timeout=RUN_TIMEOUT_SECONDS,
    cpu=8,
    memory=32768,
)
def run_calibration(
    tasks_payload: dict[str, Any],
    evaluator_source: str,
    expected_task_count: int,
) -> bytes:
    """Run only the approved five-task pilot or 40-task calibration."""

    from runeconomy.comparison import run_comparison

    if expected_task_count not in {5, 40}:
        raise ValueError(
            "Proof 2 runner permits only 5-task pilot or 40-task calibration"
        )
    tasks = tasks_payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected_task_count:
        raise ValueError(f"workload must contain exactly {expected_task_count} tasks")
    if _chat_template_digest() != CHAT_TEMPLATE_DIGEST:
        raise RuntimeError("chat-template identity differs from the pin")
    if metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError("vLLM version differs from the pin")

    work = Path(tempfile.mkdtemp(prefix="runeconomy-proof-2-"))
    dataset = work / "tasks.json"
    evaluator = work / "evaluator.py"
    artifacts = work / "run"
    dataset.write_text(
        json.dumps(tasks_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evaluator.write_text(evaluator_source, encoding="utf-8")

    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs: list[BinaryIO] = []
    started = time.monotonic()
    try:
        for profile in ("baseline", "candidate"):
            process, log = _start_server(profile, work)
            processes[profile] = process
            logs.append(log)
        for profile in ("baseline", "candidate"):
            _wait_until_healthy(profile, processes[profile])
        _assert_backend_profiles(work)
        _warm_and_reset_structured_paths()
        os.environ.update(
            {
                "RUNECONOMY_BASELINE_URL": f"http://127.0.0.1:{PORTS['baseline']}",
                "RUNECONOMY_CANDIDATE_URL": f"http://127.0.0.1:{PORTS['candidate']}",
                "RUNECONOMY_REPO_ROOT": str(REMOTE_REPOSITORY),
            }
        )
        result = run_comparison(
            baseline_command=f"python {REMOTE_AGENT}",
            candidate_command=f"python {REMOTE_AGENT}",
            dataset_path=dataset,
            evaluator_path=evaluator,
            output_path=artifacts / "result.json",
            artifacts_dir=artifacts,
            workdir=REMOTE_REPOSITORY,
            repetitions=1,
            order="randomized",
            seed=20261123,
            timeout_seconds=180,
            evaluator_timeout_seconds=30,
            max_task_success_regression=0.1 if expected_task_count == 40 else 1.0,
            min_task_success_rate=0.8 if expected_task_count == 40 else 0.0,
            min_completed_rate=1.0,
            optimization_target="engine_change_latency",
            latency_semantics="superiority",
            max_latency_regression_ms=0.0,
        )
        result["provenance"]["driver_location"] = (
            "same Modal container as both vLLM servers"
        )
        (artifacts / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata_payload = {
            "purpose": (
                "40-task calibration"
                if expected_task_count == 40
                else "five-task pilot"
            ),
            "held_out": False,
            "task_count": expected_task_count,
            "remote_elapsed_seconds": time.monotonic() - started,
            "unmeasured_structured_warmup_requests_per_server": 2,
            "source_model": SOURCE_MODEL,
            "model_revision": MODEL_REVISION,
            "chat_template_digest": CHAT_TEMPLATE_DIGEST,
            "backend_version": VLLM_VERSION,
            "hardware": GPU,
            "profiles": {
                profile: {
                    "served_model": SERVED_MODELS[profile],
                    "quantization": QUANTIZATION[profile] or "none_bf16",
                    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                }
                for profile in ("baseline", "candidate")
            },
            "decision": result["decision"]["status"],
        }
        return _private_archive(
            work,
            metadata_payload,
            metadata_name=(
                "calibration-metadata.json"
                if expected_task_count == 40
                else "pilot-metadata.json"
            ),
        )
    finally:
        for process in processes.values():
            _stop_process(process)
        for log in logs:
            log.close()


@app.function(
    image=proof_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
        FLASHINFER_CACHE_PATH: flashinfer_cache,
    },
    timeout=RUN_TIMEOUT_SECONDS,
    cpu=8,
    memory=32768,
)
def run_crossover_validation(
    tasks_payload: dict[str, Any],
    evaluator_source: str,
) -> bytes:
    """Validate crossover mechanics only on the consumed 40-task calibration."""

    from runeconomy.comparison import _decision, _summarize, run_comparison

    if _canonical_payload_digest(tasks_payload) != CONSUMED_CALIBRATION_TASKS_SHA256:
        raise ValueError("crossover validation requires the consumed calibration tasks")
    if (
        _digest_bytes(evaluator_source.encode())
        != CONSUMED_CALIBRATION_EVALUATOR_SHA256
    ):
        raise ValueError(
            "crossover validation requires the consumed calibration evaluator"
        )
    tasks = tasks_payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 40:
        raise ValueError("crossover validation requires exactly 40 consumed tasks")
    blocks = _balanced_blocks(tasks)
    if any(len(block_tasks) != 20 for block_tasks in blocks.values()):
        raise ValueError("crossover validation requires two 20-task blocks")
    if _chat_template_digest() != CHAT_TEMPLATE_DIGEST:
        raise RuntimeError("chat-template identity differs from the pin")
    if metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError("vLLM version differs from the pin")

    work = Path(tempfile.mkdtemp(prefix="runeconomy-proof-2-crossover-"))
    evaluator = work / "evaluator.py"
    evaluator.write_text(evaluator_source, encoding="utf-8")
    gpu_inventory = _physical_gpu_inventory()
    pooled_attempts: list[dict[str, Any]] = []
    block_results: dict[str, dict[str, Any]] = {}
    procedures: dict[str, dict[str, Any]] = {}
    started = time.monotonic()

    for block, block_tasks in blocks.items():
        block_root = work / block
        dataset = block_root / "tasks.json"
        artifacts = block_root / "run"
        block_root.mkdir(parents=True, exist_ok=True)
        dataset.write_text(
            json.dumps({"tasks": block_tasks}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        processes: dict[str, subprocess.Popen[bytes]] = {}
        logs: list[BinaryIO] = []
        try:
            assignment = CROSSOVER_DEVICE_ASSIGNMENTS[block]
            for profile in ("baseline", "candidate"):
                process, log = _start_server(
                    profile,
                    work,
                    block=block,
                    device_index=assignment[profile],
                )
                processes[profile] = process
                logs.append(log)
            for profile in ("baseline", "candidate"):
                _wait_until_healthy(profile, processes[profile])
            _assert_backend_profiles(work, block)
            procedures[block] = _warm_and_reset_structured_paths()
            os.environ.update(
                {
                    "RUNECONOMY_BASELINE_URL": (
                        f"http://127.0.0.1:{PORTS['baseline']}"
                    ),
                    "RUNECONOMY_CANDIDATE_URL": (
                        f"http://127.0.0.1:{PORTS['candidate']}"
                    ),
                    "RUNECONOMY_REPO_ROOT": str(REMOTE_REPOSITORY),
                }
            )
            result = run_comparison(
                baseline_command=f"python {REMOTE_AGENT}",
                candidate_command=f"python {REMOTE_AGENT}",
                dataset_path=dataset,
                evaluator_path=evaluator,
                output_path=artifacts / "result.json",
                artifacts_dir=artifacts,
                workdir=REMOTE_REPOSITORY,
                repetitions=1,
                order="randomized",
                seed=CROSSOVER_SEED,
                timeout_seconds=180,
                evaluator_timeout_seconds=30,
                max_task_success_regression=0.1,
                min_task_success_rate=0.8,
                min_completed_rate=1.0,
                optimization_target="engine_change_latency",
                latency_semantics="superiority",
                max_latency_regression_ms=0.0,
            )
            result["provenance"].update(
                {
                    "crossover_block": block,
                    "physical_device_assignment": assignment,
                    "driver_location": "same Modal container as both vLLM servers",
                    "warmup_and_cache_reset": procedures[block],
                }
            )
            (artifacts / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            block_results[block] = result
            pooled_attempts.extend(
                _annotated_attempts(result, block=block, artifacts=artifacts)
            )
        finally:
            for process in processes.values():
                _stop_process(process)
            for log in logs:
                log.close()

        if block == "block_a" and time.monotonic() - started >= 15 * 60:
            raise TimeoutError(
                "crossover abort: block A consumed the preregistered half-window"
            )

    pooled_summary = _summarize(pooled_attempts)
    pooled_decision = _decision(
        pooled_summary,
        max_task_success_regression=0.1,
        min_task_success_rate=0.8,
        min_completed_rate=1.0,
        optimization_target="engine_change_latency",
        latency_semantics="superiority",
        max_latency_regression_ms=0.0,
    )
    crossover_result = {
        "result_type": "runeconomy.proof_2_crossover_validation",
        "schema_version": "0.1",
        "purpose": "consumed calibration task crossover validation",
        "held_out": False,
        "design": {
            "seed": CROSSOVER_SEED,
            "tasks_sha256": CONSUMED_CALIBRATION_TASKS_SHA256,
            "evaluator_sha256": CONSUMED_CALIBRATION_EVALUATOR_SHA256,
            "device_assignments": CROSSOVER_DEVICE_ASSIGNMENTS,
            "block_task_counts": {
                block: len(block_tasks) for block, block_tasks in blocks.items()
            },
            "block_family_counts": {
                block: dict(Counter(task["family"] for task in block_tasks))
                for block, block_tasks in blocks.items()
            },
            "warmup_and_cache_reset": procedures,
        },
        "gpu_inventory": gpu_inventory,
        "primary_pooled_summary": pooled_summary,
        "validation_decision": pooled_decision,
        "descriptive_sensitivity": _descriptive_crossover_sensitivity(pooled_attempts),
        "block_results": {
            block: {
                "summary": result["summary"],
                "decision": result["decision"],
                "provenance": result["provenance"],
            }
            for block, result in block_results.items()
        },
        "attempts": pooled_attempts,
        "provenance": {
            "source_model": SOURCE_MODEL,
            "model_revision": MODEL_REVISION,
            "chat_template_digest": CHAT_TEMPLATE_DIGEST,
            "backend_version": VLLM_VERSION,
            "hardware": GPU,
            "remote_elapsed_seconds": time.monotonic() - started,
        },
    }
    (work / "crossover-result.json").write_text(
        json.dumps(crossover_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _private_archive(
        work,
        {
            "purpose": "consumed calibration task crossover validation",
            "held_out": False,
            "tasks_sha256": CONSUMED_CALIBRATION_TASKS_SHA256,
            "evaluator_sha256": CONSUMED_CALIBRATION_EVALUATOR_SHA256,
            "device_assignments": CROSSOVER_DEVICE_ASSIGNMENTS,
            "warmup_and_cache_reset": procedures,
            "remote_elapsed_seconds": time.monotonic() - started,
        },
        metadata_name="crossover-validation-metadata.json",
    )


def _execute_frozen_heldout(
    tasks_payload: dict[str, Any],
    evaluator_source: str,
    *,
    projected_cost_usd: float,
) -> bytes:
    from runeconomy.comparison import _decision, _summarize, run_comparison

    blocks = _frozen_heldout_blocks(tasks_payload, evaluator_source)
    if _chat_template_digest() != CHAT_TEMPLATE_DIGEST:
        raise RuntimeError("chat-template identity differs from the freeze")
    if metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError("vLLM version differs from the freeze")

    work = Path(tempfile.mkdtemp(prefix="runeconomy-proof-2-heldout-"))
    evaluator = work / "evaluator.py"
    evaluator.write_text(evaluator_source, encoding="utf-8")
    gpu_inventory = _physical_gpu_inventory()
    pooled_attempts: list[dict[str, Any]] = []
    block_results: dict[str, dict[str, Any]] = {}
    procedures: dict[str, dict[str, Any]] = {}
    started = time.monotonic()

    for block, block_tasks in blocks.items():
        block_root = work / block
        dataset = block_root / "tasks.json"
        artifacts = block_root / "run"
        block_root.mkdir(parents=True, exist_ok=True)
        dataset.write_text(
            json.dumps({"tasks": block_tasks}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        processes: dict[str, subprocess.Popen[bytes]] = {}
        logs: list[BinaryIO] = []
        try:
            assignment = CROSSOVER_DEVICE_ASSIGNMENTS[block]
            for profile in ("baseline", "candidate"):
                process, log = _start_server(
                    profile,
                    work,
                    block=block,
                    device_index=assignment[profile],
                )
                processes[profile] = process
                logs.append(log)
            for profile in ("baseline", "candidate"):
                _wait_until_healthy(profile, processes[profile])
            _assert_backend_profiles(work, block)
            procedures[block] = _warm_and_reset_structured_paths()
            os.environ.update(
                {
                    "RUNECONOMY_BASELINE_URL": (
                        f"http://127.0.0.1:{PORTS['baseline']}"
                    ),
                    "RUNECONOMY_CANDIDATE_URL": (
                        f"http://127.0.0.1:{PORTS['candidate']}"
                    ),
                    "RUNECONOMY_REPO_ROOT": str(REMOTE_REPOSITORY),
                }
            )
            result = run_comparison(
                baseline_command=f"python {REMOTE_AGENT}",
                candidate_command=f"python {REMOTE_AGENT}",
                dataset_path=dataset,
                evaluator_path=evaluator,
                output_path=artifacts / "result.json",
                artifacts_dir=artifacts,
                workdir=REMOTE_REPOSITORY,
                repetitions=1,
                order="randomized",
                seed=CROSSOVER_SEED,
                timeout_seconds=180,
                evaluator_timeout_seconds=30,
                max_task_success_regression=0.05,
                min_task_success_rate=0.90,
                min_completed_rate=1.0,
                optimization_target="engine_change_latency",
                latency_semantics="superiority",
                max_latency_regression_ms=0.0,
            )
            result["provenance"].update(
                {
                    "crossover_block": block,
                    "physical_device_assignment": assignment,
                    "driver_location": "same Modal container as both vLLM servers",
                    "warmup_and_cache_reset": procedures[block],
                    "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
                }
            )
            (artifacts / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            block_results[block] = result
            pooled_attempts.extend(
                _annotated_attempts(result, block=block, artifacts=artifacts)
            )
        finally:
            for process in processes.values():
                _stop_process(process)
            for log in logs:
                log.close()

        if block == "block_a" and time.monotonic() - started >= 15 * 60:
            raise TimeoutError(
                "held-out abort: block A consumed the frozen half-window"
            )

    pooled_summary = _summarize(pooled_attempts)
    pooled_decision = _decision(
        pooled_summary,
        max_task_success_regression=0.05,
        min_task_success_rate=0.90,
        min_completed_rate=1.0,
        optimization_target="engine_change_latency",
        latency_semantics="superiority",
        max_latency_regression_ms=0.0,
    )
    heldout_result = {
        "result_type": "runeconomy.proof_2_frozen_heldout",
        "schema_version": "0.1",
        "purpose": "one-shot frozen Proof 2 held-out execution",
        "held_out": True,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "design": {
            "seed": CROSSOVER_SEED,
            "tasks_sha256": FROZEN_HELDOUT_TASKS_SHA256,
            "evaluator_sha256": FROZEN_HELDOUT_EVALUATOR_SHA256,
            "task_count": 180,
            "device_assignments": CROSSOVER_DEVICE_ASSIGNMENTS,
            "block_task_counts": {
                block: len(block_tasks) for block, block_tasks in blocks.items()
            },
            "block_family_counts": {
                block: dict(Counter(task["family"] for task in block_tasks))
                for block, block_tasks in blocks.items()
            },
            "warmup_and_cache_reset": procedures,
            "spending": {
                "hard_ceiling_usd": HELDOUT_HARD_CEILING_USD,
                "projected_cost_usd": projected_cost_usd,
            },
        },
        "gpu_inventory": gpu_inventory,
        "primary_pooled_summary": pooled_summary,
        "decision": pooled_decision,
        "descriptive_sensitivity": _descriptive_crossover_sensitivity(pooled_attempts),
        "block_results": {
            block: {
                "summary": result["summary"],
                "decision": result["decision"],
                "provenance": result["provenance"],
            }
            for block, result in block_results.items()
        },
        "attempts": pooled_attempts,
        "provenance": {
            "source_model": SOURCE_MODEL,
            "model_revision": MODEL_REVISION,
            "chat_template_digest": CHAT_TEMPLATE_DIGEST,
            "backend_version": VLLM_VERSION,
            "hardware": GPU,
            "remote_elapsed_seconds": time.monotonic() - started,
        },
    }
    (work / "heldout-result.json").write_text(
        json.dumps(heldout_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _private_archive(
        work,
        {
            "purpose": "one-shot frozen Proof 2 held-out execution",
            "held_out": True,
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "tasks_sha256": FROZEN_HELDOUT_TASKS_SHA256,
            "evaluator_sha256": FROZEN_HELDOUT_EVALUATOR_SHA256,
            "device_assignments": CROSSOVER_DEVICE_ASSIGNMENTS,
            "warmup_and_cache_reset": procedures,
            "projected_cost_usd": projected_cost_usd,
            "remote_elapsed_seconds": time.monotonic() - started,
        },
        metadata_name="heldout-metadata.json",
    )


@app.function(
    image=proof_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
        FLASHINFER_CACHE_PATH: flashinfer_cache,
    },
    timeout=RUN_TIMEOUT_SECONDS,
    cpu=8,
    memory=32768,
)
def run_heldout(
    tasks_payload: dict[str, Any],
    evaluator_source: str,
    authorization: str,
    frozen_contract_sha256: str,
    projected_cost_usd: float,
) -> bytes:
    """Execute the frozen held-out contract at most once after authorization."""

    if authorization != FROZEN_HELDOUT_AUTHORIZATION:
        raise PermissionError("held-out execution lacks the frozen authorization")
    if frozen_contract_sha256 != FROZEN_CONTRACT_SHA256:
        raise ValueError("held-out invocation does not name the frozen contract")
    if projected_cost_usd <= 0 or projected_cost_usd > HELDOUT_HARD_CEILING_USD:
        raise ValueError("held-out projected cost exceeds the frozen ceiling")
    if (
        PHASE_4_SPEND_AT_FREEZE_USD + projected_cost_usd
        > PHASE_4_CUMULATIVE_CEILING_USD
    ):
        raise ValueError("held-out projected cost exceeds the Phase 4 ceiling")
    _frozen_heldout_blocks(tasks_payload, evaluator_source)

    ledger_key = FROZEN_CONTRACT_SHA256
    started = {
        "state": "started",
        "contract_sha256": FROZEN_CONTRACT_SHA256,
        "tasks_sha256": FROZEN_HELDOUT_TASKS_SHA256,
        "evaluator_sha256": FROZEN_HELDOUT_EVALUATOR_SHA256,
    }
    if not heldout_run_ledger.put(ledger_key, started, skip_if_exists=True):
        raise RuntimeError("the frozen held-out contract has already been invoked")
    try:
        archive = _execute_frozen_heldout(
            tasks_payload,
            evaluator_source,
            projected_cost_usd=projected_cost_usd,
        )
    except BaseException as exc:
        heldout_run_ledger.put(
            ledger_key,
            {**started, "state": "invalid", "error_type": type(exc).__name__},
        )
        raise
    heldout_run_ledger.put(
        ledger_key,
        {**started, "state": "completed", "archive_sha256": _digest_bytes(archive)},
    )
    return archive


@app.local_entrypoint()
def prepare() -> None:
    """Cache the pinned source checkpoint without allocating a GPU."""

    path = download_model.remote()
    print(f"pinned model cached at {path}")


def _run_local(tasks: str, evaluator: str, output: str, count: int) -> None:
    task_path = Path(tasks).resolve()
    evaluator_path = Path(evaluator).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    archive = run_calibration.remote(task_payload, evaluator_source, count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(archive)
    temporary.replace(output_path)
    print(f"wrote private calibration evidence to {output_path}")


@app.local_entrypoint()
def pilot(
    tasks: str = "artifacts/proof-2-preparation/pilot-tasks.json",
    evaluator: str = "artifacts/proof-2-preparation/pilot-evaluator.py",
    output: str = "artifacts/proof-2-pilot.tar.gz",
) -> None:
    """Run the approved five-task calibration-only pilot."""

    _run_local(tasks, evaluator, output, 5)


@app.local_entrypoint()
def calibration(
    tasks: str = "artifacts/proof-2-preparation/calibration-tasks.json",
    evaluator: str = "artifacts/proof-2-preparation/calibration-evaluator.py",
    output: str = "artifacts/proof-2-calibration.tar.gz",
) -> None:
    """Run the approved 40-task calibration split."""

    _run_local(tasks, evaluator, output, 40)


@app.local_entrypoint()
def crossover_validation(
    tasks: str = "artifacts/proof-2-preparation/calibration-tasks.json",
    evaluator: str = "artifacts/proof-2-preparation/calibration-evaluator.py",
    output: str = "artifacts/proof-2-crossover-validation.tar.gz",
) -> None:
    """Validate the two-block crossover only on consumed calibration tasks."""

    task_path = Path(tasks).resolve()
    evaluator_path = Path(evaluator).resolve()
    output_path = Path(output).resolve()
    if _digest_bytes(task_path.read_bytes()) != CONSUMED_CALIBRATION_TASKS_SHA256:
        raise ValueError("crossover validation tasks are not the consumed calibration")
    if (
        _digest_bytes(evaluator_path.read_bytes())
        != CONSUMED_CALIBRATION_EVALUATOR_SHA256
    ):
        raise ValueError(
            "crossover validation evaluator is not the consumed calibration evaluator"
        )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    archive = run_crossover_validation.remote(task_payload, evaluator_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(archive)
    temporary.replace(output_path)
    print(f"wrote private crossover validation evidence to {output_path}")


@app.local_entrypoint()
def heldout(
    authorization: str,
    current_l40s_rate_per_second: float,
    current_cpu_rate_per_core_second: float,
    current_memory_rate_per_gib_second: float,
    tasks: str = "artifacts/proof-2-proposed-crossover-contract-v2/tasks.json",
    evaluator: str = "artifacts/proof-2-proposed-crossover-contract-v2/evaluator.py",
    contract: str = "docs/proof-2-preregistration-proposal.json",
    output: str = "artifacts/proof-2-heldout.tar.gz",
) -> None:
    """Invoke the immutable one-shot runner only after separate authorization."""

    if authorization != FROZEN_HELDOUT_AUTHORIZATION:
        raise PermissionError("held-out execution lacks the frozen authorization")
    task_path = Path(tasks).resolve()
    evaluator_path = Path(evaluator).resolve()
    contract_path = Path(contract).resolve()
    output_path = Path(output).resolve()
    if _digest_bytes(task_path.read_bytes()) != FROZEN_HELDOUT_TASKS_SHA256:
        raise ValueError("held-out tasks do not match the frozen dataset digest")
    if _digest_bytes(evaluator_path.read_bytes()) != FROZEN_HELDOUT_EVALUATOR_SHA256:
        raise ValueError("held-out evaluator does not match the frozen digest")
    if _digest_bytes(contract_path.read_bytes()) != FROZEN_CONTRACT_SHA256:
        raise ValueError("local held-out contract differs from the immutable freeze")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite held-out evidence: {output_path}")

    projected_cost_usd = _projected_resource_cost_usd(
        l40s_per_gpu_second=current_l40s_rate_per_second,
        cpu_per_core_second=current_cpu_rate_per_core_second,
        memory_per_gib_second=current_memory_rate_per_gib_second,
    )
    if projected_cost_usd > HELDOUT_HARD_CEILING_USD:
        raise ValueError(
            f"projected held-out cost ${projected_cost_usd:.6f} exceeds "
            f"the frozen ${HELDOUT_HARD_CEILING_USD:.2f} ceiling"
        )
    if (
        PHASE_4_SPEND_AT_FREEZE_USD + projected_cost_usd
        > PHASE_4_CUMULATIVE_CEILING_USD
    ):
        raise ValueError("projected held-out cost exceeds the Phase 4 ceiling")

    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    _frozen_heldout_blocks(task_payload, evaluator_source)
    archive = run_heldout.remote(
        task_payload,
        evaluator_source,
        authorization,
        FROZEN_CONTRACT_SHA256,
        projected_cost_usd,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(archive)
    temporary.replace(output_path)
    print(f"wrote one-shot private held-out evidence to {output_path}")
