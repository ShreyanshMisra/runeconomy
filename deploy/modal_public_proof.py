"""Colocated, cost-bounded Modal pilot for the reliable proof agent."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

import modal


APP_NAME = "runeconomy-public-proof-pilot"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
CHAT_TEMPLATE_DIGEST = (
    "sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
VLLM_VERSION = "0.33.0"
TRANSFORMERS_VERSION = "5.16.1"
HUGGINGFACE_HUB_VERSION = "1.31.0"

GPU = "A10"
PORT = 8000
BLOCK_SIZE = 16
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_SEQS = 8
PILOT_TIMEOUT_SECONDS = 12 * 60

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"
REMOTE_REPOSITORY = Path("/root/proof-repo")
REMOTE_AGENT = Path("/root/reliable-proof/agent.py")

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

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"huggingface_hub[hf_xet]=={HUGGINGFACE_HUB_VERSION}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

pilot_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
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
        }
    )
    .add_local_file(
        "examples/reliable_proof/agent.py", str(REMOTE_AGENT), copy=True
    )
    .add_local_dir(
        "src/runeconomy", str(REMOTE_REPOSITORY / "src" / "runeconomy"), copy=True
    )
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
    """Cache the exact model revision without allocating a GPU."""

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=HF_CACHE_PATH,
    )
    hf_cache.commit()
    return path


def vllm_command() -> list[str]:
    return [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--enable-prefix-caching",
        "--block-size",
        str(BLOCK_SIZE),
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        "--enable-scale-out",
    ]


def _wait_until_healthy(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 8 * 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy within eight minutes")


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
            repo_id=MODEL_NAME,
            revision=MODEL_REVISION,
            cache_dir=HF_CACHE_PATH,
            local_files_only=True,
        )
    )
    if snapshot.name != MODEL_REVISION:
        raise RuntimeError("resolved model revision differs from the pinned revision")
    tokenizer = json.loads(
        (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    template = tokenizer.get("chat_template")
    if not isinstance(template, str):
        raise RuntimeError("tokenizer configuration has no chat template")
    return "sha256:" + hashlib.sha256(template.encode()).hexdigest()


def _warm_structured_paths() -> None:
    from runeconomy.vllm import VLLMClient

    client = VLLMClient(
        f"http://127.0.0.1:{PORT}",
        api_dialect="chat_completions",
        timeout_seconds=120,
    )
    schemas = (
        (
            "repository_search_plan",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        (
            "repository_identifier_answer",
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
    for name, schema in schemas:
        client.capture(
            {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": "Return a JSON object containing the value OK.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
                "cache_salt": "runeconomy-proof-pilot-warmup",
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


@app.function(
    image=pilot_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
        FLASHINFER_CACHE_PATH: flashinfer_cache,
    },
    timeout=PILOT_TIMEOUT_SECONDS,
    cpu=4,
    memory=16384,
)
def run_pilot(
    tasks_payload: dict[str, Any],
    evaluator_source: str,
    expected_task_count: int = 5,
) -> bytes:
    """Run an approved calibration workload beside vLLM."""

    from runeconomy.comparison import run_comparison

    tasks = tasks_payload.get("tasks")
    if expected_task_count not in {5, 20, 40}:
        raise ValueError(
            "only the approved five-task pilot, 20-task calibration, or "
            "40-task held-out proof is allowed"
        )
    if not isinstance(tasks, list) or len(tasks) != expected_task_count:
        raise ValueError(
            f"workload requires exactly {expected_task_count} calibration tasks"
        )
    if _chat_template_digest() != CHAT_TEMPLATE_DIGEST:
        raise RuntimeError("chat-template identity differs from the pin")
    if metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError("vLLM version differs from the pin")

    work = Path(tempfile.mkdtemp(prefix="runeconomy-public-proof-pilot-"))
    dataset = work / "pilot-tasks.json"
    evaluator = work / "pilot-evaluator.py"
    artifacts = work / "run"
    dataset.write_text(
        json.dumps(tasks_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evaluator.write_text(evaluator_source, encoding="utf-8")

    command = vllm_command()
    started = time.monotonic()
    process = subprocess.Popen(command)
    try:
        _wait_until_healthy(process)
        _warm_structured_paths()
        os.environ.update(
            {
                "VLLM_BASE_URL": f"http://127.0.0.1:{PORT}",
                "RUNECONOMY_REPO_ROOT": str(REMOTE_REPOSITORY),
                "RUNECONOMY_MODEL": MODEL_NAME,
                "RUNECONOMY_MODEL_REVISION": MODEL_REVISION,
                "RUNECONOMY_TOKENIZER": MODEL_NAME,
                "RUNECONOMY_TOKENIZER_REVISION": MODEL_REVISION,
                "RUNECONOMY_CHAT_TEMPLATE_DIGEST": CHAT_TEMPLATE_DIGEST,
                "RUNECONOMY_BACKEND_VERSION": VLLM_VERSION,
            }
        )
        result = run_comparison(
            baseline_command=(
                f"python {REMOTE_AGENT} --metadata-position prefix"
            ),
            candidate_command=(
                f"python {REMOTE_AGENT} --metadata-position suffix"
            ),
            dataset_path=dataset,
            evaluator_path=evaluator,
            output_path=artifacts / "result.json",
            artifacts_dir=artifacts,
            workdir=REMOTE_REPOSITORY,
            repetitions=1,
            order="randomized",
            seed=20261114,
            timeout_seconds=180,
            evaluator_timeout_seconds=30,
            max_task_success_regression=(
                0.1 if expected_task_count in {20, 40} else 1.0
            ),
            min_task_success_rate=(
                0.8 if expected_task_count in {20, 40} else 0.0
            ),
            min_completed_rate=1.0,
        )
        result["provenance"]["driver_location"] = "same Modal container as vLLM"
        (artifacts / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_payload = {
            "purpose": (
                "frozen 40-task held-out proof"
                if expected_task_count == 40
                else (
                    "full 20-task live calibration"
                    if expected_task_count == 20
                    else "five-task calibration-only infrastructure pilot"
                )
            ),
            "task_count": expected_task_count,
            "remote_elapsed_seconds": time.monotonic() - started,
            "unmeasured_structured_warmup_requests": 2,
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "chat_template_digest": CHAT_TEMPLATE_DIGEST,
            "backend_version": VLLM_VERSION,
            "decision": result["decision"]["status"],
        }
        return _private_archive(
            work,
            metadata_payload,
            metadata_name=(
                "heldout-metadata.json"
                if expected_task_count == 40
                else (
                    "calibration-metadata.json"
                    if expected_task_count == 20
                    else "pilot-metadata.json"
                )
            ),
        )
    finally:
        _stop_process(process)


@app.local_entrypoint()
def prepare() -> None:
    """Cache the pinned model without allocating a GPU."""

    path = download_model.remote()
    print(f"pinned model cached at {path}")


@app.local_entrypoint()
def pilot(
    tasks: str = "artifacts/public-proof-preparation/pilot-tasks.json",
    evaluator: str = "artifacts/public-proof-preparation/pilot-evaluator.py",
    output: str = "artifacts/public-proof-pilot.tar.gz",
) -> None:
    """Run the approved five-task calibration pilot and save private evidence."""

    task_path = Path(tasks).resolve()
    evaluator_path = Path(evaluator).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    archive = run_pilot.remote(task_payload, evaluator_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(archive)
    temporary.replace(output_path)
    print(f"wrote private pilot evidence to {output_path}")


@app.local_entrypoint()
def calibration(
    tasks: str = "artifacts/public-proof-preparation/calibration-tasks.json",
    evaluator: str = "artifacts/public-proof-preparation/calibration-evaluator.py",
    output: str = "artifacts/public-proof-calibration-live.tar.gz",
) -> None:
    """Run the approved full calibration and save private evidence."""

    task_path = Path(tasks).resolve()
    evaluator_path = Path(evaluator).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    archive = run_pilot.remote(task_payload, evaluator_source, 20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(archive)
    temporary.replace(output_path)
    print(f"wrote private calibration evidence to {output_path}")


@app.local_entrypoint()
def heldout(
    tasks: str = "artifacts/public-proof-preparation/heldout-tasks.json",
    evaluator: str = "artifacts/public-proof-preparation/heldout-evaluator.py",
    output: str = "artifacts/public-proof-heldout-live.tar.gz",
) -> None:
    """Run the approved frozen held-out proof and save private evidence."""

    task_path = Path(tasks).resolve()
    evaluator_path = Path(evaluator).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output_path}")
    task_payload = json.loads(task_path.read_text(encoding="utf-8"))
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    archive = run_pilot.remote(task_payload, evaluator_source, 40)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(archive)
    temporary.replace(output_path)
    print(f"wrote private held-out evidence to {output_path}")
