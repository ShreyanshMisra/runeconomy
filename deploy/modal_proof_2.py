"""Cost-bounded dual-L40S pilot and calibration for Proof 2."""

from __future__ import annotations

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
from typing import Any, BinaryIO

import modal


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

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"huggingface_hub[hf_xet]=={HUGGINGFACE_HUB_VERSION}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": HF_CACHE_PATH})
)

proof_image = (
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
            "VLLM_STRICT_QUANT_SCHEME": "1",
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


def _warm_structured_paths() -> None:
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
    profile: str, work: Path
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    log = (work / f"{profile}-vllm.log").open("wb")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0" if profile == "baseline" else "1",
    }
    process = subprocess.Popen(
        vllm_command(profile),
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


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
        raise ValueError("Proof 2 runner permits only 5-task pilot or 40-task calibration")
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
        _warm_structured_paths()
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
        result["provenance"]["driver_location"] = "same Modal container as both vLLM servers"
        (artifacts / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata_payload = {
            "purpose": (
                "40-task calibration" if expected_task_count == 40 else "five-task pilot"
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
