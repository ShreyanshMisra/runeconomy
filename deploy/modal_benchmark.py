"""Cost-bounded, colocated Modal runner for the held-out NegativeSplit proof.

This App is intentionally separate from the persistent HTTP Server definition
in ``modal_vllm.py``. ``modal run`` therefore allocates exactly one ephemeral
GPU function and cannot accidentally start a second minimum replica.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

APP_NAME = "runeconomy-benchmark"
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
GPU_TIMEOUT_SECONDS = 70 * 60
PHASE2_TIMEOUT_SECONDS = 30 * 60
PHASE2_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PHASE2_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"

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

benchmark_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        "jsonschema>=4.23,<5",
        f"huggingface_hub[hf_xet]=={HUGGINGFACE_HUB_VERSION}",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": HF_CACHE_PATH,
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
            "VLLM_SERVER_DEV_MODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
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
    """Cache the exact model revision before allocating a GPU."""

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=HF_CACHE_PATH,
    )
    hf_cache.commit()
    return path


@app.function(image=benchmark_image, timeout=5 * 60, cpu=2, memory=4096)
def verify_image() -> str:
    """Fail on CPU if the pinned image lacks required benchmark capabilities."""

    version = metadata.version("vllm")
    package_files = metadata.files("vllm") or []
    source_text = "\n".join(
        path.locate().read_text(encoding="utf-8", errors="ignore")
        for path in package_files
        if path.suffix == ".py" and path.locate().is_file()
    )
    required_flags = (
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        "--enable-scale-out",
    )
    missing = [flag for flag in required_flags if flag not in source_text]
    if version != VLLM_VERSION or missing:
        raise RuntimeError(
            f"unexpected vLLM image {version}; missing flags: {', '.join(missing)}"
        )
    return version


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


def phase2_vllm_command() -> list[str]:
    """Return a normal non-development server command without render APIs."""

    return [
        "vllm",
        "serve",
        PHASE2_MODEL_NAME,
        "--revision",
        PHASE2_MODEL_REVISION,
        "--served-model-name",
        PHASE2_MODEL_NAME,
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
    ]


def _wait_until_healthy(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 12 * 60
    health_url = f"http://127.0.0.1:{PORT}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy within twelve minutes")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _served_model_id() -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as response:
        payload = json.load(response)
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or len(models) != 1:
        raise RuntimeError("vLLM did not expose exactly one served model")
    model_id = models[0].get("id") if isinstance(models[0], dict) else None
    if not isinstance(model_id, str):
        raise RuntimeError("vLLM model identity response had no string id")
    return model_id


def _observed_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=MODEL_NAME,
            revision=MODEL_REVISION,
            cache_dir=HF_CACHE_PATH,
            local_files_only=True,
        )
    ).resolve()
    if path.name != MODEL_REVISION:
        raise RuntimeError(
            f"resolved model snapshot {path.name!r} differs from {MODEL_REVISION!r}"
        )
    return path


def _observed_chat_template_digest(snapshot: Path) -> str:
    tokenizer = json.loads(
        (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    template = tokenizer.get("chat_template")
    if not isinstance(template, str):
        raise RuntimeError("tokenizer_config.json contains no string chat_template")
    return "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()


def _observed_engine_config(command: list[str]) -> dict[str, Any]:
    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    return {
        "enable_prefix_caching": "--enable-prefix-caching" in command,
        "tensor_parallel_size": int(value("--tensor-parallel-size")),
        "dtype": value("--dtype"),
        "max_model_len": int(value("--max-model-len")),
        "gpu_memory_utilization": float(value("--gpu-memory-utilization")),
        "max_num_seqs": int(value("--max-num-seqs")),
    }


class _ObservedLocalClient:
    """Delegate capture while independently observing the live local identity."""

    def __init__(self, process: subprocess.Popen[bytes], command: list[str]) -> None:
        from runeconomy.vllm import VLLMClient

        self._client = VLLMClient(f"http://127.0.0.1:{PORT}")
        self._process = process
        self._command = command

    def capture(self, request_body: dict[str, Any]):
        return self._client.capture(request_body)

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        self._client.reset_prefix_cache(reset_external=reset_external)

    def environment_identity(self) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError("vLLM exited before environment identity was observed")
        snapshot = _observed_snapshot()
        return {
            "model": _served_model_id(),
            "model_revision": snapshot.name,
            "tokenizer_revision": snapshot.name,
            "chat_template_digest": _observed_chat_template_digest(snapshot),
            "backend_version": metadata.version("vllm"),
            "engine_config": _observed_engine_config(self._command),
        }


def _validate_manifest_identity(payload: dict[str, Any]) -> None:
    backend = payload["backend"]
    expected = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template_digest": CHAT_TEMPLATE_DIGEST,
        "backend_version": VLLM_VERSION,
        "hardware": "1x NVIDIA A10 24GB on Modal",
        "block_size": BLOCK_SIZE,
        "engine_config": _observed_engine_config(vllm_command()),
    }
    mismatches = {
        field: {"expected": value, "manifest": backend.get(field)}
        for field, value in expected.items()
        if backend.get(field) != value
    }
    if mismatches:
        raise ValueError(f"manifest does not match benchmark deployment: {mismatches}")


@app.function(
    image=benchmark_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
        FLASHINFER_CACHE_PATH: flashinfer_cache,
    },
    timeout=GPU_TIMEOUT_SECONDS,
    cpu=4,
    memory=16384,
)
def run_collocated(manifest_payload: dict[str, Any]) -> bytes:
    """Run one resolved experiment beside vLLM and return compressed raw evidence."""

    from runeconomy.experiments import ExperimentManifest, run_experiment

    _validate_manifest_identity(manifest_payload)
    manifest = ExperimentManifest.from_dict(manifest_payload)
    command = vllm_command()
    process = subprocess.Popen(command)
    try:
        _wait_until_healthy(process)
        client = _ObservedLocalClient(process, command)
        # Warm the Responses streaming path and compiled kernels without adding
        # a warm-up observation to the predeclared evaluation population.
        client.capture(
            {
                "model": MODEL_NAME,
                "instructions": "Return only OK.",
                "input": "OK",
                "temperature": 0.0,
                "max_output_tokens": 8,
                "seed": 404,
            }
        )
        client.reset_prefix_cache()
        result = run_experiment(manifest, client)
        result["provenance"]["driver_location"] = "same Modal container as vLLM"
        result["provenance"]["driver_base_url"] = f"http://127.0.0.1:{PORT}"
        result["provenance"]["unmeasured_driver_warmup_requests"] = 1
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return gzip.compress(encoded, compresslevel=9)
    finally:
        _stop_process(process)


@app.function(
    image=benchmark_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
        FLASHINFER_CACHE_PATH: flashinfer_cache,
    },
    timeout=PHASE2_TIMEOUT_SECONDS,
    cpu=4,
    memory=16384,
)
def run_phase2_acceptance() -> bytes:
    """Validate ordinary Chat Completions and salt isolation beside vLLM."""

    from runeconomy.backends import discover_vllm_capabilities
    from runeconomy.experiments import ExperimentManifest, run_experiment
    from runeconomy.vllm import VLLMClient

    # The shared benchmark image enables development mode for the Phase 1
    # runner. Remove it before starting this subprocess so reset endpoints are
    # genuinely absent from the Phase 2 server.
    os.environ.pop("VLLM_SERVER_DEV_MODE", None)
    command = phase2_vllm_command()
    started = time.monotonic()
    process = subprocess.Popen(command)
    try:
        _wait_until_healthy(process)
        base_url = f"http://127.0.0.1:{PORT}"
        capabilities = discover_vllm_capabilities(base_url).to_dict()
        client = VLLMClient(
            base_url,
            api_dialect="chat_completions",
            timeout_seconds=120,
        )
        smoke = client.capture(
            {
                "model": PHASE2_MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "Reply with exactly OK."}
                ],
                "temperature": 0.0,
                "max_tokens": 4,
                "seed": 2601,
            }
        ).to_dict()
        manifest = ExperimentManifest.from_dict(
            {
                "schema_version": "0.3",
                "experiment_id": "phase2-standard-vllm-acceptance",
                "backend": {
                    "kind": "vllm",
                    "api_dialect": "chat_completions",
                    "base_url": base_url,
                    "model": PHASE2_MODEL_NAME,
                    "model_revision": PHASE2_MODEL_REVISION,
                    "tokenizer_revision": PHASE2_MODEL_REVISION,
                    "chat_template_digest": "observed-separately",
                    "backend_version": VLLM_VERSION,
                    "hardware": "1x NVIDIA A10 24GB on Modal",
                    "block_size": BLOCK_SIZE,
                    "engine_config": _observed_engine_config(command),
                },
                "sampling": {
                    "temperature": 0.0,
                    "max_tokens": 4,
                    "seed": 2601,
                },
                "execution": {
                    "repetitions": 2,
                    "warmup_repetitions": 0,
                    "order": "alternating",
                    "seed": 2601,
                    "cache_treatment": "cold",
                    "cache_isolation": "salt_isolated",
                    "arrival_pattern": "sequential",
                    "require_evaluation": True,
                },
                "acceptance": {
                    "primary_metric": "total_latency_ms",
                    "optimization_target": "cache_reuse",
                    "max_task_success_regression": 0.5,
                    "min_task_success_rate": 0.0,
                    "task_success_method": "paired_exact",
                    "min_completed_rate": 1.0,
                    "min_cached_token_gain": 0.0,
                    "require_latency_improvement": False,
                },
                "design": {
                    "purpose": "Phase 2 compatibility and isolation acceptance only",
                    "claim_boundary": "No performance or product claim",
                },
                "constants": {},
                "cases": [
                    {
                        "case_id": "reply-ok",
                        "evaluator": {"kind": "regex", "pattern": "^OK[.!]?$"},
                        "variants": {
                            "baseline": {
                                "request": {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "Reply with exactly OK.",
                                        }
                                    ]
                                }
                            },
                            "candidate": {
                                "request": {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "Reply with exactly OK.",
                                        }
                                    ]
                                }
                            },
                        },
                    },
                    {
                        "case_id": "reply-yes",
                        "evaluator": {"kind": "regex", "pattern": "^YES[.!]?$"},
                        "variants": {
                            "baseline": {
                                "request": {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "Reply with exactly YES.",
                                        }
                                    ]
                                }
                            },
                            "candidate": {
                                "request": {
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": "Reply with exactly YES.",
                                        }
                                    ]
                                }
                            },
                        },
                    },
                ],
            }
        )
        result = run_experiment(manifest, client)
        evidence = {
            "validation": "phase2-standard-vllm",
            "elapsed_seconds": time.monotonic() - started,
            "command": command,
            "development_mode": False,
            "capabilities": capabilities,
            "smoke_capture": smoke,
            "experiment": result,
        }
        return gzip.compress(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(),
            compresslevel=9,
        )
    finally:
        _stop_process(process)


def _resolved_payload(manifest_path: Path, workflow_limit: int) -> dict[str, Any]:
    from runeconomy.experiments import ExperimentManifest

    manifest = ExperimentManifest.from_path(manifest_path)
    if workflow_limit:
        workflow_ids = list(
            dict.fromkeys(case.workflow_id for case in manifest.cases)
        )
        if workflow_limit < 1 or workflow_limit > len(workflow_ids):
            raise ValueError(
                f"workflow_limit must be between 1 and {len(workflow_ids)}"
            )
        included = set(workflow_ids[:workflow_limit])
        design = {
            **manifest.design,
            "pilot_subset": {
                "workflow_limit": workflow_limit,
                "selection": "first predeclared workflow IDs",
                "eligible_for_final_claim": False,
            },
        }
        manifest = replace(
            manifest,
            design=design,
            cases=tuple(
                case for case in manifest.cases if case.workflow_id in included
            ),
        )
    payload = manifest.to_resolved_dict()
    payload["backend"]["base_url"] = f"http://127.0.0.1:{PORT}"
    payload["backend"]["api_key_env"] = None
    return payload


@app.local_entrypoint()
def prepare() -> None:
    """Cache weights and verify the serving image without a GPU allocation."""

    path = download_model.remote()
    print(f"Pinned model cached at {path}")
    version = verify_image.remote()
    print(f"Verified required benchmark capabilities in vLLM {version}")


@app.local_entrypoint()
def pilot(
    manifest: str = "examples/experiments/repository-agent-tool-pruning-heldout.json",
    output: str = "reports/repository-agent-tool-pruning-heldout-pilot.json",
    workflow_limit: int = 2,
) -> None:
    """Run a bounded pilot or the full predeclared workload when limit is zero."""

    manifest_path = Path(manifest).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output_path}")
    payload = _resolved_payload(manifest_path, workflow_limit)
    _validate_manifest_identity(payload)
    compressed = run_collocated.remote(payload)
    raw = gzip.decompress(compressed)
    # Validate before making the evidence visible at its final path.
    result = json.loads(raw)
    if not isinstance(result, dict) or "decision" not in result:
        raise RuntimeError("remote benchmark returned an invalid result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(raw + b"\n")
    temporary.replace(output_path)
    print(f"wrote {result['decision']['status']} experiment result to {output_path}")


@app.local_entrypoint()
def phase2(
    output: str = "reports/phase2-standard-vllm-acceptance.json",
) -> None:
    """Run the approved bounded Phase 2 acceptance and save raw evidence."""

    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {output_path}")
    compressed = run_phase2_acceptance.remote()
    raw = gzip.decompress(compressed)
    evidence = json.loads(raw)
    capabilities = evidence.get("capabilities", {})
    if not isinstance(capabilities, dict) or "experiment_modes" not in capabilities:
        raise RuntimeError("remote Phase 2 acceptance returned invalid evidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(raw + b"\n")
    temporary.replace(output_path)
    print(
        "wrote Phase 2 acceptance evidence to "
        f"{output_path} ({evidence['elapsed_seconds']:.1f}s remote elapsed)"
    )
