"""Private, cost-bounded vLLM benchmark server on Modal.

The server uses Modal's proxy authentication (enabled by default). Never set
``unauthenticated=True`` while vLLM development endpoints are enabled.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request

import modal

from runeconomy.backends import DEFAULT_BENCHMARK_FLAGS, missing_vllm_flags

APP_NAME = "runeconomy-vllm"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
VLLM_VERSION = "0.33.0"
TRANSFORMERS_VERSION = "5.16.1"
HUGGINGFACE_HUB_VERSION = "1.31.0"

GPU = "A10"
PORT = 8000
BLOCK_SIZE = 16
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85

HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"

app = modal.App(APP_NAME)

# Version 2 is selected explicitly because Modal still defaults to v1 when the
# version is omitted. These pre-RunEconomy volume identifiers are intentionally
# retained so the brand change does not discard downloaded model and compilation
# caches.
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

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
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
)


@app.function(
    image=download_image,
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=30 * 60,
    cpu=2,
    memory=4096,
)
def download_model() -> str:
    """Download pinned weights using CPU resources before allocating a GPU."""

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=HF_CACHE_PATH,
    )
    hf_cache.commit()
    return path


@app.function(image=vllm_image, timeout=5 * 60, cpu=2, memory=4096)
def verify_vllm_image() -> str:
    """Fail on CPU if the pinned release lacks required serving flags."""

    from importlib import metadata

    version = metadata.version("vllm")
    package_files = metadata.files("vllm") or []
    source_text = "\n".join(
        path.locate().read_text(encoding="utf-8", errors="ignore")
        for path in package_files
        if path.suffix == ".py" and path.locate().is_file()
    )
    missing = missing_vllm_flags(source_text, DEFAULT_BENCHMARK_FLAGS)
    if missing:
        raise RuntimeError(
            f"{version} lacks required flags: {', '.join(missing)}"
        )
    return version


def vllm_command() -> list[str]:
    """Return the fully pinned serving command recorded by the manifest."""

    return [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
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
        "8",
        "--enable-prefix-caching",
        "--block-size",
        str(BLOCK_SIZE),
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        "--enable-scale-out",
    ]


@app.server(
    image=vllm_image,
    gpu=GPU,
    volumes={
        HF_CACHE_PATH: hf_cache,
        VLLM_CACHE_PATH: vllm_cache,
        FLASHINFER_CACHE_PATH: flashinfer_cache,
    },
    port=PORT,
    # Private Modal proxy authentication remains enabled.
    # Modal Servers reject requests at zero replicas instead of cold-starting.
    # Keep exactly one replica only for a validation window, then stop the app.
    max_containers=1,
    min_containers=1,
    target_concurrency=1,
    scaledown_window=60,
    startup_timeout=10 * 60,
    exit_grace_period=30,
    routing_region="us-east",
)
class VLLMServer:
    process: subprocess.Popen[bytes]

    @modal.enter()
    def start(self) -> None:
        command = vllm_command()
        print("Starting:", *command, flush=True)
        self.process = subprocess.Popen(command)

        deadline = time.monotonic() + 9 * 60
        health_url = f"http://127.0.0.1:{PORT}/health"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {self.process.returncode}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        print("vLLM is healthy", flush=True)
                        return
            except OSError:
                time.sleep(2)
        raise TimeoutError("vLLM did not become healthy before the startup deadline")

    @modal.exit()
    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


@app.local_entrypoint()
def prepare() -> None:
    """Preload the model without allocating a GPU."""

    path = download_model.remote()
    print(f"Pinned model cached at {path}")
    version = verify_vllm_image.remote()
    print(f"Verified required CLI flags in {version}")
