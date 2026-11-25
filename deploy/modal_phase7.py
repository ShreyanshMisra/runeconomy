"""Cost-bounded Modal runner for Phase 6/7 bundles.

A bundle (``scripts/prepare_phase7.py``) carries frozen or calibration
workspaces, the lab agent, the frozen repository snapshot, the task splits,
and ``plan.json``. For each stage this runner:

1. starts the declared vLLM servers on their GPUs and waits for health
2. verifies quantization from the startup logs, the vLLM version, and the
   chat-template digest (a CPU preflight first confirms this vLLM accepts
   every launch flag, before any GPU is requested)
3. warms the structured-output paths, checks that responses carry cached
   prompt tokens and server TTFT, and resets every prefix cache
4. runs each workspace with ``runeconomy experiment run``, adding
   ``--require-frozen`` for frozen workspaces and sampling every server's
   ``/metrics``
5. resets the prefix caches between runs

Every frozen digest is enforced by RunEconomy itself. A drifted workspace
refuses to run. The runner returns a private archive with results, server
logs and a timing/cost record, and it restores nothing it did not start.

    uv run modal run deploy/modal_phase7.py --bundle artifacts/phase-7-bundles/calibration \\
        --output artifacts/phase-7-calibration.tar.gz --authorize-usd 15
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

import modal


APP_NAME = "runeconomy-phase-7"
VLLM_VERSION = "0.34.0"
TRANSFORMERS_VERSION = "5.18.0"
HUGGINGFACE_HUB_VERSION = "2.0.0"
GPU = "L40S:2"
CPU = 16
MEMORY_MB = 65536
TIMEOUT_SECONDS = 150 * 60
CHAT_TEMPLATE_DIGEST = "sha256:cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
MODELS = {
    "14b": ("Qwen/Qwen2.5-14B-Instruct", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"),
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
}
# Published Modal rates used for the post-run estimate (USD per second).
RATES = {"l40s_gpu": 1.95 / 3600, "cpu_core": 0.0473 / 3600, "memory_gib": 0.008 / 3600}
FIRST_PORT = 8000
BLOCK_SIZE = 16
# Every optional flag path in _command, for the CPU preflight.
PREFLIGHT_SPEC = {
    "model": "14b", "served_model": "preflight", "max_model_len": 8192,
    "gpu_memory_utilization": 0.9, "max_num_seqs": 8, "quantization": "fp8_per_tensor",
}
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
FLASHINFER_CACHE_PATH = "/root/.cache/flashinfer"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("agent-inference-doctor-hf-cache", create_if_missing=True, version=2)
vllm_cache = modal.Volume.from_name("agent-inference-doctor-vllm-cache", create_if_missing=True, version=2)
flashinfer_cache = modal.Volume.from_name("agent-inference-doctor-flashinfer-cache", create_if_missing=True, version=2)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
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
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
    .add_local_python_source("runeconomy")
)


def projected_cost_usd(seconds: float) -> float:
    return seconds * (
        2 * RATES["l40s_gpu"] + CPU * RATES["cpu_core"] + (MEMORY_MB / 1024) * RATES["memory_gib"]
    )


def _snapshot(model: str) -> Path:
    from huggingface_hub import snapshot_download

    repo, revision = MODELS[model]
    try:
        path = snapshot_download(repo_id=repo, revision=revision, cache_dir=HF_CACHE_PATH, local_files_only=True)
    except Exception:  # not cached yet: fetch the pinned revision once
        path = snapshot_download(repo_id=repo, revision=revision, cache_dir=HF_CACHE_PATH)
        hf_cache.commit()
    snapshot = Path(path)
    if snapshot.name != revision:
        raise RuntimeError(f"{repo} resolved to a revision other than the pin")
    template = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template")
    if "sha256:" + hashlib.sha256(str(template).encode()).hexdigest() != CHAT_TEMPLATE_DIGEST:
        raise RuntimeError(f"{repo} chat-template identity differs from the pin")
    return snapshot


def _command(spec: dict[str, Any], port: int) -> list[str]:
    repo, revision = MODELS[spec["model"]]
    command = [
        "vllm", "serve", repo, "--revision", revision,
        "--served-model-name", spec["served_model"],
        "--dtype", "bfloat16",
        "--max-model-len", str(spec["max_model_len"]),
        "--gpu-memory-utilization", str(spec["gpu_memory_utilization"]),
        "--max-num-seqs", str(spec["max_num_seqs"]),
        "--enable-prefix-caching",
        "--block-size", str(BLOCK_SIZE),
        # Per-request usage details (cached prompt tokens) and per-request
        # metrics (server TTFT, queue time) on every response. Static
        # replays read both; without them those measurements are unknown.
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
        # Request logging is off by default in vLLM 0.34 (the earlier
        # --disable-log-requests was a no-op absent from its non-default args).
        "--port", str(port),
    ]
    if spec.get("quantization"):
        command += ["--quantization", spec["quantization"]]
    return command


@app.function(image=image, cpu=2, memory=8192, timeout=15 * 60)
def preflight_flags() -> dict[str, Any]:
    """Report every server flag the runner would pass that this vLLM lacks.

    Runs on CPU before any GPU is requested, so a renamed or removed flag
    fails in seconds instead of after a paid model load. ``vllm serve
    --help`` cannot build its parser without a device, so this reads the
    installed package instead: vLLM derives each CLI flag from a config
    field (``--max-num-seqs`` from ``max_num_seqs``) or declares it literally.
    """

    import vllm

    source = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in Path(vllm.__file__).parent.rglob("*.py")
    )
    flags = sorted({token for token in _command({**PREFLIGHT_SPEC}, FIRST_PORT) if token.startswith("--")})
    missing = [
        flag for flag in flags
        # A literal declaration or alias in either quote style, a deprecation
        # notice that still accepts it, or the config field it derives from.
        if flag not in source and f"{flag[2:].replace('-', '_')}:" not in source
    ]
    return {"missing": missing}


def _wait_healthy(name: str, process: subprocess.Popen[bytes], port: int, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server {name} exited during startup ({process.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"server {name} did not become healthy")


def _verify_log(name: str, spec: dict[str, Any], log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="replace")
    quantization = spec.get("quantization")
    if quantization and f"quantization={quantization}" not in text:
        raise RuntimeError(f"server {name} did not start with {quantization}")
    if not quantization and "quantization=None" not in text:
        raise RuntimeError(f"server {name} did not start unquantized")


def _warm_and_reset(name: str, spec: dict[str, Any], port: int) -> dict[str, Any]:
    from runeconomy.vllm import VLLMClient

    client = VLLMClient(f"http://127.0.0.1:{port}", api_dialect="chat_completions", timeout_seconds=180)
    schemas = {
        "lab_search_plan": {"type": "object", "properties": {"query": {"type": "string", "enum": ["OK"]}},
                            "required": ["query"], "additionalProperties": False},
        "lab_answer": {"type": "object", "properties": {"answer": {"type": "string", "enum": ["OK", "NO"]}},
                       "required": ["answer"], "additionalProperties": False},
        "verify_line": {"type": "object", "properties": {"verified": {"type": "boolean"}},
                        "required": ["verified"], "additionalProperties": False},
    }
    captured = []
    for schema_name, schema in schemas.items():
        captured.append(client.capture({
            "model": spec["served_model"],
            "messages": [{"role": "user", "content": "Return a JSON object."}],
            "temperature": 0, "max_tokens": 16, "cache_salt": "runeconomy-phase-7-warmup",
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        }))
    client.reset_prefix_cache()
    # The launch flags promise these per-response fields; check that the
    # server delivers them before any experiment depends on them.
    telemetry = {
        "cached_tokens_reported": all(item.cached_tokens is not None for item in captured),
        "server_ttft_reported": all(item.server_ttft_ms is not None for item in captured),
    }
    if not all(telemetry.values()):
        raise RuntimeError(f"server {name} omitted per-request telemetry: {telemetry}")
    return {"server": name, "warmup_requests": len(schemas), "prefix_cache_reset": True,
            "response_telemetry": telemetry}


def _reset_all(servers: dict[str, tuple[dict[str, Any], int]]) -> None:
    from runeconomy.vllm import VLLMClient

    for spec, port in servers.values():
        VLLMClient(f"http://127.0.0.1:{port}", api_dialect="chat_completions").reset_prefix_cache()


def _gpu_inventory() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    rows = [line.split(", ") for line in completed.stdout.strip().splitlines()]
    return [{"index": row[0], "name": row[1], "memory": row[2]} for row in rows]


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={HF_CACHE_PATH: hf_cache, VLLM_CACHE_PATH: vllm_cache, FLASHINFER_CACHE_PATH: flashinfer_cache},
)
def run_bundle(bundle_archive: bytes, only: list[str] | None = None) -> bytes:
    started = time.monotonic()
    if metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError("vLLM version differs from the pin")
    root = Path(tempfile.mkdtemp(prefix="runeconomy-phase-7-"))
    with tarfile.open(fileobj=io.BytesIO(bundle_archive), mode="r:gz") as archive:
        archive.extractall(root, filter="data")
    bundle = root / "bundle"
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    results = bundle / "results"
    results.mkdir(exist_ok=True)
    record: dict[str, Any] = {
        "window": plan["window"], "gpu": GPU, "gpu_inventory": _gpu_inventory(),
        "vllm_version": metadata.version("vllm"), "chat_template_digest": CHAT_TEMPLATE_DIGEST,
        "stages": [],
    }
    for model in {spec["model"] for stage in plan["stages"] for spec in stage["servers"].values()}:
        _snapshot(model)
    for stage in plan["stages"]:
        runs = [run for run in stage["runs"] if only is None or run["experiment"] in only]
        if not runs:
            continue
        stage_started = time.monotonic()
        processes: list[subprocess.Popen[bytes]] = []
        servers: dict[str, tuple[dict[str, Any], int]] = {}
        stage_record: dict[str, Any] = {"name": stage["name"], "servers": {}, "runs": []}
        try:
            for index, (name, spec) in enumerate(sorted(stage["servers"].items())):
                port = FIRST_PORT + index
                log = results / f"{stage['name']}-{name}-vllm.log"
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(spec["gpu"])}
                processes.append(subprocess.Popen(_command(spec, port), env=environment,
                                                  stdout=log.open("wb"), stderr=subprocess.STDOUT))
                servers[name] = (spec, port)
            for (name, (spec, port)), process in zip(sorted(servers.items()), processes):
                _wait_healthy(name, process, port)
                _verify_log(name, spec, results / f"{stage['name']}-{name}-vllm.log")
                stage_record["servers"][name] = {**spec, "startup_verified": True,
                                                 "command": _command(spec, port),
                                                 "warmup": _warm_and_reset(name, spec, port)}
            stage_record["servers_ready_seconds"] = time.monotonic() - stage_started
            for name, (spec, port) in sorted(servers.items()):
                # Record exposed metric names only (no values or labels) so the
                # telemetry allowlist can be checked against this vLLM version.
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
                    names = sorted({
                        line.split("{")[0].split(" ")[0]
                        for line in response.read().decode("utf-8", "replace").splitlines()
                        if line and not line.startswith("#")
                    })
                stage_record["servers"][name]["metric_names"] = [n for n in names if n.startswith("vllm:")]
            for run in runs:
                experiment = plan["experiments"][run["experiment"]]
                environment = {
                    **os.environ,
                    "RUNECONOMY_REPO_ROOT": str(bundle / "repo"),
                    # Modal mounts the local runeconomy package under /root.
                    "PYTHONPATH": os.pathsep.join(filter(None, ["/root", os.environ.get("PYTHONPATH")])),
                }
                for prefix, server in run["variants"].items():
                    spec, port = servers[server]
                    environment[f"{prefix}_URL"] = f"http://127.0.0.1:{port}"
                    environment[f"{prefix}_MODEL"] = spec["served_model"]
                command = [
                    sys.executable, "-m", "runeconomy", "experiment", "run", experiment["workspace"],
                    "--output", f"results/{run['experiment']}", "--json",
                ]
                if experiment.get("frozen"):
                    command.append("--require-frozen")
                if run.get("server_metrics"):
                    command.append("--server-metrics")
                for target, server in (run.get("metrics") or {}).items():
                    command += ["--server-metrics-url", f"{target}=http://127.0.0.1:{servers[server][1]}/metrics"]
                run_started = time.monotonic()
                completed = subprocess.run(command, cwd=bundle, env=environment, capture_output=True, text=True)
                (results / f"{run['experiment']}-run.json").write_text(completed.stdout, encoding="utf-8")
                (results / f"{run['experiment']}-run.stderr").write_text(completed.stderr[-20000:], encoding="utf-8")
                stage_record["runs"].append({
                    "experiment": run["experiment"], "exit_code": completed.returncode,
                    "seconds": time.monotonic() - run_started,
                })
                _reset_all(servers)
        finally:
            for process in processes:
                _stop(process)
            stage_record["seconds"] = time.monotonic() - stage_started
            record["stages"].append(stage_record)
    elapsed = time.monotonic() - started
    record["remote_elapsed_seconds"] = elapsed
    record["projected_cost_usd"] = projected_cost_usd(elapsed)
    (results / "window-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(results, arcname="results")
    return buffer.getvalue()


@app.local_entrypoint()
def main(bundle: str, output: str, authorize_usd: float, only: str = "") -> None:
    ceiling = projected_cost_usd(TIMEOUT_SECONDS)
    if authorize_usd < ceiling:
        raise SystemExit(
            f"the worst-case cost at the {TIMEOUT_SECONDS // 60}-minute timeout is ${ceiling:.2f}; "
            f"--authorize-usd {authorize_usd:.2f} is below it"
        )
    destination = Path(output)
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    source = Path(bundle)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(source, arcname="bundle")
    print(f"RunEconomy Phase 7 · bundle {len(buffer.getvalue()) / 1e6:.1f} MB · worst case ${ceiling:.2f}")
    preflight = preflight_flags.remote()
    if preflight["missing"]:
        raise SystemExit(f"vLLM {VLLM_VERSION} does not accept {', '.join(preflight['missing'])}; no GPU was started")
    print(f"RunEconomy Phase 7 · vLLM {VLLM_VERSION} accepts every server flag")
    archive_bytes = run_bundle.remote(buffer.getvalue(), [item for item in only.split(",") if item] or None)
    destination.write_bytes(archive_bytes)
    print(f"RunEconomy Phase 7 · wrote {destination} (sha256:{hashlib.sha256(archive_bytes).hexdigest()})")
