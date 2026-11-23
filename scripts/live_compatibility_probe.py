"""Run one authenticated, scale-to-zero vLLM compatibility validation.

The persisted artifact deliberately excludes the Modal endpoint, Server ID,
credentials, response text, and response IDs. It records only the pinned public
combination, capability outcomes, bounded timings, and content digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from runeconomy.experiments import ExperimentManifest, run_experiment
from runeconomy.vllm import CapturedResponse
try:
    from scripts.modal_validation_window import (
        autoscaler_minimum,
        resolve_server,
        set_autoscaler_minimum,
        wait_until_healthy,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    from modal_validation_window import (  # type: ignore[no-redef]
        autoscaler_minimum,
        resolve_server,
        set_autoscaler_minimum,
        wait_until_healthy,
    )


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _token_ids(payload: Mapping[str, Any]) -> tuple[int, ...]:
    tokens = payload.get("token_ids", payload.get("prompt_token_ids"))
    engine_input = payload.get("engine_input")
    if tokens is None and isinstance(engine_input, Mapping):
        tokens = engine_input.get("prompt_token_ids")
    if not isinstance(tokens, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in tokens
    ):
        raise RuntimeError("render response did not contain integer token IDs")
    return tuple(tokens)


def _sse_events(lines: Iterator[str]) -> Iterator[tuple[float, dict[str, Any]]]:
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                encoded = "\n".join(data)
                data.clear()
                if encoded != "[DONE]":
                    payload = json.loads(encoded)
                    if isinstance(payload, dict):
                        yield time.monotonic(), payload
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        encoded = "\n".join(data)
        if encoded != "[DONE]":
            payload = json.loads(encoded)
            if isinstance(payload, dict):
                yield time.monotonic(), payload


class ModalCurlClient:
    """Small authenticated client backed by Modal's credential-aware curl CLI."""

    def __init__(
        self,
        url: str,
        *,
        model: str,
        identity: dict[str, Any],
        timeout_seconds: int = 180,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.identity = identity
        self.timeout_seconds = timeout_seconds

    def _command(self, path: str, payload: Mapping[str, Any] | None) -> list[str]:
        command = [
            "uv",
            "run",
            "modal",
            "curl",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--max-time",
            str(self.timeout_seconds),
        ]
        if payload is not None:
            command.extend(
                [
                    "-H",
                    "Content-Type: application/json",
                    "-X",
                    "POST",
                    "--data-binary",
                    json.dumps(payload, separators=(",", ":")),
                ]
            )
        command.append(self.url + path)
        return command

    def json(
        self, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        completed = subprocess.run(
            self._command(path, payload),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"authenticated request to {path} failed")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"authenticated request to {path} returned malformed JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"authenticated request to {path} returned non-object JSON")
        return result

    def stream(
        self, path: str, payload: Mapping[str, Any]
    ) -> tuple[float, float | None, list[dict[str, Any]]]:
        command = self._command(path, payload)
        command.insert(-1, "--no-buffer")
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        events: list[dict[str, Any]] = []
        first_output_at: float | None = None
        try:
            for observed_at, event in _sse_events(iter(process.stdout.readline, "")):
                events.append(event)
                event_type = event.get("type")
                choices = event.get("choices")
                if first_output_at is None and (
                    event_type in {"response.output_text.delta", "response.function_call_arguments.delta"}
                    or (
                        isinstance(choices, list)
                        and any(
                            isinstance(choice, Mapping)
                            and isinstance(choice.get("delta"), Mapping)
                            and bool(choice["delta"].get("content"))
                            for choice in choices
                        )
                    )
                ):
                    first_output_at = observed_at
            stderr = process.stderr.read() if process.stderr is not None else ""
            returncode = process.wait(timeout=self.timeout_seconds)
            if returncode != 0:
                raise RuntimeError(
                    f"authenticated stream to {path} failed: {stderr[:160]}"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        ended = time.monotonic()
        first_ms = None if first_output_at is None else (first_output_at - started) * 1000
        return (ended - started) * 1000, first_ms, events

    def prompt_tokens(self, request: Mapping[str, Any]) -> tuple[int, ...]:
        payload = dict(request)
        payload.pop("stream", None)
        return _token_ids(self.json("/v1/chat/completions/render", payload))

    def capture(self, request_body: dict[str, Any]) -> CapturedResponse:
        prompt_tokens = self.prompt_tokens(request_body)
        payload = dict(request_body)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        latency, client_ttft, events = self.stream("/v1/chat/completions", payload)
        usage: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        pieces: list[str] = []
        status = None
        for event in events:
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if isinstance(event.get("metrics"), dict):
                metrics = event["metrics"]
            choices = event.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                    pieces.append(delta["content"])
                if isinstance(choice.get("finish_reason"), str):
                    status = "completed"
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, Mapping) else {}
        output_text = "".join(pieces)
        return CapturedResponse(
            prompt_token_ids=prompt_tokens,
            input_tokens=_integer(usage.get("prompt_tokens")),
            output_tokens=_integer(usage.get("completion_tokens")),
            cached_tokens=_integer(details.get("cached_tokens")),
            client_ttft_ms=client_ttft,
            total_latency_ms=latency,
            server_ttft_ms=_number(metrics.get("time_to_first_token_ms")),
            server_queue_ms=_number(metrics.get("queue_time_ms")),
            generation_time_ms=_number(metrics.get("generation_time_ms")),
            mean_itl_ms=_number(metrics.get("mean_itl_ms")),
            tokens_per_second=_number(metrics.get("tokens_per_second")),
            response_id=None,
            status=status,
            output_digest=_digest(events),
            output_text=output_text,
        )

    def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
        suffix = "?reset_external=true" if reset_external else ""
        response = self.json("/reset_prefix_cache" + suffix, {})
        if response.get("success") is not True:
            raise RuntimeError("vLLM did not confirm prefix-cache reset")

    def environment_identity(self) -> dict[str, Any]:
        return dict(self.identity)


def _nonstream_checks(client: ModalCurlClient) -> dict[str, Any]:
    repeated = "stable-prefix " * 256
    chat = {
        "model": client.model,
        "messages": [{"role": "user", "content": repeated + "Reply only OK."}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 2301,
    }
    client.reset_prefix_cache()
    first = client.json("/v1/chat/completions", chat)
    second = client.json("/v1/chat/completions", chat)
    usage = second.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    cached = _integer(details.get("cached_tokens"))
    metrics = second.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}

    responses_payload = {
        "model": client.model,
        "instructions": "Reply only OK.",
        "input": "OK",
        "temperature": 0.0,
        "max_output_tokens": 4,
        "seed": 2302,
    }
    response = client.json("/v1/responses", responses_payload)
    return {
        "chat_completions_nonstream": bool(first.get("choices")),
        "responses_nonstream": response.get("object") == "response",
        "cached_tokens_reported": cached is not None,
        "cached_tokens_positive_on_repeat": isinstance(cached, int) and cached > 0,
        "server_ttft_reported": _number(metrics.get("time_to_first_token_ms")) is not None,
        "chat_response_digest": _digest(first.get("choices")),
        "responses_response_digest": _digest(response.get("output")),
    }


def _stream_checks(client: ModalCurlClient) -> dict[str, Any]:
    chat_payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": "Reply only OK."}],
        "temperature": 0.0,
        "max_tokens": 4,
        "seed": 2303,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chat_latency, chat_ttft, chat_events = client.stream(
        "/v1/chat/completions", chat_payload
    )
    responses_payload = {
        "model": client.model,
        "instructions": "Reply only OK.",
        "input": "OK",
        "temperature": 0.0,
        "max_output_tokens": 4,
        "seed": 2304,
        "stream": True,
    }
    response_latency, response_ttft, response_events = client.stream(
        "/v1/responses", responses_payload
    )
    return {
        "chat_completions_stream": any(event.get("choices") for event in chat_events),
        "responses_stream": any(
            event.get("type") in {"response.completed", "response.incomplete"}
            for event in response_events
        ),
        "chat_client_ttft_ms": chat_ttft,
        "chat_total_latency_ms": chat_latency,
        "responses_client_ttft_ms": response_ttft,
        "responses_total_latency_ms": response_latency,
    }


def _checkpoint_resume(
    client: ModalCurlClient,
    *,
    checkpoint_path: Path,
    version: str,
    model_revision: str,
    gpu: str,
    chat_template_digest: str,
) -> dict[str, Any]:
    manifest = ExperimentManifest.from_dict(
        {
            "schema_version": "0.3",
            "experiment_id": "live-compatibility-checkpoint",
            "backend": {
                "kind": "vllm",
                "api_dialect": "chat_completions",
                "base_url": "authenticated-modal-endpoint",
                "model": client.model,
                "model_revision": model_revision,
                "tokenizer_revision": model_revision,
                "chat_template_digest": chat_template_digest,
                "backend_version": version,
                "hardware": f"1x NVIDIA {gpu} on Modal",
                "block_size": 16,
                "engine_config": {
                    "enable_prefix_caching": True,
                    "dtype": "bfloat16",
                    "tensor_parallel_size": 1,
                },
            },
            "sampling": {"temperature": 0.0, "max_tokens": 4, "seed": 2305},
            "execution": {
                "repetitions": 1,
                "warmup_repetitions": 0,
                "order": "alternating",
                "seed": 2305,
                "cache_treatment": "cold",
                "cache_isolation": "salt_isolated",
                "arrival_pattern": "sequential",
                "require_evaluation": True,
            },
            "acceptance": {
                "primary_metric": "total_latency_ms",
                "optimization_target": "cache_reuse",
                "max_task_success_regression": 1.0,
                "min_task_success_rate": 0.0,
                "task_success_method": "paired_exact",
                "min_completed_rate": 1.0,
                "min_cached_token_gain": 0.0,
                "require_latency_improvement": False,
                "require_environment_identity": True,
            },
            "design": {
                "purpose": "live checkpoint and resume compatibility only",
                "claim_boundary": "no performance claim",
            },
            "constants": {},
            "cases": [
                {
                    "case_id": "reply-ok",
                    "evaluator": {"kind": "regex", "pattern": "(?s).+"},
                    "variants": {
                        "baseline": {
                            "request": {
                                "messages": [
                                    {"role": "user", "content": "Reply only OK."}
                                ]
                            }
                        },
                        "candidate": {
                            "request": {
                                "messages": [
                                    {"role": "user", "content": "Reply only OK."}
                                ]
                            }
                        },
                    },
                }
            ],
        }
    )

    class Interrupted:
        def __init__(self) -> None:
            self.calls = 0

        def capture(self, request: dict[str, Any]) -> CapturedResponse:
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt("planned compatibility interruption")
            return client.capture(request)

        def reset_prefix_cache(self, *, reset_external: bool = False) -> None:
            client.reset_prefix_cache(reset_external=reset_external)

        def environment_identity(self) -> dict[str, Any]:
            return client.environment_identity()

    interrupted = False
    try:
        run_experiment(manifest, Interrupted(), checkpoint_path=checkpoint_path)
    except KeyboardInterrupt:
        interrupted = True
    resumed = run_experiment(
        manifest,
        client,
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    return {
        "planned_interruption_observed": interrupted,
        "checkpoint_record_count": len(checkpoint_path.read_text().splitlines()) - 1,
        "resumed_trial_count": len(resumed["trials"]),
        "resumed_decision": resumed["decision"]["status"],
        "environment_identity_verified": resumed["provenance"]["environment_identity"][
            "verified"
        ],
    }


def run_probe(
    *,
    app_name: str,
    output_path: Path,
    checkpoint_path: Path,
    expected_version: str,
    model: str,
    model_revision: str,
    gpu: str,
    chat_template_digest: str,
    runeconomy_commit: str,
) -> dict[str, Any]:
    if output_path.exists() or checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite compatibility evidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    server_id, url = resolve_server(app_name)
    if url is None:
        raise RuntimeError("Modal Server discovery returned no endpoint")
    if autoscaler_minimum(server_id) != 0:
        raise RuntimeError("compatibility window must begin at zero warm containers")
    opened = False
    window_started = time.monotonic()
    try:
        opened = True
        set_autoscaler_minimum(server_id, 1)
        health_started = time.monotonic()
        wait_until_healthy(
            url,
            timeout_seconds=12 * 60,
            interval_seconds=5.0,
        )
        cold_start_seconds = time.monotonic() - health_started
        info = subprocess.run(
            ["uv", "run", "modal", "server", "info", server_id, "--json"],
            text=True,
            capture_output=True,
            check=True,
        )
        server = json.loads(info.stdout)
        if server.get("unauthenticated") is not False:
            raise RuntimeError("compatibility endpoint is not authenticated")
        observed_gpus = server.get("gpus")
        client = ModalCurlClient(
            url,
            model=model,
            identity={},
        )
        version_payload = client.json("/version")
        observed_version = version_payload.get("version")
        if observed_version != expected_version:
            raise RuntimeError(
                f"expected vLLM {expected_version}, observed {observed_version!r}"
            )
        models = client.json("/v1/models").get("data")
        if not isinstance(models, list) or len(models) != 1 or models[0].get("id") != model:
            raise RuntimeError("served model identity did not match the pinned model")
        if not chat_template_digest.startswith("sha256:"):
            raise ValueError("chat template digest must be a sha256 value")
        identity = {
            "model": model,
            "model_revision": model_revision,
            "tokenizer_revision": model_revision,
            "chat_template_digest": chat_template_digest,
            "backend_version": observed_version,
            "engine_config": {
                "enable_prefix_caching": True,
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
            },
        }
        client.identity = identity
        openapi = client.json("/openapi.json")
        paths = openapi.get("paths")
        paths = paths if isinstance(paths, Mapping) else {}
        required_paths = {
            "/v1/chat/completions",
            "/v1/responses",
            "/tokenize",
            "/v1/chat/completions/render",
            "/v1/responses/render",
            "/reset_prefix_cache",
        }
        missing_paths = sorted(required_paths - set(paths))
        if missing_paths:
            raise RuntimeError(f"required API paths are absent: {missing_paths}")
        sample_chat = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply only OK."}],
        }
        tokenized = client.json("/tokenize", sample_chat)
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError("tokenize did not return tokens")
        chat_render = _token_ids(client.json("/v1/chat/completions/render", sample_chat))
        response_render = _token_ids(
            client.json(
                "/v1/responses/render",
                {"model": model, "instructions": "Reply only OK.", "input": "OK"},
            )
        )
        nonstream = _nonstream_checks(client)
        streaming = _stream_checks(client)
        checkpoint = _checkpoint_resume(
            client,
            checkpoint_path=checkpoint_path,
            version=expected_version,
            model_revision=model_revision,
            gpu=gpu,
            chat_template_digest=chat_template_digest,
        )
        checks = {
            "authenticated_endpoint": True,
            "server_id_discovered": True,
            "backend_version": observed_version == expected_version,
            "model_identity": True,
            "tokenize": True,
            "chat_prompt_render": bool(chat_render),
            "responses_prompt_render": bool(response_render),
            "prefix_cache_reset": True,
            **nonstream,
            **{
                key: value
                for key, value in streaming.items()
                if key in {"chat_completions_stream", "responses_stream"}
            },
            "checkpoint_resume": (
                checkpoint["planned_interruption_observed"]
                and checkpoint["resumed_trial_count"] == 2
                and checkpoint["environment_identity_verified"]
            ),
        }
        required_boolean_checks = {
            key: value for key, value in checks.items() if isinstance(value, bool)
        }
        status = "passed" if all(required_boolean_checks.values()) else "limited"
        result = {
            "schema_version": "0.1",
            "tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "combination": {
                "runeconomy_commit": runeconomy_commit,
                "vllm_version": observed_version,
                "model": model,
                "model_revision": model_revision,
                "gpu": gpu,
                "api_surface": sorted(required_paths),
                "engine_config": identity["engine_config"],
                "chat_template_digest": chat_template_digest,
            },
            "deployment_evidence": {
                "app_state": "deployed",
                "server_id_discovered": True,
                "authenticated": True,
                "observed_gpus": observed_gpus,
                "cold_start_seconds": cold_start_seconds,
            },
            "checks": checks,
            "stream_timings": streaming,
            "checkpoint_resume": checkpoint,
            "evidence_quality": {
                "backend_version": "directly_observed",
                "served_model": "directly_observed",
                "model_revision": "reliably_derived_from_pinned_deployment",
                "tokenizer_template": "reliably_derived_from_pinned_model",
                "cached_tokens": "directly_observed",
                "server_ttft": (
                    "directly_observed"
                    if nonstream["server_ttft_reported"]
                    else "unresolved"
                ),
                "gpu": "directly_observed_from_modal_server",
            },
            "limitations": [
                "Compatibility smoke only; no throughput or superiority claim.",
                "Client TTFT includes the experimental modal curl authentication path.",
                "Model revision is pinned by deployment and not emitted by the vLLM HTTP API.",
            ],
            "provenance": {
                "raw_evidence_contains_no_endpoint_or_server_id": True,
                "checkpoint_sha256": "sha256:"
                + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            },
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_path)
        return result
    finally:
        if opened:
            set_autoscaler_minimum(server_id, 0)
        window_seconds = time.monotonic() - window_started
        print(f"validation window closed after {window_seconds:.1f}s; minimum verified at zero")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--chat-template-digest", required=True)
    parser.add_argument("--runeconomy-commit", required=True)
    args = parser.parse_args(argv)
    result = run_probe(
        app_name=args.app,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        expected_version=args.vllm_version,
        model=args.model,
        model_revision=args.model_revision,
        gpu=args.gpu,
        chat_template_digest=args.chat_template_digest,
        runeconomy_commit=args.runeconomy_commit,
    )
    print(f"wrote {result['status']} compatibility evidence to {args.output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
