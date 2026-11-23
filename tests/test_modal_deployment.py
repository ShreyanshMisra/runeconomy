from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path


def _load_deployment():
    path = Path(__file__).parents[1] / "deploy" / "modal_vllm.py"
    spec = importlib.util.spec_from_file_location("modal_vllm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Modal deployment module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModalDeploymentTests(unittest.TestCase):
    def test_benchmark_server_is_pinned_and_instrumented(self) -> None:
        deployment = _load_deployment()
        command = deployment.vllm_command()

        self.assertIn("--revision", command)
        self.assertIn(deployment.MODEL_REVISION, command)
        self.assertIn("--enable-prefix-caching", command)
        self.assertIn("--enable-prompt-tokens-details", command)
        self.assertIn("--enable-per-request-metrics", command)
        self.assertIn("--enable-scale-out", command)
        self.assertEqual(command[command.index("--block-size") + 1], "16")
        self.assertEqual(deployment.VLLM_VERSION, "0.33.0")
        self.assertGreaterEqual(
            tuple(map(int, deployment.HUGGINGFACE_HUB_VERSION.split("."))),
            (1, 5, 0),
        )

    def test_development_server_never_disables_modal_authentication(self) -> None:
        source = (Path(__file__).parents[1] / "deploy" / "modal_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("VLLM_SERVER_DEV_MODE", source)
        self.assertNotIn(
            "unauthenticated=True", source.replace("``unauthenticated=True``", "")
        )

    def test_modal_images_include_the_runeconomy_package(self) -> None:
        source = (Path(__file__).parents[1] / "deploy" / "modal_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(source.count('.add_local_python_source("runeconomy")'), 2)

    def test_benchmark_server_defaults_to_zero_warm_containers(self) -> None:
        source = (Path(__file__).parents[1] / "deploy" / "modal_vllm.py").read_text(
            encoding="utf-8"
        )
        server_decorator = re.search(r"@app\.server\((.*?)\)\nclass VLLMServer", source, re.S)
        self.assertIsNotNone(server_decorator)
        assert server_decorator is not None
        self.assertRegex(server_decorator.group(1), r"\bmin_containers\s*=\s*0\b")
        self.assertNotIn("reject requests at zero replicas", source)

    def test_compatibility_deployments_accept_explicit_pinned_overrides(self) -> None:
        source = Path(__file__).parents[1] / "deploy" / "modal_vllm.py"
        environment = {
            **os.environ,
            "RUNECONOMY_MODAL_APP_NAME": "compat-test",
            "RUNECONOMY_MODEL_NAME": "example/model",
            "RUNECONOMY_MODEL_REVISION": "revision",
            "RUNECONOMY_VLLM_VERSION": "9.9.9",
            "RUNECONOMY_GPU": "L40S",
            "RUNECONOMY_MAX_MODEL_LEN": "4096",
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util,json;"
                    f"s=importlib.util.spec_from_file_location('deployment',{str(source)!r});"
                    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                    "print(json.dumps([m.APP_NAME,m.MODEL_NAME,m.MODEL_REVISION,"
                    "m.VLLM_VERSION,m.GPU,m.MAX_MODEL_LEN]))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            completed.stdout.strip(),
            '["compat-test", "example/model", "revision", "9.9.9", "L40S", 4096]',
        )

        source_text = source.read_text(encoding="utf-8")
        self.assertIn('"RUNECONOMY_MODEL_NAME": MODEL_NAME', source_text)
        self.assertEqual(source_text.count("**RUNTIME_CONFIG_ENV"), 2)


if __name__ == "__main__":
    unittest.main()
