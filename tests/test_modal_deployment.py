from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
