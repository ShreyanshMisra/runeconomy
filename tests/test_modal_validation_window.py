from __future__ import annotations

import json
import subprocess
import unittest

from scripts.modal_validation_window import validation_window


class _Runner:
    def __init__(self, validation_code: int = 0) -> None:
        self.minimum = 0
        self.validation_code = validation_code
        self.commands: list[list[str]] = []

    def __call__(self, command):
        command = list(command)
        self.commands.append(command)
        if command[3:5] == ["app", "info"]:
            output = {
                "lifecycle": {"state": "deployed"},
                "servers": {
                    "resolved-server-name": {
                        "id": "fu-private",
                        "summary": {"web_url": "https://private.invalid"},
                    }
                },
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(output), "")
        if command[3:5] == ["server", "info"]:
            output = {
                "object_id": "fu-private",
                "server_url": "https://private.invalid",
                "unauthenticated": False,
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(output), "")
        if "autoscaler" in command and "update" in command:
            self.minimum = int(command[-1])
            return subprocess.CompletedProcess(command, 0, "", "")
        if "autoscaler" in command and "settings" in command:
            output = {"current_settings": {"min_containers": self.minimum}}
            return subprocess.CompletedProcess(command, 0, json.dumps(output), "")
        if "curl" in command:
            return subprocess.CompletedProcess(command, 0, "healthy", "")
        if command == ["run-validation"]:
            return subprocess.CompletedProcess(command, self.validation_code, "", "")
        raise AssertionError(command)


class ModalValidationWindowTests(unittest.TestCase):
    def test_success_restores_zero_and_resolves_server_from_app_info(self) -> None:
        runner = _Runner()

        result = validation_window(
            app_name="selected-app",
            validation_command=["run-validation"],
            runner=runner,
        )

        self.assertEqual(result, 0)
        self.assertEqual(runner.minimum, 0)
        updates = [command for command in runner.commands if "update" in command]
        self.assertEqual([command[-1] for command in updates], ["1", "0"])
        self.assertTrue(all("fu-private" in command for command in updates))

    def test_failed_validation_still_restores_zero(self) -> None:
        runner = _Runner(validation_code=7)

        result = validation_window(
            app_name="selected-app",
            validation_command=["run-validation"],
            runner=runner,
        )

        self.assertEqual(result, 7)
        self.assertEqual(runner.minimum, 0)

    def test_health_failure_still_restores_zero(self) -> None:
        runner = _Runner()
        original = runner.__call__

        def failing(command):
            result = original(command)
            if "curl" in command:
                return subprocess.CompletedProcess(command, 1, "", "cold")
            return result

        with self.assertRaises(TimeoutError):
            validation_window(
                app_name="selected-app",
                validation_command=["run-validation"],
                runner=failing,
                health_timeout_seconds=0.0,
            )
        self.assertEqual(runner.minimum, 0)

    def test_operator_interruption_still_restores_zero(self) -> None:
        runner = _Runner()
        original = runner.__call__

        def interrupted(command):
            result = original(command)
            if list(command) == ["run-validation"]:
                raise KeyboardInterrupt("simulated operator interruption")
            return result

        with self.assertRaises(KeyboardInterrupt):
            validation_window(
                app_name="selected-app",
                validation_command=["run-validation"],
                runner=interrupted,
            )
        self.assertEqual(runner.minimum, 0)

    def test_modal_default_null_minimum_is_interpreted_as_zero(self) -> None:
        runner = _Runner()

        def default_settings(command):
            result = runner(command)
            if "autoscaler" in command and "settings" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    json.dumps({"current_settings": {"min_containers": None}}),
                    "",
                )
            return result

        from scripts.modal_validation_window import autoscaler_minimum

        self.assertEqual(autoscaler_minimum("fu-private", default_settings), 0)


if __name__ == "__main__":
    unittest.main()
