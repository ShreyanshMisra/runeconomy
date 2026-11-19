"""Minimal runtime support for RunEconomy agent-process attempts."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from .attempt_status import AttemptStatusError, write_attempt_status


_REQUIRED_PATHS = {
    "task_path": "RUNECONOMY_TASK_PATH",
    "attempt_dir": "RUNECONOMY_ATTEMPT_DIR",
    "output_path": "RUNECONOMY_OUTPUT_PATH",
    "trace_path": "RUNECONOMY_TRACE_PATH",
    "status_path": "RUNECONOMY_STATUS_PATH",
}


class AttemptEnvironmentError(AttemptStatusError):
    """The runner environment cannot safely initialize an attempt."""


class AttemptLifecycleError(AttemptStatusError):
    """The application used the attempt lifecycle inconsistently."""


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise AttemptEnvironmentError(f"{name} is required and must be non-empty")
    return value


def _integer(
    environment: Mapping[str, str], name: str, *, minimum: int | None = None
) -> int:
    value = _required(environment, name)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise AttemptEnvironmentError(f"{name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise AttemptEnvironmentError(f"{name} must be at least {minimum}")
    return parsed


def _absolute_path(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_required(environment, name))
    if not path.is_absolute():
        raise AttemptEnvironmentError(f"{name} must be an absolute path")
    return path


@dataclass
class Attempt:
    """One status-aware execution created by RunEconomy's process runner.

    Call one terminal method exactly once. An unhandled exception records a
    conservative ``application_failure`` when no terminal state exists, then
    propagates unchanged. A clean exit without a terminal method records
    ``terminal_status_not_declared`` and raises :class:`AttemptLifecycleError`.
    """

    task_path: Path
    output_path: Path
    trace_path: Path
    status_path: Path
    attempt_dir: Path
    task_id: str
    variant: str
    repetition: int
    seed: int
    _terminal_status: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> Attempt:
        values = os.environ if environment is None else environment
        paths = {
            attribute: _absolute_path(values, variable)
            for attribute, variable in _REQUIRED_PATHS.items()
        }
        attempt_dir = paths["attempt_dir"]
        if attempt_dir.is_symlink() or not attempt_dir.is_dir():
            raise AttemptEnvironmentError(
                "RUNECONOMY_ATTEMPT_DIR must be an existing non-symlink directory"
            )
        task_path = paths["task_path"]
        if task_path.is_symlink() or not task_path.is_file():
            raise AttemptEnvironmentError(
                "RUNECONOMY_TASK_PATH must be an existing non-symlink file"
            )
        resolved_attempt = attempt_dir.resolve()
        for variable, path in (
            (_REQUIRED_PATHS[attribute], paths[attribute])
            for attribute in ("task_path", "output_path", "trace_path", "status_path")
        ):
            try:
                path.resolve().relative_to(resolved_attempt)
            except ValueError as exc:
                raise AttemptEnvironmentError(
                    f"{variable} must remain inside RUNECONOMY_ATTEMPT_DIR"
                ) from exc
        status_path = paths["status_path"]
        attempt_files = {
            paths["task_path"].resolve(),
            paths["output_path"].resolve(),
            paths["trace_path"].resolve(),
            status_path.resolve(),
        }
        if len(attempt_files) != 4:
            raise AttemptEnvironmentError(
                "task, output, trace, and status paths must be distinct"
            )
        if os.path.lexists(status_path):
            raise AttemptEnvironmentError(
                "RUNECONOMY_STATUS_PATH already exists and will not be overwritten"
            )
        task_id = _required(values, "RUNECONOMY_TASK_ID")
        variant = _required(values, "RUNECONOMY_VARIANT")
        if variant not in {"baseline", "candidate"}:
            raise AttemptEnvironmentError(
                "RUNECONOMY_VARIANT must be baseline or candidate"
            )
        return cls(
            **paths,
            task_id=task_id,
            variant=variant,
            repetition=_integer(values, "RUNECONOMY_REPETITION", minimum=0),
            seed=_integer(values, "RUNECONOMY_SEED"),
        )

    def __enter__(self) -> Attempt:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, traceback
        if exception is not None:
            if self._terminal_status is None:
                try:
                    self.application_failure(
                        stage="execution", code="unhandled_exception"
                    )
                except AttemptStatusError:
                    exception.add_note(
                        "RunEconomy could not record the conservative attempt status."
                    )
            return False
        if self._terminal_status is None:
            self.application_failure(
                stage="lifecycle", code="terminal_status_not_declared"
            )
            raise AttemptLifecycleError(
                "attempt exited without declaring exactly one terminal status"
            )
        return False

    @property
    def terminal_status(self) -> str | None:
        return self._terminal_status

    def _write(
        self,
        status: str,
        *,
        component: str | None = None,
        stage: str | None = None,
        code: str | None = None,
    ) -> None:
        if self._terminal_status is not None:
            raise AttemptLifecycleError(
                "attempt terminal status was already declared"
            )
        temporary_name: str | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.status_path.parent,
                prefix=f".{self.status_path.name}.",
                suffix=".record",
            )
            os.close(descriptor)
            write_attempt_status(
                temporary_name,
                status,
                component=component,
                stage=stage,
                code=code,
            )
            try:
                os.link(temporary_name, self.status_path)
            except FileExistsError as exc:
                raise AttemptLifecycleError(
                    "attempt status already exists and was not overwritten"
                ) from exc
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
                Path(f"{temporary_name}.tmp").unlink(missing_ok=True)
        self._terminal_status = status

    def completed(self) -> None:
        self._write("completed")

    def task_failure(self, *, stage: str, code: str) -> None:
        self._write(
            "task_failure", component="agent", stage=stage, code=code
        )

    def model_output_failure(self, *, stage: str, code: str) -> None:
        self._write(
            "model_output_failure",
            component="model_output",
            stage=stage,
            code=code,
        )

    def application_failure(self, *, stage: str, code: str) -> None:
        self._write(
            "application_failure",
            component="application",
            stage=stage,
            code=code,
        )

    def infrastructure_failure(self, *, stage: str, code: str) -> None:
        self._write(
            "infrastructure_failure",
            component="infrastructure",
            stage=stage,
            code=code,
        )
