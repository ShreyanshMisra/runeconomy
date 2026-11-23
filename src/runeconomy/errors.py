"""Stable public error codes without a wholesale exception rewrite."""

from __future__ import annotations

from dataclasses import dataclass

from .attempt_status import AttemptStatusError
from .backends import BackendProbeError
from .evaluators import EvaluatorInfrastructureError
from .experiments import ExperimentValidationError
from .models import TraceValidationError
from .tutorial import TutorialError
from .vllm import VLLMError, VLLMHTTPError


ERROR_CODE_VERSION = "0.1"


@dataclass(frozen=True)
class ErrorDescriptor:
    code: str
    category: str
    summary: str
    next_action_code: str


_ERRORS = {
    "runeconomy.evidence.invalid": ErrorDescriptor(
        "runeconomy.evidence.invalid",
        "evidence",
        "The evidence could not be parsed or did not satisfy the supported input contract.",
        "runeconomy.action.review_evidence_format",
    ),
    "runeconomy.evidence.unsupported": ErrorDescriptor(
        "runeconomy.evidence.unsupported",
        "evidence",
        "The evidence format or source type is not supported by this RunEconomy version.",
        "runeconomy.action.select_supported_evidence_type",
    ),
    "runeconomy.evidence.ambiguous": ErrorDescriptor(
        "runeconomy.evidence.ambiguous",
        "evidence",
        "RunEconomy could not select one evidence type or run without explicit guidance.",
        "runeconomy.action.select_evidence_type_or_run",
    ),
    "runeconomy.workspace.unresolved": ErrorDescriptor(
        "runeconomy.workspace.unresolved",
        "workspace",
        "The experiment workspace is structurally valid but still has unresolved required fields.",
        "runeconomy.action.resolve_workspace_fields",
    ),
    "runeconomy.workspace.invalid": ErrorDescriptor(
        "runeconomy.workspace.invalid",
        "workspace",
        "The experiment workspace or manifest is malformed or internally inconsistent.",
        "runeconomy.action.validate_workspace",
    ),
    "runeconomy.workspace.unsupported_schema": ErrorDescriptor(
        "runeconomy.workspace.unsupported_schema",
        "workspace",
        "The workspace schema is not supported by this RunEconomy version.",
        "runeconomy.action.use_matching_runeconomy_version",
    ),
    "runeconomy.backend.unsupported": ErrorDescriptor(
        "runeconomy.backend.unsupported",
        "backend",
        "The saved or observed backend capabilities do not support the requested experiment mode.",
        "runeconomy.action.review_backend_capabilities",
    ),
    "runeconomy.backend.authentication": ErrorDescriptor(
        "runeconomy.backend.authentication",
        "backend",
        "Backend authentication was unavailable or rejected.",
        "runeconomy.action.configure_backend_authentication",
    ),
    "runeconomy.backend.connection": ErrorDescriptor(
        "runeconomy.backend.connection",
        "backend",
        "RunEconomy could not establish a usable backend connection.",
        "runeconomy.action.check_backend_connection",
    ),
    "runeconomy.backend.timeout": ErrorDescriptor(
        "runeconomy.backend.timeout",
        "backend",
        "The backend did not respond within the configured timeout.",
        "runeconomy.action.check_backend_timeout",
    ),
    "runeconomy.backend.probe_failed": ErrorDescriptor(
        "runeconomy.backend.probe_failed",
        "backend",
        "Backend capability discovery did not complete.",
        "runeconomy.action.rerun_doctor",
    ),
    "runeconomy.evaluator.infrastructure": ErrorDescriptor(
        "runeconomy.evaluator.infrastructure",
        "evaluator",
        "The evaluator infrastructure failed, so task quality could not be established.",
        "runeconomy.action.repair_evaluator",
    ),
    "runeconomy.attempt.status_invalid": ErrorDescriptor(
        "runeconomy.attempt.status_invalid",
        "attempt",
        "An attempt status record did not satisfy the versioned status protocol.",
        "runeconomy.action.repair_attempt_status",
    ),
    "runeconomy.checkpoint.mismatch": ErrorDescriptor(
        "runeconomy.checkpoint.mismatch",
        "checkpoint",
        "The checkpoint does not match the resolved experiment or deterministic schedule.",
        "runeconomy.action.review_checkpoint_provenance",
    ),
    "runeconomy.identity.drift": ErrorDescriptor(
        "runeconomy.identity.drift",
        "identity",
        "The recorded environment identity changed across the experiment.",
        "runeconomy.action.restore_environment_identity",
    ),
    "runeconomy.claim.incomplete": ErrorDescriptor(
        "runeconomy.claim.incomplete",
        "claim",
        "The available evidence is insufficient for the intended verification claim.",
        "runeconomy.action.narrow_or_complete_claim",
    ),
    "runeconomy.report.generation": ErrorDescriptor(
        "runeconomy.report.generation",
        "report",
        "The experiment result could not be rendered as a report.",
        "runeconomy.action.review_result_schema",
    ),
    "runeconomy.tutorial.invalid": ErrorDescriptor(
        "runeconomy.tutorial.invalid",
        "tutorial",
        "The local tutorial could not be created or served safely.",
        "runeconomy.action.review_tutorial_setup",
    ),
    "runeconomy.support.unsafe_input": ErrorDescriptor(
        "runeconomy.support.unsafe_input",
        "support_bundle",
        "The selected support input is unsafe to inspect locally.",
        "runeconomy.action.review_support_input",
    ),
    "runeconomy.support.output_exists": ErrorDescriptor(
        "runeconomy.support.output_exists",
        "support_bundle",
        "The requested support archive already exists.",
        "runeconomy.action.choose_new_support_output",
    ),
    "runeconomy.support.invalid_bundle": ErrorDescriptor(
        "runeconomy.support.invalid_bundle",
        "support_bundle",
        "The generated support archive failed its local safety validation.",
        "runeconomy.action.keep_bundle_private_and_report_code",
    ),
    "runeconomy.command.failed": ErrorDescriptor(
        "runeconomy.command.failed",
        "command",
        "The command did not complete under the current inputs and configuration.",
        "runeconomy.action.review_command_inputs",
    ),
}


def descriptor_for_code(code: str) -> ErrorDescriptor:
    """Return only registered, content-free public error metadata."""

    return _ERRORS.get(code, _ERRORS["runeconomy.command.failed"])


def _validation_code(message: str, stage: str) -> str:
    lowered = message.lower()
    if stage == "experiment.report" or "render experiment report" in lowered:
        return "runeconomy.report.generation"
    if "checkpoint" in lowered and (
        "match" in lowered or "schedule" in lowered or "resume" in lowered
    ):
        return "runeconomy.checkpoint.mismatch"
    if "identity" in lowered and ("drift" in lowered or "changed" in lowered):
        return "runeconomy.identity.drift"
    if "api key environment variable" in lowered or "authentication" in lowered:
        return "runeconomy.backend.authentication"
    if "unsupported" in lowered and (
        "backend" in lowered or "api dialect" in lowered or "capabilit" in lowered
    ):
        return "runeconomy.backend.unsupported"
    if "unresolved" in lowered:
        return "runeconomy.workspace.unresolved"
    if "claim" in lowered and ("evidence" in lowered or "identity" in lowered):
        return "runeconomy.claim.incomplete"
    return "runeconomy.workspace.invalid"


def classify_exception(exc: Exception, *, stage: str) -> ErrorDescriptor:
    """Map existing exceptions to stable public codes without changing their classes."""

    explicit = getattr(exc, "error_code", None)
    if isinstance(explicit, str) and explicit in _ERRORS:
        return _ERRORS[explicit]
    if isinstance(exc, BackendProbeError):
        return descriptor_for_code(
            {
                "authentication": "runeconomy.backend.authentication",
                "connection": "runeconomy.backend.connection",
                "timeout": "runeconomy.backend.timeout",
                "configuration": "runeconomy.backend.unsupported",
            }.get(exc.category, "runeconomy.backend.probe_failed")
        )
    if isinstance(exc, VLLMHTTPError):
        return descriptor_for_code(
            "runeconomy.backend.authentication"
            if exc.status in {401, 403}
            else "runeconomy.backend.connection"
        )
    if isinstance(exc, VLLMError):
        return descriptor_for_code("runeconomy.backend.connection")
    if isinstance(exc, TraceValidationError):
        lowered = str(exc).lower()
        if "cannot detect" in lowered or "ambiguous" in lowered or "select one" in lowered:
            code = "runeconomy.evidence.ambiguous"
        elif "unsupported" in lowered or "requires --" in lowered:
            code = "runeconomy.evidence.unsupported"
        else:
            code = "runeconomy.evidence.invalid"
        return descriptor_for_code(code)
    if isinstance(exc, AttemptStatusError):
        return descriptor_for_code("runeconomy.attempt.status_invalid")
    if isinstance(exc, EvaluatorInfrastructureError):
        return descriptor_for_code("runeconomy.evaluator.infrastructure")
    if isinstance(exc, ExperimentValidationError):
        return descriptor_for_code(_validation_code(str(exc), stage))
    if isinstance(exc, TutorialError):
        return descriptor_for_code("runeconomy.tutorial.invalid")
    return descriptor_for_code("runeconomy.command.failed")


def validation_status_code(status: str) -> str | None:
    return {
        "unresolved": "runeconomy.workspace.unresolved",
        "invalid": "runeconomy.workspace.invalid",
        "unsupported_backend": "runeconomy.backend.unsupported",
        "claim_evidence_incomplete": "runeconomy.claim.incomplete",
    }.get(status)
