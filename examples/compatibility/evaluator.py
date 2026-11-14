"""Deterministic answer key for the frozen compatibility workloads."""

from __future__ import annotations

import json
import sys


ANSWERS = {
    "calibration-report-renderer": "render_experiment_html",
    "calibration-vllm-client": "VLLMClient",
    "heldout-critical-path": "calculate_critical_path",
    "heldout-backend-discovery": "discover_vllm_capabilities",
    "heldout-openinference-import": "import_openinference",
    "heldout-export-safety": "assert_safe_for_export",
    "heldout-closed-loop": "run_comparison",
    "heldout-experiment-init": "initialize_experiment",
}


def main() -> int:
    request = json.load(sys.stdin)
    task_id = request.get("task_id")
    expected = ANSWERS.get(task_id)
    if expected is None:
        raise ValueError(f"unknown compatibility task {task_id!r}")
    observed = str(request.get("output_text", "")).strip()
    success = observed == expected
    json.dump(
        {
            "success": success,
            "score": float(success),
            "details": {
                "comparison": "case-sensitive exact identifier",
                "observed_sha256": __import__("hashlib").sha256(
                    observed.encode()
                ).hexdigest(),
            },
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
