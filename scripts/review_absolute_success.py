"""Apply an explicit post-measurement absolute-success methodology review.

This never rewrites the raw result. The reviewed copy records the original
predeclared decision and labels the new floor as post hoc.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def review_result(raw: dict[str, Any], minimum_rate: float) -> dict[str, Any]:
    if not 0 <= minimum_rate <= 1:
        raise ValueError("minimum_rate must be between 0 and 1")
    reviewed = copy.deepcopy(raw)
    basis = reviewed["summary"].get("workflow", reviewed["summary"])
    baseline = basis["variants"]["baseline"]["outcomes"]
    candidate = basis["variants"]["candidate"]["outcomes"]
    baseline_interval = baseline.get("task_success_95_ci")
    candidate_interval = candidate.get("task_success_95_ci")
    if baseline_interval is None or candidate_interval is None:
        raise ValueError("result has no absolute task-success intervals")

    baseline_ok = baseline_interval[0] >= minimum_rate
    candidate_ok = candidate_interval[0] >= minimum_rate
    original_decision = copy.deepcopy(reviewed["decision"])
    check = {
        "name": "posthoc_minimum_task_success_rate",
        "passed": baseline_ok and candidate_ok,
        "required": True,
        "predeclared": False,
        "observed": {
            "baseline": baseline.get("task_success_rate"),
            "candidate": candidate.get("task_success_rate"),
            "baseline_95_ci": baseline_interval,
            "candidate_95_ci": candidate_interval,
        },
        "threshold": {
            "minimum_95_ci_lower_bound": minimum_rate,
            "method": "Clopper-Pearson exact binomial",
        },
    }
    if not baseline_ok:
        status = "invalid"
        reason = (
            "post-measurement methodology review found that the baseline did not "
            "establish the absolute task-success floor; the raw predeclared verdict "
            "is not eligible for a public verification claim"
        )
    elif not candidate_ok:
        status = "regression"
        reason = (
            "post-measurement methodology review found that the candidate did not "
            "establish the absolute task-success floor"
        )
    else:
        status = original_decision["status"]
        reason = original_decision["reason"]
    reviewed["decision"] = {
        **original_decision,
        "status": status,
        "reason": reason,
        "checks": [check, *original_decision["checks"]],
    }
    reviewed["provenance"]["methodology_review"] = {
        "reviewed_at": datetime.now(UTC).isoformat(),
        "review_timing": "post_measurement",
        "original_predeclared_decision": original_decision,
        "minimum_task_success_rate": minimum_rate,
        "claim": (
            "The raw measurement is preserved; this reviewed verdict adds a "
            "non-predeclared safeguard and cannot strengthen the original claim."
        ),
    }
    return reviewed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rate", type=float, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    reviewed = review_result(raw, args.minimum_rate)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(
        json.dumps(reviewed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {reviewed['decision']['status']} reviewed result to {args.output}")


if __name__ == "__main__":
    main()
