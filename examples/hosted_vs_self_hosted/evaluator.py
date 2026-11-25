"""Deterministic answer key for the hosted-versus-self-hosted example tasks."""

from __future__ import annotations

import hashlib
import json
import sys


ANSWERS = {
    "carrier-a1042": "Parcelway",
    "status-a1057": "delivered",
    "eta-a1063": "2027-01-15",
    "warehouse-a1071": "Allentown",
    "carrier-a1088": "Parcelway",
    "status-a1094": "returned",
}


def main() -> int:
    request = json.load(sys.stdin)
    expected = ANSWERS.get(request.get("task_id"))
    if expected is None:
        raise ValueError(f"unknown task {request.get('task_id')!r}")
    observed = str(request.get("output_text", "")).strip()
    success = observed.casefold() == expected.casefold()
    json.dump(
        {
            "success": success,
            "score": float(success),
            "details": {
                "comparison": "case-insensitive exact field value",
                "observed_sha256": hashlib.sha256(observed.encode()).hexdigest(),
            },
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
