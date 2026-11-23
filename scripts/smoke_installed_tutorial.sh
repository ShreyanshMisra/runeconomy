#!/usr/bin/env bash
set -euo pipefail

wheel="${1:-}"
if [[ -z "$wheel" || ! -f "$wheel" ]]; then
  echo "usage: $0 /absolute/path/to/runeconomy.whl" >&2
  exit 2
fi
wheel="$(cd "$(dirname "$wheel")" && pwd)/$(basename "$wheel")"
smoke_root="$(mktemp -d)"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$smoke_root"
}
trap cleanup EXIT INT TERM

uv venv --quiet "$smoke_root/venv"
uv pip install --quiet --python "$smoke_root/venv/bin/python" "$wheel"
mkdir "$smoke_root/work"
cd "$smoke_root/work"
unset PYTHONPATH

python_path="$smoke_root/venv/bin/python"
cli="$smoke_root/venv/bin/runeconomy"

"$python_path" - "$smoke_root/venv" <<'PY'
import pathlib
import runeconomy
import sys

venv = pathlib.Path(sys.argv[1]).resolve()
module = pathlib.Path(runeconomy.__file__).resolve()
if venv not in module.parents:
    raise SystemExit(f"runeconomy resolved outside the clean environment: {module}")
PY

"$cli" tutorial init ./runeconomy-tutorial
"$cli" tutorial serve > tutorial-server.log 2>&1 &
server_pid=$!

"$python_path" - "$server_pid" <<'PY'
import json
import os
import sys
import time
import urllib.request

pid = int(sys.argv[1])
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/healthz", timeout=0.25) as response:
            payload = json.load(response)
        if payload == {"status": "ok", "tutorial_schema_version": "0.1"}:
            break
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError as exc:
            raise SystemExit("tutorial server exited before readiness") from exc
        time.sleep(0.05)
else:
    raise SystemExit("tutorial server did not become ready within 10 seconds")
PY

"$cli" import runeconomy-tutorial/evidence.json --output evidence.json
"$cli" experiment init --from-evidence evidence.json --output experiment/
"$cli" experiment validate experiment/
plan_exit=0
"$cli" experiment plan experiment/ --json > plan.json || plan_exit=$?
if [[ "$plan_exit" -gt 2 ]]; then
  echo "installed experiment plan returned unexpected exit $plan_exit" >&2
  exit 1
fi
"$cli" experiment run experiment/ --output results/
"$cli" experiment report results/result.json --output report.html
"$cli" support-bundle experiment/ --output runeconomy-support.zip

"$python_path" - <<'PY'
import json
import math
import pathlib
import zipfile
from importlib.resources import files

import jsonschema
from runeconomy.agent import Attempt
from runeconomy.decision_artifacts import enrich_decision_artifact
from runeconomy.planning import plan_experiment

result = json.loads(pathlib.Path("results/result.json").read_text(encoding="utf-8"))
report = pathlib.Path("report.html")
support = pathlib.Path("runeconomy-support.zip")
plan = json.loads(pathlib.Path("plan.json").read_text(encoding="utf-8"))
attempt_dir = pathlib.Path("runtime-helper-smoke").resolve()
attempt_dir.mkdir()
task_path = attempt_dir / "task.json"
task_path.write_text('{"task_id":"installed-smoke"}', encoding="utf-8")
environment = {
    "RUNECONOMY_TASK_PATH": str(task_path),
    "RUNECONOMY_TASK_ID": "installed-smoke",
    "RUNECONOMY_VARIANT": "baseline",
    "RUNECONOMY_REPETITION": "0",
    "RUNECONOMY_SEED": "17",
    "RUNECONOMY_ATTEMPT_DIR": str(attempt_dir),
    "RUNECONOMY_OUTPUT_PATH": str(attempt_dir / "output.txt"),
    "RUNECONOMY_TRACE_PATH": str(attempt_dir / "trace.json"),
    "RUNECONOMY_STATUS_PATH": str(attempt_dir / "status.json"),
}
with Attempt.from_environment(environment) as attempt:
    attempt.completed()
status = json.loads((attempt_dir / "status.json").read_text(encoding="utf-8"))
if status != {"schema_version": "0.1", "status": "completed"}:
    raise SystemExit(f"unexpected installed runtime-helper status: {status}")
if plan.get("artifact_type") != "runeconomy.experiment_plan":
    raise SystemExit("installed experiment plan artifact is missing")
if plan.get("state") not in {
    "blocked", "needs_design_input", "ready_for_calibration", "ready_to_freeze"
}:
    raise SystemExit(f"unexpected installed experiment plan state: {plan.get('state')}")
completion = plan.get("completion", {})
planned = completion.get("planned_attempts_per_variant")
minimum = completion.get("minimum_completed_attempts_per_variant")
maximum = completion.get("maximum_failed_or_incomplete_attempts_per_variant")
if not isinstance(planned, int) or completion.get("threshold") != 1.0:
    raise SystemExit(f"unexpected installed completion plan: {completion}")
if minimum != planned or maximum != 0:
    raise SystemExit(f"100% completion tolerance is not exact: {completion}")

manifest_path = pathlib.Path("experiment/manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["acceptance"]["min_completed_rate"] = 0.95
manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
rounded_completion = plan_experiment(pathlib.Path("experiment"))["completion"]
rounded_planned = rounded_completion["planned_attempts_per_variant"]
rounded_minimum = rounded_completion["minimum_completed_attempts_per_variant"]
rounded_maximum = rounded_completion[
    "maximum_failed_or_incomplete_attempts_per_variant"
]
expected_minimum = math.ceil(0.95 * rounded_planned - 1e-12)
if rounded_minimum != expected_minimum or rounded_maximum != rounded_planned - expected_minimum:
    raise SystemExit(f"95% completion tolerance is not exact: {rounded_completion}")

summary = {
    "variants": {
        name: {
            "attempt_count": 1,
            "evaluated_count": 1,
            "task_success_count": 1,
            "task_success_rate": 1.0,
            "status_counts": {"completed": 1},
        }
        for name in ("baseline", "candidate")
    },
    "effects": {
        "task_success": {
            "outcome_counts": {
                "both_success": 1,
                "both_failure": 0,
                "baseline_only": 0,
                "candidate_only": 0,
            }
        }
    },
}
base_decision = {
    "status": "verified",
    "reason": "sanitized installed-wheel semantic smoke",
}
unknown = enrich_decision_artifact(
    {**base_decision, "checks": [{"name": "workflow_latency_ms_improvement", "passed": True}]},
    summary,
)
valid = enrich_decision_artifact(
    {**base_decision, "checks": [{"name": "evaluator_integrity", "passed": True}]},
    summary,
)
invalid = enrich_decision_artifact(
    {
        **base_decision,
        "status": "invalid",
        "checks": [{"name": "evaluator_integrity", "passed": False}],
    },
    summary,
)
if unknown["evidence_validity"]["status"] != "unknown" or unknown["performance_eligibility"]["eligible"]:
    raise SystemExit("missing evidence checks were not conservative")
if valid["evidence_validity"]["status"] != "valid" or not valid["performance_eligibility"]["eligible"]:
    raise SystemExit("passing required evidence was not valid and eligible")
if invalid["evidence_validity"]["status"] != "invalid" or invalid["performance_eligibility"]["eligible"]:
    raise SystemExit("failed required evidence was not invalid and ineligible")
schema_path = files("runeconomy").joinpath("data/decision-0.2.schema.json")
if not schema_path.is_file():
    raise SystemExit("decision schema 0.2 is missing from the installed wheel")
schema = json.loads(schema_path.read_text(encoding="utf-8"))
for artifact in (unknown, valid, invalid):
    if artifact.get("schema_version") != "0.2":
        raise SystemExit("installed decision artifact does not use schema 0.2")
    jsonschema.validate(artifact, schema)
if result["decision"]["status"] != "verified":
    raise SystemExit(f"unexpected tutorial verdict: {result['decision']['status']}")
effect = result["summary"]["effects"]["cached_tokens"]["candidate_minus_baseline_mean"]
if effect != 8.0:
    raise SystemExit(f"unexpected cached-token effect: {effect}")
if not report.is_file() or report.stat().st_size == 0:
    raise SystemExit("tutorial report is missing or empty")
if not support.is_file() or support.stat().st_size == 0:
    raise SystemExit("tutorial support bundle is missing or empty")
text = report.read_text(encoding="utf-8")
with zipfile.ZipFile(support) as archive:
    if archive.namelist() != [
        "manifest.json",
        "environment.json",
        "workspace.json",
        "capabilities.json",
        "validation.json",
        "errors.json",
    ]:
        raise SystemExit("tutorial support bundle members differ from the allowlist")
    support_text = "\n".join(
        archive.read(name).decode("utf-8") for name in archive.namelist()
    )
for forbidden in (
    "Stable instructions",
    "Run metadata",
    "Authorization:" + " Bearer",
    "modal" + ".direct",
    "/home/runner/work/",
    "/" + "Users/",
):
    if forbidden in text:
        raise SystemExit(f"tutorial report contains forbidden material: {forbidden}")
    if forbidden in support_text:
        raise SystemExit(f"tutorial support bundle contains forbidden material: {forbidden}")
print("Installed-wheel tutorial and decision semantics completed successfully")
PY
