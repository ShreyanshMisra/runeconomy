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
"$cli" experiment run experiment/ --output results/
"$cli" experiment report results/result.json --output report.html

"$python_path" - <<'PY'
import json
import pathlib

result = json.loads(pathlib.Path("results/result.json").read_text(encoding="utf-8"))
report = pathlib.Path("report.html")
if result["decision"]["status"] != "verified":
    raise SystemExit(f"unexpected tutorial verdict: {result['decision']['status']}")
effect = result["summary"]["effects"]["cached_tokens"]["candidate_minus_baseline_mean"]
if effect != 8.0:
    raise SystemExit(f"unexpected cached-token effect: {effect}")
if not report.is_file() or report.stat().st_size == 0:
    raise SystemExit("tutorial report is missing or empty")
text = report.read_text(encoding="utf-8")
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
print("Installed-wheel tutorial completed with the expected verified plumbing verdict")
PY
