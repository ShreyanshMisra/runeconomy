# Evidence imports

Use the unified importer first:

```bash
runeconomy import SOURCE --output evidence.json
```

Use `--type` only when auto-detection is ambiguous. Supported evidence includes
OpenInference or OpenTelemetry OTLP JSON, Phoenix JSON trace exports, exact-token
vLLM capture JSONL, Prefixwise ledger JSONL, and RunEconomy proxy sessions.

The assessment distinguishes basic diagnosis, structural attribution, exact
cache analysis, critical-path analysis, static replay, agent-process comparison,
and public verification claims.

A capability may be supported, unsupported, or unknown. Follow the assessment's
next action. Structural parentage is not an execution dependency. Token counts
or hashes are not token sequences. Prefixwise conclusions may be imported, but
prompts and token arrays must never be reconstructed from hashes.

OpenInference exact segment attribution is accepted only when explicit segment
token arrays exactly reproduce the complete recorded prompt token sequence.
Otherwise preserve block-level, structural, or unknown attribution.

Directory imports may contain several runs. Select one run deliberately before
initializing a runnable experiment.


## Capture a client that has no trace export

Work in a private directory outside version control. Review the
[recording privacy rules](privacy.md) before starting full capture.

```bash
runeconomy proxy --upstream "$VLLM_BASE_URL" --record session.jsonl
```

Point the client at the printed loopback base URL and run the workflow.
Use distinct workflow IDs for overlapping agent runs. After the final request,
stop the proxy with Ctrl-C and let it flush. Check the final `session_close`
record’s counters. Dropped exchanges or recording errors can leave out the
call you want to replay. Recapture if that call is missing.

```bash
runeconomy import session.jsonl --type proxy --output evidence.json
```

For static replay, supply an evidence-backed `--replay-context` file on
import with the evaluator, serving identity and proposed intervention.
