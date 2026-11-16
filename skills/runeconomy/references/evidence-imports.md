# Evidence imports

Use the unified importer first:

```bash
runeconomy import SOURCE --output evidence.json
```

Use `--type` only when auto-detection is ambiguous. Supported evidence includes
OpenInference or OpenTelemetry OTLP JSON, Phoenix JSON trace exports, exact-token
vLLM capture JSONL, and Prefixwise ledger JSONL.

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
