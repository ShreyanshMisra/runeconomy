# Evidence-to-report tutorial

The canonical golden path is packaged with RunEconomy. It works from an
installed wheel without cloning this repository:

```bash
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

Follow the generated `runeconomy-tutorial/README.md` in another terminal.

The tutorial uses a RunEconomy-enriched OpenInference fixture with explicit
prompt tokens, prompt segments, workflow dependencies, and experiment context.
Ordinary OpenTelemetry or OpenInference evidence may support diagnosis while
still producing an unresolved workspace.

The deterministic local server validates product plumbing only. It is not a
complete vLLM implementation, does not run a model, and is not a performance
benchmark or live-vLLM validation.

`mock_vllm.py` remains as a compatibility entry point for repository developers.
It delegates to the same server implementation used by the installed CLI.
