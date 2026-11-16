# Evidence-to-report golden path

This public fixture exercises the complete local workflow with sanitized
OpenInference evidence and a deterministic vLLM-compatible mock server. It
uses no private trace, credential, endpoint, model output, paid compute or GPU.

Start the mock server in one terminal:

```bash
python examples/golden_path/mock_vllm.py
```

Then run:

```bash
runeconomy import examples/golden_path/openinference-evidence.json \
  --output evidence.json
runeconomy experiment init --from-evidence evidence.json \
  --output experiment/
runeconomy experiment validate experiment/
runeconomy experiment run experiment/ --output results/
runeconomy experiment report results/result.json \
  --output report.html
```

The mock result validates workflow plumbing and verdict gating. It is not a
live-vLLM performance result and must not be presented as one. The separately
documented real-vLLM path requires an approved private benchmark server.
