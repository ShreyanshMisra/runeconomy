# Verify your inference optimizations.

Agents make several model calls to complete one task. A serving change can
improve cache reuse, time to first token or another inference metric without
making the completed workflow faster.

RunEconomy connects those measurements to complete workflow latency and task
success. It can verify a change, identify a regression or decline to verify an
attractive result when the quality evidence is not strong enough.

## Try the complete workflow locally

```bash
pip install --pre runeconomy==0.2.0a4
runeconomy tutorial init ./runeconomy-tutorial
runeconomy tutorial serve
```

The deterministic localhost tutorial validates the complete product workflow.
It does not run a model or measure real inference performance. Its
RunEconomy-enriched OpenInference fixture contains the explicit evidence needed
for an immediately runnable controlled experiment.

Users can import their own traces. Ordinary traces may support diagnosis while
still producing an unresolved workspace. RunEconomy generates a report from
newly executed controlled trials, not by relabelling a historical trace.

[Follow the import, initialization, validation, run and report steps](/start/).
