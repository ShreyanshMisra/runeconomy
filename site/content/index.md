# Verify the optimization, not just the metric

RunEconomy identifies inference waste on the critical path of an agent workflow
and tests whether a change improves successful-task performance.

[Install and run the five-minute proof](/getting-started/) ·
[Read the rejected optimization](/proof/)

## Trace → diagnosis → experiment → gated verdict

Import real workflow evidence, turn a structured finding into a controlled
baseline/candidate experiment, and evaluate task quality before performance.
RunEconomy sits between agent observability and self-hosted inference
infrastructure; it is not another trace dashboard or cache router.

## The flagship result is a rejection

The candidate gained about 105 cached tokens, reduced server TTFT by about
9.08 ms, and reduced successful-pair workflow latency by about 62.43 ms. It
still received an `invalid` verdict because the frozen task-quality gates did
not pass. All 80 attempts completed with valid traces.

[See every gate, interval, hash, and limitation](/proof/)

## Start with compatibility

`runeconomy doctor` reads vLLM metadata and safe render capabilities, reports
which experiment modes are supported, and gives an exact next action. It never
generates text or resets caches.

[Diagnose a vLLM server](/doctor/) · [Install the agent skill](/skill/)

## Honest alpha boundaries

RunEconomy is an early CLI for private testers. It currently focuses on vLLM,
static request experiments, and closed-loop process comparison using durable
OpenTelemetry/OpenInference evidence. There is no hosted dashboard, managed
collector, framework-specific SDK, or SGLang runner.

[Read the alpha boundaries](/alpha/) ·
[View on PyPI](https://pypi.org/project/runeconomy/) ·
[View on GitHub](https://github.com/ShreyanshMisra/runeconomy)
