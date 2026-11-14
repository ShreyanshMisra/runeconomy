# Reliable proof agent

This is the frozen agent used for RunEconomy's calibration and 40-task
held-out proof. It uses two schema-constrained inference calls around one real,
bounded repository search tool:

1. Copy the exact quoted phrase from the task into a structured search plan.
2. Search every candidate path and derive the enclosing identifier from the
   parsed Python structure rather than asking the model to guess a path or
   owner.
3. Return one schema-constrained Python identifier from that evidence.

The baseline places dynamic run metadata before static instructions. The
candidate places the same metadata after them. Generated content remains in
private attempt artifacts; the shareable comparison contains digests,
validated status provenance, and telemetry only.

The private 20-task calibration split was used for development. The separate
40-task workload was then frozen and executed once. Its verdict is `invalid`,
not verified: cache reuse and TTFT improved, and latency among both-successful
pairs appeared favorable, but the baseline absolute task-success lower bound
and paired task-success non-inferiority bound failed their predeclared gates.

That held-out workload is consumed. Do not rerun it, tune this agent against
its failures, change its thresholds, or present its conditional mechanism and
latency observations as a verified improvement. Raw tasks, answer keys, model
outputs, traces, endpoints, and evidence remain private and ignored.
