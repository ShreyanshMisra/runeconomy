# Verdicts

Read the gates in their recorded order.

- `verified`: evidence and identity checks passed, task quality passed, and the
  declared performance criterion passed. The candidate is eligible for
  deployment review, not automatic deployment.
- `regression`: a declared quality or performance guardrail failed. Do not ship
  the candidate without a new intervention and newly frozen design.
- `inconclusive`: evidence was usable, but uncertainty was too large. Gather
  more paired evidence under a new frozen design.
- `invalid`: the experiment cannot support a verified improvement or regression
  claim. This does not mean the candidate was slower.

Mechanism metrics and successful-pair latency may still be reported as
conditional observations. Always state which pairs were included and why the
observations were not eligible to become a verified claim.
