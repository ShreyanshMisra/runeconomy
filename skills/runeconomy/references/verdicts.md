# Verdicts

Open the gate-by-gate report with
`runeconomy experiment report results/result.json --output report.html --open`.
Read the gates in their recorded order. Report the deciding gate, its measured
value or confidence bound, the frozen threshold, and the denominator (all
attempts or successful pairs). With `--require-verified`, exit code 3 means a
result was produced with a verdict other than `verified`.

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

For a tail-latency or cost claim, name the declared percentile or cost per
1,000 successful workflows and the load at which it was measured. Proof 4
shows why the quality gate comes first: removing search halved median latency
while success fell from 100/100 to 54/100, below the 80% floor.
