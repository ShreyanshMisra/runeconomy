# Proof 2 aggregate outcome

Status: **final and invalid under the frozen contract**

This is a privacy-safe aggregate outcome record. It is separate from, and does
not modify, the immutable freeze record at commit
`3159f74023b666ace384896196c08422b5da62a5`.

Proof 2 was executed exactly once. Both BF16 and FP8 completed 179 of 180
attempts. The frozen contract required 100% completion for each variant, so
both completion gates failed and the outcome is permanently invalid. No rerun,
post-hoc reclassification, or new performance claim is permitted.

## Decisive result

| Gate | Requirement | Observation | Result |
| --- | --- | --- | --- |
| BF16 completion | 180/180 | 179/180 | Failed |
| FP8 completion | 180/180 | 179/180 | Failed |

The identity, trace-integrity, absolute task-success, task-success
noninferiority, and successful-pair latency gates passed. Those passing gates
cannot rescue an experiment that failed the frozen completion requirement.

Among the 179 pairs where both variants succeeded, the descriptive FP8-minus-
BF16 workflow-latency mean was -391.080 ms with a paired 95% interval of
[-398.606, -383.554] ms. Because the experiment is invalid, this measurement
is not eligible to support a verified performance claim.

## Prespecified descriptive sensitivity

| Slice | Successful pairs | FP8 minus BF16 mean |
| --- | ---: | ---: |
| Call-expression family | 89 | -401.658 ms |
| String-literal family | 90 | -380.619 ms |
| Block A | 89 | -392.456 ms |
| Block B | 90 | -389.718 ms |

These summaries are descriptive only and cannot overturn or reinterpret the
frozen verdict.

## Provenance and privacy

- Frozen commit: `3159f74023b666ace384896196c08422b5da62a5`
- Contract: `sha256:b349073a6f93a9391d628f9abe38d49120cbd0f4b5cc4202f190f924da6a9816`
- Dataset: `sha256:c945014bc7aed8cd6164b9c12da060a05880bea35a2ed29850e27612aa9924b2`
- Evaluator: `sha256:8de9593491d146f7d81126f911165d85ae9353b13f7fa9f9065a05325d089810`
- Private raw archive: `sha256:4252332482a369e686288b9b05033849a0d4c5f42974420f840add27040a3c13`
- Held-out cost: $0.69699111; cumulative Phase 4 cost: $3.00542935

No task text, answer map, model output, trace, endpoint, or physical GPU
identifier appears in this record. The task-level archive remains private and
uncommitted. This record is an audit artifact, not promotional material.
