# Prebound third-cumulant view reuse changed output bytes through stale symmetry metadata

2026-08-27 · incident · failed · measured

The isolated maximum-rank primitive was byte exact, but the full sequential rank ladder reused overlapping prebound FlopScope views with stale symmetry provenance. Call 0 changed output bytes and added 125,829,120 where FLOPs, so the gate stopped terminally with no plain retry.

Recorded measurement:

    The isolated maximum-rank primitive was byte exact, but the full sequential rank ladder reused overlapping prebound FlopScope views with stale symmetry provenance. Call 0 changed output bytes and added 125,829,120 where FLOPs, so the gate stopped terminally with no plain retry.

## Evidence

- [D4_CANDIDATE_RESULT.json#L1](../../../experiments/wall-and-residual/depth-4-third-cumulant-arena-full-estimator-successor-passed/D4_CANDIDATE_RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../../../experiments/wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log#L1)
- [TERMINAL_CLOSURE.json#L1](../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/TERMINAL_CLOSURE.json#L1)
- [D14_CANDIDATE_RESULT.json#L1](D14_CANDIDATE_RESULT.json#L1)

<details>
<summary>All result files and records (4)</summary>

- [D4_CANDIDATE_RESULT.json](../../../experiments/wall-and-residual/depth-4-third-cumulant-arena-full-estimator-successor-passed/D4_CANDIDATE_RESULT.json) · result
- [SUBMIT_WATCH.log](../../../experiments/wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log) · log
- [TERMINAL_CLOSURE.json](../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/TERMINAL_CLOSURE.json) · result
- [D14_CANDIDATE_RESULT.json](D14_CANDIDATE_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-prebound-third-cumulant-view-reuse-changed-output-bytes-through) · [Copy provenance](../../../PROVENANCE.json)

