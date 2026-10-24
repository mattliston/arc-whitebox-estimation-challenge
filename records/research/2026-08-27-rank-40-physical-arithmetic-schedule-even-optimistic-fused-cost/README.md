# Rank-40 physical-arithmetic schedule: even the optimistic fused cost exceeded the score boundary

2026-08-27 · finding · rejected · derived

The exact c142 R40 schedule and tiny tensor identity pass, but the unchanged-raw route is economically closed. With cap512, strict fused physical arithmetic totals 749,925,597,823 FLOPs, 176,109,214,432 above the live 573,816,383,391 boundary. Even the strongest conditional fused/free-forward floor is 582,698,554,603, still 8,882,171,212 high.

Recorded measurement:

    The exact c142 R40 schedule and tiny tensor identity pass, but the unchanged-raw route is economically closed. With cap512, strict fused physical arithmetic totals 749,925,597,823 FLOPs, 176,109,214,432 above the live 573,816,383,391 boundary. Even the strongest conditional fused/free-forward floor is 582,698,554,603, still 8,882,171,212 high.

## Programs

- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../../submissions/328586/estimator.py](../../../submissions/328586/estimator.py)
- [../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_r40_economic_falsification_v1/audit_economic_gap.py](../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_r40_economic_falsification_v1/audit_economic_gap.py)

## Evidence

- [RESULT.json#L1](../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/RESULT.json#L1)
- [ECONOMIC_GAP.json#L1](../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/ECONOMIC_GAP.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.json](../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/RESULT.json) · result
- [ECONOMIC_GAP.json](../../../experiments/flop-accounting/depth-4-arena-common-subexpression-reuse-7-643-930/ECONOMIC_GAP.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-rank-40-physical-arithmetic-schedule-even-optimistic-fused-cost) · [Copy provenance](../../../PROVENANCE.json)

