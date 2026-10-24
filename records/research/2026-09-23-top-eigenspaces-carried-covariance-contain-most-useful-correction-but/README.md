# Top eigenspaces of carried covariance contain most of the useful correction but simple structures explain <1.2%

2026-09-23T04:39Z · finding · informational · measured

Correcting directions with at least one index in the leading covariance eigenspace recovered most of the full correction benefit. Low-rank error fits and activity blocks recovered less.

Recorded measurement:

    2026-09-22 21:39 analyst: (1) ROW PACKAGE VALID40 (torch, learned ON): K31(.85,.5) .9856 40/40; K31(.8,.5) .9847 37/40; K31(.8,.5)+K22cov(.9,.5) .9901 31/40 -> K22cov increment 1.0055 (CI 1.0015-1.0097) HARMFUL; porter told not to build it. (2) dC STRUCTURE (fable_analyst_v1/dC_structure.log, dC_propagation.log, dC_swap_v1/v2.log, dC_eigsubspace.log): no cheap structure explains offdiag(P dC P) (<1.2%); 3-layer linear-response inheritance explains 23-47%. Closed-loop swaps at L13 (r35/r100; perp fix .889/.833): top-30 SVD .935/.886, top-100 .913/.843; blocks kink x on .959/.899, on x on .976/.953, kink x kink .980/.991; noise control 1.017/1.029. KEY: the part of the error with at least one side in the top-k eigen-subspace of the CARD's own P C P recovers the lever: k=30 .919/.889 (67-73% of the L13 gain), k=100 .910/.835 (81-99%); at L7+10+13 k=100 .783/.718 vs perp .766/.707 (93-96%). Both-sided k=100 only 36-52%.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L52](../../TEAM_LOG_2026-09-22_to_09-26.md#L52)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [dC_structure.log](../../../experiments/fourth-order-terms/fourth-cumulant-2-2-plane-row-factors-fail-improve/dC_structure.log) · log
- [dC_propagation.log](../../../experiments/fourth-order-terms/fourth-cumulant-2-2-plane-row-factors-fail-improve/dC_propagation.log) · log
- [dC_eigsubspace.log](../../../experiments/fourth-order-terms/fourth-cumulant-2-2-plane-row-factors-fail-improve/dC_eigsubspace.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-top-eigenspaces-carried-covariance-contain-most-useful-correction-but) · [Copy provenance](../../../PROVENANCE.json)

