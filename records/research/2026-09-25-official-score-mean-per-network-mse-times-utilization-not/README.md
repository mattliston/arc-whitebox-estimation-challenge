# Official score is mean of per-network MSE times utilization, not product of their means

2026-09-25T07:14Z · finding · informational · measured

An official-page check established the aggregation rule and justified per-network adjusted-error comparisons for allocation studies.

Recorded measurement:

    - 2026-09-25 00:14 lead: SCORE AGGREGATION CHECKED (S23 page): board score = MEAN over MLPs of (MSE_i x util_i) = 2.3950e-9; the product of means is 2.3957e-9; corr(MSE, util) -0.10. No shared budget across nets, so each net's optimum minimizes its own MSE_i x u_i. Analyst next: per-net adaptive allocation. Step 1 = the oracle gain from existing variant rows (stop if < 0.5%); step 2 = a LOO-predictable router (>= 50% of the oracle; cross-seed rule).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L817](../../TEAM_LOG_2026-09-22_to_09-26.md#L817)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-official-score-mean-per-network-mse-times-utilization-not) · [Copy provenance](../../../PROVENANCE.json)

