# Organizer cumulant-propagation baseline is less accurate and over budget at order three

2026-09-24T17:13Z · finding · informational · measured

The second-order baseline was far less accurate; its uncompressed third-order factors exceeded the budget and still underperformed the deployed estimator.

Recorded measurement:

    - 2026-09-24 10:13 [physicist] LEAN-CARD QUESTION (fable_physicist_v1/lean/RESULT_LEAN.md): organizers' mlp_kprop on our nets vs 1e9 truth: k_max 2 = 2 products/layer, MSE 3.0e-6/5.1e-6; k_max 3 factored = 143 products/layer (factor rank grows 2n per layer, never compressed; 2.2x the budget), MSE 4.26e-8/5.49e-8 (r35/r100) vs our S6b 1.56/1.57e-8. Our card = their k_max-3 design made affordable (11.9 dense-product flops/layer; Luna 9.9 plain). A ~10-product card most likely replaces our EXACT YOUNG TIER (young push + atoms read = 47% of the bill) with one fixed-rank (~768-1024) third-order carrier from birth (+ joint refit): covariance 2 + carrier ~5 + star 0-3.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L314](../../TEAM_LOG_2026-09-22_to_09-26.md#L314)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT_LEAN.md](RESULT_LEAN.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-organizer-cumulant-propagation-baseline-less-accurate-over-budget-order) · [Copy provenance](../../../PROVENANCE.json)

