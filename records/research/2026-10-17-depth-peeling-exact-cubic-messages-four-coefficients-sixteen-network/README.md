# Depth peeling, exact cubic messages and four coefficients: sixteen-network raw -0.162%, eight-network adjusted -0.505%

2026-10-17T01:04:40Z · finding · informational · measured

The numerical science source of the Oct 17 release line was frozen as candidate_s2tr_peel_cap750_w3_zb4_v3 (48c126c6) and graded on fixed networks: the first 8 read cold adjusted -0.504838% (raw -0.294331%), the fixed 16 raw -0.1624699% (pooled -0.1810248%, 8/16 wins). With the provisional warm bill of -0.81584% it projected 1.46148e-9, above the user target of 1.46e-9, so no fire; later routed pricing lowered the bill side to 0.568% and the science projection to 1.4597-1.4613e-9.

Recorded measurement:

    Fixed 8 cold adjusted mean -.504838%, raw -.294331%; fixed 16 raw -.1624699% SE .1376497%, pooled -.1810248%, control-bill-weighted -.1870027%, 8/16 wins; provisional warm bill -.81584% -> 1.46148e-9. Final projection v2 envelope 1.45970924-1.46132515e-9; v3 conditional point 1.460684154e-9 (envelope 1.460515523-1.460716331e-9); v5 recurring 1.461387092e-9 (range 1.461218380-1.461419285).

## Programs

- [../../../experiments/arithmetic-kernels/rank-four-exact-cubic-message-kernel-saves-355-273/code/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_v3.py](../../../experiments/arithmetic-kernels/rank-four-exact-cubic-message-kernel-saves-355-273/code/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_v3.py)
- [candidate_s2tr_peel_release_v11.py](candidate_s2tr_peel_release_v11.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4602](../../TEAM_LOG_2026-10-01_to_10-22.md#L4602)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4604](../../TEAM_LOG_2026-10-01_to_10-22.md#L4604)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4652](../../TEAM_LOG_2026-10-01_to_10-22.md#L4652)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4653](../../TEAM_LOG_2026-10-01_to_10-22.md#L4653)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4654](../../TEAM_LOG_2026-10-01_to_10-22.md#L4654)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4701](../../TEAM_LOG_2026-10-01_to_10-22.md#L4701)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4734](../../TEAM_LOG_2026-10-01_to_10-22.md#L4734)

<details>
<summary>All result files and records (9)</summary>

- [SCORE_ESTIMATE_v2.json](SCORE_ESTIMATE_v2.json) · result
- [SCORE_ESTIMATE_v3.json](SCORE_ESTIMATE_v3.json) · result
- [FIXED16_SCIENCE_v1.json](FIXED16_SCIENCE_v1.json) · result
- [FIXED8_RESULT_v1.json](FIXED8_RESULT_v1.json) · result
- [ACTUAL16_WARM_TARIFF_v1.json](ACTUAL16_WARM_TARIFF_v1.json) · result
- [PUSH736_EXCLUSION_COST_v1.json](PUSH736_EXCLUSION_COST_v1.json) · result
- [ROOT_PUBLIC_GRADE_v1-428ebecc.json](../../../submissions/336412/ROOT_PUBLIC_GRADE_v1-428ebecc.json) · result
- [P40_RECOVERY_v1.json](P40_RECOVERY_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-depth-peeling-exact-cubic-messages-four-coefficients-sixteen-network) · [Copy provenance](../../../PROVENANCE.json)

