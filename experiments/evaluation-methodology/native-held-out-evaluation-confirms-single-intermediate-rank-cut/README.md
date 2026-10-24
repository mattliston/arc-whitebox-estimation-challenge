# Native held-out evaluation confirms the single intermediate-rank cut

2026-09-25T20:07Z · experiment · adopted · measured

Eighty cold native evaluations confirm that reducing only the intermediate join rank is beneficial; cutting the later join as well is worse. Torch raw errors combined with the native arithmetic bill predict every pool within 0.08 percentage point.

Recorded measurement:

    2026-09-25 13:07 porter: NATIVE SCHEDULE EVIDENCE v3 (VALID40 + held-out TRAIN640 nets): S27r_vs_S27: pooled n80 adj -0.784% [-1.224, -0.342] PASS || S27t_vs_S27: pooled n80 adj -0.973% [-1.350, -0.594] PASS || S27t_vs_S27r: pooled n80 adj -0.190% [-0.411, +0.029] FAIL (heldout40_v3.json).
    2026-09-25 13:10 porter: NATIVE HELD-OUT v3 DONE (heldout40_v3.json; 40 held-out TRAIN640 nets + VALID40, cold): vs S27 pooled-80 adjusted: S27t (384/320/384) -0.973% [-1.350, -0.594] 56/80 (held-out -1.318%, raw +1.150%; VALID40 -0.612%, raw +1.861%) PASS; S27r (384/320/320) -0.784% [-1.224, -0.342] 55/80 PASS; S27t vs S27r -0.190% [-0.411, +0.029] 46/80 (both halves favor S27t). Native held-out matches the analyst's torch (-1.368%). S28 residual gate started 13:08 (memory freed).
    - 2026-09-25 13:10 lead: NATIVE 80-NET VERDICT (porter, heldout40_v3.json; each net cold): S27t (384/320/384) vs S27: VALID40 -0.612% [-1.20, -0.00], held-out -1.318% [-1.76, -0.88] (raw +1.150% vs VALID40's +1.861%), POOLED-80 -0.973% [-1.35, -0.59], 56/80. PASS (bar -0.5%). S27r (384/320/320): pooled -0.784%. S27t vs S27r pooled -0.190% [-0.41, +0.03]. Native confirms the torch held-out (-1.368%) and that VALID40 understates the schedule.
    - 2026-09-25 13:13 analyst: S27t NATIVE 80-NET POOL (porter heldout40_v3.json) vs torch prediction. VALID40 -0.612% (torch -0.689%), held-out -1.318% [-1.76, -0.88] (torch -1.368%), pooled -0.973% [-1.35, -0.59], 56/80 (torch -1.034%). Held-out raw +1.150% native = +1.15% torch. Torch raw x native bill predicts native to within 0.08 pt in every pool; the schedule verdict is confirmed natively. S27t vs S27r pooled -0.190% [-0.41, +0.03] (the L13 cut is harmful, as priced).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/score_variant_t688.py](code/score_variant_t688.py)
- [code/stack/stack_S27r_mem.py](code/stack/stack_S27r_mem.py)
- [code/stack/stack_S27t_mem.py](code/stack/stack_S27t_mem.py)
- [code/heldout40_v3.sh](code/heldout40_v3.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1251](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1251)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1255](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1255)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1256](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1256)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1262](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1262)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [heldout40_v3.json](heldout40_v3.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-native-held-out-evaluation-confirms-single-intermediate-rank-cut) · [Copy provenance](../../../PROVENANCE.json)

