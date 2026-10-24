# Learned covariance edge cut deeper (off, hard clip, tight rosters): each cut costs 1.4-2x its bill; off +1.31% MSE

2026-10-16T15:15:33Z · experiment · rejected · measured

The learned covariance edge module at layers 8-14 was cut in three deeper ways on the estimator submitted as #336412 (S2tr): switched off (EDGE0), its tanh stand-in replaced by a hard clip (ECUTC), and its rosters tightened (ECUTP, rows |mu/sigma| <= 1.5, columns gate >= 0.05). All lost: the kept block of the edge is load-bearing at about 1.4-2x its bill, with the loss at layers 10-11. The 24-seed confirm of edge-off made it dearer per request than the five-full sandwich diet it was meant to replace. Edge extension to all rows at layers 12-14 had already been answered by the exact arm ECUTX (-0.035%, nothing to buy).

Recorded measurement:

    EDGE0 (74052476) 8 seeds: xMSE +0.763% (SE .60, 4/8 lower), bill -0.544%, xnet +0.219%, aggregate +0.283%; per layer L10 +0.61 (.26), L11 +0.94 (.21). ECUTC (e4942294): xMSE +0.265% (SE .134, 3/8 lower), bill -0.195%, xnet +0.070%. ECUTP (3d64bb8d): xMSE +0.422% (SE .275, 2/8 lower), bill -0.198%, xnet +0.223%. EDGE-off 24 seeds (676c1e72): xMSE +1.313% (SE .36, 6/24 lower; A80 alone +1.59%), aggregate +1.37%, bill -0.546%, xnet +0.767% (+1.35% per 1,000 requests vs five-full ~+1.0%).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_lowres_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_modsw_edge_v1.py](code/cand_modsw_edge_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4209](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4209)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4222](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4222)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4241](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4241)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4244](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4244)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4267](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4267)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4279](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4279)

<details>
<summary>All result files and records (5)</summary>

- [L2H_edge0_v1.json](L2H_edge0_v1.json) · result
- [L2H_ecutc_v1.json](L2H_ecutc_v1.json) · result
- [L2H_ecutp_v1.json](L2H_ecutp_v1.json) · result
- [MODSW_GRADE_latest.json](../../../records/research/2026-10-16-physics-modules-priced-per-1-000-requests-fourth-order/MODSW_GRADE_latest.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-learned-covariance-edge-cut-deeper-off-hard-clip-tight) · [Copy provenance](../../../PROVENANCE.json)

