# Post-optimization work census: young pushes and reads still consume about half of total work

2026-09-25T18:26Z · finding · informational · measured

Updated attribution includes the final request and splitting wrappers. Low-rank star transport and Nyström joins explain the largest reductions, while old-plane rank-dependent work remains substantial.

Recorded measurement:

    - 2026-09-25 11:26 [engineer2] S27 BILL CENSUS (fired S27 9ff80a12; fable_engineer2_v1/bill27/: bill_census_layers_e2.py = the porter's PT-38 + setup first + op + innermost frame + treuse unwrap; census_S27_report.py = the S21 layout with S27 lines mapped by source text). The porter's PT-38 on S26+ files sees treuse's _predict_binary wrapper as the loop: unwrap __wrapped__ first.
      NET 100 (warm, 2nd after setup): 141.97 U (304.87 G, util .13864) vs S21 154.97 U. Net 35 (cold after setup): 142.45 U (util .13911). Setup: 12.68 GF off-budget.
      | push_young | 35.28 | 24.85% | -0.08 |
      | read_young | 34.64 | 24.40% | -1.41 (k-split) |
      | old_reconstruct | 17.50 | 12.33% | -0.08 |
      | covariance | 13.24 | 9.32% | 0 |
      | read_old | 11.52 | 8.12% | -0.41 |
      | join | 8.80 | 6.20% | -3.15 (Nystrom join) |
      | push_newborn | 5.46 | 3.85% | -0.14 |
      | star_lowrank | 4.32 | 3.04% | -7.58 (low-rank star, s1=0, Nystrom) |
      | pair_map / gnn / edge | 2.45 / 2.06 / 1.86 | | 0 |
      | push_old_basis | 1.46 | | |
      | read_prep / births_planes | 0.83 / 0.79 | | births_planes -0.13 (j3) |
      By role: leaf matmul 83.3%, transform add/sub 9.1%, physics ops 6.0%, LAZY BINDS 1.02% (1.45 U; read_young 0.71, push_young 0.34), copies 0.5%. L15 bank pass (all L15 work) 5.57 U = 3.9%.
      OLD TIER (rank-dependent: rebuild + old read + old-basis push + join) = 39.3 U = 27.7% of the bill. L7-9: 9.59 U (6.8%). L10-12: 14.82 U (10.4%). L13-15: 14.87 U (10.5%); L15 has the rebuild 2.48 U but no old read. Joins L7/L10/L13 = 2.90/3.49/2.41 U.

## Programs

- [bill_census_layers_e2.py](bill_census_layers_e2.py)
- [census_S27_report.py](census_S27_report.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1112](../../TEAM_LOG_2026-09-22_to_09-26.md#L1112)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1113](../../TEAM_LOG_2026-09-22_to_09-26.md#L1113)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1116](../../TEAM_LOG_2026-09-22_to_09-26.md#L1116)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1117](../../TEAM_LOG_2026-09-22_to_09-26.md#L1117)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1118](../../TEAM_LOG_2026-09-22_to_09-26.md#L1118)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1119](../../TEAM_LOG_2026-09-22_to_09-26.md#L1119)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1120](../../TEAM_LOG_2026-09-22_to_09-26.md#L1120)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1121](../../TEAM_LOG_2026-09-22_to_09-26.md#L1121)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1122](../../TEAM_LOG_2026-09-22_to_09-26.md#L1122)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1123](../../TEAM_LOG_2026-09-22_to_09-26.md#L1123)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1124](../../TEAM_LOG_2026-09-22_to_09-26.md#L1124)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1125](../../TEAM_LOG_2026-09-22_to_09-26.md#L1125)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1126](../../TEAM_LOG_2026-09-22_to_09-26.md#L1126)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1128](../../TEAM_LOG_2026-09-22_to_09-26.md#L1128)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1129](../../TEAM_LOG_2026-09-22_to_09-26.md#L1129)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-post-optimization-work-census-young-pushes-reads-still-consume) · [Copy provenance](../../../PROVENANCE.json)

