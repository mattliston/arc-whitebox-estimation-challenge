# Wall-capped extra read plans: only 37 plans satisfy 0.1% bill per added server-second

2026-10-16T07:52:12Z · experiment · rejected · measured

The added read grid failed the wall-efficiency target at useful incidence. The 375-key capped table was kept as a possible small component, while the full extension was dropped.

Recorded measurement:

    37of1710plans;hit.05/net;mean-.014%;375keyfresh4first-.160%,SE.037,warm-.265%,SE.143;MSE+.009%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_smtable_v1/build_smtable_v1.py](code/lead/runs/opus_smtable_v1/build_smtable_v1.py)
- [code/lead/runs/opus_smtable_v1/price_grid_v1.py](code/lead/runs/opus_smtable_v1/price_grid_v1.py)
- [code/lead/runs/opus_smtable_v1/price_grid_v2.py](code/lead/runs/opus_smtable_v1/price_grid_v2.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py](code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py)
- [code/lead/runs/opus_governor_v1/candidate_GOV_v3.py](code/lead/runs/opus_governor_v1/candidate_GOV_v3.py)
- [code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_diag_v1.py](code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_diag_v1.py)
- [code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py](code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4017](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4017)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4006](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4006)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BUILD_smtable_v1.json](../expanded-read-kernel-grid-warm-bill-0-1666-but/BUILD_smtable_v1.json) · result
- [fragment_smprune_r002_v1.py.txt](../expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smprune_r002_v1.py.txt) · result
- [fragment_smprune_r005_v1.py.txt](../expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smprune_r005_v1.py.txt) · result
- [fragment_smtable_cap_v2.py.txt](../expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smtable_cap_v2.py.txt) · result
- [fragment_smtable_ratio_v3.py.txt](../expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smtable_ratio_v3.py.txt) · result
- [PRICE_grid_v1_s0.json](../expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_grid_v1_s0.json) · result
- [PRICE_grid_v1_s1.json](../expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_grid_v1_s1.json) · result
- [PRICE_missed_v2.json](../expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_missed_v2.json) · result
- [PRICE_sanity_v1.json](../expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_sanity_v1.json) · result
- [fragment_smcov_diag_v1.py.txt](../expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smcov_diag_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-wall-capped-extra-read-plans-only-37-plans-satisfy) · [Copy provenance](../../../PROVENANCE.json)

