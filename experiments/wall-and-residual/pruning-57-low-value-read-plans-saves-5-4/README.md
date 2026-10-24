# Pruning 57 low-value read plans saves 5.4 projected wall seconds for about 0.15% warm bill

2026-10-16T08:10:06Z · experiment · inconclusive · measured

Removing low arithmetic-saving plans freed runtime and request headroom, but was later dropped when the intended costly extensions failed. A stronger 123-key cut was priced as another option.

Recorded measurement:

    57keys:requests-128first/-188warm;FLOPs+.015%first/+.151%warm;risk12.24%;worst103.9sfrom109.5;123keys:risk11.31%,worst95.7s,bill+.35..+.41%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L4025](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4025)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4018](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4018)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4029](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4029)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BUILD_smtable_v1.json](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/BUILD_smtable_v1.json) · result
- [fragment_smprune_r002_v1.py.txt](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smprune_r002_v1.py.txt) · result
- [fragment_smprune_r005_v1.py.txt](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smprune_r005_v1.py.txt) · result
- [fragment_smtable_cap_v2.py.txt](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smtable_cap_v2.py.txt) · result
- [fragment_smtable_ratio_v3.py.txt](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smtable_ratio_v3.py.txt) · result
- [PRICE_grid_v1_s0.json](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_grid_v1_s0.json) · result
- [PRICE_grid_v1_s1.json](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_grid_v1_s1.json) · result
- [PRICE_missed_v2.json](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_missed_v2.json) · result
- [PRICE_sanity_v1.json](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_sanity_v1.json) · result
- [fragment_smcov_diag_v1.py.txt](../../arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smcov_diag_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-pruning-57-low-value-read-plans-saves-5-4) · [Copy provenance](../../../PROVENANCE.json)

