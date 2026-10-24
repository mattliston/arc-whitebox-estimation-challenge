# Grouped read plans reduce metered FLOPs while adding 47-441% processor instructions

2026-10-16T07:33:58Z · finding · rejected · measured

A real-compute benchmark of 40 shapes exposed the memory-bound cost hidden by tariff-only comparisons. This led to wall-aware plan selection and pruning.

Recorded measurement:

    +1.5..18.3Ginstructions/+47..441%;+.25..1.35slocal;40..560MFLOPssaved;instructions/FLOP.7-1.2incumbentvs1.6-4.5grouped

## Programs

- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/build_smtable_v1.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/build_smtable_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/price_grid_v1.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/price_grid_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/price_grid_v2.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/price_grid_v2.py)
- [../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](../../../experiments/third-order-planes/protected-difference-read-truncated-224-coordinates-bill-0-61/code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [../../../experiments/arithmetic-kernels/grouped-rank-40-matrix-kernels-corrected-all-24-adjusted/code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py](../../../experiments/arithmetic-kernels/grouped-rank-40-matrix-kernels-corrected-all-24-adjusted/code/lead/runs/astra_smirnov_cfull_v1/smirnov_cfull_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_governor_v1/candidate_GOV_v3.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_governor_v1/candidate_GOV_v3.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_diag_v1.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_diag_v1.py)
- [../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/code/lead/runs/opus_smtable_v1/cand_GOV3_smtable_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4003](../../TEAM_LOG_2026-10-01_to_10-22.md#L4003)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BUILD_smtable_v1.json](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/BUILD_smtable_v1.json) · result
- [fragment_smprune_r002_v1.py.txt](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smprune_r002_v1.py.txt) · result
- [fragment_smprune_r005_v1.py.txt](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smprune_r005_v1.py.txt) · result
- [fragment_smtable_cap_v2.py.txt](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smtable_cap_v2.py.txt) · result
- [fragment_smtable_ratio_v3.py.txt](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smtable_ratio_v3.py.txt) · result
- [PRICE_grid_v1_s0.json](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_grid_v1_s0.json) · result
- [PRICE_grid_v1_s1.json](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_grid_v1_s1.json) · result
- [PRICE_missed_v2.json](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_missed_v2.json) · result
- [PRICE_sanity_v1.json](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/PRICE_sanity_v1.json) · result
- [fragment_smcov_diag_v1.py.txt](../../../experiments/arithmetic-kernels/expanded-read-kernel-grid-warm-bill-0-1666-but/fragment_smcov_diag_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-grouped-read-plans-reduce-metered-flops-while-adding-47) · [Copy provenance](../../../PROVENANCE.json)

