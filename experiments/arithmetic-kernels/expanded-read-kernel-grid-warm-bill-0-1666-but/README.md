# Expanded read-kernel grid: warm bill -0.1666% but projected worst wall 116.7 seconds

2026-10-16T07:01:32Z · experiment · inconclusive · measured

A tariff grid of 4140 shapes added 1710 profitable first-call plans, but more memory-bound linear operations consumed nearly all remaining wall headroom.

Recorded measurement:

    1862positive/1710new;fresh4bill-.183%,xMSE+.066%;boundaryfirst-.2265%/warm-.1666%;requests+362/+326;risk15.90%;worst116.7sfrom109.5s;instructions+4.3..4.9%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3984](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3984)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3981](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3981)

<details>
<summary>All result files and records (11)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BUILD_smtable_v1.json](BUILD_smtable_v1.json) · result
- [fragment_smprune_r002_v1.py.txt](fragment_smprune_r002_v1.py.txt) · result
- [fragment_smprune_r005_v1.py.txt](fragment_smprune_r005_v1.py.txt) · result
- [fragment_smtable_cap_v2.py.txt](fragment_smtable_cap_v2.py.txt) · result
- [fragment_smtable_ratio_v3.py.txt](fragment_smtable_ratio_v3.py.txt) · result
- [PRICE_grid_v1_s0.json](PRICE_grid_v1_s0.json) · result
- [PRICE_grid_v1_s1.json](PRICE_grid_v1_s1.json) · result
- [PRICE_missed_v2.json](PRICE_missed_v2.json) · result
- [PRICE_sanity_v1.json](PRICE_sanity_v1.json) · result
- [fragment_smcov_diag_v1.py.txt](fragment_smcov_diag_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-expanded-read-kernel-grid-warm-bill-0-1666-but) · [Copy provenance](../../../PROVENANCE.json)

