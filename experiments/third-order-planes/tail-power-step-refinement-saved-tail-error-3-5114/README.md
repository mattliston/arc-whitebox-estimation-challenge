# Tail power-step refinement: saved tail error -3.5114% for 546,008,000 extra FLOPs

2026-10-15T18:47:36Z · experiment · rejected · measured

A fixed power refinement on the omitted tail improved its saved training-factor error but added 0.22286% of the full-network bill before any final-accuracy claim. The route closed without native model evaluation.

Recorded measurement:

    TRAIN16tailerror-3.5114%;546.008M/.22286%bill;actual546008000FLOPs;chargedopdelta-22

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_rtt2_tail_power_source_v1/price_tail_power_HELD_v1.py](code/lead/runs/astra_rtt2_tail_power_source_v1/price_tail_power_HELD_v1.py)
- [code/lead/runs/astra_rtt2_tail_power_source_v1/offline_tail_power_HELD_v1.py](code/lead/runs/astra_rtt2_tail_power_source_v1/offline_tail_power_HELD_v1.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_read_column_oracle_v1/inspect_fixed_band_oracle_HELD_v1.py](code/lead/runs/astra_read_column_oracle_v1/inspect_fixed_band_oracle_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3479](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3479)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [OFFLINE_TAIL_POWER_RESULT_v1.json](OFFLINE_TAIL_POWER_RESULT_v1.json) · result
- [OFFLINE_RULE_FROZEN_v1.json](OFFLINE_RULE_FROZEN_v1.json) · result
- [TAIL_POWER_TARIFF_v1.json](TAIL_POWER_TARIFF_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-tail-power-step-refinement-saved-tail-error-3-5114) · [Copy provenance](../../../PROVENANCE.json)

