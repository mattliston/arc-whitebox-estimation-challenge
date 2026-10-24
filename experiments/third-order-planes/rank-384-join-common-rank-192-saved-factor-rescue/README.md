# Rank-384 join with common rank 192: saved-factor rescue recovers only 1.29% of added error

2026-10-15T16:20:26Z · experiment · rejected · measured

The already-used rank-384 cache and selected-Gram symmetry did not supply a cheaper exact rescue. Reducing the common leg to 192 saved cost but the full rank 384 join recovered little of its added error.

Recorded measurement:

    selected-Gram symmetry adds 922M FLOPs; common192 potential 2.572G/~1.00%; historical raw +1.288%/net +.448%; full384 recovers 1.29%; combined factor proxy +6.50% vs320/u256

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_full384_exact_rescue_audit_v1/check_saved_common192_v1.py](code/lead/runs/astra_full384_exact_rescue_audit_v1/check_saved_common192_v1.py)
- [code/lead/runs/astra_full384_exact_rescue_audit_v1/check_saved_common192_v2.py](code/lead/runs/astra_full384_exact_rescue_audit_v1/check_saved_common192_v2.py)
- [code/lead/runs/astra_full384_exact_rescue_audit_v1/price_ru192_v1.py](code/lead/runs/astra_full384_exact_rescue_audit_v1/price_ru192_v1.py)
- [code/lead/runs/astra_full384_exact_rescue_audit_v1/price_selected_gram_symmetry_v1.py](code/lead/runs/astra_full384_exact_rescue_audit_v1/price_selected_gram_symmetry_v1.py)
- [code/lead/runs/astra_full384_exact_rescue_audit_v1/price_selected_gram_symmetry_v2.py](code/lead/runs/astra_full384_exact_rescue_audit_v1/price_selected_gram_symmetry_v2.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py](code/lead/runs/astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3419](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3419)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3425](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3425)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [CLOSURE_v1.json](CLOSURE_v1.json) · result
- [ADAPTIVE_CAPACITY_TRAIN8_ORACLE_CLOSE_v1.json](ADAPTIVE_CAPACITY_TRAIN8_ORACLE_CLOSE_v1.json) · result
- [RU192_WARM_TARIFF_v1.json](RU192_WARM_TARIFF_v1.json) · result
- [SAVED_COMMON192_PROXY_v1.json](SAVED_COMMON192_PROXY_v1.json) · result
- [SAVED_COMMON192_PROXY_v2.json](SAVED_COMMON192_PROXY_v2.json) · result
- [SELECTED_GRAM_SYMMETRY_TARIFF_v2.json](SELECTED_GRAM_SYMMETRY_TARIFF_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-rank-384-join-common-rank-192-saved-factor-rescue) · [Copy provenance](../../../PROVENANCE.json)

