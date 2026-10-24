# Protected difference read truncated to 224 coordinates: bill -0.61% but MSE +2.54%, adjusted +1.91%, closed

2026-10-15T00:58:31Z · experiment · rejected · measured

Truncating the protected difference read of the plane bank from 256 to 224 coordinates saved 0.6% of bill and lost 2.5% of accuracy on the two fixed networks; its prepared 24-network science was never run.

Recorded measurement:

    MSE +2.53942%, bill -0.60936%, adjusted +1.91459%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_protected_difference_read256_v1/fragment_prd224_HELD_v1.py](code/lead/runs/astra_protected_difference_read256_v1/fragment_prd224_HELD_v1.py)
- [code/lead/runs/astra_protected_difference_read256_v1/price_read224_cached_v1.py](code/lead/runs/astra_protected_difference_read256_v1/price_read224_cached_v1.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py](code/lead/runs/astra_yr7_setup_headroom_v1/candidate_yr7_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3230](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3230)

<details>
<summary>All result files and records (2)</summary>

- [DIAGNOSTIC_PROTOCOL_HELD_v1.json](DIAGNOSTIC_PROTOCOL_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Offline rescues of the 224-coordinate read recover only 14 to 39% of the dropped read error (gates half, 75%), closed](../../../records/research/2026-10-15-offline-rescues-224-coordinate-read-recover-only-14-39/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-protected-difference-read-truncated-224-coordinates-bill-0-61) · [Copy provenance](../../../PROVENANCE.json)

