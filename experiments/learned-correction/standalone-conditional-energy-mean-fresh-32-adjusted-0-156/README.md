# Standalone conditional energy mean: fresh-32 adjusted -0.156%, native parity delivery fails

2026-10-14T03:45:56Z · experiment · rejected · measured

The E1 energy conditional-mean closure was run alone on the unchanged carried state (without the failed E3 and E23 arms). It priced at a 0.18% bill saving and passed the fresh screen with a tiny MSE cost, but its native port failed the matched-C parity gate on both fixed networks (negative D statistic, correlation about 0.6), so it was excluded.

Recorded measurement:

    native price MAX log bill -0.0018089749120291227 (geometric saving 0.1807339703%); fresh 32 (16 pairs): mse_log +0.025052895%, bill_log -0.180897491%, net_log -0.155844596% (SE 0.047289687%); native parity 5 calls on 7102096/7102097: FAIL, D -0.810272 / -1.011242, corr 0.603854 / 0.651639

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_science_UNVALIDATED_v2.py](code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_science_UNVALIDATED_v2.py)
- [code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_smoke_UNVALIDATED_v3.py](code/lead/runs/astra_energy_e1_standalone_v1/finite_e1_smoke_UNVALIDATED_v3.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2530](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2530)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2546](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2546)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2560](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2560)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2575](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2575)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2582](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2582)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2600](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2600)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2609](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2609)

<details>
<summary>All result files and records (3)</summary>

- [E1_CLOSURE_OWNER_RECEIPT_v1.json](E1_CLOSURE_OWNER_RECEIPT_v1.json) · result
- [SOURCE_HYPOTHESIS_v1.json](SOURCE_HYPOTHESIS_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-standalone-conditional-energy-mean-fresh-32-adjusted-0-156) · [Copy provenance](../../../PROVENANCE.json)

