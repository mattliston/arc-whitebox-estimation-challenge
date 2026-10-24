# Negative-only row cut of the read (forecast -0.57% bill): pilot-8 adjusted +1.05% vs the fired file, closed

2026-10-15T05:44:06Z · experiment · rejected · measured

Cutting only the negative rows of the read while keeping the positive rows exact saved 0.4% of bill and lost 1.5% of accuracy on the pilot.

Recorded measurement:

    vs e406: adjusted +1.048086088% (MSE +1.461896046%, bill -0.413809958%); vs held QR +1.082195823%; vs d401 +0.592209450%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_read_negative_row2_native_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_negative_row2_HELD_v1.py](code/fragment_negative_row2_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3358](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3358)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3362](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3362)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3364](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3364)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3366](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3366)

<details>
<summary>All result files and records (2)</summary>

- [NR2_PILOT8_DECISION_v1.json](NR2_PILOT8_DECISION_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-negative-only-row-cut-read-forecast-0-57-bill) · [Copy provenance](../../../PROVENANCE.json)

