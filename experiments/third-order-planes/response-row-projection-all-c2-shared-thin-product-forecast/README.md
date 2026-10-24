# Response-row projection with an all-c2 shared thin product (forecast -1.84% bill): pilot-8 adjusted +1.53%, closed

2026-10-15T05:29:09Z · experiment · rejected · measured

Projecting the read onto response rows with a shared thin product across all c2 terms (forecast 4.734 GFLOP saving per network) cut the bill 1.7% but raised the MSE 3.2% on the 8-network pilot.

Recorded measurement:

    vs e406: adjusted +1.526024055% (MSE +3.205658465%, bill -1.679634410%); vs d401 +1.070147416%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_read_response_row2_native_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_row2_HELD_v2.py](code/fragment_row2_HELD_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3343](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3343)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3348](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3348)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3350](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3350)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3352](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3352)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3354](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3354)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3358](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3358)

<details>
<summary>All result files and records (2)</summary>

- [ACTUAL_TARIFF_v2.json](ACTUAL_TARIFF_v2.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-response-row-projection-all-c2-shared-thin-product-forecast) · [Copy provenance](../../../PROVENANCE.json)

