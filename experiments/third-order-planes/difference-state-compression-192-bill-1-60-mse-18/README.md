# Difference-state compression to 192: bill -1.60%, MSE +18.7%, adjusted +16.8% on the first network

2026-10-14T19:20:03Z · experiment · rejected · measured

Compressing the common-u difference state and its reconstruction to 192 directions (modeled 1.8 to 1.9% compute) destroyed accuracy on the first fixed network and was closed before the second.

Recorded measurement:

    7100136: bill -1.60017%, MSE +18.72933%, adjusted +16.82945%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_difference_v192_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_difference_v192_HELD_v1.py](code/fragment_difference_v192_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2997](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2997)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3005](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3005)

<details>
<summary>All result files and records (3)</summary>

- [OWNER_SMOKE_RETIREMENT_v1.json](OWNER_SMOKE_RETIREMENT_v1.json) · record
- [TARIFF_AND_HISTORY_PROPOSAL_v1.json](TARIFF_AND_HISTORY_PROPOSAL_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-difference-state-compression-192-bill-1-60-mse-18) · [Copy provenance](../../../PROVENANCE.json)

