# Antipodal projection of the final-layer learned head (odd component onto W^T state): two-network adjusted +0.14%, closed

2026-10-14T20:26:42Z · experiment · rejected · measured

A fixed final-layer correction that keeps the even component of the learned head under the final-weight sign flip and projects the odd component onto W^T applied to the previous 64-state and mean (48 new signed directions, one extra encoder, a 65-column SVD) made both fixed networks slightly worse.

Recorded measurement:

    adjusted +0.102412197% / +0.178753136%, geometric +0.140575392% (mean log MSE +0.041238136%, log bill +0.099238541%); +260,403,116 FLOPs each

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_terminal_antipodal_projection_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_antipodal_projection_HELD_v3.py](code/fragment_antipodal_projection_HELD_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3047](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3047)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3050](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3050)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3052](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3052)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3054](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3054)

<details>
<summary>All result files and records (2)</summary>

- [DIAGNOSTIC_PROTOCOL_HELD_v3.json](DIAGNOSTIC_PROTOCOL_HELD_v3.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-antipodal-projection-final-layer-learned-head-odd-component-onto) · [Copy provenance](../../../PROVENANCE.json)

