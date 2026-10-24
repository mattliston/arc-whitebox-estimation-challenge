# Jensen mean floor on the post-ReLU mean: two-network adjusted +0.12%/+0.70%, closed

2026-10-14T18:54:47Z · experiment · rejected · measured

Flooring the propagated mean by Jensen's inequality (28,672 extra FLOPs per network) made both fixed networks worse.

Recorded measurement:

    adjusted +0.117956% / +0.696997%, paired +0.407059%; extra bill 28,672 FLOPs each

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_jensen_mean_floor_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_jensen_mean_floor_HELD_v1.py](code/fragment_jensen_mean_floor_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2984](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2984)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2987](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2987)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2989](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2989)

<details>
<summary>All result files and records (2)</summary>

- [SOURCE_HYPOTHESIS_HELD_v1.json](SOURCE_HYPOTHESIS_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-jensen-mean-floor-post-relu-mean-two-network-adjusted) · [Copy provenance](../../../PROVENANCE.json)

