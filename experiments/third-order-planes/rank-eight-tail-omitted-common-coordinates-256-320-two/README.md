# Rank-eight tail for omitted common coordinates 256:320: two-network -0.83%, pilot-8 adjusted +0.03%

2026-10-14T23:31:48Z · experiment · rejected · measured

A fixed rank-8 approximation of the omitted common-u coordinates 256 to 320, added to the unchanged rank-256 reconstruction and transpose read at layers 10 to 14, was the strongest two-network survivor of the day, yet on an 8-network pilot against the fired young-read file its accuracy gain barely covered its bill. The historical full-u320 native result (MSE -0.92% at +1.26% bill) bounded what the tail could recover.

Recorded measurement:

    fixed 2 vs 017 controls: geometric adjusted -0.825885233% (7100136 -0.845898661%, 7100137 -0.805867765%), MSE -1.164520479%, whole bill +0.342625186%; pilot 8 on YR7: MSE -0.305396%, bill +0.337079%, adjusted +0.030653%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_tail8_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_ut8_HELD_v1.py](code/fragment_ut8_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3174](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3174)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3186](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3186)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3188](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3188)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3194](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3194)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3198](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3198)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3200](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3200)

<details>
<summary>All result files and records (4)</summary>

- [ROOT_DIAGNOSTIC_RETIREMENT_v1.json](ROOT_DIAGNOSTIC_RETIREMENT_v1.json) · record
- [ROOT_PILOT_RETIREMENT_v1.json](ROOT_PILOT_RETIREMENT_v1.json) · record
- [OWNER_PILOT_RETIREMENT_v1.json](OWNER_PILOT_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Common coordinates compressed to 248 plus rank eight: all-24 adjusted -0.055%, held without release](../common-coordinates-compressed-248-plus-rank-eight-all-24/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-rank-eight-tail-omitted-common-coordinates-256-320-two) · [Copy provenance](../../../PROVENANCE.json)

