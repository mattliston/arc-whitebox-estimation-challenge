# Cubic marginal closure of the final layer (1023/1024 units accepted): two-network adjusted +0.33%, closed

2026-10-14T19:24:25Z · experiment · rejected · measured

A fixed cubic (third-moment) marginal closure of the terminal layer was accepted on 1023 of 1024 units on each network and still raised the final MSE, so it is a substantive accuracy failure rather than a fallback null.

Recorded measurement:

    adjusted -0.041213% / +0.703079%, geometric +0.330243%; MSE meanlog +0.327653%; bill meanlog +0.002046% (5,396,478 extra FLOPs each)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_cubic_marginal_closure_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cubic_marginal_math_HELD_v1.py](code/cubic_marginal_math_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3001](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3001)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3007](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3007)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3009](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3009)

<details>
<summary>All result files and records (2)</summary>

- [ROOT_SMOKE_RETIREMENT_v1.json](ROOT_SMOKE_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-cubic-marginal-closure-final-layer-1023-1024-units-accepted) · [Copy provenance](../../../PROVENANCE.json)

