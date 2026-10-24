# Power-step sketch only at layer 7 gave +0.01% adjusted; later errors reversed the local improvement

2026-10-18T00:06:42Z · experiment · rejected · measured

Restricting the power-step join sketch to layer 7 improved layers 8–10 on most networks while leaving later join plane errors unchanged. Final errors reversed on most networks, consistent with dial co-adaptation; changing the old-tier read scale did not fix the reversal.

Recorded measurement:

    7 ranks: adjusted +0.01% (SE .48), raw +0.12%, bill -0.11%, 2/7 lower; layers 8–10 better on 6/7, layers 11+ reverse on 5/7; read scale .99/1.01 gives adjusted +0.11%/+0.07%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/joinsketch/join_state_v16.py](code/joinsketch/join_state_v16.py)
- [code/jsk/make_joinsketch_v5.py](code/jsk/make_joinsketch_v5.py)
- [code/jsk/make_joinsketch_v4.py](code/jsk/make_joinsketch_v4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5014](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5014)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5018](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5018)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5020](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5020)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5022](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5022)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-power-step-sketch-only-layer-7-gave-0-01) · [Copy provenance](../../../PROVENANCE.json)

