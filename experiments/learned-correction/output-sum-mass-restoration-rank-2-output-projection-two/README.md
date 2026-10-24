# Output-sum (mass) restoration on the rank-2 output projection: two-network -0.07% but pilot-8 adjusted +0.03%, closed

2026-10-14T22:18:32Z · experiment · rejected · measured

Restoring the output sum lost by the rank-2 output projection survived the two-network smoke narrowly and failed the frozen sign rule on an 8-network pilot; a layer-10 tail-only core Ritz refinement (LTR) was a similarly narrow two-network survivor that was not pursued.

Recorded measurement:

    GOPsum fixed 2: adjusted -0.073374%; pilot 8: adjusted +0.031913%, MSE -0.022817%, bill +0.054742%; LTR fixed 2: adjusted -0.018959%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_gop2_output_mass_v1/fragment_gop2_output_mass_HELD_v1.py](code/astra_gop2_output_mass_v1/fragment_gop2_output_mass_HELD_v1.py)
- [code/astra_l10_tail_ritz_cost_rescue_v1/fragment_ltr_HELD_v1.py](code/astra_l10_tail_ritz_cost_rescue_v1/fragment_ltr_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3124](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3124)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3132](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3132)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3136](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3136)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3152](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3152)

<details>
<summary>All result files and records (3)</summary>

- [GOPSUM_PILOT8_DECISION_v1.json](GOPSUM_PILOT8_DECISION_v1.json) · result
- [OWNER_RETIREMENT_v1.json](OWNER_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-output-sum-mass-restoration-rank-2-output-projection-two) · [Copy provenance](../../../PROVENANCE.json)

