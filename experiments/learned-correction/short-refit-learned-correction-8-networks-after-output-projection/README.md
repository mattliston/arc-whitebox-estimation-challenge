# Short refit of the learned correction on 8 networks after the output projection: held-out MSE -0.014% (4/8), parked

2026-10-14T22:43:19Z · experiment · rejected · measured

One fixed half refit-after-drop update (1,043 head scalars plus three dials) trained on 8 fixed networks with the output projection in place changed the 8 held-out networks by a hair, so it was parked without a native run.

Recorded measurement:

    held-out 8: MSE -0.014124% (SE log 0.312235%, 4/8 lower)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gop2_short_refit_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/apply_fixed_short_step_HELD_v1.py](code/apply_fixed_short_step_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3142](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3142)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3146](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3146)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3148](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3148)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3161](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3161)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3163](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3163)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3171](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3171)

<details>
<summary>All result files and records (2)</summary>

- [FIXED_CHECK_DECISION_v1.json](FIXED_CHECK_DECISION_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-short-refit-learned-correction-8-networks-after-output-projection) · [Copy provenance](../../../PROVENANCE.json)

