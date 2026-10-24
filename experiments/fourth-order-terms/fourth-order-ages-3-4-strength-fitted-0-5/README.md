# Fourth-order ages 3-4 strength (fitted 0.5 at layers 8-15): 0.7 raw +0.198%, 0.3 flat, 1.0 late +0.221%, layers 4-7 null

2026-10-19T17:53:30Z · experiment · rejected · measured

The fourth-order ages 3-4 term (K4L34, the native port of a torch gain, active at layers 8-15 at strengths (0.5, 0.5)) was re-centered with zero-bill finite differences: 0.7 is worse, 0.3 is flat (a plateau 0.3-0.5), 1.0 at layers 13-15 is worse, and adding ages 3-4 at layers 4-7 is null for +0.125% bill: the seed-fitted strength and layer set are at the store optimum.

Recorded measurement:

    k4l34_s07 (28 ranks vs stk1): raw +0.198% (se .08, 9/28 lower), bill 0; k4l34_s03: raw -0.029% (15/28 lower); k4l34_late1: raw +0.221%; k4l34_e47: raw -0.025%, bill +0.125% (~+0.05% vs the #337126 physics)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_k4l34_s07_sci_v1.py](code/candidate_k4l34_s07_sci_v1.py)
- [code/candidate_k4l34_s03_sci_v1.py](code/candidate_k4l34_s03_sci_v1.py)
- [code/candidate_k4l34_late1_sci_v1.py](code/candidate_k4l34_late1_sci_v1.py)
- [code/candidate_k4l34_e47_sci_v1.py](code/candidate_k4l34_e47_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5991](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5991)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5995](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5995)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5996](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5996)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5997](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5997)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-fourth-order-ages-3-4-strength-fitted-0-5) · [Copy provenance](../../../PROVENANCE.json)

