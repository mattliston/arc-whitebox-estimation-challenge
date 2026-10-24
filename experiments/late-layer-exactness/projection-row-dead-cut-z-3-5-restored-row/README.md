# Projection-row dead cut z 3.5 restored on the row-cut lineage: +0.159% adjusted (22 ranks); at layer 14 only, null

2026-10-19T13:47:39Z · experiment · rejected · measured

Restoring the projection-row dead cut from 3.0902 to 3.5 (fewer rows cut at layers 8-14), which had read -0.234% cross-half MSE for +0.065% bill on 24 seeds on 10-16, is dead on store ranks inside the row-cut stack (raw +0.24% worse and bill +0.18% for fewer rows cut), another restored approximation losing to co-adaptation; at layer 14 alone the 64-rounded kept count barely moves and the arm is null.

Recorded measurement:

    stk8 (kc + rc30 + deadcut 3.5) 22 ranks vs the kc arm: +0.159% (se .136; raw +0.363%, bill -0.201%) against stk1 -0.176% (raw +0.205%, bill -0.378%); dc14_35: raw -0.046% (se .07), bill +0.082% vs stk1 (~0 vs the #337090 physics)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_stk8_sci_v1.py](code/candidate_stk8_sci_v1.py)
- [code/candidate_stk9_sci_v1.py](code/candidate_stk9_sci_v1.py)
- [code/candidate_dc14_35_sci_v1.py](code/candidate_dc14_35_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5931](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5931)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5954](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5954)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-projection-row-dead-cut-z-3-5-restored-row) · [Copy provenance](../../../PROVENANCE.json)

