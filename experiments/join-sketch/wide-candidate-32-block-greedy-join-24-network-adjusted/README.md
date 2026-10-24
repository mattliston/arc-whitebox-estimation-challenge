# Wide-candidate 32-block greedy join: 24-network adjusted +0.919% versus block-greedy reference

2026-10-16T05:11:14Z · experiment · rejected · measured

Better offline trace error failed to predict final accuracy for the wide-candidate implementation. The candidate also exceeded the request-risk gate.

Recorded measurement:

    versusblockgreedyxMSE+1.086%,bill-.167%,net+.919%,SE.49,9/24;versusbaseMSE+.063%,SE.47,bill+.522%;risk29.36%,worst110.0s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_sketchgb_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/grade_osk_v1.py](code/grade_osk_v1.py)
- [code/cand_2b108_gb128r_osk_m64b128_v1.py](code/cand_2b108_gb128r_osk_m64b128_v1.py)
- [code/cand_2b108_gb128r_osk_m64b32_v1.py](code/cand_2b108_gb128r_osk_m64b32_v1.py)
- [code/cand_2b108_gb128r_osk_m64b64_v1.py](code/cand_2b108_gb128r_osk_m64b64_v1.py)
- [code/cand_S2t_gb128r_osk_m64b32_v1.py](code/cand_S2t_gb128r_osk_m64b32_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3895](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3895)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [grade_osk.txt](../../../records/research/2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/grade_osk.txt) · result
- [grade_osk128.txt](../../../records/research/2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/grade_osk128.txt) · result
- [OSK128_GRADE_v1.json](../../../records/research/2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/OSK128_GRADE_v1.json) · result
- [OSK_GRADE_v1.json](../../../records/research/2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/OSK_GRADE_v1.json) · result
- [fragment_join_candgreedy_v1.py.txt](../../../records/research/2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/fragment_join_candgreedy_v1.py.txt) · result
- [bnd_osk_project.txt](../../../records/research/2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/bnd_osk_project.txt) · result
- [bnd_osk_project_v2.txt](../../../records/research/2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/bnd_osk_project_v2.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-wide-candidate-32-block-greedy-join-24-network-adjusted) · [Copy provenance](../../../PROVENANCE.json)

