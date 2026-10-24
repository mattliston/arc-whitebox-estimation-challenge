# Wide-candidate 128-block greedy join: adjusted +0.46%; column ordering changes the common read

2026-10-16T05:25:19Z · experiment · rejected · measured

Even almost-identical pivots did not preserve the previous final accuracy. The diagnosis found that index sorting changed which leading 256 frame coordinates entered the common read, while product rounding was too small to explain the difference.

Recorded measurement:

    24net+.46%,SE.30;MSE+.636%,bill-.176%;vsbaseMSE-.387%,SE.42;Yerror1.5e-6narrow/3.5e-6wide;<1%truncationerror

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_sketchgb_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/grade_osk_v1.py](code/grade_osk_v1.py)
- [code/cand_2b108_gb128r_osk_m64b128_v1.py](code/cand_2b108_gb128r_osk_m64b128_v1.py)
- [code/cand_2b108_gb128r_osk_m64b32_v1.py](code/cand_2b108_gb128r_osk_m64b32_v1.py)
- [code/cand_2b108_gb128r_osk_m64b64_v1.py](code/cand_2b108_gb128r_osk_m64b64_v1.py)
- [code/cand_S2t_gb128r_osk_m64b32_v1.py](code/cand_S2t_gb128r_osk_m64b32_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3904](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3904)

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

[Research graph](../../../graph/index.html#node=2026-10-16-wide-candidate-128-block-greedy-join-adjusted-0-46) · [Copy provenance](../../../PROVENANCE.json)

