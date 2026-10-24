# Join pivot choice is worth only about three to four rank units in captured Gram matrices

2026-10-16T05:02:47Z · finding · rejected · measured

An offline trace/Frobenius comparison showed that adding retained rank changes representation error much more than changing pivots at fixed rank. Better pivots could not support a rank reduction.

Recorded measurement:

    rank-32trace+86..120%;rank+32-51..58%;rank+64-77..85%;bestgreedy-4..15%;rank~8xthelever

## Programs

- [../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/grade_osk_v1.py](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/grade_osk_v1.py)
- [../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_2b108_gb128r_osk_m64b128_v1.py](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_2b108_gb128r_osk_m64b128_v1.py)
- [../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_2b108_gb128r_osk_m64b32_v1.py](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_2b108_gb128r_osk_m64b32_v1.py)
- [../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_2b108_gb128r_osk_m64b64_v1.py](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_2b108_gb128r_osk_m64b64_v1.py)
- [../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_S2t_gb128r_osk_m64b32_v1.py](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/cand_S2t_gb128r_osk_m64b32_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3888](../../TEAM_LOG_2026-10-01_to_10-22.md#L3888)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [grade_osk.txt](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/grade_osk.txt) · result
- [grade_osk128.txt](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/grade_osk128.txt) · result
- [OSK128_GRADE_v1.json](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/OSK128_GRADE_v1.json) · result
- [OSK_GRADE_v1.json](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/OSK_GRADE_v1.json) · result
- [fragment_join_candgreedy_v1.py.txt](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/fragment_join_candgreedy_v1.py.txt) · result
- [bnd_osk_project.txt](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/bnd_osk_project.txt) · result
- [bnd_osk_project_v2.txt](../2026-10-16-random-join-sketches-fail-trace-error-43-77-wide/bnd_osk_project_v2.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-join-pivot-choice-worth-only-about-three-four-rank) · [Copy provenance](../../../PROVENANCE.json)

