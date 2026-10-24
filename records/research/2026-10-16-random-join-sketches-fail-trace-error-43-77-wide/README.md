# Random join sketches fail: trace error +43 to +77%; wide-candidate greedy improves the proxy

2026-10-16T04:59:54Z · finding · rejected · measured

Residual join energy was too small for the random sketch noise. Greedy selection from a wider diagonal-ranked candidate set improved trace error, motivating native tests whose final accuracy later failed.

Recorded measurement:

    random+43..77%traceerror;residual~1e-3trace;candidate64/block128pivotsoverlap.992;tracevsstaticblockgreedy-5.6%,candidate64/block32-9.0%,fullpivot-11.0%

## Programs

- [grade_osk_v1.py](grade_osk_v1.py)
- [cand_2b108_gb128r_osk_m64b128_v1.py](cand_2b108_gb128r_osk_m64b128_v1.py)
- [cand_2b108_gb128r_osk_m64b32_v1.py](cand_2b108_gb128r_osk_m64b32_v1.py)
- [cand_2b108_gb128r_osk_m64b64_v1.py](cand_2b108_gb128r_osk_m64b64_v1.py)
- [cand_S2t_gb128r_osk_m64b32_v1.py](cand_S2t_gb128r_osk_m64b32_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3882](../../TEAM_LOG_2026-10-01_to_10-22.md#L3882)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [grade_osk.txt](grade_osk.txt) · result
- [grade_osk128.txt](grade_osk128.txt) · result
- [OSK128_GRADE_v1.json](OSK128_GRADE_v1.json) · result
- [OSK_GRADE_v1.json](OSK_GRADE_v1.json) · result
- [fragment_join_candgreedy_v1.py.txt](fragment_join_candgreedy_v1.py.txt) · result
- [bnd_osk_project.txt](bnd_osk_project.txt) · result
- [bnd_osk_project_v2.txt](bnd_osk_project_v2.txt) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-random-join-sketches-fail-trace-error-43-77-wide) · [Copy provenance](../../../PROVENANCE.json)

