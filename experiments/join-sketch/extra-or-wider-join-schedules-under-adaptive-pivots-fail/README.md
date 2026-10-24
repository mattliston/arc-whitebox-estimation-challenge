# Extra or wider join schedules under adaptive pivots fail: late absorption adjusted +1.62%

2026-10-16T03:41:40Z · experiment · rejected · measured

Absorbing one more recent cohort saved little push arithmetic but damaged final accuracy. An extra layer 14 join increased cost before offering an accuracy case.

Recorded measurement:

    extraL14bill+1.74%/+930ops;L13absorbcohort9xMSE+1.90%,0/8,bill-.28%,net+1.62%;widerwindowbill-1.21%,MSE+2.66%oneseed

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_jsched_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_2b108h128_j14_v1.py](code/cand_2b108h128_j14_v1.py)
- [code/cand_2b108h128_w1013_v1.py](code/cand_2b108h128_w1013_v1.py)
- [code/cand_2b108h128_w13_v1.py](code/cand_2b108h128_w13_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3805](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3805)

<details>
<summary>All result files and records (10)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [JSCHED_RESULT_v1.json](JSCHED_RESULT_v1.json) · result
- [diag_base_7103248.json](diag_base_7103248.json) · result
- [diag_j14_7103248.json](diag_j14_7103248.json) · result
- [diag_w1013_7103248.json](diag_w1013_7103248.json) · result
- [diag_w13_7103248.json](diag_w13_7103248.json) · result
- [guard_diag_base.json](guard_diag_base.json) · result
- [guard_diag_j14.json](guard_diag_j14.json) · result
- [guard_diag_w1013.json](guard_diag_w1013.json) · result
- [guard_diag_w13.json](guard_diag_w13.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-extra-or-wider-join-schedules-under-adaptive-pivots-fail) · [Copy provenance](../../../PROVENANCE.json)

