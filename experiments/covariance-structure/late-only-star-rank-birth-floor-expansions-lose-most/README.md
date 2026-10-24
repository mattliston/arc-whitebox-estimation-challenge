# Late-only star rank and birth-floor expansions lose most of their all-layer accuracy benefit

2026-10-16T06:47:53Z · experiment · rejected · measured

Restricting expensive refinements to the last layers made their raw gains too small to cover even the reduced cost. The late common-leg rank 384 form also hit assertions.

Recorded measurement:

    star8L13-15MSE-.071%,bill+.115%;star8L15-.000%;star32late-.051%/+.465%;birth.005L13-14-.022%/+.077%;joincohortrestore+.639%/+145ops

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_late_v1/grade_late_v1.py](code/opus_late_v1/grade_late_v1.py)
- [code/opus_late_v1/build_dials_v1.py](code/opus_late_v1/build_dials_v1.py)
- [code/opus_late_v1/build_kfix_v1.py](code/opus_late_v1/build_kfix_v1.py)
- [code/opus_late_v1/build_late_v1.py](code/opus_late_v1/build_late_v1.py)
- [code/opus_late_v1/cand_s2tr_late_bfL2_v1.py](code/opus_late_v1/cand_s2tr_late_bfL2_v1.py)
- [code/opus_late_v1/cand_s2tr_late_gk04_v1.py](code/opus_late_v1/cand_s2tr_late_gk04_v1.py)
- [code/opus_late_v1/cand_s2tr_late_j13r_v1.py](code/opus_late_v1/cand_s2tr_late_j13r_v1.py)
- [code/opus_late_v1/cand_s2tr_late_k21k00_v1.py](code/opus_late_v1/cand_s2tr_late_k21k00_v1.py)
- [code/opus_late_v1/cand_s2tr_late_k21k10_v1.py](code/opus_late_v1/cand_s2tr_late_k21k10_v1.py)
- [code/opus_late_v1/cand_s2tr_late_k21km01_v1.py](code/opus_late_v1/cand_s2tr_late_k21km01_v1.py)
- [code/opus_late_v1/cand_s2tr_late_k21kp01_v1.py](code/opus_late_v1/cand_s2tr_late_k21kp01_v1.py)
- [code/opus_late_v1/cand_s2tr_late_kfix_v1.py](code/opus_late_v1/cand_s2tr_late_kfix_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3969](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3969)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3963](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3963)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [LATE_GRADE_latest.json](LATE_GRADE_latest.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-late-only-star-rank-birth-floor-expansions-lose-most) · [Copy provenance](../../../PROVENANCE.json)

