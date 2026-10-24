# Layer and inner-dimension subsets of two-strip covariance traded requests for bill at the same rate

2026-10-18T02:44:43Z · experiment · rejected · measured

Restricting the two-strip sandwich by inner dimension or layer reduced its bill premium but also reduced request savings. The layers-4–15 variant with prebinding had useful runtime margin, yet platform-calibrated cost exceeded the available score improvement.

Recorded measurement:

    Everywhere: requests -7.4/-7.1%, bill +0.65%; inner dimension <=960: requests -3.8/-2.5%, bill +0.30/+0.22%; layers >=8: requests -3.1/-3.5%, bill +0.26/+0.31%; layers 4–15: requests -5.3/-5.5%, bill +0.45/+0.50%; 7-rank adjusted -1.20% (SE .27) versus control.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_v55j.py](code/candidate_v55j.py)
- [code/candidate_v58j.py](code/candidate_v58j.py)
- [code/candidate_v59j.py](code/candidate_v59j.py)
- [code/candidate_v60j.py](code/candidate_v60j.py)
- [code/candidate_v62j.py](code/candidate_v62j.py)
- [code/grade_store_v1.py](code/grade_store_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5073](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5073)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5075](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5075)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5084](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5084)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5094](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5094)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5095](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5095)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [fragment_two_strip_from_v44s.py.txt](fragment_two_strip_from_v44s.py.txt) · code
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-layer-inner-dimension-subsets-two-strip-covariance-traded-requests) · [Copy provenance](../../../PROVENANCE.json)

