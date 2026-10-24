# Eigen-truncating the 384-row sketch to rank 352 loses more energy at every captured join

2026-10-16T11:57:24Z · experiment · rejected · measured

A better subspace at lower rank could not replace the discarded 32 rank units. Even exact rank 352 eigenvectors lost more trace energy than the static rank 384 construction.

Recorded measurement:

    trace1.57-1.72x,Frobenius1.13-1.15xstatic384;exact352trace1.36-1.49x;Frobenius.49-.53x;~40%ofrankcutlossrecovered

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_u64idea_v1/grade_nt_v1.py](code/opus_u64idea_v1/grade_nt_v1.py)
- [code/opus_u64idea_v1/grade_price_v1.py](code/opus_u64idea_v1/grade_price_v1.py)
- [code/opus_u64idea_v1/build_nt_v1.py](code/opus_u64idea_v1/build_nt_v1.py)
- [code/opus_u64idea_v1/build_nt_v2.py](code/opus_u64idea_v1/build_nt_v2.py)
- [code/opus_u64idea_v1/build_ntdiag_v1.py](code/opus_u64idea_v1/build_ntdiag_v1.py)
- [code/opus_u64idea_v1/build_ntsched_v1.py](code/opus_u64idea_v1/build_ntsched_v1.py)
- [code/opus_u64idea_v1/build_s2trdiag_v1.py](code/opus_u64idea_v1/build_s2trdiag_v1.py)
- [code/opus_u64idea_v1/cand_s2tr_a4s_v1.py](code/opus_u64idea_v1/cand_s2tr_a4s_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)
- [code/opus_u64idea_v1/cand_s2tr_ntb32_v2.py](code/opus_u64idea_v1/cand_s2tr_ntb32_v2.py)
- [code/opus_u64idea_v1/cand_s2tr_ntb32diag_v1.py](code/opus_u64idea_v1/cand_s2tr_ntb32diag_v1.py)
- [code/opus_u64idea_v1/cand_s2trdiag_v1.py](code/opus_u64idea_v1/cand_s2trdiag_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4163](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4163)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [NT_GRADE_latest.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_GRADE_latest.json) · result
- [NT_RESULT_v1.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_RESULT_v1.json) · result
- [NT_RESULT_v2.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_RESULT_v2.json) · result
- [NTSCHED_RESULT_v1.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NTSCHED_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-eigen-truncating-384-row-sketch-rank-352-loses-more) · [Copy provenance](../../../PROVENANCE.json)

