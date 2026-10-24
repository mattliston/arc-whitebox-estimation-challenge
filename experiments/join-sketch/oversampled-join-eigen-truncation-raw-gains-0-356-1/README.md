# Oversampled join with eigen truncation: raw gains -0.356 to -1.298% are canceled by bill

2026-10-16T10:01:30Z · experiment · rejected · measured

Full and block eigen truncations recovered the near-eigenspace accuracy opportunity but failed the adjusted-score promotion criterion across full 64 and block 32/64 variants.

Recorded measurement:

    full64MSE-1.298%,SE.40,bill+2.019%;block64-1.220%,SE.52,+1.231%;block32all-.751%,SE.50,+.658%,net-.093%;block32L10/13-.356%,SE.37,+.450%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L4088](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4088)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4081](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4081)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4084](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4084)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [NT_GRADE_latest.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_GRADE_latest.json) · result
- [NT_RESULT_v1.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_RESULT_v1.json) · result
- [NT_RESULT_v2.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_RESULT_v2.json) · result
- [NTSCHED_RESULT_v1.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NTSCHED_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-oversampled-join-eigen-truncation-raw-gains-0-356-1) · [Copy provenance](../../../PROVENANCE.json)

