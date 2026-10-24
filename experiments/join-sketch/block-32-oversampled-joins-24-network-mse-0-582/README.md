# Block-32 oversampled joins: 24-network MSE -0.582%, adjusted +0.068%

2026-10-16T10:09:07Z · experiment · rejected · measured

The independent 16-network confirmation reduced the apparent gain and left pooled adjusted error slightly positive. Source cost optimization could not remove enough arithmetic or request overhead.

Recorded measurement:

    all24MSE-.582%,SE.352;bill+.650%;net+.068%,SE.353,aggregate+.075%,12/24;only-.089G/-.037%available;ops+262..278

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ntconfirm_v1/grade_ntconfirm_v1.py](code/opus_ntconfirm_v1/grade_ntconfirm_v1.py)
- [code/opus_ntconfirm_v1/cand_s2tr_ntb32_v2_snapshot.py](code/opus_ntconfirm_v1/cand_s2tr_ntb32_v2_snapshot.py)
- [code/opus_ntconfirm_v1/cand_s2tr_ntb32l7_v2_snapshot.py](code/opus_ntconfirm_v1/cand_s2tr_ntb32l7_v2_snapshot.py)
- [code/opus_ntconfirm_v1/cand_s2tr_w13n_v1_snapshot.py](code/opus_ntconfirm_v1/cand_s2tr_w13n_v1_snapshot.py)
- [code/opus_u64idea_v1/grade_nt_v1.py](code/opus_u64idea_v1/grade_nt_v1.py)
- [code/opus_u64idea_v1/grade_price_v1.py](code/opus_u64idea_v1/grade_price_v1.py)
- [code/opus_u64idea_v1/build_nt_v1.py](code/opus_u64idea_v1/build_nt_v1.py)
- [code/opus_u64idea_v1/build_nt_v2.py](code/opus_u64idea_v1/build_nt_v2.py)
- [code/opus_u64idea_v1/build_ntdiag_v1.py](code/opus_u64idea_v1/build_ntdiag_v1.py)
- [code/opus_u64idea_v1/build_ntsched_v1.py](code/opus_u64idea_v1/build_ntsched_v1.py)
- [code/opus_u64idea_v1/build_s2trdiag_v1.py](code/opus_u64idea_v1/build_s2trdiag_v1.py)
- [code/opus_u64idea_v1/cand_s2tr_a4s_v1.py](code/opus_u64idea_v1/cand_s2tr_a4s_v1.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)
- [code/opus_u64idea_v1/cand_s2tr_ntb32_v2.py](code/opus_u64idea_v1/cand_s2tr_ntb32_v2.py)
- [code/opus_u64idea_v1/cand_s2tr_ntb32diag_v1.py](code/opus_u64idea_v1/cand_s2tr_ntb32diag_v1.py)
- [code/opus_u64idea_v1/cand_s2trdiag_v1.py](code/opus_u64idea_v1/cand_s2trdiag_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4095](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4095)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4098](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4098)

<details>
<summary>All result files and records (9)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [_stale_NTCONFIRM_ntb32l7_v2_incomplete.json](_stale_NTCONFIRM_ntb32l7_v2_incomplete.json) · result
- [NTCONFIRM_ntb32_v2.json](NTCONFIRM_ntb32_v2.json) · result
- [NTCONFIRM_ntb32l7_v2.json](NTCONFIRM_ntb32l7_v2.json) · result
- [W13N_8seed_v1.json](W13N_8seed_v1.json) · result
- [NT_GRADE_latest.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_GRADE_latest.json) · result
- [NT_RESULT_v1.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_RESULT_v1.json) · result
- [NT_RESULT_v2.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NT_RESULT_v2.json) · result
- [NTSCHED_RESULT_v1.json](../warm-started-join-subspace-old-basis-heavy-young-columns/NTSCHED_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-block-32-oversampled-joins-24-network-mse-0-582) · [Copy provenance](../../../PROVENANCE.json)

