# Warm-started join subspace from the old basis and heavy young columns fails saved-Gram accuracy

2026-10-16T09:20:15Z · experiment · rejected · measured

Rayleigh-Ritz on the previous basis plus a small young-column set did not approximate most dominant join subspaces. A full Gram product would still be needed, restoring the expensive power-iteration cost.

Recorded measurement:

    L7p64Frobenius1.2e-2vs1.15e-3;L13p641.23e-3vsp1285.6e-4vsstatic7.4e-5;L10p1284.1e-4vs8.7e-4,eigen1.3e-4

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L4074](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4074)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [NT_GRADE_latest.json](NT_GRADE_latest.json) · result
- [NT_RESULT_v1.json](NT_RESULT_v1.json) · result
- [NT_RESULT_v2.json](NT_RESULT_v2.json) · result
- [NTSCHED_RESULT_v1.json](NTSCHED_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-warm-started-join-subspace-old-basis-heavy-young-columns) · [Copy provenance](../../../PROVENANCE.json)

