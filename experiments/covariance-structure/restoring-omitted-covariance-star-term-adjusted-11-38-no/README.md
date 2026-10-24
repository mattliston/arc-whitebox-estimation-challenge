# Restoring the omitted covariance-star term: adjusted +11.38% with no accuracy gain

2026-10-16T14:02:45Z · experiment · rejected · measured

The exact off-diagonal term required two additional dense products at each live sandwich. After refreshing the source-context guard it ran successfully but worsened the final objective.

Recorded measurement:

    MSE+.483%,SE.642,2/8;bill+10.89%,SE.09;net+11.38%;aggregate+11.46%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2r_v1/build_ladder2r_v1.py](code/opus_ladder2r_v1/build_ladder2r_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_DCV2_v1.py](code/opus_ladder2r_v1/cand_s2tr_DCV2_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_DCV3_v1.py](code/opus_ladder2r_v1/cand_s2tr_DCV3_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_DCV4_v1.py](code/opus_ladder2r_v1/cand_s2tr_DCV4_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_DCV5_v1.py](code/opus_ladder2r_v1/cand_s2tr_DCV5_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_l2r_DCVX_v1.py](code/opus_ladder2r_v1/cand_s2tr_l2r_DCVX_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_l2r_PSDMX_v1.py](code/opus_ladder2r_v1/cand_s2tr_l2r_PSDMX_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_l2r_S1X_v1.py](code/opus_ladder2r_v1/cand_s2tr_l2r_S1X_v1.py)
- [code/opus_ladder2r_v1/cand_s2tr_l2r_S1X_v2.py](code/opus_ladder2r_v1/cand_s2tr_l2r_S1X_v2.py)
- [code/opus_ladder2r_v1/cand_s2tr_l2r_S1X_v3.py](code/opus_ladder2r_v1/cand_s2tr_l2r_S1X_v3.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4194](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4194)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [jobs_dcv3_a80_v1.json](../exact-covariance-sandwich-padded-symmetric-kernel-adjusted-0-808/jobs_dcv3_a80_v1.json) · result
- [jobs_dcv45_v1.CANCELLED.json](../exact-covariance-sandwich-padded-symmetric-kernel-adjusted-0-808/jobs_dcv45_v1.CANCELLED.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-restoring-omitted-covariance-star-term-adjusted-11-38-no) · [Copy provenance](../../../PROVENANCE.json)

