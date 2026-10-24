# Exact covariance sandwich and padded symmetric kernel: adjusted +0.808% and +0.174%

2026-10-16T13:58:06Z · experiment · rejected · measured

Removing the covariance cut worsened accuracy in the co-adapted estimator. Replacing the inner-size symmetric kernel was a rounding-level twin with additional arithmetic.

Recorded measurement:

    exactMSE+.304%,SE.156,2/8,bill+.504%,net+.808%,aggregate+.863%;paddedMSE+.034%,SE.066,4/8,bill+.139%,net+.174%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L4188](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4188)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [jobs_dcv3_a80_v1.json](jobs_dcv3_a80_v1.json) · result
- [jobs_dcv45_v1.CANCELLED.json](jobs_dcv45_v1.CANCELLED.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-exact-covariance-sandwich-padded-symmetric-kernel-adjusted-0-808) · [Copy provenance](../../../PROVENANCE.json)

