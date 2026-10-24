# Dead-side row cutoff relaxed to 3.5: eight-network adjusted -0.111%; removing it gives +0.053%

2026-10-16T14:01:03Z · experiment · inconclusive · measured

Both variants failed the larger raw-accuracy carrier threshold. The cheaper cutoff 3.5 variant was queued for independent confirmation beyond this slice.

Recorded measurement:

    offMSE-.144%,SE.26,bill+.197%,net+.053%;3.5MSE-.168%,SE.17,bill+.057%,net-.111%,SE.17

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2_v1/grade_ladder2_24_v1.py](code/opus_ladder2_v1/grade_ladder2_24_v1.py)
- [code/opus_ladder2_v1/grade_ladder2_v1.py](code/opus_ladder2_v1/grade_ladder2_v1.py)
- [code/opus_ladder2_v1/build_ladder2_v1.py](code/opus_ladder2_v1/build_ladder2_v1.py)
- [code/opus_ladder2_v1/agg_v1.py](code/opus_ladder2_v1/agg_v1.py)
- [code/opus_ladder2_v1/cand_s2tr_l2_DC35_v1.py](code/opus_ladder2_v1/cand_s2tr_l2_DC35_v1.py)
- [code/opus_ladder2_v1/cand_s2tr_l2_DCoff_v1.py](code/opus_ladder2_v1/cand_s2tr_l2_DCoff_v1.py)
- [code/opus_ladder2_v1/cand_s2tr_l2_L15RC25_v1.py](code/opus_ladder2_v1/cand_s2tr_l2_L15RC25_v1.py)
- [code/opus_ladder2_v1/cand_s2tr_l2_L15RC30_v1.py](code/opus_ladder2_v1/cand_s2tr_l2_L15RC30_v1.py)
- [code/opus_ladder2_v1/cand_s2tr_l2_L15RC_v1.py](code/opus_ladder2_v1/cand_s2tr_l2_L15RC_v1.py)
- [code/opus_ladder2_v1/cand_s2tr_l2_OTT225_v1.py](code/opus_ladder2_v1/cand_s2tr_l2_OTT225_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4191](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4191)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [LADDER2_GRADE24_latest.json](../../late-layer-exactness/removing-final-layer-row-cut-or-common-read-split/LADDER2_GRADE24_latest.json) · result
- [LADDER2_GRADE_latest.json](../../late-layer-exactness/removing-final-layer-row-cut-or-common-read-split/LADDER2_GRADE_latest.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-dead-side-row-cutoff-relaxed-3-5-eight-network) · [Copy provenance](../../../PROVENANCE.json)

