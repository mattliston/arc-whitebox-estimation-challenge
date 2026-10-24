# Removing final-layer row cut or common-read split: no accuracy gain for +1.354/+0.372% bill

2026-10-16T13:54:07Z · experiment · rejected · measured

Exact final-layer variants showed that these two engineering shortcuts were effectively lossless at the final objective.

Recorded measurement:

    rowcutoffMSE+.135%,SE.12,3/8,bill+1.354%,net+1.489%;splitoffMSE+.001%,SE.06,3/8,bill+.372%,net+.373%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L4185](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4185)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [LADDER2_GRADE24_latest.json](LADDER2_GRADE24_latest.json) · result
- [LADDER2_GRADE_latest.json](LADDER2_GRADE_latest.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-removing-final-layer-row-cut-or-common-read-split) · [Copy provenance](../../../PROVENANCE.json)

