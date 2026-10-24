# Removing low-value grouped-read plans: worst wall 109.0 to 103.6 seconds for +0.09% bill

2026-10-16T02:37:59Z · experiment · adopted · measured

The lean stack removed 62 added plans and the layer 6 group 4 plans. It passed all release gates and became the base of subsequent greedy-join submissions.

Recorded measurement:

    167keys109.0s/17.44%;remove62:106.6s/16.62%/+0.05%;92keys103.6s/15.96%/+.09%;78keys102.5s/15.53%/+.16..+.18%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_lean_v1/build_lean_v1.py](code/opus_lean_v1/build_lean_v1.py)
- [code/opus_lean_v1/candidate_S2t_lean_VA_v1.py](code/opus_lean_v1/candidate_S2t_lean_VA_v1.py)
- [code/opus_lean_v1/candidate_S2t_lean_VB_v1.py](code/opus_lean_v1/candidate_S2t_lean_VB_v1.py)
- [code/opus_lean_v1/candidate_S2t_lean_VC_v1.py](code/opus_lean_v1/candidate_S2t_lean_VC_v1.py)
- [code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py](code/opus_f384_v1/stack/candidate_f384_compactB_S1_v1.py)
- [code/opus_release_v1/s2t/candidate_S2t_cblazy_v1.py](code/opus_release_v1/s2t/candidate_S2t_cblazy_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3726](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3726)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3730](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3730)

<details>
<summary>All result files and records (9)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [LEAN_DECISION_v1.json](LEAN_DECISION_v1.json) · result
- [diag_VB_7103248.json](diag_VB_7103248.json) · result
- [diag_VC_7103248.json](diag_VC_7103248.json) · result
- [guard_diag_VB.json](guard_diag_VB.json) · result
- [guard_diag_VC.json](guard_diag_VC.json) · result
- [JUDGE_OFFICIAL120_leanvb_20261016_opus_v1.txt](JUDGE_OFFICIAL120_leanvb_20261016_opus_v1.txt) · result
- [JUDGE_OFFICIAL120_leanvc_20261016_opus_v1.txt](JUDGE_OFFICIAL120_leanvc_20261016_opus_v1.txt) · result
- [JUDGE_OFFICIAL120_s4tables_20261016_opus_v2.txt](JUDGE_OFFICIAL120_s4tables_20261016_opus_v2.txt) · result

</details>

## Related work

- shipped in: [Submission #336381: one adaptive 128-column join block on compact kernels](../../../submissions/336381/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-removing-low-value-grouped-read-plans-worst-wall-109) · [Copy provenance](../../../PROVENANCE.json)

