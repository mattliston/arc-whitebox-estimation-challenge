# Full census refit makes the block-greedy join release pass: request risk 23.10 to 17.21%

2026-10-16T04:06:36Z · experiment · adopted · measured

Rebuilding the entire prebind list for the new join shapes succeeded where adding a small list of extra keys had failed. Predictions remained identical and the source entered submission #336397.

Recorded measurement:

    first48594/warm45809;bound601/649;setup4.441s,slow4.581s,worst104.2s;nativebill-.189%;risk23.10to17.21%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_gbfix_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_S4_vnr256_pbt_v1.py](code/candidate_S4_vnr256_pbt_v1.py)
- [code/candidate_S4gb_vngb128_pbt_v1.py](code/candidate_S4gb_vngb128_pbt_v1.py)
- [code/cand_2b108_frag_r256_v1.py](code/cand_2b108_frag_r256_v1.py)
- [code/cand_VN_gb128r_pb_v2.py](code/cand_VN_gb128r_pb_v2.py)
- [code/cand_VN_gb128r_pb_v3.py](code/cand_VN_gb128r_pb_v3.py)
- [code/cand_VN_r256_v1.py](code/cand_VN_r256_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3828](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3828)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3791](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3791)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3819](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3819)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_join_greedy_gb128r_pb_v2.py.txt](../../join-sketch/static-plus-one-256-column-greedy-block-eight-network/fragment_join_greedy_gb128r_pb_v2.py.txt) · result
- [fragment_join_greedy_gb128r_pb_v3.py.txt](../../join-sketch/static-plus-one-256-column-greedy-block-eight-network/fragment_join_greedy_gb128r_pb_v3.py.txt) · result
- [fragment_join_greedy_r256_2b108lineage_v1.py.txt](../../join-sketch/static-plus-one-256-column-greedy-block-eight-network/fragment_join_greedy_r256_2b108lineage_v1.py.txt) · result
- [fragment_prebind_vngb128_v1.py.txt](../../join-sketch/static-plus-one-256-column-greedy-block-eight-network/fragment_prebind_vngb128_v1.py.txt) · result
- [fragment_prebind_vnr256_v1.py.txt](../../join-sketch/static-plus-one-256-column-greedy-block-eight-network/fragment_prebind_vnr256_v1.py.txt) · result
- [diag_base_7103248.json](../../join-sketch/static-plus-one-256-column-greedy-block-eight-network/diag_base_7103248.json) · result

</details>

## Related work

- shipped in: [Submission #336397: three greedy 128-column join blocks with prebind refit; three residual failures](../../../submissions/336397/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-full-census-refit-makes-block-greedy-join-release-pass) · [Copy provenance](../../../PROVENANCE.json)

