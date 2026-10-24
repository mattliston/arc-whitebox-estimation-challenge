# Static plus one 256-column greedy block: eight-network adjusted -0.655%

2026-10-16T03:38:59Z · experiment · adopted · measured

The larger single adaptive block improved the eight-network screen while remaining within the original request gate. Later 24-network evidence did not reproduce this apparent gain.

Recorded measurement:

    xMSE-1.027%,SE.789,5/8;bill+.372%;xnet-.655%,SE.785;first48755/warm46089;risk19.70%;worst104.07s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_gbfix_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_S4_vnr256_pbt_v1.py](code/candidate_S4_vnr256_pbt_v1.py)
- [code/candidate_S4gb_vngb128_pbt_v1.py](code/candidate_S4gb_vngb128_pbt_v1.py)
- [code/cand_2b108_frag_r256_v1.py](code/cand_2b108_frag_r256_v1.py)
- [code/cand_VN_gb128r_pb_v2.py](code/cand_VN_gb128r_pb_v2.py)
- [code/cand_VN_gb128r_pb_v3.py](code/cand_VN_gb128r_pb_v3.py)
- [code/cand_VN_r256_v1.py](code/cand_VN_r256_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3798](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3798)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3800](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3800)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_join_greedy_gb128r_pb_v2.py.txt](fragment_join_greedy_gb128r_pb_v2.py.txt) · result
- [fragment_join_greedy_gb128r_pb_v3.py.txt](fragment_join_greedy_gb128r_pb_v3.py.txt) · result
- [fragment_join_greedy_r256_2b108lineage_v1.py.txt](fragment_join_greedy_r256_2b108lineage_v1.py.txt) · result
- [fragment_prebind_vngb128_v1.py.txt](fragment_prebind_vngb128_v1.py.txt) · result
- [fragment_prebind_vnr256_v1.py.txt](fragment_prebind_vnr256_v1.py.txt) · result
- [diag_base_7103248.json](diag_base_7103248.json) · result

</details>

## Related work

- shipped in: [Submission #336392: one adaptive 256-column join block with census-refitted prebinding](../../../submissions/336392/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-static-plus-one-256-column-greedy-block-eight-network) · [Copy provenance](../../../PROVENANCE.json)

