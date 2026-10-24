# Block-greedy join pivots: pilot adjusted -0.760%, but request risk 30.95%

2026-10-16T02:43:08Z · experiment · adopted · measured

A block 128 greedy residual-diagonal selection improved noise-free accuracy more cheaply than full rank 384. Its initial production requests failed the gate, motivating smaller adaptive blocks and later census refits.

Recorded measurement:

    block128xMSE-1.455%,SE.80,5/8;bill+.695%;xnet-.760%;requests+1290/+666;risk30.95%;hybrid64xMSE-.969%,bill+.535%,xnet-.434%,risk26.71%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_joinx_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_2b108_frag_gb128r_v1.py](code/cand_2b108_frag_gb128r_v1.py)
- [code/cand_2b108_frag_h128_v1.py](code/cand_2b108_frag_h128_v1.py)
- [code/cand_2b108_ojx_GB128_v1.py](code/cand_2b108_ojx_GB128_v1.py)
- [code/cand_2b108_ojx_GB128r_v2.py](code/cand_2b108_ojx_GB128r_v2.py)
- [code/cand_2b108_ojx_H64_v1.py](code/cand_2b108_ojx_H64_v1.py)
- [code/cand_VB_ojx_gb128_v1.py](code/cand_VB_ojx_gb128_v1.py)
- [code/cand_VN_gb128r_v1.py](code/cand_VN_gb128r_v1.py)
- [code/cand_VN_h128_v1.py](code/cand_VN_h128_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3733](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3733)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3740](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3740)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3754](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3754)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_join_greedy_gb128_v1.py.txt](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/fragment_join_greedy_gb128_v1.py.txt) · result
- [fragment_join_greedy_gb128r_2b108lineage_v1.py.txt](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/fragment_join_greedy_gb128r_2b108lineage_v1.py.txt) · result
- [fragment_join_greedy_h128_2b108lineage_v1.py.txt](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/fragment_join_greedy_h128_2b108lineage_v1.py.txt) · result
- [JOINX_PRICE_v1.json](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/JOINX_PRICE_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336397: three greedy 128-column join blocks with prebind refit; three residual failures](../../../submissions/336397/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-block-greedy-join-pivots-pilot-adjusted-0-760-but) · [Copy provenance](../../../PROVENANCE.json)

