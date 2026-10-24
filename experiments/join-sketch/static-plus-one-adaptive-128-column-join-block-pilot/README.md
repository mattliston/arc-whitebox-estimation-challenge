# Static-plus-one adaptive 128-column join block: pilot adjusted -0.475%, request risk 19.64%

2026-10-16T03:08:32Z · experiment · adopted · measured

Restricting adaptive selection to one 128-column block supplied a smaller but gate-compatible improvement. The full block-greedy variant needed further first-network request reductions.

Recorded measurement:

    xMSE-.991%,SE.863,5/8;bill+.516%;xnet-.475%;first48875/warm45993;risk19.64%;worst104.1s;fullblockrisk23.10%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3763](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3763)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3761](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3761)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3766](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3766)

<details>
<summary>All result files and records (5)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_join_greedy_gb128_v1.py.txt](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/fragment_join_greedy_gb128_v1.py.txt) · result
- [fragment_join_greedy_gb128r_2b108lineage_v1.py.txt](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/fragment_join_greedy_gb128r_2b108lineage_v1.py.txt) · result
- [fragment_join_greedy_h128_2b108lineage_v1.py.txt](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/fragment_join_greedy_h128_2b108lineage_v1.py.txt) · result
- [JOINX_PRICE_v1.json](../../../records/research/2026-10-16-join-sketch-alternatives-priced-adaptive-pivots-0-325-0/JOINX_PRICE_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336381: one adaptive 128-column join block on compact kernels](../../../submissions/336381/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-static-plus-one-adaptive-128-column-join-block-pilot) · [Copy provenance](../../../PROVENANCE.json)

