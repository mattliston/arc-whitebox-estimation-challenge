# Density-weighted greedy pivots: MSE -0.141/-0.217% with standard errors above 1%

2026-10-16T04:14:58Z · experiment · rejected · measured

Pure Gaussian-density weighting and a half-weighted variant did not meet the fixed 0.3% gain criterion. Pivot response varied strongly in sign across networks.

Recorded measurement:

    purepdf-.141%,SE1.09,5/8;half-.217%,SE1.06,4/8;bill0;ops+27;perseeds-4.3..+4.9%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_wpivot_v1/build_wpivot_v1.py](code/opus_wpivot_v1/build_wpivot_v1.py)
- [code/opus_wpivot_v1/cand_2b108_r256_wpivotP1_v1.py](code/opus_wpivot_v1/cand_2b108_r256_wpivotP1_v1.py)
- [code/opus_wpivot_v1/cand_2b108_r256_wpivotP2_v1.py](code/opus_wpivot_v1/cand_2b108_r256_wpivotP2_v1.py)
- [code/opus_gbfix_v1/cand_2b108_frag_r256_v1.py](code/opus_gbfix_v1/cand_2b108_frag_r256_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3841](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3841)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [WPIVOT_RESULT_v1.json](WPIVOT_RESULT_v1.json) · result
- [fragment_join_greedy_r256_2b108lineage_v1.py.txt](../static-plus-one-256-column-greedy-block-eight-network/fragment_join_greedy_r256_2b108lineage_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-density-weighted-greedy-pivots-mse-0-141-0-217) · [Copy provenance](../../../PROVENANCE.json)

