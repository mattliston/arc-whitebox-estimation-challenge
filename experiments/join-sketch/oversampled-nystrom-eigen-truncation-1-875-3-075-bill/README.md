# Oversampled Nystrom with eigen truncation: +1.875/+3.075% bill for 64/128 extra pivots

2026-10-16T04:05:33Z · experiment · rejected · measured

The small-core eigendecomposition tariff and wider products exceeded the remaining join oracle opportunity. Even free truncation would leave excessive cost, so no multi-network screen ran.

Recorded measurement:

    p64+1.875%/+360ops;p128+3.075%/+470;plainMSE~-.09%;p64+4.52G=eigh2.13G+products~1.5G+solve.63G+QR.41G

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_oversample_v1/build_ovs_v1.py](code/opus_oversample_v1/build_ovs_v1.py)
- [code/opus_oversample_v1/cand_2b108_r256_ovs128_v1.py](code/opus_oversample_v1/cand_2b108_r256_ovs128_v1.py)
- [code/opus_oversample_v1/cand_2b108_r256_ovs64_v1.py](code/opus_oversample_v1/cand_2b108_r256_ovs64_v1.py)
- [code/opus_gbfix_v1/cand_2b108_frag_r256_v1.py](code/opus_gbfix_v1/cand_2b108_frag_r256_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3824](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3824)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [OVERSAMPLE_RESULT_v1.json](OVERSAMPLE_RESULT_v1.json) · result
- [diag_ovs64_7103248.json](diag_ovs64_7103248.json) · result
- [diag_r256_7103248.json](diag_r256_7103248.json) · result
- [guard_diag64.json](guard_diag64.json) · result
- [guard_diagr256.json](guard_diagr256.json) · result
- [fragment_join_greedy_r256_2b108lineage_v1.py.txt](../static-plus-one-256-column-greedy-block-eight-network/fragment_join_greedy_r256_2b108lineage_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-oversampled-nystrom-eigen-truncation-1-875-3-075-bill) · [Copy provenance](../../../PROVENANCE.json)

