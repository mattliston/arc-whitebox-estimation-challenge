# Pair-map and readout-feature vector batching removed 97–98 requests with bit-identical predictions

2026-10-17T17:11:39Z · experiment · informational · measured

Broadcast and stacked multiplies combined repeated pair-map operands, while stacked divides and square roots combined readout-feature operations. The fragment preserved prediction bytes and rebound the dense-core function identity; matrix covariance batching was omitted because copying cost more than the request saving.

Recorded measurement:

    Ranks 35/100 bit-identical; requests -97/-98; pair calls 1872→1817; feature calls 630→588; bill +~0.0001%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_vec_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_vecB_v1.py](code/candidate_vecB_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4894](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4894)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [fragment_pair_vec_v1.py.txt](fragment_pair_vec_v1.py.txt) · code
- [fragment_features_vec_v1.py.txt](fragment_features_vec_v1.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-pair-map-readout-feature-vector-batching-removed-9798-requests) · [Copy provenance](../../../PROVENANCE.json)

