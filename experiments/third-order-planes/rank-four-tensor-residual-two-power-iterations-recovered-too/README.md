# Rank-four tensor residual with two power iterations recovered too little error for its bill

2026-10-18T12:49:44Z · experiment · rejected · measured

A fixed rank-four tensor residual approximation and two power iterations recovered only a small fraction of signed-read error. A near-complete first-order oracle showed missing signal, rather than an algebra bug, limited the approximation.

Recorded measurement:

    Recovery 14.87/8.01%; bill +.100/.107%; full first-order oracle ~95%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_accuracy_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/tensor_pair_gram_probe_v1.py](code/tensor_pair_gram_probe_v1.py)
- [code/tensor_residual_r4_probe_v3.py](code/tensor_residual_r4_probe_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5676](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5676)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [TENSOR_RESIDUAL_R4_RESULT_v3.json](TENSOR_RESIDUAL_R4_RESULT_v3.json) · result
- [TENSOR_PAIR_GRAM_RESULT_v1.json](TENSOR_PAIR_GRAM_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-rank-four-tensor-residual-two-power-iterations-recovered-too) · [Copy provenance](../../../PROVENANCE.json)

