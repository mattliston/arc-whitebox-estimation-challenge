# Rank-128 covariance approximation: next-variance disturbance 0.81-0.92% RMS on saved networks

2026-10-15T15:41:47Z · experiment · rejected · measured

The historical training-factor spectral oracle showed that rank-128 covariance truncation injected mean-error energy exceeding the whole mean squared error, so no candidate was evaluated.

Recorded measurement:

    nextvariance disturbance .81-.92% RMS on historic TRAIN16; mean error energy > whole MSE

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_diagonal_lowrank_covariance_oracle_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/inspect_train16_rank128_v1.py](code/inspect_train16_rank128_v1.py)
- [code/inspect_train16_rank128_v2.py](code/inspect_train16_rank128_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3406](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3406)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [TRAIN16_SPECTRAL128_DIAGONAL_v2.json](TRAIN16_SPECTRAL128_DIAGONAL_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-rank-128-covariance-approximation-next-variance-disturbance-0-81) · [Copy provenance](../../../PROVENANCE.json)

