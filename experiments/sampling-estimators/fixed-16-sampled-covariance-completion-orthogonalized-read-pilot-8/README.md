# Fixed-16 sampled covariance completion on the orthogonalized read: pilot-8 adjusted +0.049%

2026-10-15T05:55:20Z · experiment · rejected · measured

A 16-sample covariance completion of the refined read (combined sample-plus-tail feature with a remote float64 coefficient; about 0.036% bill, 336 charged ops) did not beat the QR read on the 8-network pilot.

Recorded measurement:

    vs QR 4d: adjusted +0.048680457% (MSE +0.012647924%, bill +0.036032533%); vs e406 +0.014570723%; vs d401 -0.441305916%; synthetic price 86,538,958 FLOPs (~0.03574%), 336 ops

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_rtt2_covariate_feasibility_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/kernel_fixed16_covariate_HELD_v1.py](code/kernel_fixed16_covariate_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3368](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3368)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3370](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3370)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3372](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3372)

<details>
<summary>All result files and records (3)</summary>

- [FIXED16_FORMULA_HELD_v1.json](FIXED16_FORMULA_HELD_v1.json) · record
- [FIXED16_SAMPLED_COVARIATE_TARIFF_v1.json](FIXED16_SAMPLED_COVARIATE_TARIFF_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-fixed-16-sampled-covariance-completion-orthogonalized-read-pilot-8) · [Copy provenance](../../../PROVENANCE.json)

