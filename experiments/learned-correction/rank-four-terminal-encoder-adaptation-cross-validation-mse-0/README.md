# Rank-four terminal encoder adaptation: cross-validation MSE +0.035696%

2026-10-15T20:42:48Z · experiment · rejected · measured

Recovering the positive-SiLU hidden correction enabled a genuine encoder-only fixed 500-step adaptation. The fit failed the required error and fold-consistency gates.

Recorded measurement:

    hidden1/2maxabs9.54e-7;CV+.035696%,SE.054476%,3/8folds,15/32wins;gate-.3%/6folds;24.98s585MB

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_enc0_rank4_fit_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fit_enc0_rank4_HELD_v1.py](code/fit_enc0_rank4_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3527](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3527)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3525](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3525)

<details>
<summary>All result files and records (12)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BLOCKED_CV_RESULT_v1.json](BLOCKED_CV_RESULT_v1.json) · result
- [COMPUTE_SOURCE_PEER_PASS_v1.json](COMPUTE_SOURCE_PEER_PASS_v1.json) · result
- [ACTUAL_FIT_RETIREMENT_v1.json](ACTUAL_FIT_RETIREMENT_v1.json) · result
- [fold0_FREEZE_v1.json](fold0_FREEZE_v1.json) · result
- [fold1_FREEZE_v1.json](fold1_FREEZE_v1.json) · result
- [fold2_FREEZE_v1.json](fold2_FREEZE_v1.json) · result
- [fold3_FREEZE_v1.json](fold3_FREEZE_v1.json) · result
- [fold4_FREEZE_v1.json](fold4_FREEZE_v1.json) · result
- [fold5_FREEZE_v1.json](fold5_FREEZE_v1.json) · result
- [fold6_FREEZE_v1.json](fold6_FREEZE_v1.json) · result
- [fold7_FREEZE_v1.json](fold7_FREEZE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-rank-four-terminal-encoder-adaptation-cross-validation-mse-0) · [Copy provenance](../../../PROVENANCE.json)

