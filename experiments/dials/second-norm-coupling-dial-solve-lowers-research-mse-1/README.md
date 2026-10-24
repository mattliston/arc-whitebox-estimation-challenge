# Second norm-coupling dial solve lowers research MSE 1.03% on 25 ranks; native benefit disappears

2026-10-02T16:57Z · experiment · rejected · measured

The second norm-coupling dial solve, including young-pair plane strengths by cohort age, lowers research final-layer MSE 1.03% on 25 judging ranks against its constant-regressor reference; four fifths comes from the law dials and the (3,1) row factor. The complete native four-file comparison later showed the first dials on that reference were 1.31% worse than the submitted table-based file, so the second solve recovered that loss and supplied no measurable improvement over submission #333587.

Recorded measurement:

    model -0.57% (leave-one-out +0.15%), real -0.50% (se 0.24, 13 of 16); 25 ranks: 1.4243e-8 vs 1.4390e-8 = -1.027% (se 0.206, 19 of 25); without age 3 -0.840% (se 0.194, 21 of 25); ages 1-2 at 1.02 / 0.77 no age 3 -0.762%; age-3 term -0.267% (se 0.121); values: c0 0.6954, k31 -0.1451, fold strength 1.850 / 1.207 / 1.167, factor 1.307, births 1.305, rows 0.854, diag 0.905, k31l 1.023 / 0.773 / 0.716; native 651 / 378 vs S41: table -1.87% / -1.06% (pooled -1.46%), constant regressor +0.31% / -0.65% (pooled -0.18%); final native ten-network comparison vs submission #333587: first dials with constant regressor +1.31% (se 0.24, 10 of 10 worse), second dials with constant regressor +0.14% (se 0.30), second dials with table -0.18% (se 0.20); native second vs first on constant regressor -1.15% (se 0.27, 9 of 10), research -0.84% (se 0.19)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/candidates/law_dials_d7`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L377](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L377)
- [TEAM_LOG_2026-10-01_to_10-22.md#L380](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L380)
- [TEAM_LOG_2026-10-01_to_10-22.md#L388](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L388)
- [TEAM_LOG_2026-10-01_to_10-22.md#L390](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L390)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L600](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L600)
- [RESULT.md#L877](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md#L877)
- [TEAM_LOG_2026-10-01_to_10-22.md#L394](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L394)
- [TEAM_LOG_2026-10-01_to_10-22.md#L396](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L396)
- [TEAM_LOG_2026-10-01_to_10-22.md#L399](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L399)
- [TEAM_LOG_2026-10-01_to_10-22.md#L404](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L404)

<details>
<summary>All result files and records (5)</summary>

- [physics_dials.json](physics_dials.json) · config
- [physics_dials-4f238e34.json](physics_dials-4f238e34.json) · config
- [RESULT.md](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-second-norm-coupling-dial-solve-lowers-research-mse-1) · [Copy provenance](../../../PROVENANCE.json)

