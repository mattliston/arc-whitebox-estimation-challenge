# Gradient-directed dial line search lowers error by 1.44% on 50 held-out networks

2026-10-02T07:00Z · experiment · adopted · measured

Candidate 1 (learned mean correction x 0.8, per-unit kurtosis term 0.75 on both ages, covariance knob k21n -0.048) gave -0.98% on the 50 held-out judging ranks; candidate 2 added a line-search step along the exact dial gradient chosen on 16 other ranks (b-source 0.965, bscale 1.03414, X1 0.59158, column-shift strength 0.656 at age 1) and gave -1.44%. All changes are literals (no bill); the native build on ten fresh ranks read -2.03% against S37 (-0.98% against candidate 1) and the board -1.65%.

Recorded measurement:

    candidate 1: -0.980% pooled (se 0.234, 38 of 50), all layers -0.573%; candidate 2: final-layer MSE 1.5985e-8 vs 1.6218e-8 = -1.442% (mean -1.450%, se 0.255, 40 of 50), all layers -1.148%; line search on 16 other ranks -0.42% at half step, -0.29% at full, minimum near 0.8; values: correction x0.8 [1], K4L 0.75 / 0.75 [1 / 1], k21n -0.048 [-0.028], b-source 0.965 [1], bscale 1.03414 [1.03834], X1 0.59158 [0.55658], F4 age 1 0.656 [0.6]; native ten fresh vs S37: -2.03% [-3.12,-0.99], 9 of 10; vs dials-1 build -0.98% [-1.56,-0.35]

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/candidates`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/runM_dials_ds08/candidate.py](code/runM_dials_ds08/candidate.py)
- [code/runM_dials2/candidate.py](code/runM_dials2/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L148](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L148)
- [TEAM_LOG_2026-10-01_to_10-22.md#L164](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L164)
- [TEAM_LOG_2026-10-01_to_10-22.md#L165](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L165)
- [TEAM_LOG_2026-10-01_to_10-22.md#L171](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L171)
- [RESULT.md#L551](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md#L551)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [quick_s38_dials2_fresh_a.json](quick_s38_dials2_fresh_a.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333481: fitted dials, masked covariance correction and three-request operand loads](../../../submissions/333481/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-gradient-directed-dial-line-search-lowers-error-1-44) · [Copy provenance](../../../PROVENANCE.json)

