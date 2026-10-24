# Kink-, dead-unit and top-mode features fail to predict per-network dial gradients

2026-10-13T01:13Z · experiment · rejected · measured

candidate generation 53 (S53)'s per-network dial gradients on the A36 networks were correlated with per-network kink fraction, dead fraction and the ratio of the top two pre-activation eigenvalues at layer 15. A first read flagged kink-gated dial candidates (strongest k4la2 band 12-14 x lambda_2/lambda_1 r -0.70, n 22), but the count of hits across 556 series per feature is at or below what noise gives, so the kink-gated-dial line was closed and the gated-beta STEP 0 rule was narrowed to three features.

Recorded measurement:

    2780 tests; kink L12-15 (n 64): 2 at |r| >= 0.4 and 6 at >= 0.3 (chance about 0.8 and 11), largest 0.45; kink L8-11 0 / 7 (0.33); dead L12-15 0 / 9 (0.40); dead L8-11 0 / 5 (0.34); lambda_2 / lambda_1 L15 (n 22): 22 / 68 against about 36 / 95 by chance, largest k4la2 12-14 r -0.70; STEP 0 pass chance on noise alone about 1 in 3

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/grads_kink_corr_v1.py](code/grads_kink_corr_v1.py)
- [code/gbeta2_step0_v3.sh](code/gbeta2_step0_v3.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1852](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1852)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1854](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1854)

<details>
<summary>All result files and records (3)</summary>

- [pernet_grads_v1.json](pernet_grads_v1.json) · result
- [hard_nets_featA36_v1.json](hard_nets_featA36_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-kink-dead-unit-top-mode-features-fail-predict-per) · [Copy provenance](../../../PROVENANCE.json)

