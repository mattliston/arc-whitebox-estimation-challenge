# Higher Gaussian pair-series and second-order mean terms are too small and uncorrelated with remaining errors

2026-10-20T08:31:31Z · experiment · rejected · measured

The omitted Gaussian pair-series remainder was 30x smaller than measured pre-variance error and uncorrelated with it. Second-order Edgeworth mean terms were 12x smaller than pre-mean error and also uncorrelated.

Recorded measurement:

      (4) Gaussian Mehler truncation (the pair map keeps n = 1, 2 of sum_n C^n/n! h_n h_n^T): |rho| mean 0.07-0.10 (max 0.58); the n >= 3 remainder is 2-3e-6 rms vs T2 6-10e-5, pushed through W it is 6-8e-6 rms vs the measured dv 2e-4 (30x smaller) and uncorrelated (|corr| < 0.09). Second-order Edgeworth mean terms (k3^2/72 h6 + k3 k4/144 h7 + k4^2/1152 h8): 8-10e-6 rms vs first-order terms 5e-4, pushed 1.1-1.4e-5 vs the measured pre-mean error 1.3-1.6e-4 (12x smaller), uncorrelated (|corr| <= 0.10). Both DEAD as zero-bill terms.
    [fable-lead] 2026-10-20T08:31:31Z FOUR OFFLINE CLOSURES on the late-layer error (nets 35/100, card quantities from the k21dump files at L8-14 vs the 32M mcall truth; lead/runs/fable_oct19_kc_v1/{dcov_structure,dv_regress,k22cov_offline,mehler_offline,edge2_offline}_v1.py):

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/mehler_offline_v1.py](code/mehler_offline_v1.py)
- [code/edge2_offline_v1.py](code/edge2_offline_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6151](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6151)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6147](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6147)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-higher-gaussian-pair-series-second-order-mean-terms-too) · [Copy provenance](../../../PROVENANCE.json)

