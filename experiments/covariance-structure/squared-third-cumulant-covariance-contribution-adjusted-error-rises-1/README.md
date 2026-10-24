# Squared third-cumulant covariance contribution: adjusted error rises 1.59060%

2026-09-26T22:37Z · experiment · rejected · measured

The r=0 connected covariance contribution from half the squared third-cumulant operator was added at layers2–14 using existing modeled cumulants. Algebra and native-kernel checks passed, but the native endpoint worsened.

Recorded measurement:

    TRAIN2 versus #332517: raw+1.47445%, compute+0.113824%, adjusted+1.59060%,0/2 wins;605 rational moment cases pass; FP32 relative kernel error9.83e-8; free optimal signed output-component diagnostic only0.21027% pooled raw gain.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_quadratic_third_covariance_v1.py](code/candidate_quadratic_third_covariance_v1.py)

## Evidence

- [RESULTS.md#L668](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L668)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-squared-third-cumulant-covariance-contribution-adjusted-error-rises-1) · [Copy provenance](../../../PROVENANCE.json)

