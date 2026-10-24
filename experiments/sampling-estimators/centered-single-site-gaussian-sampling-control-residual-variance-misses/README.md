# Centered single-site Gaussian sampling control: residual variance misses optimistic score threshold by 544–625x

2026-09-27 · experiment · rejected · measured

A centered smoothed single-site Gaussian control was measured on actual network suffixes using antithetic pairs. It reduced antithetic variance but remained far above the optimistic classical independent-sampling threshold.

Recorded measurement:

    4096 antithetic pairs on35/100;residual variances.01869705/.02150322;captures38.6% of antithetic variance;misses2e-9 threshold544/625x.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/smoothed_site_cv_capture_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/control.py](code/control.py)
- [code/run.py](code/run.py)
- [code/unit.py](code/unit.py)
- [code/verify.py](code/verify.py)

## Evidence

- [RESULTS.md#L283](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L283)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-centered-single-site-gaussian-sampling-control-residual-variance-misses) · [Copy provenance](../../../PROVENANCE.json)

