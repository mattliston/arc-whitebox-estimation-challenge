# Float64 covariance diagonal reductions: native adjusted gain is 0.251635%, below expansion gate

2026-09-27 · experiment · inconclusive · measured

Input-layer weight norms and hidden outer-diagonal reductions were widened to float64 while original matrix products and off-diagonal arithmetic remained. Both cases improved, but the estimate missed the fixed1% expansion gate.

Recorded measurement:

    TRAIN35/100 raw-0.294540%,bill+0.043215%,adjusted-0.251635%,2/2 wins;scalar helper agreement5.56e-17;hidden variances remain approximate.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/covariance_diagonal_reduction_v1/assess.py](code/covariance_diagonal_reduction_v1/assess.py)
- [code/covariance_diagonal_reduction_v1/assess_v2.py](code/covariance_diagonal_reduction_v1/assess_v2.py)
- [code/covariance_diagonal_reduction_v1/build.py](code/covariance_diagonal_reduction_v1/build.py)
- [code/covariance_diagonal_reduction_v1/candidate.py](code/covariance_diagonal_reduction_v1/candidate.py)
- [code/covariance_diagonal_reduction_v1/candidate_v2.py](code/covariance_diagonal_reduction_v1/candidate_v2.py)
- [code/covariance_diagonal_reduction_v1/unit.py](code/covariance_diagonal_reduction_v1/unit.py)
- [code/covariance_diagonal_reduction_v1/unit_v2.py](code/covariance_diagonal_reduction_v1/unit_v2.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [RESULTS.md#L1480](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1480)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-float64-covariance-diagonal-reductions-native-adjusted-gain-0-251635) · [Copy provenance](../../../PROVENANCE.json)

