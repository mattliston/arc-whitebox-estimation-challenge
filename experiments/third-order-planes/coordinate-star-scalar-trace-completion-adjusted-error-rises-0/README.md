# Coordinate-star scalar trace completion: adjusted error rises 0.18467%

2026-09-27 · experiment · rejected · measured

For residual D=A-QB, the scalar third-moment read gained the spherical trace contribution ||w||²/(n+2)*w.T@(D@1+2*diag(D)). Residual row sums and diagonals avoid materializing D, but the complete native endpoint worsened.

Recorded measurement:

    TRAIN2 raw+0.11600%,adjusted+0.18467%,0/2 wins;bill+0.06849%;CPU+2.64511%;15 executed star calls;explicit signed tensor certificate9.77e-15.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_star_residual_trace_v1.py](code/build_star_residual_trace_v1.py)
- [code/candidate_star_residual_trace_v1.py](code/candidate_star_residual_trace_v1.py)
- [code/certify_star_residual_trace_v1.py](code/certify_star_residual_trace_v1.py)
- [code/native_eval_star_trace_v1.py](code/native_eval_star_trace_v1.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [RESULTS.md#L1070](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1070)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-coordinate-star-scalar-trace-completion-adjusted-error-rises-0) · [Copy provenance](../../../PROVENANCE.json)

