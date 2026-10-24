# Covariance negative-row cut with exact omitted variances: adjusted error rises 2.74827%

2026-09-27 · experiment · rejected · measured

The active principal covariance block and all omitted variances were retained, while covariances touching rows below -2.5 were removed. The omitted diagonal was transported as a fully charged weighted Gram.

Recorded measurement:

    TRAIN35/100:10/15 covariance transports cut per network;raw+3.11812%,bill-0.36239%,adjusted+2.74827%,0/2 wins;all pre-cut layers byte-identical.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_covariance_tail_block_v1.py](code/build_covariance_tail_block_v1.py)
- [code/candidate_covariance_tail_block_v1.py](code/candidate_covariance_tail_block_v1.py)
- [code/certify_covariance_tail_block_v1.py](code/certify_covariance_tail_block_v1.py)
- [code/native_eval_covariance_tail_block_v1.py](code/native_eval_covariance_tail_block_v1.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [RESULTS.md#L1123](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1123)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [covariance_tail_block_closeout_v1.json](covariance_tail_block_closeout_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-covariance-negative-row-cut-exact-omitted-variances-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

