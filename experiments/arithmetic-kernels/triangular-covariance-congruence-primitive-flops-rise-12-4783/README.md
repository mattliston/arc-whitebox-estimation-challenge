# Triangular covariance congruence: primitive FLOPs rise 12.4783%

2026-09-26T23:30Z · experiment · rejected · measured

A recursive triangular product with64-row leaves and a block symmetric Gram was tested on a synthetic positive-definite input. Lower CPU time did not compensate for a larger metered primitive bill.

Recorded measurement:

    Primitive bill1,837,433,984 ->2,066,714,069 FLOPs,+12.4783%; FP64-reference errors3.56e-5(parent)/1.58e-5(proposal); CPU roughly halved.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/price_triangular_covariance_v1.py](code/price_triangular_covariance_v1.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)

## Evidence

- [RESULTS.md#L820](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L820)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-triangular-covariance-congruence-primitive-flops-rise-12-4783) · [Copy provenance](../../../PROVENANCE.json)

