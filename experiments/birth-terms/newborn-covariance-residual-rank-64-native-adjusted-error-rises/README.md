# Newborn covariance-residual rank 64: native adjusted error rises 32.4964%

2026-09-26 · experiment · rejected · measured

Newborn transport reused a scaled covariance column plus a rank-64 residual while restoring its diagonal exactly. A corrected nonsymmetric algebra witness passed, but a small transported-matrix residual did not preserve the native endpoint.

Recorded measurement:

    Low-rank transported relative error 3.8–5.4%; native TRAIN2 raw +35.1188%, bill -1.9346%, adjusted +32.4964%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/capture_newborn_residual_v1.py](code/capture_newborn_residual_v1.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [RESULTS.md#L378](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L378)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [newborn_residual_k64_train2_comparison_v1.json](newborn_residual_k64_train2_comparison_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-newborn-covariance-residual-rank-64-native-adjusted-error-rises) · [Copy provenance](../../../PROVENANCE.json)

