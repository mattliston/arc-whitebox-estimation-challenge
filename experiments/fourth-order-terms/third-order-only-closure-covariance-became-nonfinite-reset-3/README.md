# Third-order-only closure: covariance became nonfinite at reset 3

2026-08-24 · experiment · rejected · measured

The exact A2_THIRD_ONLY source (SHA-256 43ee43386c450b33...) was independently audited as the intended no-fourth-cumulant canonical-polyadic carrier (CP4), no-fourth-order Hermite state (H4)/fourth-cumulant Edgeworth operator (L4), third-cumulant canonical-polyadic carrier (CP3)+third-order Hermite state (H3)+Gaussian+third-cumulant Edgeworth operator (L3)+one-half-L3-squared ablation with no source bug. It nevertheless failed its preregistered target-free scientific gate: the official-He width-32 fixture became nonfinite first at covariance reset 3 and third-cumulant repeated-index plane (K21) reset 4 while byte-identical V4 remained finite. Its finite width-8 and width-64 outputs also disagreed catastrophically with V4 (final mean squared error (MSE) 3.7206637e6 and 4.8400454e11).

Recorded measurement:

    The exact A2_THIRD_ONLY source (SHA-256 43ee43386c450b33...) was independently audited as the intended no-CP4, no-H4/L4, CP3+H3+Gaussian+L3+one-half-L3-squared ablation with no source bug. It nevertheless failed its preregistered target-free scientific gate: the official-He width-32 fixture became nonfinite first at covariance reset 3 and K21 reset 4 while byte-identical V4 remained finite. Its finite width-8 and width-64 outputs also disagreed catastrophically with V4 (final MSE 3.7206637e6 and 4.8400454e11). The registered exact resource-projection gate also failed, so no literal production meter, target/raw-MSE access, package, or submission was permitted. This closes only these exact A2 bytes; it does not supply the originally planned target-score contrast or close every materially different coherent quartic construction.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fullbudget_k21_k4_closure_accuracy_frontier_v1/verify_design.py](code/fullbudget_k21_k4_closure_accuracy_frontier_v1/verify_design.py)
- [code/fullbudget_k21_k4_closure_v1/estimator.py](code/fullbudget_k21_k4_closure_v1/estimator.py)
- [code/fullbudget_k21_k4_closure_v2/estimator.py](code/fullbudget_k21_k4_closure_v2/estimator.py)
- [code/fullbudget_k21_k4_closure_v3/estimator.py](code/fullbudget_k21_k4_closure_v3/estimator.py)

## Evidence

- [DESIGN_MANIFEST.json#L1](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/DESIGN_MANIFEST.json#L1)
- [PREREGISTER.md#L1](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/PREREGISTER.md#L1)
- [VARIANT_MATRIX.json#L1](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/VARIANT_MATRIX.json#L1)
- [SOURCE_BINDINGS.json#L1](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/SOURCE_BINDINGS.json#L1)
- [GRAPH_CHILDREN.json#L1](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/GRAPH_CHILDREN.json#L1)
- [REJECTION_RECEIPT-468120ba.json#L1](../../wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/REJECTION_RECEIPT-468120ba.json#L1)

<details>
<summary>All result files and records (6)</summary>

- [DESIGN_MANIFEST.json](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/DESIGN_MANIFEST.json) · result
- [PREREGISTER.md](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/PREREGISTER.md) · record
- [VARIANT_MATRIX.json](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/VARIANT_MATRIX.json) · result
- [SOURCE_BINDINGS.json](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/SOURCE_BINDINGS.json) · result
- [GRAPH_CHILDREN.json](../../wall-and-residual/pair-series-order-32-reduced-24-final-prediction-disagreement/GRAPH_CHILDREN.json) · result
- [REJECTION_RECEIPT-468120ba.json](../../wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/REJECTION_RECEIPT-468120ba.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-third-order-only-closure-covariance-became-nonfinite-reset-3) · [Copy provenance](../../../PROVENANCE.json)

