# Hermite pair-series order sweep: order 4 reached 3.849300358689406e-6 final raw error; orders 4–10 tied

2026-08-23T17:40Z · experiment · adopted · measured

A fresh full-covariance Hermite/Mehler analytic host (float64, 1/sqrt(k!)-normalised coefficients verified against direct Gauss-Hermite integrals to 1.381e-7) was scored once on the sealed mini100 targets at orders 1, 2, 4, 6 and 10. Order 4 is the best pure analytic baseline and higher univariate Hermite order is not the remaining bottleneck. The first-layer prediction matches the exact Gaussian/ReLU formula sqrt(sum_i W_ij^2)/sqrt(2 pi) to float32 precision and its scored residual equals the reference Monte Carlo noise floor, so layer 1 is exact and layer 3 is the first material failure.

Recorded measurement:

    final raw MSE: order 1 4.050566386127281e-6, order 2 3.850312259546773e-6, order 4 3.849300358689406e-6 (all-layer 2.1180294255017804e-6, layer 2 1.5536452260972563e-7), order 10 3.849300547399477e-6; metered FLOPs per MLP: order 4 103,519,094,784, order 10 103,923,320,832 (0.04726 of budget); layer-1 prediction vs exact formula MSE 2.944948632584076e-16, scored layer-1 MSE 6.881554782955935e-10 vs expected reference noise 6.818022375182014e-10 (ratio 1.0093183043818075); order-10 bivariate truncation error 3.763e-5

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analytic_hermite10_full_covariance/estimator.py](code/analytic_hermite10_full_covariance/estimator.py)
- [code/data_gate/target_blind_mini.py](code/data_gate/target_blind_mini.py)
- [code/exact_layer_foothold/verify_exact_layer_foothold.py](code/exact_layer_foothold/verify_exact_layer_foothold.py)

## Evidence

- [BUDGET_AUDIT.json#L1](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/BUDGET_AUDIT.json#L1)
- [L1_NOISE_FLOOR.json#L1](L1_NOISE_FLOOR.json#L1)
- [MATH_VERIFICATION.json#L1](MATH_VERIFICATION.json#L1)

<details>
<summary>All result files and records (7)</summary>

- [MATH_VERIFICATION.json](MATH_VERIFICATION.json) · result
- [PREREGISTRATION.md](PREREGISTRATION.md) · record
- [seal.json](seal.json) · result
- [0b3c04441938e2930f4d8fc92e9ca788d0dcfcc4f0107a8b152867df735d5a7e.joint-score.json](0b3c04441938e2930f4d8fc92e9ca788d0dcfcc4f0107a8b152867df735d5a7e.joint-score.json) · result
- [L1_NOISE_FLOOR.json](L1_NOISE_FLOOR.json) · result
- [DIAGNOSTIC.md](DIAGNOSTIC.md) · record
- [BUDGET_AUDIT.json](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/BUDGET_AUDIT.json) · result

</details>

## Related work

- ← supersedes: [Third-cumulant carrier caps 512–1792: small endpoint gains, layer-2 error saturated at cap 1024](../../third-order-planes/third-cumulant-carrier-caps-5121792-small-endpoint-gains-layer/README.md)

[Research graph](../../../graph/index.html#node=2026-08-23-hermite-pair-series-order-sweep-order-4-reached-3) · [Copy provenance](../../../PROVENANCE.json)

