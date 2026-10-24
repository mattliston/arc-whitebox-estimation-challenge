# One fixed full-skew-plane covariance information test

Phase2 only, September5,2026. Design before new target measurement.
No bank edits, submissions or full-card fitting authorized by this test.

## New information and its limitation

For standardized covariance R and full repeated third plane S_ij=T_iij,
the known quadratic-variable regression Gram is D=S R^{-1} S^T. The
existing covariance-innovation basis and shared-vertex dictionary do not
contain S or this Gram. The earlier Schur audit tested only positivity,
not whether this Gram predicts the missing covariance innovation.

The perturbative Gaussian quadratic transport

  X_i = Z_i + T_iab [(R^{-1} Z)_a (R^{-1} Z)_b - (R^{-1})_ab]/6

reproduces symmetric T to first order and has connected fourth cumulant
at quadratic order

  Q_ijkl = (4/9) [T_ij: R^{-1} T_kl: + two other pairings].

Full T_ij: is expensive. For i != j, use ONLY its minimum R^{-1}-norm
extension from the two known entries S_ij and S_ji. Thus

  H_ij = (S_ij^2 + S_ji^2 - 2 R_ij S_ij S_ji)/(1-R_ij^2),
  J_ij = [S_ii(S_ij-R_ij S_ji) + S_ij(S_ji-R_ij S_ij)]/(1-R_ij^2),
  Q22 = (4/9)(D+2H), Q31=(4/3)J, q4_i=(4/3)D_ii.

For i=j set H_ii=J_ii=D_ii. This is a PROXY, not a certified global
fourth tensor: omitted conditional components need not be small, and the
pairwise extensions need not be mutually consistent. No positivity repair,
rank sweep, claim of exact closure, or use of target-derived directions.

Use the current active block (gate floor .01, same rounding/order as the
parent), a float64 positive-definite solve, and a float64 Gram. Reject a
failed Cholesky or relative solve residual >1e-8; no regularization sweep.
Small Gaussian-Wick and conditional-projection certificates come first.

## Fixed inference cost gate

Only L8..14 writes are proposed. Count symmetric full-RHS solve, Gram,
assembly and three linear pair-response columns. Price float64 honestly.
Allow at most 15% extra total FLOPs on both frozen35/100 active tapes.
This is an analytical pre-gate, not a native memory/latency/bill receipt.

## Information protocol

Replay the exact coherent-8M oracle-previous-state protocol from
`generate_covariance_innovation_probe_v2.py`. Freeze R, S, marginal
quantities and host mean before current-target feedback. Assert the old
14 input columns, R and covariance targets match the frozen v2 rows.
No new Monte Carlo or sealed targets. Train35/100/150/200/250/300;
holdout400/500. Layers8..14, same as the old covariance diagnostic.

Build exactly three physical response columns, using the parent's
first-order Mehler covariance response through correlation order2:

1. scaling of the current third plane including its marginal diagonal;
2. scaling of the current radial fourth tensor;
3. the new quadratic proxy's Q22, Q31 and q4 jointly.

Also fit the two-column control (1,2). Fixed normalized ridge .001;
objective equally weights covariance MSE divided by per-net baseline
energy and next-Gaussian-mean response MSE divided by its baseline energy.
Training coefficients are frozen before holdout rows are evaluated. No
coefficient, layer, architecture, rank, clipping or ridge sweep.

To continue, BOTH holdouts must have covariance and next-mean response
error ratios <=.75 versus no correction, and the three-column result must
beat the two-column control by >=10% in next-mean error on EACH holdout.
Otherwise close this fixed proxy. A pass is only an information result;
it still requires an honest current-state rollout, coadaptation, independent
14-net panel, native metering, memory and persistent wall-time gates.
