# Direct fourth-plane pair-kernel information test

2026-09-06, Phase2. This tests a new supervised object: actual missing
pre13 K22/K31/k4 planes, not the previous total covariance innovation,
mean-head features, or a fourth-trace carrier. Third-order errors dominated
the old covariance labels; they are held fixed in this isolated test.
No old optimizer, feature-width or trace-metric sweep is reopened.

Use only the already frozen rank35 pre13 bank inputs and the two qualified
independent8M fourth-plane A/B statistic files. No new MC, nets or layers.
The result is an ORACLE COEFFICIENT-SPAN test on one cell, not cross-network
generalization, trained-estimator accuracy, or a deployable correction.

## Fixed input-only features and models

Reuse the established14 analytic endpoint responses:
g,pdf,alpha*pdf,(alpha^2-1)*pdf,host/sigma,alpha*g,
skew times(g,pdf,alpha*pdf,host/sigma), and kurt times those same four.
Whiten the exact column space with an SVD, removing only singular values
<=1e-12 times the largest (numerical dependencies, not a rank search).
Require the retained span reconstructs the original features to1e-10.
Use three off-diagonal correlation powers R^0,R^1,R^2, each divided by its
input-only RMS. Explicitly zero their diagonals. No additional features.

Standardized K22 and K31 increments are
sum_r Rpow_r * H@Theta_r@H.T, symmetric Theta for K22, unrestricted for
K31. Standardized marginal k4 increment is H@beta. K22/K31 diagonal
entries are not fitted; the separate marginal supplies their effect.

Fit each stream independently with ridge1e-3 and the exact linear normal
operator, using fixed diagonally preconditioned CG: relative residual
<=1e-7, maximum300 iterations. Scalar ridge solve is analytic after input
whitening. No target-dependent clipping or checkpoint/regularizer/iteration
selection. A numerical failure stops the round as unqualified; no solver
sweep. First certify small adjoint/normal-operator and dense-solve parity.

## Independent-replica value gate

Each independently fitted fourth field enters the same exact finite pair
response and next W14 variance-to-mean sensitivity used by the two earlier
trace oracle tests. All six target response arrays must be byte-identical
to those receipts. Report raw plane and response signal/noise estimates.

Require positive cross-signal and, for BOTH covariance and next-mean
responses, residual/baseline cross-energy<=0.20 and BOTH cross-replica
ordinary prediction errors<=0.30. This strong information gate is warranted
before acquiring cross-net joint-plane labels or building a supplier.
Because coefficients are fitted separately on the same physical cell,
passing is only evidence of a simple representational opportunity.

Failure closes this specified field span: no feature/correlation-order/
rank/regularization/iteration/sampling/fit expansion. A pass would need
a separately designed genuine cross-net test, co-adaptation, cost, and
14-panel/runtime gates. No state writes, uploads, forum posts or commits.
