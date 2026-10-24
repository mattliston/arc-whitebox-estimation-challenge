# Audit a score-derived covariance-error metric before any new fitting

September6,2026, Phase2. This is a mathematical/read-only loss audit,
NOT a restart of the closed endpoint regression. That fixed ridge optimum
bound applies to its K21-weighted objective, not every objective or an
unregularized capacity ceiling. No coefficients are changed here.

For a fresh Gaussian weight column w~N(0,tau2 I), freeze variance at
v0=tau2 tr(Cpost), mean at m^T w. The Gaussian variance-to-ReLU-mean
derivative is phi(m^T w/sqrt(v0))/(2sqrt(v0)). For symmetric error E,
integrating the SQUARED mean response reweights w by a Gaussian tilt:

    t = 2 tau2 ||m||^2 / v0
    A = I - [t/(1+t)] u u^T, u=m/||m||
    risk(E) = (1+t)^(-1/2) tau2^2/(8 pi v0)
              * [2 tr(A E A E) + tr(A E)^2].

This exactly integrates the FROZEN-variance surrogate, not actual varying
w^T Cpost w, finite-network final score or the learned final-head response.
It is different from the closed conditional-weight attribution, which
estimated the conditional MEAN quadratic action from trace/mean-ray only;
here the full squared random quadratic fluctuation is retained. It also
differs from a generic unweighted covariance Frobenius norm.

Use the already frozen mini57/post13 and the existing third-plane span's
target/predicted covariance matrices. Reconstruct current mean feedback
and mask from target-free inputs; require exact next pre14 covariance
and mean parity with saved bank arrays. No true state enters this geometry.
Certify the formula using explicit small Gaussian fourth moments and the
equivalent tilted covariance matrix, including sign-indefinite E.

Then two fixed independent4096 fresh Gaussian WEIGHT directions (not MLP
input samples or a sampling estimator), PCG64DXSM seeds930084/930085,
batch512, tau2=2/1024. Compute squared responses of target, prediction,
and residual using frozen v0 and actual varying w^T Cpost w. Keep all three
fields; no selected arm, weight/direction count or seed sweep. Require
frozen-variance analytic energy within4 batch standard errors of MC for
all fields; variable-variance energy within25% of analytic; and analytic
residual/baseline ratio within15% relative of variable-variance MC ratio.
Report uncertainty and unweighted Frobenius/actual W14 ratios separately.

This audit can establish a useful risk metric, not that a new optimum is
learnable. No new fit, target download, field/basis/objective sweep, carrier
write, raw/adjusted score, panel or submission is authorized by this file.
A pass needs a separately reasoned model/optimization decision; a fail
closes this constant-variance risk approximation without tuning it.
