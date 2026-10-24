# Spectral sample-correction opportunity bound

2026-09-05, approximately 12:41 PDT. Phase 2 only. No new sampling run,
training, estimator changes or submission. Official bank remains 329926.

Distinct question: an analytic mean can be corrected selectively in linear
combinations of output neurons, rather than mixing each coordinate with a
noisy sample mean. Previously measured affine/low-degree control variates
do not directly test this covariance-aware vector shrinkage. This audit
does not dispute their negative results or claim a new control variate.

Let Q be an orthonormal basis determined only by the current card's final
post covariance, e=Q.T@(true_mean-card_mean), and let c_j be the sampling
covariance diagonal in this basis. A fixed diagonal correction with gains
a_j has expected squared error sum_j [(1-a_j)^2 e_j^2+a_j^2 c_j/N]/1024.
The per-mode oracle a_j=e_j^2/(e_j^2+c_j/N) is not deployable: it knows the
actual deterministic error. Its risk is an optimistic opportunity screen
for diagonal shrinkage in this particular target-blind basis, not a lower
bound on every possible statistical estimator. The risk identity uses
only iid sample-mean covariance, not a Gaussian-noise approximation.

Use existing rank35/100 documented1e9 means and coherent8M post covariances.
The latter are float32 and noisy: flag negative projected variances, and
clip them to zero only for this explicitly optimistic bound. Also report
the current card's variance model separately. Current scalar-final mean
must match the archived bank reference to 1e-12 or better.

Evaluate only N=512,2048,8192,32768,131072. Price the sample forwards with
the already certified/metered matrix-kernel polynomial, and include the
existing bank's measured first-context FLOPs. Omit eigensolver, random
generation, correction and sample reduction costs to remain optimistic;
do not mislabel this as an actual candidate bill or runtime measurement.

If no common sample count predicts >=15% adjusted improvement on BOTH
development networks even with oracle gains, close this exact approach
without samples or a fit. If it passes, gains still require target-blind
cross-network calibration, actual independent samples, final-layer
accuracy replication, complete billing and runtime checks before any
submission. A per-mode fitted oracle number is never a local score.
