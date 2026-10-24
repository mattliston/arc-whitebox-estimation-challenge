# Local Gaussian three-residual source: fixed precision screen

This is a research-only necessary-precision test, not a submitted estimator
or a final-score experiment. It uses the sealed c2 rank35 pre8 packet and
consumer W9. No Monte Carlo truth is opened and no coefficients are fitted.
The source differs from the closed physical-prefix tangent innovation test:
it samples only the local Gaussian reference and contains no tangents.

For Z ~ N(0,Cpre), define

    R = alive * (ReLU(mu+Z) - E_G ReLU(mu+Z) - Phi(mu/sigma)*Z).

The all-distinct third-ray contribution for row q is

    (q R)^3 - 3 (q R)(q^2 R^2) + 2(q^3 R^3).

The measured response multiplies this by the frozen consumer9 scalar
third-cumulant mean derivative, -alpha9*pdf9/(6*sigma9^2).
This is one Gaussian source sector, not the full physical innovation, and
does not establish that it is disjoint from every term in the learned bank.

The primary estimator averages +/-Z antithetic pairs. Two independent
batches of exactly512 pairs use fixed seeds1909198 and1909199. The one-sided
IID variance is reported as a diagnostic, not as a separately selected arm.
The necessary precision gate is mean per-output variance of a512-pair mean
<=1e-10 in BOTH batches. This reserves a small fraction of the approximately
5e-9 raw-error scale needed for the1e-9 adjusted goal. It is a declared local
engineering gate, not an impossibility theorem about other Gaussian rules.
Failure closes this sampled constructor without a sample/rank/seed sweep.

Use the symmetric part of the captured covariance and report its original
asymmetry. Require an ordinary positive-definite Cholesky factorization:
no PSD projection, diagonal loading, or correlation clipping. Check the
Gaussian marginal formulas against the packet. Hash every selected span
and the weight file. Report an optimistic dense arithmetic acquisition/read
cost; there is no claim of a complete flopscope deployment bill.

Run serially under the existing8GB/256MiB-swap process-group guard. Save a
small JSON receipt only. The accepted Cost150 submission stays byte exact.
