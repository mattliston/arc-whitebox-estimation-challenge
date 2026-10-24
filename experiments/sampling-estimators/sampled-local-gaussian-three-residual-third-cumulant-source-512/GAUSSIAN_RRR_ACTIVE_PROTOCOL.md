# Active-reference Gaussian RRR precision screen

The strict full-covariance experiment failed BEFORE sampling. Preserve that
failure. The separate spectrum diagnostic finds33 negative eigenvalues with
only2.94e-13 of squared covariance energy; the incoming active896x896 block
is strictly positive, while16 omitted-coordinate variances are slightly
negative. An FP64 push alone does not restore the full Gaussian domain.

This second experiment uses a declared different reference:

    Z = W8_active.T L_active epsilon,
    L_active L_active.T = symmetric(C7_active).

No eigensolver repair, diagonal loading, or clipping is used. This reference
omits the old inactive variances completely, including positive ones. It is
an approximate Gaussian source and makes no claim of equality to captured
Cpre or the physical network distribution. The covariance difference is
reported. Marginal Gaussian means/gates are recomputed for this reference.
The nonlinear residual is masked by the saved layer8 alive mask. Queries
are the COLUMNS of W9, since the estimator applies W.T to column vectors.
The initial unexecuted sampling branch mistakenly used rows; v1 failed
before that branch and supplies no variance numbers. Its original source
and failure receipt remain intact.

The same prespecified two512-pair batches, seeds1909198/1909199, scalar
consumer response and1e-10 per-output mean-variance gate apply unchanged.
No truth or accuracy results informed this reference definition. Failures
close this sampled constructor without a sample/seed/reference sweep.
Passing precision alone would still leave signed value, overlap subtraction,
complete cost and estimator deployment unqualified.

This is a bounded research diagnostic, never an estimator payload. Keep all
source/result files and the accepted submission unchanged.
