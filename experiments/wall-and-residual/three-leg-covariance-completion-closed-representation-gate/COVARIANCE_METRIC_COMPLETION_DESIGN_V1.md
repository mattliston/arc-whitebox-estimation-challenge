# Three-leg covariance-metric repeated-plane completion

Phase2 only, September5 2026,22:48PDT. Target-free representation test.

The earlier covariance-weighted star used factors(C, identity, A) and a
two-by-two solve. This proposal uses covariance in ALL THREE tensor legs:

  T_ijk = sum_ab H_ab (R_ia R_ja R_kb + R_ia R_ka R_jb
                       + R_ja R_ka R_ib).

R is the current post covariance standardized on the existing live block.
The repeated plane is the linear, symmetric positive-definite operator

  M(H) = [(R*R)@H + 2*R*(R@H.T)]@R.

It is a Gaussian-metric minimum-norm tensor completion, not a normalizable
exponential-cubic probability model. Positivity follows by contracting
the symmetric repeated-coordinate tensor with R in all three legs. The
first term supplies a Kronecker preconditioner with inverse
solve(R*R, K)@inverse(R). No dense n^3 tensor is required in application.
Store two rank-n factors, U=R and V=R@H.T. Physical sigma factors are put
back on their rows. The remaining repeated-plane discrepancy is restored
by the existing coordinate star. Original non-CP star and trace tensors
are unchanged. This changes the CP all-distinct completion only.

Before any width1024 work, independently materialize synthetic n6/n9
dense tensors and verify the plane, operator self-adjointness, positive
quadratic form, factor push/read, and coordinate-star restoration to
1e-10 relative. Covariances must be positive definite, no relaxed checks.

One fixed current-bank cell: rank35 post12 -> actual W13. No mean/cov
truth injection and no labels. Reconstruct the actual frozen P/Q prefix,
capture CP factors, covariance and original star/trace. Retain existing
live support and all repeated CP entries by star restoration. Compare
next K21 on the fixed32 rows0:32:1024 against the exact original tensor,
weighted by the existing Gaussian pair response of the actual next state.

Two PREDECLARED solver outputs serve different purposes: two
preconditioned conjugate-gradient iterations are the sole practical arm;
up to30 iterations to1e-10 residual are a representation reference, not a
candidate or permission to increase inference iterations. Report both
linear residual and next-query error. The practical arm must have <=3%
weighted next-K21 relative error. Failure closes this two-step completion
without rollout, rank/iteration/regularizer sweep, training or raw scoring.
If R is not SPD, stop and report it; do not introduce a fitted/jitter repair.

A pass is ONLY representation fidelity, not final raw accuracy. Complete
metered-like cost including factorization, solve, every PCG matmul/dot,
restoration, allocations, factor pushes and reads must save >=20% of the
whole current bank on BOTH tapes before a full-depth pilot. Then all
co-adaptation,14-net and persistent-runtime release gates still apply.
One nice10 CPU process, at most4 BLAS threads. No forum writes or uploads.
