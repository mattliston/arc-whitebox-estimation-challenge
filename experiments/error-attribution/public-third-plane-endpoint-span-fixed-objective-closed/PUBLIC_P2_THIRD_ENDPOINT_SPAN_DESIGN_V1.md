# Frozen recurrent endpoint span for the missing third plane

September6,2026. One admitted TRAINING cell only: mini57/pre13. No
cross-network training, new sample, extra download or state mutation.
This differs from the closed history decoder of total fresh covariance
errors: the target is now the missing full off-diagonal K21 plane on the
actual on-policy bank trajectory, using newly qualified 100M labels.
Marginal K3 is held fixed; no per-unit variance/moment head is retried.

Features F=[1,H64] are the actual bank's frozen current recurrent state
before mean feedback. Obtain an equivalent input-only orthonormal basis
by SVD, retaining singular values >1e-12 times the largest. Normalize
off-diagonal R^r, r=0,1,2, by their input RMS. Predict standardized plane

    S_ij = sum_r Rpower_r,ij (F_i^T Theta_r F_j), diag(S)=0.

The three coefficient matrices are unrestricted (K21 is not symmetric).
Use the exact frozen off-diagonal third-to-covariance kernel L, where
DeltaC=L*S+(L*S)^T. Weight the K21 ridge objective by
(L^2+(L^T)^2) normalized to mean1. Fixed ridge .001, zero start,
matrix-free PCG tolerance1e-7, at most300 iterations, diagonal preconditioner.
No feature, rank, regularization, weight, iteration, target or cell sweep.
This is a fixed regularized in-sample oracle-coefficient experiment, NOT
an unregularized capacity ceiling or a generalizing predictor.

Before fitting: qualify the weighted adjoint/solve against a dense small
system; certify L against the existing exact finite pair substitution.
Freeze actual pre14 Gaussian variance-to-mean response in a separate
target-free parent replay that reproduces all original pre13 inputs and
prefix bytes. Existing data-qualified targets do not enter that replay.

Evaluate natural full-strength predicted K21 increments through the same
exact third-only covariance and next-mean consumers; target is full true
off-diagonal K21 minus actual bank K21, all other inputs fixed. Require
both ordinary response error ratios <=.50, converged normal solve, and
no diagonal/marginal write. Report input-weighted plane error too.
No independent replicate is available for these public labels; do not
noise-correct or claim a confidence-certified ceiling.

Fail closes this endpoint dictionary/objective on the specified source;
no remaining seven-file download or six/two fit. Pass justifies a separate
cross-network test with this already frozen split and later full-carrier
and release qualification. No adjusted/raw score or bank follows here.

Clarification of prior receipt field mean_score_targets_decoded=false:
the official1e9 gt_mean was not decoded and no final score was calculated.
The100M mean array WAS decoded for covariance centering/affine consistency,
including its unused final-layer row. It should not be described as zero
post-mean label access. No held-out file was downloaded or opened.
