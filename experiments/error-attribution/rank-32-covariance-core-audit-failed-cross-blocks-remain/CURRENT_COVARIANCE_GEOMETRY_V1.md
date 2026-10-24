# Current-bank covariance geometry, read only

September 5, 2026. Ranks35/100, frozen matched full-pair learned host,
trajectory byte-identical to each saved reference. Archived pre-covariance
truth has 8M samples; cov/cov2 are IDENTICAL, not independent replicas.
No oracle values enter features or change the trajectory.

At layer15, retaining covariance-error terms with AT LEAST ONE index in
the top8 host covariance eigenvectors captures 81.23% /83.41% of observed
Frobenius error. For rank35, keeping BOTH indices in that subspace captures
only 8.06%. The corresponding top8 cross-term diagonal correlates .882
with separate32M variance-error labels; top32 .938. These are observed
geometry, not a deployable correction or a noise-corrected ceiling.

Gaussian sampling-energy estimates are 3.30% /2.99% of observed layer15
covariance error (fourth-cumulant sampling contribution omitted). At
layer4, sampling is ~84%, so early residual spectra should NOT be interpreted.
Raw-second-moment error is larger than centered-covariance error, showing
substantial cancellation with mean centering rather than an obvious
universal covariance coefficient.

Receipts: `current_covariance_error_r35_v1.json` and rank100 counterpart.
This supports targeting cross-direction interactions, not repeating the
closed projection of all state into a leading covariance block.
