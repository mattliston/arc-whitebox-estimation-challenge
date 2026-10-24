# Current degree-two moment-matrix consistency: target-free audit

For any centered random vector Z with covariance C, let U_i=Z_i^2-C_ii.
Then Cov(U_i,Z_j)=K21_ij and Cov(U_i,U_j)=K22_ij+2*C_ij^2.
The block covariance of (Z,U) is positive semidefinite. If C is positive
definite, its Schur complement must satisfy

    K22 + 2*(C*C) - K21 @ inverse(C) @ K21.T >= 0.

This is an exact necessary moment condition, not a Gaussian assumption.
Existing PSD checks examined C alone, not this joint third/fourth condition.
The current fourth-order plane is radial; the full carried third plane
could violate this stronger condition even if C is positive definite.

Read only the already frozen current-parent pre-state tapes for public
ranks 35/100, layers 4/8/12/15. Restrict to the same active output indices
so intentionally absent cold cross-plane entries are not treated as data.
Standardize by marginal sigma; record the covariance eigenvalues, scalar
Cauchy ratios, and negative Schur eigenvalue energy/trace. Use a symmetric
positive-definite solve when allowed by the actual spectrum; do not silently
regularize or repair an indefinite/near-singular input.

No truth labels, model changes, fits, eigenspace truncation or estimator
port. If the exact necessary condition holds to numerical tolerance, close
this proposed moment-PSD repair without trying correction strengths.
If it fails materially, inspect conditioning and independently check a
small true-state moment block before designing a repair. A violation by
itself does not identify whether K3 or K4 is wrong or establish a raw gain.
