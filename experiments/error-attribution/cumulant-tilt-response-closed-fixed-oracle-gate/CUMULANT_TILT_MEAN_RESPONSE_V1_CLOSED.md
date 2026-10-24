# Cumulant tilt response: closed at the fixed oracle gate

September 5, 2026, 20:32 PDT. Phase 2 only; no estimator or submission.

The exponential-tilt response uses the exact distributional identities
dm = C eta and dC = T3 contracted with eta. At rank35, post-layer12,
eta solves the active covariance system against the independent 1B-sample
mean correction. The next-layer variance query uses the existing complete
CP, coordinate-star and trace representation. It does not update the
current third state with the target correction.

Small exact-tensor and finite-probability derivative certificates passed.
The 832-dimensional active covariance is positive definite; solve relative
error is 7.28e-15, tilt energy .0002412, and the largest relative next
variance change is .001189. This is a small, numerically well-resolved
perturbation, not a singular solve.

Against the two independent 8M covariance labels, the value-weighted
signal-error ratio is 1.1402001, with individual ratios 1.0907013 and
1.1075480 and cosine .03065. The preregistered .75/.85/.85 gate fails.
Do not expand to training, other strengths, ridges or selected cells.

Receipt: cumulant_tilt_response_r35_l12_v1.json. This oracle diagnostic
does not change the official bank or provide a raw/adjusted score.
