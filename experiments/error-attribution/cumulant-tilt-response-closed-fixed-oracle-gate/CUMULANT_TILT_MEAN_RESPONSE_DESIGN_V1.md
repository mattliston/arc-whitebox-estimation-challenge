# Cumulant-derived covariance response to a mean correction

September 5, 2026, Phase 2. A different mean-feedback law, not interpolation
between the closed centered/consistent feedback endpoints.

For exponential reweighting exp(eta dot X), derivatives of the cumulant
generating function give d mean = C eta and d covariance = T3 contracted
with eta. Thus eta=C^{-1} delta_mean supplies a first-order distribution
response to a mean correction. The existing raw-second-preserving update
and deterministic translation are different distribution assumptions.

First an oracle information test, not a new estimator: rank35 post12,
same true previous-m/C host, true1e9 mean correction on the existing active
neurons. Construct the CURRENT host post-third tensor without applying
the current oracle mean/covariance correction. Require positive definite
active covariance; solve exactly, no tuned ridge or damping. Check small
tilt energy and the mean equation numerically.

Query only diag(W13^T (T3 contracted eta) W13). CP projected factors and
the coordinate-star product are already next-layer consumers; this query
has a possible cheap implementation but no complete price is assumed.
The trace component is included exactly. First certify against a small
dense tensor and the actual derivative of a finite-distribution tilt.

Judge the same next-Gaussian-mean linear-response error using the two
existing independent8M post12 covariance streams. Centered host covariance
is the baseline; compare at natural coefficient1. Pass only if weighted
signal ratio<=.75, both stream ratios<=.85, .5*delta^T eta<=.01 and max
relative next-variance change<=.10. Failure closes this response without
regularization/strength/source/cell sweeps. Success still needs replication,
complete inference price, honest learned-mean response and release gates.

This diagnostic uses current mean truth explicitly; it is NOT a deployable
correction, feature, score, or authorization to fit on leaked inputs.
