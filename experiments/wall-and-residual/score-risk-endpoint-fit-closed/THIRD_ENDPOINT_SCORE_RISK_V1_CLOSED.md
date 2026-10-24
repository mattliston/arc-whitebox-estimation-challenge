# Score-risk endpoint fit: closed

September 6, 2026. The new metric passed its separate independent-weight
audit, but changing the objective did not create enough useful prediction.
Same mini57/pre13 cell and H65 x H65 x R^(0,1,2) dictionary, ridge .001:

- Analytic mean-response risk residual .772535 (old fit .828883).
- Actual W14 mean-response residual .801489 (old fit .859562).
- Ordinary covariance residual .460450, old weighted-plane residual1.22507.
- Exactly zero marginal mean write; pair-kernel parity1.63e-19.
- Fixed1000-step solve true relative residual4.20655e-7, missing1e-7 gate.

No extra iterations were run. The one-normal-application, same-ridge
spectral displacement bound gives exact-optimum risk ratio >=.77252535,
far above the required .50. This bound is algebraic with a1.001 numerical
margin, not interval arithmetic, a statistical certificate, an
unregularized capacity bound, or a statement about all possible predictors.

The fit took20.61s, peak host RSS1,145,618,432 bytes. Fit finished05:22:42
PDT and bound05:28:12 PDT, September6,2026. No new labels or files downloaded,
no held-out access, carrier write, raw/adjusted score, or estimator change.
No regularizer, endpoint rank, metric or solver-budget continuation.

Fit receipt: third_endpoint_score_risk_wide_v1.json.
Output SHA256 823935bc1a2fff311c0092dd3cbc8327f524d358f9a6348a82a9704e637cfab3.
Bound: third_endpoint_score_risk_bound_v1.json.
