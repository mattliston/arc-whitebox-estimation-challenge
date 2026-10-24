# Addendum: exact layer-1 prediction is at the reference-noise floor

This file is additive.  It does not modify or supersede the frozen
`DIAGNOSTIC.md` whose SHA-256 is
`1a0e3b63c454b543793d2e3195a61eda638b1a958f27651980093f62ec1acb0a`.

Across the target-blind mini100 first-layer weights, the analytic post-ReLU
variance predicts a `10^9`-sample reference-noise MSE of
`6.818022375182014e-10`.  The sealed score is
`6.881554782955935e-10`, only `0.9318%` higher.  CP1792's stored layer-1
prediction agrees with the exact Gaussian formula to MSE
`2.944948632584076e-16` and maximum absolute difference
`2.9802219692065535e-8`, consistent with float32 rounding.

Consequently, a learned layer-1 correction would fit the published Monte Carlo
reference noise rather than improve the mathematical expectation.  The
observer must keep its layer-1 correction structurally zero.

Evidence:

- `L1_NOISE_FLOOR.json` SHA-256
  `44b5018dfaafcd433aaa8319385bc4b3edef3758c6678e5529affdb16f82db4a`
- `verify_l1_noise_floor.py` SHA-256
  `067769ecfbc834d509decbb6f769bde76a3f2b058865232b7c3133f2ef6bb0e1`
- no new target iterator call, target decode, score computation, or fit.
